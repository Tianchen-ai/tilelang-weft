"""Quantized projections with explicit interleaved weights and live tile sums."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F, layouts
from .preparation import quantize_rows


@tilelang.jit(target="weft")
def staged_projection(M, N, K, format="Q4_0", batch_rows=16, staged_columns=32,
                      rows=4, columns=1, extreme_policy="first", inverse_policy="direct"):
    """Reuse natural encoded stages while keeping each output tile live over K."""
    assert format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "MXFP4")
    assert min(M, N, K, batch_rows, staged_columns, rows, columns) > 0
    assert M % batch_rows == N % staged_columns == K % 32 == 0
    assert batch_rows % rows == staged_columns % columns == 0
    encoding = getattr(F, format)
    affine = format in ("Q4_1", "Q5_1")
    lookup = format in ("IQ4_NL", "MXFP4")
    partner = F.Q8_1 if affine else F.Q8_0
    weight_bytes = K // 32 * (encoding.schema.storage_bits // 8)
    activation_bytes = K // 32 * (partner.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if format == "MXFP4" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Codebook": "input", "Scale": "input",
                     "Quantized": "quantized", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if lookup:
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        if format == "MXFP4":
            scale_table = T.view(Scale, shape=(256,), axes=("exponent",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if lookup:
                table = T.stage(T.load(codebook))
            quantize_rows(source, prepared, M, K, "Q8_1" if affine else "Q8_0", 32,
                          extreme_policy, inverse_policy, "dynamic", 32)
            with T.level.tiles(N, extent=staged_columns, axis="n") as nc:
                weight_stage = T.stage(T.load(weights[nc, :]))
                with T.level.tiles(M, extent=batch_rows, axis="m") as mc:
                    activation_stage = T.stage(T.load(prepared[mc, :]))
                    with T.level.cols(nc, group=columns) as col:
                        with T.level.rows(mc, group=rows) as row:
                            total = T.state((rows, columns), "float32", init=0)
                            with T.level.blocks(K, extent=32) as block:
                                weight = T.load(weight_stage[col, block])
                                activation = T.load(activation_stage[row, block])
                                low, inputs, xd = weight.q, activation.q, activation.d
                                if format in ("Q5_0", "Q5_1"):
                                    high = weight.qh
                                if lookup:
                                    codes = T.load(table, at=low, bounds="in_bounds")
                                else:
                                    codes = T.alloc_local((columns, 32), "int8")
                                    for c, k in T.Parallel(columns, 32):
                                        if format in ("Q5_0", "Q5_1"):
                                            code = T.cast(low[c, k], "uint8") | (T.cast(high[c, k], "uint8") << T.uint8(4))
                                            codes[c, k] = T.cast(code, "int8") - T.int8(16 if format == "Q5_0" else 0)
                                        else:
                                            codes[c, k] = T.cast(low[c, k], "int8") - T.int8(8 if format == "Q4_0" else 0)
                                products = T.alloc_local((rows, columns), "int32")
                                T.reduce_dot(inputs, codes, products, over="k", acc_dtype="int32", engine="vector")
                                if format == "MXFP4":
                                    wd = T.load(scale_table, at=weight.e, bounds="in_bounds")
                                else:
                                    wd = weight.d
                                if affine:
                                    minimum, activation_sum = weight.m, activation.s
                                for r, c in T.Parallel(rows, columns):
                                    contribution = (T.cast(xd[r], "float32") * T.cast(wd[c], "float32")) * T.cast(products[r, c], "float32")
                                    if affine:
                                        contribution += T.cast(minimum[c], "float32") * T.cast(activation_sum[r], "float32")
                                    total[r, c] += contribution
                            T.copy(total, output[row, col])

    return main


@tilelang.jit(target="weft")
def projection(M, N, K, format="Q4_0", rows=4, columns=16,
               extreme_policy="first", inverse_policy="direct"):
    """Consume persistent row interleave; prepare activation inside this call.

    Each output tile retains its floating accumulator across all weight records.
    The encoding, cohort and separate preparation phase are author choices.
    Local vector layout and instruction selection remain compiler decisions.
    """
    assert format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "MXFP4")
    assert M > 0 and N > 0 and K > 0 and K % 32 == 0
    assert M % rows == 0 and N % columns == 0
    encoding = getattr(F, format)
    persistent = getattr(layouts, format + "Rows")[columns]
    affine = format in ("Q4_1", "Q5_1")
    lookup = format in ("IQ4_NL", "MXFP4")
    bias = {"Q4_0": 8, "Q5_0": 16, "Q8_0": 128, "IQ4_NL": 128, "MXFP4": 128}.get(format, 0)
    partner = F.Q8_1 if affine else F.Q8_0
    weight_bytes = K // 32 * (encoding.schema.storage_bits // 8)
    activation_bytes = K // 32 * (partner.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Sums: T.Tensor((M, K // 32 * 2) if bias else (0,), "uint8"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if format == "MXFP4" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Quantized": "quantized", "Output": "output", "Sums": "sums",
                     "Codebook": "input", "Scale": "input"}})
        weights = T.view(Weights, shape=(N, K), encoding=persistent, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if bias:
            sums = T.view(Sums, shape=(M, K), encoding=layouts.BlockSum32, axes=("m", "k"), abi=True)
        if lookup:
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        if format == "MXFP4":
            scale_table = T.view(Scale, shape=(256,), axes=("exponent",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if lookup:
                table = T.load(codebook)
            quantize_rows(source, prepared, M, K, "Q8_1" if affine else "Q8_0", 32,
                          extreme_policy, inverse_policy, "dynamic", 32)
            if bias:
                for m in T.serial(M):
                    with T.level.blocks(K, extent=32) as block:
                        values = T.load(prepared[m, block].q)
                        widened = T.alloc_local((32,), "int16")
                        reduced = T.alloc_local((1,), "int16")
                        for k in T.Parallel(32):
                            widened[k] = T.cast(values[k], "int16")
                        T.reduce_sum(widened, reduced, dim=0)
                        T.copy(reduced, sums[m, block].value)
            with T.level.cols(N, group=columns) as col:
                with T.level.rows(M, group=rows) as row:
                    total = T.state((rows, columns), "float32", init=0)
                    correction = T.state((rows, columns), "float32", init=0)
                    with T.level.blocks(K, extent=32) as block:
                        weight, activation = weights[col, block], prepared[row, block]
                        low = T.load(weight.q)
                        if format in ("Q5_0", "Q5_1"):
                            high = T.load(weight.qh)
                        inputs = T.load(activation.q)
                        codes = T.alloc_local((columns, 32), "uint8")
                        products = T.alloc_local((rows, columns), "int32")
                        for c, k in T.Parallel(columns, 32):
                            if format in ("Q5_0", "Q5_1"):
                                codes[c, k] = T.cast(low[c, k], "uint8") | (T.cast(high[c, k], "uint8") << T.uint8(4))
                            elif lookup:
                                codes[c, k] = T.cast(table[T.cast(low[c, k], "uint8")], "uint8") ^ T.uint8(128)
                            elif format == "Q8_0":
                                codes[c, k] = T.cast(low[c, k], "uint8") ^ T.uint8(128)
                            else:
                                codes[c, k] = T.cast(low[c, k], "uint8")
                        T.reduce_dot(inputs, codes, products, over="k", acc_dtype="int32", engine="vector")
                        xd = T.load(activation.d)
                        if format == "MXFP4":
                            exponent = T.load(weight.e)
                            wd = T.load(scale_table, at=exponent, bounds="in_bounds")
                        else:
                            wd = T.load(weight.d)
                        if bias:
                            block_sums = T.load(sums[row, block].value)
                        if affine:
                            minimum, activation_sum = T.load(weight.m), T.load(activation.s)
                        for r, c in T.Parallel(rows, columns):
                            integer = products[r, c] - T.int32(bias) * T.cast(block_sums[r], "int32") if bias else products[r, c]
                            total[r, c] += (T.cast(xd[r], "float32") * T.cast(wd[c], "float32")) * T.cast(integer, "float32")
                            if affine:
                                correction[r, c] += T.cast(activation_sum[r], "float32") * T.cast(minimum[c], "float32")
                    if affine:
                        for r, c in T.Parallel(rows, columns):
                            total[r, c] += correction[r, c]
                    T.copy(total, output[row, col])

    return main


@tilelang.jit(target="weft")
def affine_projection(M, N, K, format="Q4_K", columns=16):
    """Keep grouped integer branches over an explicitly interleaved record tile."""
    assert format in ("Q2_K", "Q4_K", "Q5_K", "IQ4_XS")
    assert M > 0 and N > 0 and N % columns == 0 and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    persistent = {"Q2_K": layouts.Q2KRows, "Q4_K": layouts.Q4KRows,
                  "Q5_K": layouts.Q5KRows, "IQ4_XS": layouts.IQ4XSRows}[format][columns]
    terms = 16 if format == "Q2_K" else 32
    groups = 256 // terms
    record_bytes = encoding.schema.storage_bits // 8

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Codebook: T.Tensor((16 if format == "IQ4_XS" else 0,), "int8"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Codebook": "input", "Quantized": "quantized", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=persistent, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if format == "IQ4_XS":
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if format == "IQ4_XS":
                table = T.load(codebook)
                biased = T.alloc_local((16,), "uint8")
                for i in T.Parallel(16):
                    biased[i] = T.cast(table[i], "uint8") + T.uint8(128)
            quantize_rows(source, prepared, M, K, "Q8_K", 256, "first", "direct", "rne", 256)
            for m in T.serial(M):
                with T.level.cols(N, group=columns) as col:
                    total = T.state((columns,), "float32", init=0)
                    correction = T.state((columns,), "float32", init=0)
                    with T.level.blocks(K, extent=256) as block:
                        weight = weights[col, block]
                        x = T.load(prepared[m, block])
                        activation, sums, xd = x.q, x.bsum, x.ds
                        wd = T.load(weight.d)
                        if format != "IQ4_XS":
                            dmin = T.load(weight.dmin)
                        if format == "Q2_K":
                            metadata = T.load(weight.scales)
                        elif format == "IQ4_XS":
                            scale_low, scale_high = T.load(weight.scales_l), T.load(weight.scales_h)
                        else:
                            scales, minima = T.load(weight.sc), T.load(weight.m)
                        integer = T.state((columns,), "int32", init=0)
                        minimum = T.state((columns,), "int32", init=0)
                        for group in T.unroll(groups):
                            low = T.load(weight.q[:, group * terms : (group + 1) * terms])
                            if format == "Q5_K":
                                high = T.load(weight.qh[:, group * terms : (group + 1) * terms])
                            if format == "IQ4_XS":
                                codes = T.load(biased, at=low, bounds="in_bounds")
                            else:
                                codes = T.alloc_local((columns, terms), "uint6" if format == "Q5_K" else "uint4" if format == "Q4_K" else "uint2")
                                for c, k in T.Parallel(columns, terms):
                                    code = T.cast(low[c, k], "uint8")
                                    if format == "Q5_K":
                                        code |= T.cast(high[c, k], "uint8") << T.uint8(4)
                                    codes[c, k] = code
                            inputs = T.alloc_local((terms,), "int8")
                            for k in T.Parallel(terms):
                                inputs[k] = activation[T.uint32(group * terms) + T.cast(k, "uint32")]
                            products = T.alloc_local((columns,), "int32")
                            T.reduce_dot(codes, inputs, products, over="k", acc_dtype="int32", engine="vector")
                            for c in T.Parallel(columns):
                                if format == "Q2_K":
                                    factor = T.cast(metadata[c, group] & T.uint8(15), "int32")
                                    subtract = T.cast(metadata[c, group] >> T.uint8(4), "int32")
                                elif format == "IQ4_XS":
                                    lower = (scale_low[c, group // 2] >> T.uint8(group % 2 * 4)) & T.uint8(15)
                                    upper = (scale_high[c] >> T.uint16(group * 2)) & T.uint16(3)
                                    factor = T.cast(lower | (T.cast(upper, "uint8") << T.uint8(4)), "int32") - 32
                                    subtract = factor * 128
                                else:
                                    factor, subtract = T.cast(scales[c, group], "int32"), T.cast(minima[c, group], "int32")
                                group_sum = T.cast(sums[group], "int32") if terms == 16 else T.cast(sums[group * 2], "int32") + T.cast(sums[group * 2 + 1], "int32")
                                integer[c] += products[c] * factor
                                minimum[c] += group_sum * subtract
                        for c in T.Parallel(columns):
                            if format == "IQ4_XS":
                                total[c] += (T.cast(wd[c], "float32") * xd[()]) * T.cast(integer[c] - minimum[c], "float32")
                            else:
                                total[c] += (T.cast(wd[c], "float32") * xd[()]) * T.cast(integer[c], "float32")
                                correction[c] += (T.cast(dmin[c], "float32") * xd[()]) * T.cast(minimum[c], "float32")
                    result = T.alloc_local((columns,), "float32")
                    for c in T.Parallel(columns):
                        result[c] = total[c] - correction[c]
                    T.copy(result, output[m, col])

    return main
