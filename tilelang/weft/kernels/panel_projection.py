"""Complete packed projections with invocation-local, shared integer panels."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows
from .decoding import ternary_digits


@tilelang.jit(target="weft")
def projection(M, N, K, format="Q2_K", rows=16, columns=16,
               engine="auto", extreme_policy="first", inverse_policy="direct", pairwise=False,
               unroll_decode=False):
    """Decode each record once; retain integer groups and all preparation in-call."""
    assert format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL",
                      "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "TQ1_0", "TQ2_0")
    assert M > 0 and N > 0 and M % rows == N % columns == 0
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    assert K > 0 and K % elements == 0
    short_affine = format in ("Q4_1", "Q5_1")
    k_affine = format in ("Q2_K", "Q4_K", "Q5_K")
    activation_format = "Q8_1" if short_affine else "Q8_0" if elements == 32 else "Q8_K"
    partner = getattr(F, activation_format)
    weight_bytes = K // elements * (encoding.schema.storage_bits // 8)
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)
    width = 16 if format in ("Q2_K", "Q3_K", "Q6_K") else 32
    assert not pairwise or (elements == 256 and engine == "vector")
    panel_dtype = "int16" if pairwise or format in ("Q4_K", "Q5_K", "Q6_K") else "uint8"
    bias = {"Q4_0": 8, "Q5_0": 16, "Q8_0": 128, "IQ4_NL": 128,
            "Q3_K": 124, "TQ1_0": 1, "TQ2_0": 1}.get(format, 0)
    if pairwise:
        bias = 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Panel: T.Tensor((elements // 2, columns, 2) if pairwise else (columns, elements), panel_dtype),
        PackedActivation: T.Tensor((elements // 2, M, 2) if pairwise else (M, elements), "int16" if pairwise else "int8"),
        ActivationSums: T.Tensor((M,) if bias else (0,), "int32"),
        Codebook: T.Tensor((16 if format == "IQ4_NL" else 0,), "int8"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Quantized": "workspace", "Output": "output", "Codebook": "input",
                     "Panel": "weights", "PackedActivation": "activation", "ActivationSums": "sums"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if pairwise:
            panel = T.view(Panel, shape=(elements // 2, columns, 2), axes=("pair", "n", "term"), abi=True)
            packed_activation = T.view(PackedActivation, shape=(elements // 2, M, 2), axes=("pair", "m", "term"), abi=True)
        else:
            panel = T.view(Panel, shape=(columns, elements), axes=("n", "k"), abi=True)
            packed_activation = T.view(PackedActivation, shape=(M, elements), axes=("m", "k"), abi=True)
        if bias:
            activation_sums = T.view(ActivationSums, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, activation_format, partner.schema.elements,
                          extreme_policy, inverse_policy, "rne" if elements == 256 else "dynamic", 64)
            if format == "IQ4_NL":
                table = T.alloc_local((16,), "int8")
                T.copy(Codebook, table)
            first = T.alloc_local((1,), "bool")
            first[0] = True
            with T.level.blocks(K, extent=elements) as block:
                for m in T.serial(M):
                    activation_codes = T.load(prepared[m, block].q)
                    if pairwise:
                        widened = T.alloc_local((elements,), "int16")
                        for k in T.Parallel(elements):
                            widened[k] = T.cast(activation_codes[k], "int16")
                        T.copy(T.reshape(widened, (elements // 2, 2)), packed_activation[:, m, :])
                    else:
                        T.copy(activation_codes, packed_activation[m, :])
                    if bias:
                        stored_sums = T.load(prepared[m, block].bsum) if elements == 256 else activation_codes
                        widened = T.alloc_local((16 if elements == 256 else 32,), "int32")
                        total_sum = T.alloc_local((1,), "int32")
                        for k in T.Parallel(16 if elements == 256 else 32):
                            widened[k] = T.cast(stored_sums[k], "int32")
                        T.reduce_sum(widened, total_sum, dim=0)
                        T.copy(total_sum, activation_sums[m])
                with T.level.cols(N, group=columns) as col:
                    wd = T.alloc_local((columns,), "float32")
                    if short_affine:
                        wm = T.alloc_local((columns,), "float32")
                    if k_affine:
                        dmin = T.alloc_local((columns,), "float32")
                        minimum_values = T.alloc_local((columns, 16), "int16")
                    for c in T.unroll(columns) if unroll_decode else T.serial(columns):
                        weight = T.load(weights[col, block][c, :])
                        d = weight.d
                        wd[c] = T.cast(d[()], "float32")
                        if short_affine:
                            minimum = weight.m
                            wm[c] = T.cast(minimum[()], "float32")
                        if k_affine:
                            minimum_scale = weight.dmin
                            dmin[c] = T.cast(minimum_scale[()], "float32")
                        if format == "Q2_K":
                            metadata = weight.scales
                        elif format == "Q3_K":
                            scale_low, scale_high = weight.scale_low, weight.scale_high
                        elif format in ("Q4_K", "Q5_K"):
                            scales, minima = weight.sc, weight.m
                        elif format == "Q6_K":
                            scales = weight.scales
                        low = weight.ql if format == "Q6_K" else weight.q
                        if format in ("Q5_0", "Q5_1", "Q5_K", "Q6_K"):
                            high = weight.qh
                        elif format == "Q3_K":
                            high = weight.hmask
                        if format == "TQ1_0":
                            digits = T.alloc_local((1, elements), "uint2")
                            ternary_digits(weight, digits, 1, single=True)
                        codes = T.alloc_local((elements,), panel_dtype)
                        for k in T.Parallel(elements):
                            if format in ("Q5_0", "Q5_1", "Q5_K", "Q6_K"):
                                code = T.cast(low[k], "int16") | (T.cast(high[k], "int16") << T.int16(4))
                                code -= 16 if format == "Q5_0" else 32 if format == "Q6_K" else 0
                            elif format == "Q3_K":
                                code = T.cast(low[k], "int16") + T.cast(high[k], "int16") * 4 - 4
                            elif format == "IQ4_NL":
                                code = T.cast(table[T.cast(low[k], "uint8")], "int16")
                            elif format == "TQ1_0":
                                code = T.cast(digits[0, k], "int16") - 1
                            else:
                                code = T.cast(low[k], "int16") - (8 if format == "Q4_0" else 1 if format == "TQ2_0" else 0)
                            group = T.cast(k, "uint32") // T.uint32(width)
                            if format == "Q2_K":
                                coefficient = T.cast(metadata[group] & T.uint8(15), "int16")
                            elif format == "Q3_K":
                                coefficient = (T.cast(scale_low[group], "int16") | (T.cast(scale_high[group], "int16") << T.int16(4))) - 32
                            elif format in ("Q4_K", "Q5_K", "Q6_K"):
                                coefficient = T.cast(scales[group], "int16")
                            else:
                                coefficient = 1
                            codes[k] = T.cast(code * coefficient + bias, panel_dtype)
                        if pairwise:
                            T.copy(T.reshape(codes, (elements // 2, 2)), panel[:, c, :])
                        else:
                            T.copy(codes, panel[c, :])
                        if k_affine:
                            for g in T.Parallel(16):
                                minimum_values[c, g] = T.cast(metadata[g] >> T.uint8(4), "int16") if format == "Q2_K" else T.cast(minima[T.cast(g, "uint32") // T.uint32(2)], "int16")
                    if not pairwise:
                        weight_values = T.load(panel)
                    with T.level.rows(M, group=rows) as row:
                        if pairwise:
                            products = T.state((rows, columns), "int32", init=0)
                            for pair in T.serial(elements // 2):
                                weight_values = T.load(panel[pair, :, :])
                                activation_values = T.load(packed_activation[pair, row, :])
                                partial = T.alloc_local((rows, columns, 1), "int32")
                                T.mac_groups(activation_values, weight_values, partial, n=2, into="int32", over="term")
                                for r, c in T.Parallel(rows, columns):
                                    products[r, c] += partial[r, c, 0]
                        else:
                            activation_values = T.load(packed_activation[row, :])
                            products = T.alloc_local((rows, columns), "int32")
                            T.reduce_dot(activation_values, weight_values, products, over="k", acc_dtype="int32", engine=engine)
                        activation = prepared[row, block]
                        xd = T.load(activation.d if elements == 32 else activation.ds)
                        if bias:
                            sums = T.load(activation_sums[row])
                        if k_affine:
                            block_sums = T.load(activation.bsum)
                            correction = T.alloc_local((rows, columns), "int32")
                            T.reduce_dot(block_sums, minimum_values, correction, over="k", acc_dtype="int32", engine=engine)
                        elif short_affine:
                            xs = T.load(activation.s)
                        result = T.alloc_local((rows, columns), "float32")
                        for r, c in T.Parallel(rows, columns):
                            integer = products[r, c] - bias * sums[r] if bias else products[r, c]
                            result[r, c] = T.cast(integer, "float32") * (T.cast(wd[c], "float32") * T.cast(xd[r], "float32"))
                            if k_affine:
                                result[r, c] -= T.cast(correction[r, c], "float32") * (T.cast(dmin[c], "float32") * xd[r])
                            elif short_affine:
                                result[r, c] += T.cast(wm[c], "float32") * T.cast(xs[r], "float32")
                        if not first[0]:
                            previous = T.load(output[row, col])
                            for r, c in T.Parallel(rows, columns):
                                result[r, c] += previous[r, c]
                        T.copy(result, output[row, col])
                first[0] = False

    return main


@tilelang.jit(target="weft")
def staged_affine(M, N, K, format="Q4_K", staged_columns=64, block_k=256,
                  batch_rows=16, columns=16, rows=1, partials=8):
    """Reuse an encoded weight stage across rows without a decoded panel."""
    assert format in ("Q4_K", "Q5_K")
    assert M > 0 and N > 0 and K > 0
    assert M % batch_rows == 0 and N % staged_columns == 0
    assert rows > 0 and batch_rows % rows == 0
    assert partials in (1, 8)
    assert K % block_k == 0 and block_k % 256 == 0 and staged_columns % columns == 0
    encoding = getattr(F, format)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K // 256 * (encoding.schema.storage_bits // 8)), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Quantized": "quantized", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_K", 256, "first", "direct", "rne", 64)
            if block_k != K:
                with T.level.rows(M, group=1) as row:
                    with T.level.cols(N, group=columns) as col:
                        zero = T.alloc_local((1, columns), "float32")
                        T.fill(zero, 0)
                        T.copy(zero, output[row, col])
            with T.level.tiles(N, extent=staged_columns, axis="n") as weight_columns:
                with T.level.tiles(K, extent=block_k, axis="k") as weight_blocks:
                    staged = T.stage(T.load(weights[weight_columns, weight_blocks]))
                    with T.level.rows(M, group=batch_rows) as batch:
                        staged_activation = T.stage(T.load(prepared[batch, weight_blocks]))
                        with T.level.cols(weight_columns, group=columns) as col:
                            with T.level.rows(batch, group=rows) as row:
                                if block_k == K:
                                    total = T.state((rows, columns), "float32", init=0)
                                else:
                                    stored = T.load(output[row, col])
                                    total = T.state((rows, columns), "float32", init=stored)
                                with T.level.blocks(weight_blocks, extent=256) as block:
                                    w = T.load(staged[col, block])
                                    x = T.load(staged_activation[row, block])
                                    integer = T.state((rows, columns), "int32", init=0)
                                    with T.level.subtiles(block, extent=32) as sub:
                                        factors = T.load(w.sc, at=sub)
                                        low = T.load(w.q, at=sub)
                                        inputs = T.load(x.q, at=sub)
                                        if format == "Q5_K":
                                            upper = T.load(w.qh, at=sub)
                                            codes = T.alloc_local((columns, 32), "uint8")
                                            for c, k in T.Parallel(columns, 32):
                                                codes[c, k] = T.cast(low[c, k], "uint8") | (T.cast(upper[c, k], "uint8") << T.uint8(4))
                                        else:
                                            codes = low
                                        products = T.alloc_local((rows, columns), "int32")
                                        if partials == 1:
                                            T.gemm(inputs, codes, products, transpose_B=True, clear_accum=True, engine="vector")
                                        else:
                                            partial = T.alloc_local((rows, columns, 8), "int16")
                                            T.mac_groups(inputs, codes, partial, n=4, into="int16", over="k")
                                            widened = T.alloc_local((rows, columns, 8), "int32")
                                            for i, j, p in T.Parallel(rows, columns, 8):
                                                widened[i, j, p] = T.cast(partial[i, j, p], "int32")
                                            T.reduce_sum(widened, products, dim=2)
                                        for r, c in T.Parallel(rows, columns):
                                            integer[r, c] += products[r, c] * T.cast(factors[c], "int32")
                                    minima, d, dmin, ds = w.m, w.d, w.dmin, x.ds
                                    paired_sums = T.sum_pairs(x.bsum)
                                    correction = T.alloc_local((rows, columns), "int32")
                                    T.gemm(paired_sums, minima, correction, transpose_B=True, clear_accum=True, engine="vector")
                                    for r, c in T.Parallel(rows, columns):
                                        total[r, c] += ds[r] * (T.cast(d[c], "float32") * T.cast(integer[r, c], "float32") - T.cast(dmin[c], "float32") * T.cast(correction[r, c], "float32"))
                                T.copy(total, output[row, col])

    return main
