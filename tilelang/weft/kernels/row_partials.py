"""Author-selected row partials, retaining vector sums across encoded blocks."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows


@tilelang.jit(target="weft")
def block32(M, K, format="Q4_0", activation="Q8_0", rows=1, lanes=8):
    assert format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "MXFP4")
    assert lanes in (8, 16) and K > 0 and K % 32 == 0
    encoding, partner = getattr(F, format), getattr(F, activation)
    weight_bytes, activation_bytes = K // 32 * (encoding.schema.storage_bits // 8), K // 32 * (partner.schema.storage_bits // 8)
    lookup = format in ("IQ4_NL", "MXFP4")
    affine = format in ("Q4_1", "Q5_1")
    centered = M == rows == 1 and format in ("Q4_0", "Q5_0")
    unsigned = format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0") and not centered
    code_type = "uint8" if unsigned else "int8"
    group, partial_type = 32 // lanes, "int16" if lanes == 16 and format != "Q8_0" else "int32"

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"),
        Activation: T.Tensor((activation_bytes,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if format == "MXFP4" else 0,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=partner, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if lookup:
                codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
                table = T.load(codebook)
            ones = T.alloc_local((32,), "uint1")
            T.fill(ones, 1)
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows, lanes), "float32", init=0)
                offset = T.state((rows,), "float32", init=0)
                with T.level.blocks(K, extent=32) as block:
                    w, x = T.load(weights[row, block]), T.load(values[block])
                    q, xq, xd = w.q, x.q, x.d
                    d = w.e if format == "MXFP4" else w.d
                    if format in ("Q5_0", "Q5_1"):
                        high = w.qh
                    if affine:
                        minimum, activation_sum = w.m, x.s
                    codes = T.alloc_local((rows, 32), code_type)
                    partial = T.alloc_local((rows, lanes), partial_type)
                    sums = T.alloc_local((lanes,), partial_type)
                    for r, k in T.Parallel(rows, 32):
                        if lookup:
                            codes[r, k] = table[T.cast(q[r, k], "uint8")]
                        elif format in ("Q5_0", "Q5_1"):
                            code = T.cast(q[r, k], "uint8") | (T.cast(high[r, k], "uint8") << T.uint8(4))
                            codes[r, k] = T.cast(code, "int8") - T.int8(16) if centered else code
                        elif format == "Q8_0":
                            codes[r, k] = T.cast(q[r, k], "uint8") ^ T.uint8(128)
                        else:
                            codes[r, k] = T.cast(q[r, k], "int8") - T.int8(8) if centered else T.cast(q[r, k], code_type)
                    T.mac_groups(codes, xq, partial, n=group, into=partial_type, over="k")
                    if not centered and format in ("Q4_0", "Q5_0", "Q8_0"):
                        T.mac_groups(ones, xq, sums, n=group, into=partial_type, over="k")
                    for r, p in T.Parallel(rows, lanes):
                        scale = Scale[T.cast(d[r], "uint8")] if format == "MXFP4" else T.cast(d[r], "float32")
                        integer = T.cast(partial[r, p], "int32")
                        if not centered and format in ("Q4_0", "Q5_0", "Q8_0"):
                            integer -= (8 if format == "Q4_0" else 16 if format == "Q5_0" else 128) * T.cast(sums[p], "int32")
                        total[r, p] += T.cast(integer, "float32") * (scale * T.cast(xd[()], "float32"))
                    if affine:
                        for r in T.Parallel(rows):
                            offset[r] += T.cast(minimum[r], "float32") * T.cast(activation_sum[()], "float32")
                result = T.alloc_local((rows,), "float32")
                T.reduce_sum(total, result, dim=1)
                for r in T.Parallel(rows):
                    result[r] += offset[r]
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def k_groups(M, K, format="Q4_K", rows=1, lanes=16):
    """Retain scaled pair partials and the affine correction independently."""
    assert format in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K") and lanes in (8, 16) and K > 0 and K % 256 == 0
    assert format not in ("Q2_K", "Q3_K") or lanes == 16
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)
    activation_bytes = K // 256 * (F.Q8_K.schema.storage_bits // 8)
    code_type = "uint6" if format in ("Q5_K", "Q6_K") else "uint4"

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"), Activation: T.Tensor((activation_bytes,), "uint8"), Output: T.Tensor((M,), "float32")
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            k_group_rows(weights, values, output, M, K, format, rows, lanes, code_type, "m")

    return main


@T.macro
def k_group_rows(weights, values, output, M, K, format, rows, lanes, code_type, axis):
    with T.level.rows(M, group=rows, axis=axis) as row:
        total = T.state((rows, lanes), "float32", init=0)
        if format in ("Q2_K", "Q4_K", "Q5_K"):
            minimum = T.state((rows, 16 if format == "Q2_K" else 8), "float32", init=0)
        with T.level.blocks(K, extent=256) as block:
            integer = T.state((rows, lanes), "int32", init=0)
            weight = weights[row, block]
            if format == "Q6_K":
                quantized, high, scales = T.load(weight.ql), T.load(weight.qh), T.load(weight.scales)
            elif format == "Q3_K":
                quantized, high = T.load(weight.q), T.load(weight.hmask)
                scale_low, scale_high = T.load(weight.scale_low), T.load(weight.scale_high)
                scales = T.alloc_local((rows, 16), "int8")
                for r, p in T.Parallel(rows, 16):
                    scales[r, p] = T.cast(
                        T.cast(scale_low[r, p], "uint8") | (T.cast(scale_high[r, p], "uint8") << T.uint8(4)), "int8"
                    ) - T.int8(32)
            elif format == "Q2_K":
                quantized, metadata = T.load(weight.q), T.load(weight.scales)
                dmin = T.load(weight.dmin)
                scales = T.alloc_local((rows, 16), "uint4")
                minima = T.alloc_local((rows, 16), "uint4")
                for r, p in T.Parallel(rows, 16):
                    scales[r, p] = T.cast(metadata[r, p], "uint4")
                    minima[r, p] = T.cast(metadata[r, p] >> T.uint8(4), "uint4")
            else:
                quantized, scales, minima = T.load(weight.q), T.load(weight.sc), T.load(weight.m)
                dmin = T.load(weight.dmin)
            d = T.load(weight.d)
            activation = T.load(values[block])
            x, xd, sums = activation.q, activation.ds, activation.bsum
            if format == "Q5_K":
                high = T.load(weight.qh)
            for group in T.unroll(64 // lanes):
                codes = T.alloc_local((rows, lanes * 4), code_type)
                inputs = T.alloc_local((lanes * 4,), "int8")
                products = T.alloc_local((rows, lanes * 2), "int16")
                factors = T.alloc_local((rows, lanes * 2), "int16")
                scaled = T.alloc_local((rows, lanes), "int32")
                for k in T.Parallel(lanes * 4):
                    inputs[k] = x[T.uint32(group * lanes * 4) + T.cast(k, "uint32")]
                for r, k in T.Parallel(rows, lanes * 4):
                    coordinate = T.uint32(group * lanes * 4) + T.cast(k, "uint32")
                    if format in ("Q5_K", "Q6_K"):
                        codes[r, k] = T.cast(quantized[r, coordinate], "uint8") | (T.cast(high[r, coordinate], "uint8") << T.uint8(4))
                    elif format == "Q3_K":
                        codes[r, k] = T.cast(quantized[r, coordinate], "uint8") | (T.cast(high[r, coordinate], "uint8") << T.uint8(2))
                    else:
                        codes[r, k] = quantized[r, coordinate]
                T.mac_groups(codes, inputs, products, n=2, into="int16", over="pair")
                for r, p in T.Parallel(rows, lanes * 2):
                    factors[r, p] = T.cast(
                        scales[
                            r,
                            T.uint32(group * lanes // (4 if format in ("Q2_K", "Q3_K", "Q6_K") else 8))
                            + T.cast(p, "uint32") // T.uint32(8 if format in ("Q2_K", "Q3_K", "Q6_K") else 16),
                        ],
                        "int16",
                    )
                T.mac_groups(products, factors, scaled, n=2, into="int32", over="pair")
                for r, p in T.Parallel(rows, lanes):
                    integer[r, p] += scaled[r, p]
            factors = T.alloc_local((rows, 16), "int16")
            sum_values = T.alloc_local((16,), "int16")
            correction = T.alloc_local((rows, lanes if format in ("Q2_K", "Q3_K", "Q6_K") else 8), "int32")
            for p in T.Parallel(16):
                sum_values[p] = sums[T.cast(p, "uint32")]
            for r, p in T.Parallel(rows, 16):
                factors[r, p] = T.cast(
                    scales[r, p]
                    if format in ("Q3_K", "Q6_K")
                    else minima[r, p]
                    if format == "Q2_K"
                    else minima[r, T.cast(p, "uint32") // T.uint32(2)],
                    "int16",
                )
            if format in ("Q2_K", "Q3_K", "Q6_K") and lanes == 16:
                for r, p in T.Parallel(rows, 16):
                    correction[r, p] = T.cast(factors[r, p], "int32") * T.cast(sum_values[p], "int32")
            else:
                T.mac_groups(factors, sum_values, correction, n=2, into="int32", over="min_pair")
            if format in ("Q3_K", "Q6_K"):
                for r, p in T.Parallel(rows, lanes):
                    integer[r, p] -= correction[r, p] * (4 if format == "Q3_K" else 32)
            else:
                for r, p in T.Parallel(rows, 16 if format == "Q2_K" else 8):
                    minimum[r, p] += (T.cast(dmin[r], "float32") * xd[()]) * T.cast(correction[r, p], "float32")
            for r, p in T.Parallel(rows, lanes):
                total[r, p] += (T.cast(d[r], "float32") * xd[()]) * T.cast(integer[r, p], "float32")
        result = T.alloc_local((rows,), "float32")
        T.reduce_sum(total, result, dim=1)
        if format in ("Q2_K", "Q4_K", "Q5_K"):
            correction = T.alloc_local((rows,), "float32")
            T.reduce_sum(minimum, correction, dim=1)
            for r in T.Parallel(rows):
                result[r] -= correction[r]
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def k_projection(M, N, K, format="Q4_K", rows=4, lanes=16, extreme_policy="first", inverse_policy="reciprocal"):
    """Complete projection, including activation quantization in this call."""
    assert format in ("Q4_K", "Q5_K", "Q6_K") and lanes in (8, 16) and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    weight_bytes, activation_bytes = K // 256 * (encoding.schema.storage_bits // 8), K // 256 * 292
    code_type = "uint4" if format == "Q4_K" else "uint6"

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        values = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, values, M, K, "Q8_K", 256, extreme_policy, inverse_policy, "rne")
            for m in T.serial(M):
                k_group_rows(weights, values[m, :], output[m, :], N, K, format, rows, lanes, code_type, "n")

    return main


@tilelang.jit(target="weft")
def indexed_groups(M, K, format="IQ2_XXS", rows=1):
    """Pair positive U6 grids with signs on prepared Q8_K activations.

    GGML's Q8_K producer emits [-127, 127], so sign transfer is exact in I8.
    Integer pair products and group scales remain explicit author operations.
    """
    assert format in ("IQ2_XS", "IQ2_XXS") and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)
    entries = 512 if format == "IQ2_XS" else 256

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((entries * 8,), "int8"),
        Signs: T.Tensor((128 * 8,), "int8"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        grid = T.view(Grid, shape=(entries, 8), encoding=F.I8X8, axes=("entry", "payload"), abi=True)
        signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows, 16), "float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block]), T.load(values[block])
                    q, d, activation, ds = w.q, w.d, x.q, x.ds
                    if format == "IQ2_XS":
                        metadata = w.scales
                    grid_index = T.alloc_local((rows, 32, 8), "uint16")
                    sign_index = T.alloc_local((rows, 32, 8), "uint16")
                    codes = T.alloc_local((rows, 32, 8), "uint6")
                    inputs = T.alloc_local((rows, 32, 8), "int8")
                    products = T.alloc_local((rows, 32, 4), "int16")
                    factors = T.alloc_local((rows, 32, 4), "int16")
                    quarters = T.alloc_local((rows, 32, 2), "int32")
                    scaled = T.alloc_local((rows, 4, 16), "int32")
                    integer = T.alloc_local((rows, 16), "int32")
                    for r, e, p in T.Parallel(rows, 32, 8):
                        entry = T.cast(e, "uint16")
                        payload = T.cast(p, "uint16")
                        if format == "IQ2_XS":
                            word = q[r, entry]
                            index, sign = word & T.uint16(511), word >> T.uint16(9)
                        else:
                            group = entry // T.uint16(4)
                            bits = T.cast(q[r, group * T.uint16(4) + T.uint16(2)], "uint32")
                            bits |= T.cast(q[r, group * T.uint16(4) + T.uint16(3)], "uint32") << T.uint32(16)
                            word = q[r, group * T.uint16(4) + entry % T.uint16(4) // T.uint16(2)]
                            index = word >> (entry % T.uint16(2) * T.uint16(8)) & T.uint16(255)
                            sign = T.cast(bits >> (T.cast(entry % T.uint16(4), "uint32") * T.uint32(7)), "uint16") & T.uint16(127)
                        grid_index[r, e, p] = index * T.uint16(8) + payload
                        sign_index[r, e, p] = sign * T.uint16(8) + payload
                    positive = T.load(grid.values, at=grid_index, bounds="in_bounds")
                    sign = T.load(signs.values, at=sign_index, bounds="in_bounds")
                    for r, e, p in T.Parallel(rows, 32, 8):
                        value = activation[T.cast(e, "uint16") * T.uint16(8) + T.cast(p, "uint16")]
                        inputs[r, e, p] = T.if_then_else(sign[r, e, p] < T.int8(0), -value, value)
                        codes[r, e, p] = T.cast(positive[r, e, p], "uint6")
                    T.mac_groups(codes, inputs, products, n=2, into="int16", over="payload")
                    for r, entry, p in T.Parallel(rows, 32, 4):
                        if format == "IQ2_XS":
                            sc = (T.cast(metadata[r, entry // 4], "uint32") >> T.cast(entry % 4 // 2 * 4, "uint32")) & T.uint32(15)
                        else:
                            group = entry // 4
                            bits = T.cast(q[r, group * 4 + 2], "uint32") | (T.cast(q[r, group * 4 + 3], "uint32") << T.uint32(16))
                            sc = bits >> T.uint32(28)
                        factors[r, entry, p] = T.cast(sc * T.uint32(2) + T.uint32(1), "int16")
                    T.mac_groups(products, factors, quarters, n=2, into="int32", over="payload")
                    for r, segment, lane in T.Parallel(rows, 4, 16):
                        quad = segment * 16 + lane
                        entry = quad // 2
                        scaled[r, segment, lane] = quarters[r, entry, quad % 2]
                    T.reduce_sum(scaled, integer, dim=1)
                    for r, lane in T.Parallel(rows, 16):
                        total[r, lane] += T.cast(integer[r, lane], "float32") * (T.cast(d[r], "float32") * ds[()]) * 0.125
                result = T.alloc_local((rows,), "float32")
                T.reduce_sum(total, result, dim=1)
                T.copy(result, output[row])

    return main
