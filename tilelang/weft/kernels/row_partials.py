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


@T.macro
def indexed_xxs_record(w, x, total, rows, parity, grid):
    """Expose metadata groups, codebook entries and sign payload reuse."""
    q, metadata, d, activation, sums, ds = w.q, w.metadata, w.d, x.q, x.bsum, x.ds
    scales = T.alloc_local((rows, 8), "int32")
    indices = T.alloc_local((rows, 8, 8, 4), "uint32")
    sign_rows = T.alloc_local((rows, 8, 4), "uint8")
    for r, group in T.Parallel(rows, 8):
        scales[r, group] = T.cast(metadata[r, group] >> T.uint32(28), "int32") * 2 + 1
    for r, group, entry, payload in T.Parallel(rows, 8, 8, 4):
        index = T.cast(q[r, T.cast(group, "uint32") * T.uint32(8) + T.cast(entry, "uint32")], "uint32")
        indices[r, group, entry, payload] = index * T.uint32(4) + T.cast(payload, "uint32")
    selected = T.load(grid.values, at=indices, bounds="in_bounds")
    for r, group, entry in T.Parallel(rows, 8, 4):
        stored = T.cast(metadata[r, group] >> (T.cast(entry, "uint32") * T.uint32(7)), "uint8") & T.uint8(127)
        folded = (stored ^ (stored >> T.uint8(4))) & T.uint8(15)
        sign_rows[r, group, entry] = stored | parity[folded]
    signs = T.alloc_local((rows, 8, 4, 8), "bool")
    for r, group, entry, payload in T.Parallel(rows, 8, 4, 8):
        signs[r, group, entry, payload] = (sign_rows[r, group, entry] & (T.uint8(1) << T.cast(payload, "uint8"))) != T.uint8(0)
    expanded_signs = T.reshape(signs, (rows, 8, 8, 4))
    codes = T.alloc_local((rows, 8, 8, 4), "uint8")
    inputs = T.alloc_local((rows, 8, 8, 4), "int8")
    for r, group, entry, payload in T.Parallel(rows, 8, 8, 4):
        value = selected[r, group, entry, payload]
        signed = T.if_then_else(expanded_signs[r, group, entry, payload], -value, value)
        codes[r, group, entry, payload] = T.cast(signed, "uint8") + T.uint8(128)
        coordinate = T.cast(group, "uint32") * T.uint32(32) + T.cast(entry, "uint32") * T.uint32(4) + T.cast(payload, "uint32")
        inputs[r, group, entry, payload] = activation[coordinate]
    products = T.alloc_local((rows, 8, 8, 1), "int32")
    T.mac_groups(codes, inputs, products, n=4, into="int32", over="payload")
    scaled = T.alloc_local((rows, 8, 8), "int32")
    for r, group, entry in T.Parallel(rows, 8, 8):
        scaled[r, group, entry] = products[r, group, entry, 0] * scales[r, group]
    grouped = T.reshape(scaled, (rows, 4, 16))
    integer = T.alloc_local((rows, 16), "int32")
    T.reduce_sum(grouped, integer, dim=1)
    for r, lane in T.Parallel(rows, 16):
        correction = T.cast(sums[lane], "int32") * 128 * scales[r, T.cast(lane, "uint32") // T.uint32(2)]
        total[r, lane] += T.cast(integer[r, lane] - correction, "float32") * (T.cast(d[r], "float32") * ds[()])


@T.macro
def indexed_subgroups_rows(weights, values, output, grid, M, K, format, rows, windows, axis):
    payload = 8 if format == "IQ2_S" else 4
    window_elements = 256 // windows
    entries = window_elements // payload
    if format == "IQ3_XXS":
        parity = T.alloc_local((16,), "uint8")
        for i in T.Parallel(16):
            bits = T.cast(i, "uint8")
            parity[i] = ((bits ^ (bits >> T.uint8(1)) ^ (bits >> T.uint8(2)) ^ (bits >> T.uint8(3))) & T.uint8(1)) << T.uint8(7)
    with T.level.rows(M, group=rows, axis=axis) as row:
        total = T.state((rows, 16), "float32", init=0)
        result = T.alloc_local((rows,), "float32")
        with T.level.blocks(K, extent=256) as block:
            w, x = T.load(weights[row, block]), T.load(values[block])
            if format == "IQ3_XXS":
                indexed_xxs_record(w, x, total, rows, parity, grid)
            else:
                activation, d, ds = x.q, w.d, x.ds
                quantized = w.q
                high, signs, scales = w.qh, w.signs, w.scales
                integer = T.alloc_local((rows, 16), "int32")
                T.fill(integer, 0)
                for part in T.unroll(windows):
                    entry_indices = T.alloc_local((rows, entries), "uint16")
                    entry_factors = T.alloc_local((rows, entries), "int16")
                    indices = T.alloc_local((rows, entries, payload), "uint16")
                    inputs = T.alloc_local((rows, entries, payload), "int8")
                    codes = T.alloc_local((rows, entries, payload), "uint6")
                    for r, e in T.Parallel(rows, entries):
                        code = T.uint16(part * entries) + T.cast(e, "uint16")
                        if format == "IQ2_S":
                            upper = (high[r, code // T.uint16(4)] >> (code % T.uint16(4) * T.uint16(2))) & T.uint8(3)
                            index = T.cast(quantized[r, code], "uint16") | (T.cast(upper, "uint16") << T.uint16(8))
                        else:
                            index = T.cast(quantized[r, code], "uint16") | (T.cast(high[r, code], "uint16") << T.uint16(8))
                        entry_indices[r, e] = index
                        if format == "IQ2_S":
                            group = T.uint16(part * entries // 2) + T.cast(e, "uint16") // T.uint16(2)
                            metadata_value = scales[r, group // T.uint16(2)]
                            scale = (metadata_value >> (group % T.uint16(2) * T.uint16(4))) & T.uint8(15)
                        else:
                            scale = scales[r, T.uint16(part * entries // 8) + T.cast(e, "uint16") // T.uint16(8)]
                        entry_factors[r, e] = T.cast(scale, "int16") * T.int16(2) + T.int16(1)
                    for r, e, p in T.Parallel(rows, entries, payload):
                        indices[r, e, p] = entry_indices[r, e] * T.uint16(payload) + T.cast(p, "uint16")
                    selected = T.load(grid.values, at=indices, bounds="in_bounds")
                    for r, e, p in T.Parallel(rows, entries, payload):
                        position = T.uint16(part * window_elements) + T.cast(e, "uint16") * T.uint16(payload) + T.cast(p, "uint16")
                        value = activation[position]
                        negative = signs[r, position] != 0
                        codes[r, e, p] = T.cast(selected[r, e, p], "uint6")
                        inputs[r, e, p] = T.if_then_else(negative, -value, value)
                    flat_codes = T.reshape(codes, (rows, window_elements))
                    flat_inputs = T.reshape(inputs, (rows, window_elements))
                    scaled = T.alloc_local((rows, window_elements // 4), "int32")
                    products = T.alloc_local((rows, window_elements // 2), "int16")
                    factors = T.alloc_local((rows, window_elements // 2), "int16")
                    T.mac_groups(flat_codes, flat_inputs, products, n=2, into="int16", over="pair")
                    for r, p in T.Parallel(rows, window_elements // 2):
                        factors[r, p] = entry_factors[r, T.cast(p, "uint16") // T.uint16(payload // 2)]
                    T.mac_groups(products, factors, scaled, n=2, into="int32", over="pair")
                    grouped = T.reshape(scaled, (rows, window_elements // 64, 16))
                    reduced = T.alloc_local((rows, 16), "int32")
                    T.reduce_sum(grouped, reduced, dim=1)
                    for r, p in T.Parallel(rows, 16):
                        integer[r, p] += reduced[r, p]
                for r, p in T.Parallel(rows, 16):
                    total[r, p] += T.cast(integer[r, p], "float32") * (T.cast(d[r], "float32") * ds[()])
        T.reduce_sum(total, result, dim=1)
        if format in ("IQ2_S", "IQ3_XXS"):
            for r in T.Parallel(rows):
                result[r] *= 0.125 if format == "IQ2_S" else 0.25
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def indexed_subgroups(M, K, format="IQ2_S", rows=1, windows=4):
    """Retain codebook geometry and explicitly grouped I32 scale partials.

    IQ2_S/IQ3_S transfer signs to prepared Q8_K in [-127, 127] before
    I16 pairs. IQ3_XXS exposes group/entry/payload and reuses its block sums.
    """
    assert format in ("IQ2_S", "IQ3_S", "IQ3_XXS") and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)
    assert windows in (1, 2, 4)
    payload = 8 if format == "IQ2_S" else 4
    window_elements = 256 // windows
    entries = window_elements // payload
    grid_bytes = {"IQ2_S": 8192, "IQ3_S": 2048, "IQ3_XXS": 1024}[format]
    grid_encoding = F.I8X8 if payload == 8 else F.I8X4

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((grid_bytes,), "int8"),
        Signs: T.Tensor((1024 if format == "IQ3_XXS" else 0,), "int8"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        grid = T.view(Grid, shape=(grid_bytes // payload, payload), encoding=grid_encoding, axes=("entry", "payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            indexed_subgroups_rows(weights, values, output, grid, M, K, format, rows, windows, "m")

    return main


@T.macro
def iq1_groups_rows(weights, values, output, grid, HalfBits, M, K, medium, rows, axis):
    with T.level.rows(M, group=rows, axis=axis) as row:
        total = T.state((rows, 16), "float32", init=0)
        result = T.alloc_local((rows,), "float32")
        with T.level.blocks(K, extent=256) as block:
            w, x = T.load(weights[row, block]), T.load(values[block])
            q, high, activation, ds, sums = w.q, w.qh, x.q, x.ds, x.bsum
            if medium:
                metadata = w.scales
            else:
                d = w.d
            indices = T.alloc_local((rows, 32, 8), "uint16")
            factors = T.alloc_local((rows, 32), "int16")
            deltas = T.alloc_local((rows, 32), "int16")
            for r, e in T.Parallel(rows, 32):
                entry = T.cast(e, "uint16")
                if medium:
                    bits = T.cast(high[r, entry // T.uint16(2)], "uint16") >> (entry % T.uint16(2) * T.uint16(4))
                    group = entry // T.uint16(4)
                    sc = T.cast(metadata[r, group // T.uint16(2) * T.uint16(2)], "uint16")
                    sc |= T.cast(metadata[r, group // T.uint16(2) * T.uint16(2) + T.uint16(1)], "uint16") << T.uint16(8)
                    shift = group % T.uint16(2) * T.uint16(6) + entry % T.uint16(4) // T.uint16(2) * T.uint16(3)
                    scale = (sc >> shift) & T.uint16(7)
                    delta = T.int16(1) - T.cast((bits >> T.uint16(3)) & T.uint16(1), "int16") * T.int16(2)
                else:
                    bits = T.cast(high[r, entry // T.uint16(4)], "uint16")
                    scale = (bits >> T.uint16(12)) & T.uint16(7)
                    delta = T.int16(1) - T.cast(bits >> T.uint16(15), "int16") * T.int16(2)
                factors[r, e] = T.cast(scale, "int16") * T.int16(2) + T.int16(1)
                deltas[r, e] = delta
            for r, e, p in T.Parallel(rows, 32, 8):
                entry = T.cast(e, "uint16")
                bits = T.cast(high[r, entry // T.uint16(2 if medium else 4)], "uint16")
                upper = ((bits >> (entry % T.uint16(2) * T.uint16(4))) if medium else (bits >> (entry % T.uint16(4) * T.uint16(3)))) & T.uint16(7)
                index = T.cast(q[r, e], "uint16") | (upper << T.uint16(8))
                indices[r, e, p] = index * T.uint16(8) + T.cast(p, "uint16")
            selected = T.load(grid.values, at=indices, bounds="in_bounds")
            codes = T.alloc_local((rows, 32, 8), "int16" if medium else "uint2")
            inputs = T.alloc_local((32, 8), "int16" if medium else "int8")
            products = T.alloc_local((rows, 32, 4 if medium else 2), "int32")
            grouped = T.alloc_local((rows, 8 if medium else 4, 16), "int32")
            integer = T.alloc_local((rows, 16), "int32")
            for e, p in T.Parallel(32, 8):
                inputs[e, p] = T.cast(activation[T.cast(e, "uint16") * T.uint16(8) + T.cast(p, "uint16")], "int16" if medium else "int8")
            for r, e, p in T.Parallel(rows, 32, 8):
                if medium:
                    codes[r, e, p] = (T.cast(selected[r, e, p], "int16") * T.int16(8) + deltas[r, e]) * factors[r, e]
                else:
                    codes[r, e, p] = T.cast(selected[r, e, p] + T.int8(1), "uint2")
            T.mac_groups(codes, inputs, products, n=2 if medium else 4, into="int32", over="payload")
            for r, g, p in T.Parallel(rows, 8 if medium else 4, 16):
                position = T.cast(g, "uint16") * T.uint16(16) + T.cast(p, "uint16")
                if medium:
                    grouped[r, g, p] = products[r, position // T.uint16(4), position % T.uint16(4)]
                else:
                    grouped[r, g, p] = products[r, position // T.uint16(2), position % T.uint16(2)] * T.cast(factors[r, position // T.uint16(2)], "int32") * 8
            T.reduce_sum(grouped, integer, dim=1)
            for r, p in T.Parallel(rows, 16):
                if medium:
                    sc0 = T.cast(metadata[r, 0], "uint16") | (T.cast(metadata[r, 1], "uint16") << T.uint16(8))
                    sc1 = T.cast(metadata[r, 2], "uint16") | (T.cast(metadata[r, 3], "uint16") << T.uint16(8))
                    sc2 = T.cast(metadata[r, 4], "uint16") | (T.cast(metadata[r, 5], "uint16") << T.uint16(8))
                    sc3 = T.cast(metadata[r, 6], "uint16") | (T.cast(metadata[r, 7], "uint16") << T.uint16(8))
                    bits = (sc0 >> T.uint16(12)) | ((sc1 >> T.uint16(8)) & T.uint16(240))
                    bits |= ((sc2 >> T.uint16(4)) & T.uint16(3840)) | (sc3 & T.uint16(61440))
                    scale = HalfBits[bits]
                    contribution = integer[r, p]
                else:
                    scale = T.cast(d[r], "float32")
                    contribution = integer[r, p] - (T.int32(8) - T.cast(deltas[r, p * 2], "int32")) * T.cast(factors[r, p * 2], "int32") * T.cast(sums[p], "int32")
                total[r, p] += T.cast(contribution, "float32") * (scale * ds[()])
        T.reduce_sum(total, result, dim=1)
        for r in T.Parallel(rows):
            result[r] *= 0.125
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def iq1_groups(M, K, format="IQ1_S", rows=1):
    """Keep the exact eighth correction and integer scaling before float lanes."""
    assert format in ("IQ1_S", "IQ1_M") and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8
    medium = format == "IQ1_M"

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((16384,), "int8"),
        HalfBits: T.Tensor((65536 if medium else 0,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        grid = T.view(Grid, shape=(2048, 8), encoding=F.I8X8, axes=("entry", "payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            iq1_groups_rows(weights, values, output, grid, HalfBits, M, K, medium, rows, "m")

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


@tilelang.jit(target="weft")
def affine_groups(M, K, format="Q5_K"):
    """Use named sublevels for scaled record groups and a separate minimum sum."""
    assert format in ("Q2_K", "Q4_K", "Q5_K") and M > 0 and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8
    group_size = 16 if format == "Q2_K" else 32
    groups = 256 // group_size

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    d, dmin, ds, sums = w.d, w.dmin, x.ds, x.bsum
                    if format == "Q2_K":
                        metadata = w.scales
                    else:
                        minima = w.m
                    integer = T.alloc_var("int32", init=0)
                    with T.level.subtiles(block, extent=group_size) as group:
                        low = T.load(w.q, at=group)
                        if format == "Q5_K":
                            high = T.load(w.qh, at=group)
                        activation = T.load(x.q, at=group)
                        if format == "Q2_K":
                            scale = T.load(metadata, at=group)
                        else:
                            scale = T.load(w.sc, at=group)
                        codes = T.alloc_local((group_size,), "uint8")
                        product = T.alloc_local((1,), "int32")
                        for k in T.Parallel(group_size):
                            codes[k] = T.cast(low[k], "uint8")
                            if format == "Q5_K":
                                codes[k] |= T.cast(high[k], "uint8") << T.uint8(4)
                        T.reduce_dot(codes, activation, product, over="k", acc_dtype="int32", engine="vector")
                        if format == "Q2_K":
                            integer += product[0] * (T.cast(scale[()], "int32") & T.int32(15))
                        else:
                            integer += product[0] * T.cast(scale[()], "int32")
                    terms = T.alloc_local((groups,), "int32")
                    correction = T.alloc_local((1,), "int32")
                    for g in T.Parallel(groups):
                        if format == "Q2_K":
                            terms[g] = T.cast(metadata[g] >> T.uint8(4), "int32") * T.cast(sums[g], "int32")
                        else:
                            terms[g] = T.cast(minima[g], "int32") * (T.cast(sums[g * 2], "int32") + T.cast(sums[g * 2 + 1], "int32"))
                    T.reduce_sum(terms, correction, dim=0)
                    if format == "Q2_K":
                        total += ds[()] * (T.cast(d[()], "float32") * T.cast(integer, "float32")
                                          - T.cast(dmin[()], "float32") * T.cast(correction[0], "float32"))
                    else:
                        total -= (ds[()] * T.cast(dmin[()], "float32")) * T.cast(correction[0], "float32")
                        total += (ds[()] * T.cast(d[()], "float32")) * T.cast(integer, "float32")
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def scaled_groups(M, K, format="Q3_K"):
    """Finish signed sixteen-element dots before the record's floating update."""
    assert format in ("Q3_K", "Q6_K") and M > 0 and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    low = w.q if format == "Q3_K" else w.ql
                    high = w.hmask if format == "Q3_K" else w.qh
                    activation, d, ds = x.q, w.d, x.ds
                    integer = T.alloc_var("int32", init=0)
                    with T.level.subtiles(block, extent=16) as group:
                        lows = T.load(low, at=group)
                        highs = T.load(high, at=group)
                        inputs = T.load(activation, at=group)
                        if format == "Q3_K":
                            scale_low = T.load(w.scale_low, at=group)
                            scale_high = T.load(w.scale_high, at=group)
                            scale = T.cast(T.cast(scale_low[()], "uint8")
                                           | (T.cast(scale_high[()], "uint8") << T.uint8(4)), "int8") - T.int8(32)
                        else:
                            scale_value = T.load(w.scales, at=group)
                            scale = scale_value[()]
                        codes = T.alloc_local((16,), "int8")
                        product = T.alloc_local((1,), "int32")
                        for k in T.Parallel(16):
                            codes[k] = T.cast(T.cast(lows[k], "uint8")
                                              | (T.cast(highs[k], "uint8") << T.uint8(2 if format == "Q3_K" else 4)), "int8") - T.int8(4 if format == "Q3_K" else 32)
                        T.reduce_dot(inputs, codes, product, over="k", acc_dtype="int32", engine="vector")
                        integer += product[0] * T.cast(scale, "int32")
                    total += (T.cast(d[()], "float32") * ds[()]) * T.cast(integer, "float32")
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def q2_lanes(M, K):
    """Accumulate four exact scaled products per I16 lane before widening."""
    assert M > 0 and K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 84), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=F.Q2_K, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    q, scales, d, dmin = w.q, w.scales, w.d, w.dmin
                    activation, sums, ds = x.q, x.bsum, x.ds
                    integer = T.alloc_var("int32", init=0)
                    for half in T.serial(4):
                        partial = T.alloc_local((16,), "int16")
                        T.fill(partial, 0)
                        for part in T.unroll(4):
                            group = half * 4 + part
                            scale = T.cast(scales[group] & T.uint8(15), "int8")
                            for k in T.Parallel(16):
                                coordinate = T.cast(half, "uint16") * T.uint16(64) + T.uint16(part * 16) + T.cast(k, "uint16")
                                scaled = T.cast(q[coordinate], "int8") * scale
                                partial[k] += T.cast(scaled, "int16") * T.cast(activation[coordinate], "int16")
                        reduced = T.alloc_local((1,), "int32")
                        widened = T.alloc_local((16,), "int32")
                        for k in T.Parallel(16):
                            widened[k] = T.cast(partial[k], "int32")
                        T.reduce_sum(widened, reduced, dim=0)
                        integer += reduced[0]
                    minima = T.alloc_local((16,), "int16")
                    correction = T.alloc_local((1,), "int32")
                    for g in T.Parallel(16):
                        minima[g] = T.cast(scales[g] >> T.uint8(4), "int16")
                    T.reduce_dot(sums, minima, correction, over="k", acc_dtype="int32", engine="vector")
                    total += ds[()] * (T.cast(d[()], "float32") * T.cast(integer, "float32") - T.cast(dmin[()], "float32") * T.cast(correction[0], "float32"))
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def scaled_lanes(M, K, format="Q3_K", lanes=32):
    """Keep signed group-scaled products in independent numerical lanes."""
    assert M > 0 and K > 0 and K % 256 == 0
    assert format in ("Q3_K", "Q6_K") and lanes in (16, 32)
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=1) as row:
                total = T.state((lanes,), "float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w = T.load(weights[row, block][0, :])
                    x = T.load(values[block])
                    ds, d = x.ds, w.d
                    low = w.q if format == "Q3_K" else w.ql
                    high, activation = w.hmask if format == "Q3_K" else w.qh, x.q
                    if format == "Q3_K":
                        scale_low, scale_high = w.scale_low, w.scale_high
                    else:
                        scales = w.scales
                    block_scale = T.cast(d[()], "float32") * ds[()]
                    integer = T.alloc_local((lanes,), "int32")
                    T.fill(integer, 0)
                    for group in T.unroll(256 // lanes):
                        left = group * (lanes // 16)
                        for k in T.Parallel(lanes):
                            coordinate = T.uint32(group * lanes) + T.cast(k, "uint32")
                            code = T.cast(T.cast(low[coordinate], "uint8") | (T.cast(high[coordinate], "uint8") << T.uint8(2 if format == "Q3_K" else 4)), "int8") - T.int8(4 if format == "Q3_K" else 32)
                            product = T.cast(code, "int16") * T.cast(activation[coordinate], "int16")
                            if format == "Q3_K":
                                a = T.cast(T.cast(scale_low[left], "uint8") | (T.cast(scale_high[left], "uint8") << T.uint8(4)), "int8") - T.int8(32)
                                if lanes == 32:
                                    b = T.cast(T.cast(scale_low[left + 1], "uint8") | (T.cast(scale_high[left + 1], "uint8") << T.uint8(4)), "int8") - T.int8(32)
                            else:
                                a = scales[left]
                                if lanes == 32:
                                    b = scales[left + 1]
                            scale = a if lanes == 16 else T.if_then_else(T.cast(k, "uint32") < T.uint32(16), a, b)
                            integer[k] += T.cast(product, "int32") * T.cast(scale, "int32")
                    for k in T.Parallel(lanes):
                        total[k] += T.cast(integer[k], "float32") * block_scale
                result = T.alloc_local((1,), "float32")
                T.reduce_sum(total, result, dim=0)
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def q6_planes(M, K, rows=1, lanes=64, planes=4):
    """Retain explicitly scaled signed products across the encoded row."""
    assert M == rows == 1
    assert lanes in (32, 64, 128) and planes >= 2 and 256 % (lanes * planes) == 0
    assert K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 210), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=F.Q6_K, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(1,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            total = T.alloc_local((lanes,), "float32")
            T.fill(total, 0)
            result = T.alloc_local((1,), "float32")
            with T.level.blocks(K, extent=256) as block:
                w, x = T.load(weights[0, block]), T.load(values[block])
                low, high, scales, d, activation, ds = w.ql, w.qh, w.scales, w.d, x.q, x.ds
                accumulated = T.alloc_local((lanes,), "int32")
                T.fill(accumulated, 0)
                for group in T.unroll(256 // (lanes * planes)):
                    scaled = T.alloc_local((planes, lanes), "int32")
                    partial = T.alloc_local((lanes,), "int32")
                    for p, k in T.Parallel(planes, lanes):
                        coordinate = T.uint32(group * planes * lanes) + T.cast(p, "uint32") * T.uint32(lanes) + T.cast(k, "uint32")
                        code = T.cast(low[coordinate], "uint8") | (T.cast(high[coordinate], "uint8") << T.uint8(4))
                        centered = T.cast(code, "int8") - T.int8(32)
                        product = T.cast(centered, "int16") * T.cast(activation[coordinate], "int16")
                        scaled[p, k] = T.cast(product, "int32") * T.cast(scales[coordinate // T.uint32(16)], "int32")
                    T.reduce_sum(scaled, partial, dim=0)
                    for k in T.Parallel(lanes):
                        accumulated[k] += partial[k]
                for k in T.Parallel(lanes):
                    total[k] += T.cast(accumulated[k], "float32") * (T.cast(d[()], "float32") * ds[()])
            T.reduce_sum(total, result, dim=0)
            T.copy(result, output)

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
            if format in ("Q2_K", "Q4_K", "Q5_K"):
                metadata = T.alloc_local((rows, 2), "float16")
                block_scales = T.alloc_local((rows, 2), "float32")
                for r in T.Parallel(rows):
                    metadata[r, 0] = d[r]
                    metadata[r, 1] = dmin[r]
                for r, part in T.Parallel(rows, 2):
                    block_scales[r, part] = T.cast(metadata[r, part], "float32") * xd[()]
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
                    minimum[r, p] += block_scales[r, 1] * T.cast(correction[r, p], "float32")
            for r, p in T.Parallel(rows, lanes):
                if format in ("Q2_K", "Q4_K", "Q5_K"):
                    scale = block_scales[r, 0]
                else:
                    scale = T.cast(d[r], "float32") * xd[()]
                total[r, p] += scale * T.cast(integer[r, p], "float32")
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
    assert format in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K") and lanes in (8, 16) and K > 0 and K % 256 == 0
    assert format not in ("Q2_K", "Q3_K") or lanes == 16
    encoding = getattr(F, format)
    weight_bytes, activation_bytes = K // 256 * (encoding.schema.storage_bits // 8), K // 256 * 292
    code_type = "uint6" if format in ("Q5_K", "Q6_K") else "uint4"

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
            quantize_rows(source, values, M, K, "Q8_K", 256, extreme_policy, inverse_policy, "rne", 256)
            for m in T.serial(M):
                k_group_rows(weights, values[m, :], output[m, :], N, K, format, rows, lanes, code_type, "n")

    return main


@T.macro
def indexed_groups_rows(weights, values, output, grid, M, K, format, rows, axis):
    parity = T.alloc_local((16,), "uint8")
    for i in T.Parallel(16):
        nibble = T.cast(i, "uint8")
        parity[i] = ((nibble ^ (nibble >> T.uint8(1)) ^ (nibble >> T.uint8(2)) ^ (nibble >> T.uint8(3))) & T.uint8(1)) << T.uint8(7)
    with T.level.rows(M, group=rows, axis=axis) as row:
        total = T.state((rows, 16), "float32", init=0)
        with T.level.blocks(K, extent=256) as block:
            w, x = T.load(weights[row, block]), T.load(values[block])
            q, d, activation, ds = w.q, w.d, x.q, x.ds
            if format == "IQ2_XS":
                metadata = w.scales
            entry_indices = T.alloc_local((rows, 32), "uint16")
            entry_signs = T.alloc_local((rows, 32), "uint8")
            entry_scales = T.alloc_local((rows, 32), "int16")
            grid_index = T.alloc_local((rows, 32, 8), "uint16")
            negative = T.alloc_local((rows, 32, 8), "bool")
            codes = T.alloc_local((rows, 32, 8), "uint6")
            inputs = T.alloc_local((rows, 32, 8), "int8")
            products = T.alloc_local((rows, 32, 4), "int16")
            factors = T.alloc_local((rows, 32, 4), "int16")
            quarters = T.alloc_local((rows, 32, 2), "int32")
            scaled = T.alloc_local((rows, 4, 16), "int32")
            integer = T.alloc_local((rows, 16), "int32")
            for r, e in T.Parallel(rows, 32):
                entry = T.cast(e, "uint16")
                if format == "IQ2_XS":
                    word = q[r, entry]
                    index, sign = word & T.uint16(511), word >> T.uint16(9)
                    sc = (T.cast(metadata[r, entry // T.uint16(4)], "uint32") >> T.cast(entry % T.uint16(4) // T.uint16(2) * T.uint16(4), "uint32")) & T.uint32(15)
                else:
                    group = entry // T.uint16(4)
                    bits = T.cast(q[r, group * T.uint16(4) + T.uint16(2)], "uint32")
                    bits |= T.cast(q[r, group * T.uint16(4) + T.uint16(3)], "uint32") << T.uint32(16)
                    word = q[r, group * T.uint16(4) + entry % T.uint16(4) // T.uint16(2)]
                    index = word >> (entry % T.uint16(2) * T.uint16(8)) & T.uint16(255)
                    sign = T.cast(bits >> (T.cast(entry % T.uint16(4), "uint32") * T.uint32(7)), "uint16") & T.uint16(127)
                    sc = bits >> T.uint32(28)
                entry_indices[r, e] = index
                entry_scales[r, e] = T.cast(sc * T.uint32(2) + T.uint32(1), "int16")
                stored = T.cast(sign, "uint8")
                folded = (stored ^ (stored >> T.uint8(4))) & T.uint8(15)
                entry_signs[r, e] = stored | parity[folded]
            for r, e, p in T.Parallel(rows, 32, 8):
                grid_index[r, e, p] = entry_indices[r, e] * T.uint16(8) + T.cast(p, "uint16")
                negative[r, e, p] = (entry_signs[r, e] & (T.uint8(1) << T.cast(p, "uint8"))) != T.uint8(0)
            positive = T.load(grid.values, at=grid_index, bounds="in_bounds")
            for r, e, p in T.Parallel(rows, 32, 8):
                value = activation[T.cast(e, "uint16") * T.uint16(8) + T.cast(p, "uint16")]
                inputs[r, e, p] = T.if_then_else(negative[r, e, p], -value, value)
                codes[r, e, p] = T.cast(positive[r, e, p], "uint6")
            T.mac_groups(codes, inputs, products, n=2, into="int16", over="payload")
            for r, entry, p in T.Parallel(rows, 32, 4):
                factors[r, entry, p] = entry_scales[r, entry]
            T.mac_groups(products, factors, quarters, n=2, into="int32", over="payload")
            for r, segment, lane in T.Parallel(rows, 4, 16):
                quad = segment * 16 + lane
                entry = quad // 2
                scaled[r, segment, lane] = quarters[r, entry, quad % 2]
            T.reduce_sum(scaled, integer, dim=1)
            for r, lane in T.Parallel(rows, 16):
                total[r, lane] += T.cast(integer[r, lane], "float32") * ((T.cast(d[r], "float32") * 0.125) * ds[()])
        result = T.alloc_local((rows,), "float32")
        T.reduce_sum(total, result, dim=1)
        T.copy(result, output[row])


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
        with T.Kernel(1, cpu_num_threads=1):
            indexed_groups_rows(weights, values, output, grid, M, K, format, rows, "m")

    return main


@tilelang.jit(target="weft")
def indexed_projection(M, N, K, format="IQ2_XS", columns=1, windows=4):
    """Reuse the row-dot program after one in-call activation preparation."""
    assert format in ("IQ1_S", "IQ1_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS")
    assert M > 0 and N > 0 and N % columns == 0 and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8
    entries = {"IQ1_S": 2048, "IQ1_M": 2048, "IQ2_S": 1024, "IQ2_XS": 512,
               "IQ2_XXS": 256, "IQ3_S": 512, "IQ3_XXS": 256}[format]
    payload = 4 if format in ("IQ3_S", "IQ3_XXS") else 8
    grid_encoding = F.I8X4 if payload == 4 else F.I8X8

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Grid: T.Tensor((entries * payload,), "int8"),
        HalfBits: T.Tensor((65536 if format == "IQ1_M" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Grid": "input", "HalfBits": "input", "Quantized": "quantized", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        grid = T.view(Grid, shape=(entries, payload), encoding=grid_encoding,
                      axes=("entry", "payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_K", 256, "first", "direct", "rne", 256)
            for m in T.serial(M):
                if format in ("IQ1_S", "IQ1_M"):
                    iq1_groups_rows(weights, prepared[m, :], output[m, :], grid, HalfBits,
                                    N, K, format == "IQ1_M", columns, "n")
                elif format in ("IQ2_XS", "IQ2_XXS"):
                    indexed_groups_rows(weights, prepared[m, :], output[m, :], grid,
                                        N, K, format, columns, "n")
                else:
                    indexed_subgroups_rows(weights, prepared[m, :], output[m, :], grid,
                                           N, K, format, columns, windows, "n")

    return main
