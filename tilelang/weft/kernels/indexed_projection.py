"""Integer codebook panels shared by every row of a complete projection."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows


@T.macro
def decode_panel(record, panel, block_scale, correction, columns, format, grid, signs, HalfBits):
    weight = T.load(record)
    q = weight.q
    if format in ("IQ1_S", "IQ1_M", "IQ2_S", "IQ3_S"):
        high = weight.qh
    entries = 64 if format in ("IQ3_S", "IQ3_XXS") else 32
    payload = 4 if format in ("IQ3_S", "IQ3_XXS") else 8
    entry_indices = T.alloc_local((columns, entries), "uint32")
    indices = T.alloc_local((columns, entries, payload), "uint32")
    factors = T.alloc_local((columns, entries), "int16")
    if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS"):
        entry_signs = T.alloc_local((columns, entries), "uint32")
        sign_indices = T.alloc_local((columns, entries, payload), "uint32")
    if format in ("IQ1_S", "IQ1_M"):
        deltas = T.alloc_local((columns, 32), "int16")
    for c, e in T.Parallel(columns, entries):
        entry = T.cast(e, "uint32")
        if format == "IQ1_S":
            metadata = T.cast(high[c, entry // T.uint32(4)], "uint32")
            upper = (metadata >> (entry % T.uint32(4) * T.uint32(3))) & T.uint32(7)
            factors[c, e] = T.cast((metadata >> T.uint32(12)) & T.uint32(7), "int16") * 2 + 1
            deltas[c, e] = 1 - T.cast(metadata >> T.uint32(15), "int16") * 2
            index = T.cast(q[c, e], "uint32") | (upper << T.uint32(8))
        elif format == "IQ1_M":
            metadata = T.cast(high[c, entry // T.uint32(2)], "uint32") >> (entry % T.uint32(2) * T.uint32(4))
            group = entry // T.uint32(4)
            scale_word = T.cast(weight.scales[c, group // T.uint32(2) * T.uint32(2)], "uint32") | (T.cast(weight.scales[c, group // T.uint32(2) * T.uint32(2) + T.uint32(1)], "uint32") << T.uint32(8))
            shift = group % T.uint32(2) * T.uint32(6) + entry % T.uint32(4) // T.uint32(2) * T.uint32(3)
            factors[c, e] = T.cast((scale_word >> shift) & T.uint32(7), "int16") * 2 + 1
            deltas[c, e] = 1 - T.cast((metadata >> T.uint32(3)) & T.uint32(1), "int16") * 2
            index = T.cast(q[c, e], "uint32") | ((metadata & T.uint32(7)) << T.uint32(8))
        elif format == "IQ2_XS":
            word = T.cast(q[c, e], "uint32")
            index, sign_index = word & T.uint32(511), word >> T.uint32(9)
            metadata = T.cast(weight.scales[c, entry // T.uint32(4)], "uint32")
            factors[c, e] = T.cast((metadata >> (entry % T.uint32(4) // T.uint32(2) * T.uint32(4))) & T.uint32(15), "int16") * 2 + 1
        elif format == "IQ2_XXS":
            group = entry // T.uint32(4)
            metadata = T.cast(q[c, group * T.uint32(4) + T.uint32(2)], "uint32") | (T.cast(q[c, group * T.uint32(4) + T.uint32(3)], "uint32") << T.uint32(16))
            word = T.cast(q[c, group * T.uint32(4) + entry % T.uint32(4) // T.uint32(2)], "uint32")
            index = (word >> (entry % T.uint32(2) * T.uint32(8))) & T.uint32(255)
            sign_index = (metadata >> (entry % T.uint32(4) * T.uint32(7))) & T.uint32(127)
            factors[c, e] = T.cast(metadata >> T.uint32(28), "int16") * 2 + 1
        elif format == "IQ2_S":
            upper = (T.cast(high[c, entry // T.uint32(4)], "uint32") >> (entry % T.uint32(4) * T.uint32(2))) & T.uint32(3)
            index = T.cast(q[c, e], "uint32") | (upper << T.uint32(8))
            metadata = T.cast(weight.scales[c, entry // T.uint32(4)], "uint32")
            factors[c, e] = T.cast((metadata >> (entry % T.uint32(4) // T.uint32(2) * T.uint32(4))) & T.uint32(15), "int16") * 2 + 1
        elif format == "IQ3_XXS":
            metadata = weight.metadata[c, entry // T.uint32(8)]
            index = T.cast(q[c, e], "uint32")
            sign_index = (metadata >> (entry % T.uint32(8) // T.uint32(2) * T.uint32(7))) & T.uint32(127)
            factors[c, e] = T.cast(metadata >> T.uint32(28), "int16") * 2 + 1
        else:
            index = T.cast(q[c, e], "uint32") | (T.cast(high[c, e], "uint32") << T.uint32(8))
            factors[c, e] = T.cast(weight.scales[c, entry // T.uint32(8)], "int16") * 2 + 1
        entry_indices[c, e] = index
        if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS"):
            entry_signs[c, e] = sign_index
    for c, e, p in T.Parallel(columns, entries, payload):
        indices[c, e, p] = entry_indices[c, e] * T.uint32(payload) + T.cast(p, "uint32")
        if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS"):
            sign_indices[c, e, p] = entry_signs[c, e] * T.uint32(8) + T.cast(p, "uint32")
            if format == "IQ3_XXS":
                sign_indices[c, e, p] += T.cast(e, "uint32") % T.uint32(2) * T.uint32(4)
    values = T.load(grid.values, at=indices, bounds="in_bounds")
    if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS"):
        selected_signs = T.load(signs.values, at=sign_indices, bounds="in_bounds")
    codes = T.alloc_local((columns, entries, payload), "int8" if format in ("IQ1_S", "IQ1_M") else "int16")
    for c, e, p in T.Parallel(columns, entries, payload):
        if format in ("IQ1_S", "IQ1_M"):
            code = T.cast(values[c, e, p], "int16")
        elif format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS"):
            code = T.cast(values[c, e, p], "int16") * T.cast(selected_signs[c, e, p], "int16")
        else:
            position = T.cast(e, "uint32") * T.uint32(payload) + T.cast(p, "uint32")
            code = T.cast(values[c, e, p], "int16") * (1 - T.cast(weight.signs[c, position], "int16") * 2)
        codes[c, e, p] = code * factors[c, e]
    T.copy(T.reshape(codes, (columns, 256)), panel)
    if format in ("IQ1_S", "IQ1_M"):
        for c, g in T.Parallel(columns, 8 if format == "IQ1_S" else 32):
            entry = g * 4 if format == "IQ1_S" else g
            correction[c, g] = -deltas[c, entry] * factors[c, entry]
    if format == "IQ1_M":
        half_indices = T.alloc_local((columns,), "uint16")
        for c in T.Parallel(columns):
            sc0 = T.cast(weight.scales[c, 0], "uint32") | (T.cast(weight.scales[c, 1], "uint32") << T.uint32(8))
            sc1 = T.cast(weight.scales[c, 2], "uint32") | (T.cast(weight.scales[c, 3], "uint32") << T.uint32(8))
            sc2 = T.cast(weight.scales[c, 4], "uint32") | (T.cast(weight.scales[c, 5], "uint32") << T.uint32(8))
            sc3 = T.cast(weight.scales[c, 6], "uint32") | (T.cast(weight.scales[c, 7], "uint32") << T.uint32(8))
            half_indices[c] = T.cast((sc0 >> T.uint32(12)) | ((sc1 >> T.uint32(8)) & T.uint32(240)) | ((sc2 >> T.uint32(4)) & T.uint32(3840)) | (sc3 & T.uint32(61440)), "uint16")
        scales = T.load(HalfBits, at=half_indices, bounds="in_bounds")
        for c in T.Parallel(columns):
            block_scale[c] = scales[c] * 0.125
    else:
        for c in T.Parallel(columns):
            block_scale[c] = T.cast(weight.d[c], "float32") * (1.0 if format == "IQ3_S" else 0.25 if format == "IQ3_XXS" else 0.125)


@tilelang.jit(target="weft")
def projection(M, N, K, format="IQ2_XS", rows=16, columns=16, engine="auto", decode_columns=1):
    assert format in ("IQ1_S", "IQ1_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS")
    assert M > 0 and N > 0 and K > 0 and K % 256 == 0 and M % rows == N % columns == 0
    assert decode_columns > 0 and columns % decode_columns == 0
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)
    payload = 4 if format in ("IQ3_S", "IQ3_XXS") else 8
    grid_entries = {"IQ1_S": 2048, "IQ1_M": 2048, "IQ2_S": 1024, "IQ2_XS": 512, "IQ2_XXS": 256, "IQ3_S": 512, "IQ3_XXS": 256}[format]
    grid_encoding = F.I8X4 if payload == 4 else F.I8X8
    integer_correction = format in ("IQ1_S", "IQ1_M")
    groups = 8 if format == "IQ1_S" else 32
    panel_dtype = "int8" if integer_correction else "int16"

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"), Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"), Output: T.Tensor((M, N), "float32"),
        Panel: T.Tensor((columns, 256), panel_dtype), PackedActivation: T.Tensor((M, 256), "int8"),
        ScalePanel: T.Tensor((columns,), "float32"),
        CorrectionPanel: T.Tensor((columns, groups) if integer_correction else (0,), "int8"),
        GroupSums: T.Tensor((M, groups) if integer_correction else (0,), "int16"),
        Grid: T.Tensor((grid_entries * payload,), "int8"),
        Signs: T.Tensor((1024 if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS") else 0,), "int8"),
        HalfBits: T.Tensor((65536 if format == "IQ1_M" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Grid": "input",
                     "Signs": "input", "HalfBits": "input", "Quantized": "workspace", "Output": "output",
                     "Panel": "weights", "PackedActivation": "activation", "ScalePanel": "scale",
                     "CorrectionPanel": "correction", "GroupSums": "sums"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        panel = T.view(Panel, shape=(columns, 256), axes=("n", "k"), abi=True)
        activation_panel = T.view(PackedActivation, shape=(M, 256), axes=("m", "k"), abi=True)
        scale_panel = T.view(ScalePanel, shape=(columns,), axes=("n",), abi=True)
        if integer_correction:
            correction_panel = T.view(CorrectionPanel, shape=(columns, groups), axes=("n", "k"), abi=True)
            group_sums = T.view(GroupSums, shape=(M, groups), axes=("m", "k"), abi=True)
        grid = T.view(Grid, shape=(grid_entries, payload), encoding=grid_encoding, axes=("entry", "payload"), abi=True)
        signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign", "payload"), abi=True) if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS") else 0
        half_bits = T.view(HalfBits, shape=(65536,), axes=("half_bits",), abi=True) if format == "IQ1_M" else 0
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_K", 256, "first", "direct", "rne", 256)
            first = T.alloc_local((1,), "bool")
            first[0] = True
            with T.level.blocks(K, extent=256) as block:
                for m in T.serial(M):
                    activation_codes = T.load(prepared[m, block].q)
                    T.copy(activation_codes, activation_panel[m, :])
                    if integer_correction:
                        sums = T.alloc_local((groups,), "int16")
                        if format == "IQ1_S":
                            stored_sums = T.load(prepared[m, block].bsum)
                            for g in T.Parallel(groups):
                                sums[g] = T.cast(T.cast(stored_sums[2 * g], "int32") + T.cast(stored_sums[2 * g + 1], "int32"), "int16")
                        else:
                            grouped = T.reshape(activation_codes, (groups, 256 // groups))
                            wide = T.alloc_local((groups, 256 // groups), "int16")
                            for g, p in T.Parallel(groups, 256 // groups):
                                wide[g, p] = T.cast(grouped[g, p], "int16")
                            T.reduce_sum(wide, sums, dim=1)
                        T.copy(sums, group_sums[m, :])
                with T.level.cols(N, group=columns) as col:
                    for cohort in T.serial(columns // decode_columns):
                        begin = cohort * decode_columns
                        local_scale = T.alloc_local((decode_columns,), "float32")
                        local_minima = T.alloc_local((decode_columns, groups), "int8")
                        decode_panel(weights[col, block][begin : begin + decode_columns, :],
                                     panel[begin : begin + decode_columns, :], local_scale, local_minima,
                                     decode_columns, format, grid, signs, half_bits)
                        T.copy(local_scale, scale_panel[begin : begin + decode_columns])
                        if integer_correction:
                            T.copy(local_minima, correction_panel[begin : begin + decode_columns, :])
                    block_scale = T.load(scale_panel)
                    if integer_correction:
                        minima = T.load(correction_panel)
                    weight_values = T.load(panel)
                    with T.level.rows(M, group=rows) as row:
                        activation_values = T.load(activation_panel[row, :])
                        products = T.alloc_local((rows, columns), "int32")
                        T.reduce_dot(activation_values, weight_values, products, over="k", acc_dtype="int32", engine=engine)
                        if integer_correction:
                            sums = T.load(group_sums[row, :])
                            correction = T.alloc_local((rows, columns), "int32")
                            T.reduce_dot(sums, minima, correction, over="k", acc_dtype="int32", engine="auto" if format == "IQ1_S" else "vector")
                        ds = T.load(prepared[row, block].ds)
                        result = T.alloc_local((rows, columns), "float32")
                        for r, c in T.Parallel(rows, columns):
                            integer = products[r, c] * 8 - correction[r, c] if integer_correction else products[r, c]
                            result[r, c] = T.cast(integer, "float32") * (ds[r] * block_scale[c])
                        if not first[0]:
                            previous = T.load(output[row, col])
                            for r, c in T.Parallel(rows, columns):
                                result[r, c] += previous[r, c]
                        T.copy(result, output[row, col])
                first[0] = False

    return main


@tilelang.jit(target="weft")
def packed_words(M, N, K, format="IQ3_XXS", rows=16, columns=8):
    """Pack four-byte words and retain the scale group outside each byte dot.

    The activation panel has adjacent output rows; integer and floating tiles
    use (column, row). All panels are invocation-local, including the final
    transpose. Group widths and numerical partials remain author declarations.
    """
    assert format in ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_XS")
    assert M > 0 and N > 0 and K > 0 and M % rows == N % columns == K % 256 == 0
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)
    groups = 16 if format == "IQ2_S" else 8
    words = 64 // groups
    payload = 8 if format == "IQ2_S" else 4
    entries = {"IQ2_S": 1024, "IQ3_S": 512, "IQ3_XXS": 256, "IQ4_XS": 0}[format]
    grid_encoding = F.I8X8 if payload == 8 else F.I8X4

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Panel: T.Tensor((columns, groups, words, 4), "uint8"),
        PackedActivation: T.Tensor((64, M, 4), "int8"),
        Scales: T.Tensor((groups, columns), "int32"),
        BlockScale: T.Tensor((columns,), "float32"),
        Sums: T.Tensor((groups, M), "int32"),
        ActivationScale: T.Tensor((M,), "float32"),
        Transposed: T.Tensor((N, M), "float32"),
        Grid: T.Tensor((entries * payload,), "int8"),
        Signs: T.Tensor((1024 if format == "IQ3_XXS" else 0,), "int8"),
        Codebook: T.Tensor((16 if format == "IQ4_XS" else 0,), "int8"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Grid": "input", "Codebook": "input",
                     "Signs": "input", "Quantized": "quantized", "Output": "output", "Panel": "panel",
                     "PackedActivation": "activation", "Scales": "scales", "BlockScale": "block_scale",
                     "Sums": "sums", "ActivationScale": "activation_scale", "Transposed": "transposed"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        panel = T.view(Panel, shape=(columns, groups, words, 4), axes=("n", "g", "word", "byte"), abi=True)
        packed = T.view(PackedActivation, shape=(64, M, 4), axes=("ak", "m", "byte"), abi=True)
        scales = T.view(Scales, shape=(groups, columns), axes=("g", "n"), abi=True)
        block_scale = T.view(BlockScale, shape=(columns,), axes=("n",), abi=True)
        sums = T.view(Sums, shape=(groups, M), axes=("g", "m"), abi=True)
        activation_scale = T.view(ActivationScale, shape=(M,), axes=("m",), abi=True)
        transposed = T.view(Transposed, shape=(N, M), axes=("n", "m"), abi=True)
        if format == "IQ4_XS":
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        else:
            grid = T.view(Grid, shape=(entries, payload), encoding=grid_encoding, axes=("entry", "payload"), abi=True)
        if format == "IQ3_XXS":
            signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign", "payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_K", 256, "first", "direct", "rne", 256)
            if format == "IQ4_XS":
                table = T.stage(T.load(codebook))
            with T.level.cols(N, group=columns) as col:
                with T.level.rows(M, group=rows) as row:
                    empty = T.state((columns, rows), "float32", init=0)
                    T.copy(empty, transposed[col, row])
            with T.level.blocks(K, extent=256) as block:
                with T.level.rows(M, group=rows) as row:
                    x = prepared[row, block]
                    q, stored_sums, ds = T.load(x.q), T.load(x.bsum), T.load(x.ds)
                    slab = T.alloc_local((64, rows, 4), "int8")
                    group_sums = T.alloc_local((groups, rows), "int32")
                    for w, r, p in T.Parallel(64, rows, 4):
                        slab[w, r, p] = q[r, T.cast(w, "uint32") * T.uint32(4) + T.cast(p, "uint32")]
                    for g, r in T.Parallel(groups, rows):
                        if groups == 16:
                            group_sums[g, r] = T.cast(stored_sums[r, g], "int32")
                        else:
                            group_sums[g, r] = T.cast(stored_sums[r, g * 2], "int32") + T.cast(stored_sums[r, g * 2 + 1], "int32")
                    T.copy(slab, packed[:, row, :])
                    T.copy(group_sums, sums[:, row])
                    T.copy(ds, activation_scale[row])
                with T.level.cols(N, group=columns) as col:
                    for c in T.serial(columns):
                        w = T.load(weights[col, block][c, :])
                        q, d = w.q, w.d
                        if format == "IQ4_XS":
                            scale_low, scale_high = w.scales_l, w.scales_h
                        elif format != "IQ3_XXS":
                            high, sign_bits, metadata = w.qh, w.signs, w.scales
                        else:
                            metadata = w.metadata
                        factors = T.alloc_local((groups,), "int32")
                        for g in T.Parallel(groups):
                            if format == "IQ2_S":
                                factors[g] = T.cast((metadata[T.cast(g, "uint32") // T.uint32(2)] >> (T.cast(g, "uint32") % T.uint32(2) * T.uint32(4))) & T.uint8(15), "int32") * 2 + 1
                            elif format == "IQ3_XXS":
                                factors[g] = T.cast(metadata[g] >> T.uint32(28), "int32") * 2 + 1
                            elif format == "IQ4_XS":
                                lower = (scale_low[T.cast(g, "uint32") // T.uint32(2)] >> T.cast(g % 2 * 4, "uint8")) & T.uint8(15)
                                upper = T.cast((scale_high[()] >> T.cast(g * 2, "uint16")) & T.uint16(3), "uint8")
                                factors[g] = T.cast(lower | (upper << T.uint8(4)), "int32") - 32
                            else:
                                factors[g] = T.cast(metadata[g], "int32") * 2 + 1
                        if format == "IQ4_XS":
                            decoded = T.load(table, at=q, bounds="in_bounds")
                            values = T.reshape(decoded, (groups, words, 4))
                        else:
                            entries_per_group = 2 if format == "IQ2_S" else 8
                            entry_indices = T.alloc_local((groups, entries_per_group), "uint32")
                            indices = T.alloc_local((groups, entries_per_group, payload), "uint32")
                            for g, e in T.Parallel(groups, entries_per_group):
                                entry = T.cast(g, "uint32") * T.uint32(entries_per_group) + T.cast(e, "uint32")
                                if format == "IQ2_S":
                                    upper = (T.cast(high[entry // T.uint32(4)], "uint32") >> (entry % T.uint32(4) * T.uint32(2))) & T.uint32(3)
                                    index = T.cast(q[entry], "uint32") | (upper << T.uint32(8))
                                elif format == "IQ3_S":
                                    index = T.cast(q[entry], "uint32") | (T.cast(high[entry], "uint32") << T.uint32(8))
                                else:
                                    index = T.cast(q[entry], "uint32")
                                entry_indices[g, e] = index
                            for g, e, p in T.Parallel(groups, entries_per_group, payload):
                                indices[g, e, p] = entry_indices[g, e] * T.uint32(payload) + T.cast(p, "uint32")
                            decoded = T.load(grid.values, at=indices, bounds="in_bounds")
                            values = T.reshape(decoded, (groups, words, 4))
                        if format == "IQ3_XXS":
                            sign_entries = T.alloc_local((groups, 4), "uint32")
                            sign_indices = T.alloc_local((groups, 4, 8), "uint32")
                            for g, e in T.Parallel(groups, 4):
                                sign_entries[g, e] = (metadata[g] >> (T.cast(e, "uint32") * T.uint32(7))) & T.uint32(127)
                            for g, e, p in T.Parallel(groups, 4, 8):
                                sign_indices[g, e, p] = sign_entries[g, e] * T.uint32(8) + T.cast(p, "uint32")
                            selected_signs = T.load(signs.values, at=sign_indices, bounds="in_bounds")
                            sign_values = T.reshape(selected_signs, (groups, words, 4))
                        codes = T.alloc_local((groups, words, 4), "uint8")
                        for g, widx, p in T.Parallel(groups, words, 4):
                            if format == "IQ4_XS":
                                code = values[g, widx, p]
                            elif format == "IQ3_XXS":
                                value = values[g, widx, p]
                                code = T.if_then_else(sign_values[g, widx, p] < 0, -value, value)
                            else:
                                position = T.cast(g, "uint32") * T.uint32(words * 4) + T.cast(widx, "uint32") * T.uint32(4) + T.cast(p, "uint32")
                                value = values[g, widx, p]
                                code = T.if_then_else(sign_bits[position] != 0, -value, value)
                            codes[g, widx, p] = T.cast(code, "uint8") + T.uint8(128)
                        T.copy(codes, panel[c, :, :, :])
                        T.copy(factors, scales[:, c])
                        scale = T.alloc_local((1,), "float32")
                        scale[0] = T.cast(d[()], "float32") * (0.125 if format == "IQ2_S" else 0.25 if format == "IQ3_XXS" else 1.0)
                        T.copy(scale, block_scale[c])
                    wd = T.load(block_scale)
                    with T.level.rows(M, group=rows) as row:
                        integer = T.state((columns, rows), "int32", init=0)
                        for g in T.serial(groups):
                            partial = T.alloc_local((columns, rows), "int32")
                            T.fill(partial, 0)
                            for word in T.unroll(words):
                                wv = T.load(panel[:, g, word, :])
                                xv = T.load(packed[g * words + word, row, :])
                                products = T.alloc_local((columns, rows, 1), "int32")
                                T.mac_groups(wv, xv, products, n=4, into="int32", over="byte")
                                for c, r in T.Parallel(columns, rows):
                                    partial[c, r] += products[c, r, 0]
                            factor = T.load(scales[g, :])
                            group_sums = T.load(sums[g, row])
                            for c, r in T.Parallel(columns, rows):
                                integer[c, r] += (partial[c, r] - T.int32(128) * group_sums[r]) * factor[c]
                        xd, prior = T.load(activation_scale[row]), T.load(transposed[col, row])
                        result = T.alloc_local((columns, rows), "float32")
                        for c, r in T.Parallel(columns, rows):
                            result[c, r] = prior[c, r] + T.cast(integer[c, r], "float32") * (xd[r] * wd[c])
                        T.copy(result, transposed[col, row])
            with T.level.cols(N, group=columns) as col:
                with T.level.rows(M, group=rows) as row:
                    value = T.load(transposed[col, row])
                    result = T.alloc_local((rows, columns), "float32")
                    for r, c in T.Parallel(rows, columns):
                        result[r, c] = value[c, r]
                    T.copy(result, output[row, col])

    return main
