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
    entries = 64 if format == "IQ3_S" else 32
    payload = 4 if format == "IQ3_S" else 8
    entry_indices = T.alloc_local((columns, entries), "uint32")
    indices = T.alloc_local((columns, entries, payload), "uint32")
    factors = T.alloc_local((columns, entries), "int16")
    if format in ("IQ2_XS", "IQ2_XXS"):
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
        else:
            index = T.cast(q[c, e], "uint32") | (T.cast(high[c, e], "uint32") << T.uint32(8))
            factors[c, e] = T.cast(weight.scales[c, entry // T.uint32(8)], "int16") * 2 + 1
        entry_indices[c, e] = index
        if format in ("IQ2_XS", "IQ2_XXS"):
            entry_signs[c, e] = sign_index
    for c, e, p in T.Parallel(columns, entries, payload):
        indices[c, e, p] = entry_indices[c, e] * T.uint32(payload) + T.cast(p, "uint32")
        if format in ("IQ2_XS", "IQ2_XXS"):
            sign_indices[c, e, p] = entry_signs[c, e] * T.uint32(payload) + T.cast(p, "uint32")
    values = T.load(grid.values, at=indices, bounds="in_bounds")
    if format in ("IQ2_XS", "IQ2_XXS"):
        selected_signs = T.load(signs.values, at=sign_indices, bounds="in_bounds")
    codes = T.alloc_local((columns, entries, payload), "int8" if format in ("IQ1_S", "IQ1_M") else "int16")
    for c, e, p in T.Parallel(columns, entries, payload):
        if format in ("IQ1_S", "IQ1_M"):
            code = T.cast(values[c, e, p], "int16")
        elif format in ("IQ2_XS", "IQ2_XXS"):
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
            block_scale[c] = T.cast(weight.d[c], "float32") * (1.0 if format == "IQ3_S" else 0.125)


@tilelang.jit(target="weft")
def projection(M, N, K, format="IQ2_XS", rows=16, columns=16, engine="auto", decode_columns=1):
    assert format in ("IQ1_S", "IQ1_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S")
    assert M > 0 and N > 0 and K > 0 and K % 256 == 0 and M % rows == N % columns == 0
    assert decode_columns > 0 and columns % decode_columns == 0
    encoding = getattr(F, format)
    weight_bytes = K // 256 * (encoding.schema.storage_bits // 8)
    payload = 4 if format == "IQ3_S" else 8
    grid_entries = {"IQ1_S": 2048, "IQ1_M": 2048, "IQ2_S": 1024, "IQ2_XS": 512, "IQ2_XXS": 256, "IQ3_S": 512}[format]
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
        Signs: T.Tensor((1024 if format in ("IQ2_XS", "IQ2_XXS") else 0,), "int8"),
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
        signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign", "payload"), abi=True) if format in ("IQ2_XS", "IQ2_XXS") else 0
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
