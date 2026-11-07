"""Record-local indexed dot programs with bounded numerical tiles."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows


@T.macro
def lookup_pair(table, first, second, output):
    indices = T.alloc_local((16,), "uint16")
    for k in T.Parallel(16):
        half = T.cast(k, "uint16") // T.uint16(8)
        entry = first * (T.uint16(1) - half) + second * half
        indices[k] = entry * T.uint16(8) + T.cast(k, "uint16") % T.uint16(8)
    selected = T.load(table, at=indices, bounds="in_bounds")
    T.copy(selected, output)


@tilelang.jit(target="weft")
def scaled_entries(M, K, format="IQ3_S"):
    """Contract each codebook scale group before applying its integer scale."""
    assert format in ("IQ2_S", "IQ3_S", "IQ4_XS") and M > 0 and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    payload = 8 if format == "IQ2_S" else 4
    entries = 2 if format == "IQ2_S" else 8
    groups = 4 if format == "IQ2_S" else 2
    table_entries = 1024 if format == "IQ2_S" else 512
    table_encoding = F.I8X8 if format == "IQ2_S" else F.I8X4

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * (encoding.schema.storage_bits // 8)), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((16 if format == "IQ4_XS" else table_entries * payload,), "int8"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        if format == "IQ4_XS":
            grid = T.view(Grid, shape=(16,), axes=("code",), abi=True)
        else:
            grid = T.view(Grid, shape=(table_entries, payload), encoding=table_encoding,
                          axes=("entry", "payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if format == "IQ4_XS":
                codebook = T.stage(T.load(grid))
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    q, d = w.q, w.d
                    if format == "IQ4_XS":
                        scale_low, scale_high = w.scales_l, w.scales_h
                    else:
                        high, signs, scales = w.qh, w.signs, w.scales
                    activation, ds = x.q, x.ds
                    window_index = T.alloc_var("uint16", init=0)
                    if format == "IQ4_XS":
                        with T.level.subtiles(block, extent=32) as window:
                            low = (scale_low[window_index // T.uint16(2)] >> T.cast(window_index % T.uint16(2) * T.uint16(4), "uint8")) & T.uint8(15)
                            high = T.cast((scale_high[()] >> (window_index * T.uint16(2))) & T.uint16(3), "uint8")
                            factor = T.cast(low | (high << T.uint8(4)), "int32") - T.int32(32)
                            selected = T.load(q, at=window)
                            codes = T.load(codebook, at=selected, bounds="in_bounds")
                            inputs = T.load(activation, at=window)
                            product = T.alloc_local((), "int32")
                            T.reduce_dot(inputs, codes, product, over="k", acc_dtype="int32", engine="vector")
                            scale = (T.cast(d[()], "float32") * ds[()]) * T.cast(factor, "float32")
                            total += scale * T.cast(product[()], "float32")
                            window_index += T.uint16(1)
                    else:
                        integer = T.alloc_var("int32", init=0)
                        with T.level.subtiles(block, extent=64) as window:
                            if format == "IQ2_S":
                                upper_offset = window_index * T.uint16(2)
                                upper_word = T.cast(high[upper_offset], "uint16") | (T.cast(high[upper_offset + T.uint16(1)], "uint16") << T.uint16(8))
                                scale_word = T.cast(scales[upper_offset], "uint16") | (T.cast(scales[upper_offset + T.uint16(1)], "uint16") << T.uint16(8))
                            indices = T.alloc_local((groups, entries, payload), "uint16")
                            inputs = T.alloc_local((groups, entries, payload), "int8")
                            codes = T.alloc_local((groups, entries, payload), "int8")
                            for g, e, p in T.Parallel(groups, entries, payload):
                                entry = window_index * T.uint16(64 // payload) + T.cast(g * entries + e, "uint16")
                                if format == "IQ2_S":
                                    upper = (upper_word >> T.cast((g * entries + e) * 2, "uint16")) & T.uint16(3)
                                else:
                                    upper = T.cast(high[entry], "uint16")
                                index = T.cast(q[entry], "uint16") | (upper << T.uint16(8))
                                indices[g, e, p] = index * T.uint16(payload) + T.cast(p, "uint16")
                            selected = T.load(grid.values, at=indices, bounds="in_bounds")
                            for g, e, p in T.Parallel(groups, entries, payload):
                                coordinate = window_index * T.uint16(64) + T.cast((g * entries + e) * payload + p, "uint16")
                                sign = T.int8(1) - T.cast(signs[coordinate], "int8") * T.int8(2)
                                codes[g, e, p] = selected[g, e, p] * sign
                                inputs[g, e, p] = activation[coordinate]
                            products = T.alloc_local((groups,), "int32")
                            T.reduce_dot(inputs, codes, products, over=("entry", "payload"), acc_dtype="int32", engine="vector")
                            scaled = T.alloc_local((groups,), "int32")
                            for g in T.Parallel(groups):
                                group = window_index * T.uint16(groups) + T.cast(g, "uint16")
                                if format == "IQ2_S":
                                    coefficient = (scale_word >> T.cast(g * 4, "uint16")) & T.uint16(15)
                                else:
                                    coefficient = T.cast(scales[group], "uint16")
                                factor = T.cast(coefficient * T.uint16(2) + T.uint16(1), "int32")
                                scaled[g] = products[g] * factor
                            partial = T.alloc_local((), "int32")
                            T.reduce_sum(scaled, partial, dim=0)
                            integer += partial[()]
                            window_index += T.uint16(1)
                        total += (T.cast(d[()], "float32") * ds[()]) * T.cast(integer, "float32")
                result = T.alloc_local((1,), "float32")
                result[0] = total * (0.125 if format == "IQ2_S" else 1.0)
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def iq1_planes(M, K, groups_per_step=8):
    """Expose independent record groups, codebook entries and payload lanes."""
    assert M > 0 and K > 0 and K % 256 == 0
    assert groups_per_step in (1, 2, 4, 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 50), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((16384,), "int8"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Grid": "input", "Output": "output"}})
        weights = T.view(Weights, shape=(M, K), encoding=F.IQ1_S, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        grid = T.view(Grid, shape=(2048, 8), encoding=F.I8X8, axes=("entry", "payload"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    q, metadata, d, activation, sums, ds = w.q, w.qh, w.d, x.q, x.bsum, x.ds
                    block_scale = T.cast(d[()], "float32") * ds[()]
                    main, correction = T.alloc_var("int32", init=0), T.alloc_var("int32", init=0)
                    for part in T.unroll(8 // groups_per_step):
                        indices = T.alloc_local((groups_per_step, 4, 8), "uint16")
                        inputs = T.alloc_local((groups_per_step, 4, 8), "int8")
                        for group, entry, payload in T.Parallel(groups_per_step, 4, 8):
                            g = T.uint16(part * groups_per_step) + T.cast(group, "uint16")
                            e, p = T.cast(entry, "uint16"), T.cast(payload, "uint16")
                            high = (metadata[g] >> (e * T.uint16(3))) & T.uint16(7)
                            index = T.cast(q[g * T.uint16(4) + e], "uint16") | (high << T.uint16(8))
                            indices[group, entry, payload] = index * T.uint16(8) + p
                            inputs[group, entry, payload] = activation[g * T.uint16(32) + e * T.uint16(8) + p]
                        codes = T.load(grid.values, at=indices, bounds="in_bounds")
                        products = T.alloc_local((groups_per_step,), "int32")
                        T.reduce_dot(inputs, codes, products, over=("entry", "payload"), acc_dtype="int32", engine="vector")
                        main_terms = T.alloc_local((groups_per_step,), "int32")
                        corrections = T.alloc_local((groups_per_step,), "int32")
                        for group in T.Parallel(groups_per_step):
                            g = T.uint16(part * groups_per_step) + T.cast(group, "uint16")
                            scale = T.cast(((metadata[g] >> T.uint16(12)) & T.uint16(7)) * T.uint16(2) + T.uint16(1), "int32")
                            delta = T.cast((T.cast(metadata[g], "int16") >> T.int16(15)) | T.int16(1), "int32")
                            main_terms[group] = scale * products[group]
                            corrections[group] = scale * delta * (T.cast(sums[g * T.uint16(2)], "int32") + T.cast(sums[g * T.uint16(2) + T.uint16(1)], "int32"))
                        main_sum, correction_sum = T.alloc_local((), "int32"), T.alloc_local((), "int32")
                        T.reduce_sum(main_terms, main_sum, dim=0)
                        T.reduce_sum(corrections, correction_sum, dim=0)
                        main += main_sum[()]
                        correction += correction_sum[()]
                    combined = T.cast(main, "float32") + T.cast(correction, "float32") * 0.125
                    total += block_scale * combined
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


@T.macro
def indexed_rows(weights, values, output, grid, signs, HalfBits, M, K, format, groups_per_step, signed_table, groups, axis):
    with T.level.rows(M, group=1, axis=axis) as row:
        total = T.alloc_var("float32", init=0)
        with T.level.blocks(K, extent=256) as block:
            w, x = T.load(weights[row, block][0, :]), T.load(values[block])
            q, activation, ds = w.q, x.q, x.ds
            if format == "IQ1_M":
                metadata = w.scales
                sc0 = T.cast(metadata[0], "uint16") | (T.cast(metadata[1], "uint16") << T.uint16(8))
                sc1 = T.cast(metadata[2], "uint16") | (T.cast(metadata[3], "uint16") << T.uint16(8))
                sc2 = T.cast(metadata[4], "uint16") | (T.cast(metadata[5], "uint16") << T.uint16(8))
                sc3 = T.cast(metadata[6], "uint16") | (T.cast(metadata[7], "uint16") << T.uint16(8))
                bits = (sc0 >> T.uint16(12)) | ((sc1 >> T.uint16(8)) & T.uint16(240))
                bits |= ((sc2 >> T.uint16(4)) & T.uint16(3840)) | (sc3 & T.uint16(61440))
                scale_value = HalfBits[bits]
            else:
                d = w.d
                scale_value = T.cast(d[()], "float32")
            integer = T.alloc_var("int32", init=0)
            correction = T.alloc_var("int32", init=0)
            if format in ("IQ1_S", "IQ1_M", "IQ2_S"):
                high = w.qh
            if format in ("IQ2_XS", "IQ2_S"):
                metadata = w.scales
            if format == "IQ1_S":
                sums = x.bsum
            if format == "IQ2_S":
                sign_bits = w.signs
            for step in T.serial(groups // groups_per_step):
                for local_group in T.unroll(groups_per_step):
                    group = step * groups_per_step + local_group
                    if format == "IQ1_S":
                        word = T.cast(high[group], "uint16")
                        factor = T.cast(((word >> T.uint16(12)) & T.uint16(7)) * T.uint16(2) + T.uint16(1), "int32")
                    elif format == "IQ1_M":
                        word = T.cast(high[group], "uint16")
                        scale_word = sc0 if group < 4 else sc1 if group < 8 else sc2 if group < 12 else sc3
                        factor = T.cast(((scale_word >> T.uint16(group % 4 * 3)) & T.uint16(7)) * T.uint16(2) + T.uint16(1), "int32")
                    elif format == "IQ2_XXS":
                        word = T.cast(q[group * 4 + 2], "uint32") | (T.cast(q[group * 4 + 3], "uint32") << T.uint32(16))
                        factor = T.cast((word >> T.uint32(28)) * T.uint32(2) + T.uint32(1), "int32")
                    else:
                        word = T.cast(metadata[group // 2], "uint16")
                        factor = T.cast(((word >> T.uint16(group % 2 * 4)) & T.uint16(15)) * T.uint16(2) + T.uint16(1), "int32")
                    partial = T.alloc_local((16,), "int16")
                    T.fill(partial, 0)
                    for pair in T.unroll(2 if format in ("IQ1_S", "IQ2_XXS") else 1):
                        if format == "IQ1_S":
                            first = T.cast(q[group * 4 + pair * 2], "uint16") | (((word >> T.uint16(pair * 6)) & T.uint16(7)) << T.uint16(8))
                            second = T.cast(q[group * 4 + pair * 2 + 1], "uint16") | (((word >> T.uint16(pair * 6 + 3)) & T.uint16(7)) << T.uint16(8))
                        elif format == "IQ1_M":
                            first = T.cast(q[group * 2], "uint16") | ((word & T.uint16(7)) << T.uint16(8))
                            second = T.cast(q[group * 2 + 1], "uint16") | (((word >> T.uint16(4)) & T.uint16(7)) << T.uint16(8))
                        elif format == "IQ2_XXS":
                            indices_word = T.cast(q[group * 4 + pair], "uint16")
                            first, second = indices_word & T.uint16(255), indices_word >> T.uint16(8)
                        elif format == "IQ2_XS":
                            first = T.cast(q[group * 2], "uint16") & T.uint16(511)
                            second = T.cast(q[group * 2 + 1], "uint16") & T.uint16(511)
                        else:
                            upper = T.cast(high[group // 2], "uint16") >> T.uint16(group % 2 * 4)
                            first = T.cast(q[group * 2], "uint16") | ((upper & T.uint16(3)) << T.uint16(8))
                            second = T.cast(q[group * 2 + 1], "uint16") | (((upper >> T.uint16(2)) & T.uint16(3)) << T.uint16(8))
                        weight = T.alloc_local((16,), "int8")
                        lookup_pair(grid.values, first, second, weight)
                        if signed_table:
                            if format == "IQ2_XXS":
                                sign_first = T.cast(word >> T.uint32(pair * 14), "uint16") & T.uint16(127)
                                sign_second = T.cast(word >> T.uint32(pair * 14 + 7), "uint16") & T.uint16(127)
                            else:
                                sign_first = T.cast(q[group * 2], "uint16") >> T.uint16(9)
                                sign_second = T.cast(q[group * 2 + 1], "uint16") >> T.uint16(9)
                            sign = T.alloc_local((16,), "int8")
                            lookup_pair(signs.values, sign_first, sign_second, sign)
                        for k in T.Parallel(16):
                            coordinate = T.uint16(group * (32 if format in ("IQ1_S", "IQ2_XXS") else 16) + pair * 16) + T.cast(k, "uint16")
                            if signed_table:
                                code = weight[k] * sign[k]
                            elif format == "IQ1_M":
                                delta_shift = T.uint16(3) + T.cast(k, "uint16") // T.uint16(8) * T.uint16(4)
                                delta = T.int8(1) - T.cast((word >> delta_shift) & T.uint16(1), "int8") * T.int8(2)
                                code = weight[k] * T.int8(8) + delta
                            elif format == "IQ2_S":
                                code = weight[k] * (T.int8(1) - T.cast(sign_bits[coordinate], "int8") * T.int8(2))
                            else:
                                code = weight[k]
                            partial[k] += T.cast(activation[coordinate], "int16") * T.cast(code, "int16")
                    widened = T.alloc_local((16,), "int32")
                    reduced = T.alloc_local((1,), "int32")
                    for k in T.Parallel(16):
                        widened[k] = T.cast(partial[k], "int32")
                    T.reduce_sum(widened, reduced, dim=0)
                    integer += factor * reduced[0]
                    if format == "IQ1_S":
                        delta = T.cast(T.cast(word, "int16") >> T.int16(15), "int32") | T.int32(1)
                        correction += factor * delta * (T.cast(sums[group * 2], "int32") + T.cast(sums[group * 2 + 1], "int32"))
            combined = T.cast(integer, "float32")
            if format == "IQ1_S":
                combined += T.cast(correction, "float32") * 0.125
            total += (scale_value * ds[()]) * combined
        result = T.alloc_local((1,), "float32")
        result[0] = total * (1.0 if format == "IQ1_S" else 0.125)
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def vec_dot(M, K, format="IQ1_S", groups_per_step=1):
    """Keep exact integer groups separate from each record's float scaling.

    The sixteen-element tile expresses numerical parallelism; it does not
    prescribe a vector length. Both codebook values and signs remain explicit
    input storage. The signed pair sums fit I16 before the I32 reduction.
    """
    assert format in ("IQ1_S", "IQ1_M", "IQ2_XS", "IQ2_XXS", "IQ2_S")
    assert M > 0 and K > 0 and K % 256 == 0
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8
    entries = {"IQ1_S": 2048, "IQ1_M": 2048, "IQ2_XS": 512, "IQ2_XXS": 256, "IQ2_S": 1024}[format]
    signed_table = format in ("IQ2_XS", "IQ2_XXS")
    groups = 8 if format in ("IQ1_S", "IQ2_XXS") else 16
    assert groups_per_step in (1, 2, 4, 8) and groups % groups_per_step == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((entries * 8,), "int8"),
        Signs: T.Tensor((1024 if signed_table else 0,), "int8"),
        HalfBits: T.Tensor((65536 if format == "IQ1_M" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Grid": "input", "Signs": "input", "HalfBits": "input", "Output": "output"}})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        grid = T.view(Grid, shape=(entries, 8), encoding=F.I8X8, axes=("entry", "payload"), abi=True)
        if signed_table:
            signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
        else:
            signs = grid
        with T.Kernel(1, cpu_num_threads=1):
            indexed_rows(weights, values, output, grid, signs, HalfBits, M, K, format, groups_per_step, signed_table, groups, "m")

    return main


@tilelang.jit(target="weft")
def projection(M, N, K, format="IQ1_S", groups_per_step=8, extreme_policy="first"):
    """Prepare Q8_K once and reuse the record-local indexed partials."""
    assert format in ("IQ1_S", "IQ1_M", "IQ2_XS", "IQ2_XXS", "IQ2_S")
    assert min(M, N, K) > 0 and K % 256 == 0
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8
    entries = {"IQ1_S": 2048, "IQ1_M": 2048, "IQ2_XS": 512, "IQ2_XXS": 256, "IQ2_S": 1024}[format]
    signed_table = format in ("IQ2_XS", "IQ2_XXS")
    groups = 8 if format in ("IQ1_S", "IQ2_XXS") else 16
    assert groups_per_step in (1, 2, 4, 8) and groups % groups_per_step == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K // 256 * record_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Grid: T.Tensor((entries * 8,), "int8"),
        Signs: T.Tensor((1024 if signed_table else 0,), "int8"),
        HalfBits: T.Tensor((65536 if format == "IQ1_M" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Grid": "input", "Signs": "input", "HalfBits": "input", "Quantized": "quantized", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        grid = T.view(Grid, shape=(entries, 8), encoding=F.I8X8, axes=("entry", "payload"), abi=True)
        if signed_table:
            signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
        else:
            signs = grid
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_K", 256, extreme_policy, "direct", "rne", 64)
            for row in T.serial(M):
                indexed_rows(weights, prepared[row, :], output[row, :], grid, signs, HalfBits,
                             N, K, format, groups_per_step, signed_table, groups, "n")

    return main
