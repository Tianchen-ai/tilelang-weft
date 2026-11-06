"""Record-local indexed dot programs with bounded numerical tiles."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


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
def iq1_planes(M, K):
    """Expose independent record groups, codebook entries and payload lanes."""
    assert M > 0 and K > 0 and K % 256 == 0

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
                    indices = T.alloc_local((8, 4, 8), "uint16")
                    inputs = T.alloc_local((8, 4, 8), "int8")
                    for group, entry, payload in T.Parallel(8, 4, 8):
                        g, e, p = T.cast(group, "uint16"), T.cast(entry, "uint16"), T.cast(payload, "uint16")
                        high = (metadata[g] >> (e * T.uint16(3))) & T.uint16(7)
                        index = T.cast(q[g * T.uint16(4) + e], "uint16") | (high << T.uint16(8))
                        indices[group, entry, payload] = index * T.uint16(8) + p
                        inputs[group, entry, payload] = activation[g * T.uint16(32) + e * T.uint16(8) + p]
                    codes = T.load(grid.values, at=indices, bounds="in_bounds")
                    products = T.alloc_local((8,), "int32")
                    T.reduce_dot(inputs, codes, products, over=("entry", "payload"), acc_dtype="int32", engine="vector")
                    main_terms, corrections = T.alloc_local((8,), "int32"), T.alloc_local((8,), "int32")
                    for group in T.Parallel(8):
                        g = T.cast(group, "uint16")
                        scale = T.cast(((metadata[g] >> T.uint16(12)) & T.uint16(7)) * T.uint16(2) + T.uint16(1), "int32")
                        delta = T.cast((T.cast(metadata[g], "int16") >> T.int16(15)) | T.int16(1), "int32")
                        main_terms[group] = scale * products[group]
                        corrections[group] = scale * delta * (T.cast(sums[g * T.uint16(2)], "int32") + T.cast(sums[g * T.uint16(2) + T.uint16(1)], "int32"))
                    main_sum, correction_sum = T.alloc_local((1,), "int32"), T.alloc_local((1,), "int32")
                    T.reduce_sum(main_terms, main_sum, dim=0)
                    T.reduce_sum(corrections, correction_sum, dim=0)
                    combined = T.cast(main_sum[0], "float32") + T.cast(correction_sum[0], "float32") * 0.125
                    total += (T.cast(d[()], "float32") * ds[()]) * combined
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


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
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=1) as row:
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
                                delta = T.int32(1) - T.cast(word >> T.uint16(15), "int32") * T.int32(2)
                                correction += factor * delta * (T.cast(sums[group * 2], "int32") + T.cast(sums[group * 2 + 1], "int32"))
                    combined = T.cast(integer, "float32")
                    if format == "IQ1_S":
                        combined += T.cast(correction, "float32") * 0.125
                    total += (scale_value * ds[()]) * combined
                result = T.alloc_local((1,), "float32")
                result[0] = total * (1.0 if format == "IQ1_S" else 0.125)
                T.copy(result, output[row])

    return main
