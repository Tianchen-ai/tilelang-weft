"""Explicit subrecord reuse and delayed convergence for short packed products."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows
from .decoding import ternary_digits


@tilelang.jit(target="weft")
def q6_pairs(M, K, lanes=16, records=2):
    """Six-bit pair products with explicit independent record accumulators."""
    assert M == 1 and lanes in (8, 16) and records in (1, 2)
    assert K > 0 and K % (256 * records) == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 210), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=F.Q6_K, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            first = T.state((lanes,), "float32", init=0)
            second = T.state((lanes,), "float32", init=0)
            with T.level.blocks(K, extent=256 * records) as block:
                for part in T.unroll(records):
                    w = T.load(weights[0, block][part * 256 : (part + 1) * 256])
                    x = T.load(values[block][part * 256 : (part + 1) * 256])
                    low, high, scales, d = w.ql, w.qh, w.scales, w.d
                    activation, sums, ds = x.q, x.bsum, x.ds
                    integer = T.alloc_local((lanes,), "int32")
                    T.fill(integer, 0)
                    for group in T.unroll(64 // lanes):
                        codes = T.alloc_local((lanes * 4,), "uint6")
                        inputs = T.alloc_local((lanes * 4,), "int8")
                        products = T.alloc_local((lanes * 2,), "int16")
                        factors = T.alloc_local((lanes * 2,), "int16")
                        scaled = T.alloc_local((lanes,), "int32")
                        for i in T.Parallel(lanes * 4):
                            position = T.uint32(group * lanes * 4) + T.cast(i, "uint32")
                            codes[i] = T.cast(low[position], "uint8") | (T.cast(high[position], "uint8") << T.uint8(4))
                            inputs[i] = activation[position]
                        T.mac_groups(codes, inputs, products, n=2, into="int16", over="pair")
                        for i in T.Parallel(lanes * 2):
                            factors[i] = T.cast(scales[T.uint32(group * lanes // 4) + T.cast(i, "uint32") // T.uint32(8)], "int16")
                        T.mac_groups(products, factors, scaled, n=2, into="int32", over="pair")
                        for i in T.Parallel(lanes):
                            integer[i] += scaled[i]
                    correction = T.alloc_local((16 // lanes, lanes), "int32")
                    reduced = T.alloc_local((lanes,), "int32")
                    for segment, lane in T.Parallel(16 // lanes, lanes):
                        position = T.cast(segment, "uint32") * T.uint32(lanes) + T.cast(lane, "uint32")
                        correction[segment, lane] = T.cast(sums[position], "int32") * T.cast(scales[position], "int32") * 32
                    T.reduce_sum(correction, reduced, dim=0)
                    for lane in T.Parallel(lanes):
                        contribution = T.cast(integer[lane] - reduced[lane], "float32") * (T.cast(d[()], "float32") * ds[()])
                        if part == 0:
                            first[lane] += contribution
                        else:
                            second[lane] += contribution
            if records == 2:
                for lane in T.Parallel(lanes):
                    first[lane] += second[lane]
            result = T.alloc_local((1,), "float32")
            T.reduce_sum(first, result, dim=0)
            T.copy(result, output)

    return main


@T.macro
def iq4_group_rows(weights, values, output, Codebook, M, K, partials, records, axis):
    """Retain scale groups and lane sums in the same dot/projection program."""
    codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
    table = T.load(codebook)
    positive = T.alloc_local((16,), "uint8")
    for i in T.Parallel(16):
        positive[i] = T.cast(table[i], "uint8") + T.uint8(128)
    with T.level.rows(M, group=1, axis=axis) as row:
        first = T.alloc_local((partials,), "float32")
        second = T.alloc_local((partials,), "float32")
        T.fill(first, 0)
        T.fill(second, 0)
        with T.level.blocks(K, extent=256 * records) as block:
            for part in T.unroll(records):
                w = T.load(weights[row, block][0, part * 256 : (part + 1) * 256])
                x = T.load(values[block][part * 256 : (part + 1) * 256])
                q, low, high, d = w.q, w.scales_l, w.scales_h, w.d
                activation, sums, ds = x.q, x.bsum, x.ds
                codes = T.alloc_local((256,), "uint8")
                products = T.alloc_local((64,), "int32")
                scales = T.alloc_local((8,), "int32")
                groups = T.alloc_local((64 // partials, partials), "int32")
                integer = T.alloc_local((partials,), "int32")
                correction = T.alloc_local((16 // partials, partials), "int32")
                reduced = T.alloc_local((partials,), "int32")
                for i in T.Parallel(256):
                    codes[i] = positive[T.cast(q[i], "uint8")]
                T.mac_groups(codes, activation, products, n=4, into="int32", over="k")
                for g in T.Parallel(8):
                    lower = (low[T.cast(g, "uint16") // T.uint16(2)] >> T.cast(g % 2 * 4, "uint8")) & T.uint8(15)
                    upper = T.cast((high[()] >> T.cast(g * 2, "uint16")) & T.uint16(3), "uint8")
                    scales[g] = T.cast(T.cast(lower | (upper << T.uint8(4)), "int8") - T.int8(32), "int32")
                for segment, lane in T.Parallel(64 // partials, partials):
                    position = T.cast(segment, "uint16") * T.uint16(partials) + T.cast(lane, "uint16")
                    groups[segment, lane] = products[position] * scales[position // T.uint16(8)]
                T.reduce_sum(groups, integer, dim=0)
                for segment, lane in T.Parallel(16 // partials, partials):
                    position = T.cast(segment, "uint16") * T.uint16(partials) + T.cast(lane, "uint16")
                    correction[segment, lane] = T.cast(sums[position], "int32") * scales[position // T.uint16(2)] * 128
                T.reduce_sum(correction, reduced, dim=0)
                for lane in T.Parallel(partials):
                    contribution = T.cast(integer[lane] - reduced[lane], "float32") * (T.cast(d[()], "float32") * ds[()])
                    if part == 0:
                        first[lane] += contribution
                    else:
                        second[lane] += contribution
        for lane in T.Parallel(partials):
            first[lane] += second[lane]
        result = T.alloc_local((1,), "float32")
        T.reduce_sum(first, result, dim=0)
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def iq4_groups(M, K, partials=16, records=2):
    """Keep nonlinear integer groups and independent record sums explicit."""
    assert M > 0 and partials in (8, 16) and records in (1, 2)
    assert K > 0 and K % (256 * records) == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 136), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Codebook: T.Tensor((16,), "int8"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=F.IQ4_XS, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            iq4_group_rows(weights, values, output, Codebook, M, K, partials, records, "m")

    return main


@T.macro
def block32_rows(weights, values, output, Codebook, Scale, M, K, format, records, partials, affine, lookup):
    """Share the same record program between packed dots and full projections."""
    if lookup:
        codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        table = T.load(codebook)
    for row in T.serial(M):
        if partials == 1:
            first_scalar = T.alloc_var("float32", init=0)
        else:
            first = T.alloc_local((partials,), "float32")
            T.fill(first, 0)
        if records == 2:
            if partials == 1:
                second_scalar = T.alloc_var("float32", init=0)
            else:
                second = T.alloc_local((partials,), "float32")
                T.fill(second, 0)
        if affine and partials != 1:
            correction = T.alloc_var("float32", init=0)
        result = T.alloc_local((1,), "float32")
        with T.level.blocks(K, extent=32 * records) as block:
            for part in T.unroll(records):
                w = weights[row, block][part * 32 : (part + 1) * 32]
                x = values[block][part * 32 : (part + 1) * 32]
                if partials == 1:
                    q, activation, xd = T.load(w.q), T.load(x.q), T.load(x.d)
                    if format == "MXFP4":
                        exponent = T.load(w.e)
                    else:
                        d = T.load(w.d)
                    if format in ("Q5_0", "Q5_1"):
                        high = T.load(w.qh)
                    if affine:
                        minimum, activation_sum = T.load(w.m), T.load(x.s)
                else:
                    weight_record, input_record = T.load(w), T.load(x)
                    q, activation, xd = weight_record.q, input_record.q, input_record.d
                    if format == "MXFP4":
                        exponent = weight_record.e
                    else:
                        d = weight_record.d
                    if format in ("Q5_0", "Q5_1"):
                        high = weight_record.qh
                    if affine:
                        minimum, activation_sum = weight_record.m, input_record.s
                if lookup:
                    codes = T.load(table, at=q, bounds="in_bounds")
                elif format in ("Q4_1", "Q8_0"):
                    codes = q
                else:
                    codes = T.alloc_local((32,), "int8" if format in ("Q4_0", "Q5_0") else "uint8")
                products = T.alloc_local((partials,), "int32")
                if not lookup and format not in ("Q4_1", "Q8_0"):
                    for k in T.Parallel(32):
                        if format in ("Q5_0", "Q5_1"):
                            code = T.cast(q[k], "uint8") | (T.cast(high[k], "uint8") << T.uint8(4))
                            codes[k] = T.cast(code, "int8") - T.int8(16) if format == "Q5_0" else code
                        else:
                            codes[k] = T.cast(q[k], "int8") - T.int8(8)
                if affine and partials != 1:
                    weight_meta = T.alloc_local((2,), "float16")
                    activation_meta = T.alloc_local((2,), "float16")
                    metadata = T.alloc_local((2,), "float32")
                    weight_meta[0], weight_meta[1] = d[()], minimum[()]
                    activation_meta[0], activation_meta[1] = xd[()], activation_sum[()]
                    for i in T.Parallel(2):
                        metadata[i] = T.cast(weight_meta[i], "float32") * T.cast(activation_meta[i], "float32")
                    scale, offset = metadata[0], metadata[1]
                elif format == "MXFP4":
                    scale = Scale[T.cast(exponent[()], "uint8")] * T.cast(xd[()], "float32")
                else:
                    scale = T.cast(d[()], "float32") * T.cast(xd[()], "float32")
                    if affine:
                        offset = T.cast(minimum[()], "float32") * T.cast(activation_sum[()], "float32")
                if partials == 1:
                    T.reduce_dot(codes, activation, products, over="k", acc_dtype="int32", engine="vector")
                else:
                    T.mac_groups(codes, activation, products, n=4, into="int32", over="k")
                if partials == 1:
                    contribution = scale * T.cast(products[0], "float32")
                    if affine:
                        contribution += offset
                    if records == 1 or part == 0:
                        first_scalar += contribution
                    else:
                        second_scalar += contribution
                else:
                    for p in T.Parallel(partials):
                        if records == 1:
                            first[p] += scale * T.cast(products[p], "float32")
                        else:
                            if part == 0:
                                first[p] += scale * T.cast(products[p], "float32")
                            else:
                                second[p] += scale * T.cast(products[p], "float32")
                if affine and partials != 1:
                    correction += offset
        if partials == 1:
            result[0] = first_scalar + second_scalar if records == 2 else first_scalar
        else:
            if records == 2:
                for p in T.Parallel(partials):
                    first[p] += second[p]
            T.reduce_sum(first, result, dim=0)
        if affine and partials != 1:
            result[0] += correction
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def block32(M, K, format="Q5_0", records=2, partials=8):
    """Choose explicit scalar block dots or independent grouped partials."""
    assert M > 0 and format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "MXFP4")
    assert records in (1, 2) and K > 0 and K % (32 * records) == 0
    assert partials in (1, 8)
    encoding = getattr(F, format)
    affine = format in ("Q4_1", "Q5_1")
    partner = F.Q8_1 if affine else F.Q8_0
    lookup = format in ("IQ4_NL", "MXFP4")
    weight_bytes = K // 32 * (encoding.schema.storage_bits // 8)
    activation_bytes = K // 32 * (partner.schema.storage_bits // 8)

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
            block32_rows(weights, values, output, Codebook, Scale,
                         M, K, format, records, partials, affine, lookup)

    return main


@T.macro
def subblocks_rows(weights, values, output, Codebook, Scale, M, K, format, rows, elements, lookup, lanes, axis):
    if lookup:
        codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        table = T.load(codebook)
        biased = T.alloc_local((16,), "uint8")
        for i in T.Parallel(16):
            biased[i] = T.cast(table[i], "uint8") + T.uint8(128)
    with T.level.rows(M, group=rows, axis=axis) as row:
        first = T.state((rows, lanes), "float32", init=0)
        second = T.state((rows, lanes), "float32", init=0)
        result = T.alloc_local((rows,), "float32")
        with T.level.blocks(K, extent=elements) as block:
            w = T.load(weights[row, block])
            q, d = w.q, w.d
            if lookup:
                all_codes = T.alloc_local((rows, elements), "uint8")
                for r, k in T.Parallel(rows, elements):
                    all_codes[r, k] = biased[T.cast(q[r, k], "uint8")]
            for part in T.unroll(elements // 32):
                x = T.load(values[block][part * 32 : (part + 1) * 32])
                activation, ds = x.q, x.d
                codes = T.alloc_local((rows, 32), "uint8" if lookup else "uint4")
                inputs = T.alloc_local((rows, 32), "int8")
                products = T.alloc_local((rows, lanes), "int32")
                if lookup:
                    bias = T.alloc_local((32,), "uint8")
                    correction = T.alloc_local((lanes,), "int32")
                    T.fill(bias, 128)
                    T.mac_groups(bias, activation, correction, n=4, into="int32", over="k")
                for r, k in T.Parallel(rows, 32):
                    if lookup:
                        codes[r, k] = all_codes[r, part * 32 + k]
                        inputs[r, k] = activation[k]
                    else:
                        positive = q[r, T.uint8(part * 32) + T.cast(k, "uint8")]
                        codes[r, k] = T.cast(1, "uint4")
                        inputs[r, k] = T.if_then_else(positive != 0, activation[k], -activation[k])
                T.mac_groups(codes, inputs, products, n=4, into="int32", over="k")
                for r in T.unroll(rows):
                    if lookup:
                        first_scale = Scale[T.cast(d[r, part * 2], "uint8")]
                        second_scale = Scale[T.cast(d[r, part * 2 + 1], "uint8")]
                    for p in T.Parallel(lanes):
                        if lookup:
                            scale = T.if_then_else(p < lanes // 2, first_scale, second_scale)
                            integer = products[r, p] - correction[p]
                        else:
                            scale = T.cast(d[r], "float32")
                            integer = products[r, p]
                        contribution = T.cast(integer, "float32") * (scale * T.cast(ds[()], "float32"))
                        if part % 2 == 0:
                            first[r, p] += contribution
                        else:
                            second[r, p] += contribution
        for r, p in T.Parallel(rows, lanes):
            first[r, p] += second[r, p]
        T.reduce_sum(first, result, dim=1)
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def subblocks(M, K, format="Q1_0", rows=1):
    """Reuse one weight record across Q8_0 records, with two independent sums."""
    assert format in ("Q1_0", "NVFP4") and K > 0 and K % 128 == 0
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    record_bytes = encoding.schema.storage_bits // 8
    lookup = format == "NVFP4"
    lanes = 8

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // elements * record_bytes), "uint8"),
        Activation: T.Tensor((K // 32 * 34,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if lookup else 0,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_0, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            subblocks_rows(weights, values, output, Codebook, Scale, M, K, format, rows, elements, lookup, lanes, "m")

    return main


@T.macro
def window_products(weights, activation, output, table, scale_table, M, K, format, elements):
    for row in T.serial(M):
        total = T.alloc_var("float32", init=0)
        with T.level.blocks(K, extent=elements) as block:
            weight = T.load(weights[row, block])
            if format == "TQ2_0":
                source = T.load(activation[block])
                codes = T.alloc_local((256,), "int8")
                for k in T.Parallel(256):
                    codes[k] = T.cast(weight.q[k], "int8") - T.int8(1)
                product = T.alloc_local((), "int32")
                T.reduce_dot(source.q, codes, product, over="k", acc_dtype="int32", engine="vector")
                factor = source.ds[()] * T.cast(weight.d[()], "float32")
                total += T.cast(product[()], "float32") * factor
            else:
                subtotal = T.alloc_var("float32", init=0)
                with T.level.subtiles(block, extent=32) as input_block:
                    source = T.load(activation[input_block])
                    if format == "Q1_0":
                        positive = T.load(weight.q, at=input_block)
                        codes = T.alloc_local((32,), "int8")
                        for k in T.Parallel(32):
                            codes[k] = T.cast(positive[k], "int8") * T.int8(2) - T.int8(1)
                        product = T.alloc_local((), "int32")
                        T.reduce_dot(source.q, codes, product, over="k", acc_dtype="int32", engine="vector")
                        subtotal += T.cast(source.d[()], "float32") * T.cast(product[()], "float32")
                    else:
                        with T.level.subtiles(input_block, extent=16) as payload:
                            indices = T.load(weight.q, at=payload)
                            inputs = T.load(source.q, at=payload)
                            exponent = T.load(weight.d, at=payload)
                            codes = T.load(table, at=indices, bounds="in_bounds")
                            product = T.alloc_local((), "int32")
                            T.reduce_dot(inputs, codes, product, over="k", acc_dtype="int32", engine="vector")
                            factor = scale_table[T.cast(exponent[()], "uint8")] * T.cast(source.d[()], "float32")
                            total += factor * T.cast(product[()], "float32")
                if format == "Q1_0":
                    total += T.cast(weight.d[()], "float32") * subtotal
        result = T.alloc_local((1,), "float32")
        result[0] = total
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def window_dot(M, K, format="Q1_0", rows_per_program=1):
    """Close integer dots in source sub-Levels before each scale update."""
    assert format in ("Q1_0", "NVFP4", "TQ2_0") and M > 0 and K > 0
    encoding = getattr(F, format)
    partner = F.Q8_K if format == "TQ2_0" else F.Q8_0
    elements = encoding.schema.elements
    assert K % elements == 0 and rows_per_program > 0 and M % rows_per_program == 0
    lookup = format == "NVFP4"

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // elements * (encoding.schema.storage_bits // 8)), "uint8"),
        Activation: T.Tensor((K // partner.schema.elements * (partner.schema.storage_bits // 8),), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if lookup else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Output": "output", "Codebook": "input", "Scale": "input"}})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        activation = T.view(Activation, shape=(K,), encoding=partner, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(M // rows_per_program, cpu_num_threads=1) as program:
            first = program * rows_per_program
            if lookup:
                codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
                table = T.stage(T.load(codebook))
                window_products(weights[first : first + rows_per_program, :], activation,
                                output[first : first + rows_per_program], table, Scale,
                                rows_per_program, K, format, elements)
            else:
                window_products(weights[first : first + rows_per_program, :], activation,
                                output[first : first + rows_per_program], Codebook, Scale,
                                rows_per_program, K, format, elements)

    return main


@tilelang.jit(target="weft")
def window_projection(M, N, K, format="Q1_0", extreme_policy="first", inverse_policy="reciprocal"):
    """Prepare Q8 activation in the call, then reuse the explicit window dots."""
    assert format in ("Q1_0", "NVFP4", "TQ2_0") and min(M, N, K) > 0
    encoding = getattr(F, format)
    partner = F.Q8_K if format == "TQ2_0" else F.Q8_0
    elements = encoding.schema.elements
    assert K % elements == 0
    lookup = format == "NVFP4"
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K // elements * (encoding.schema.storage_bits // 8)), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if lookup else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Quantized": "quantized", "Output": "output", "Codebook": "input", "Scale": "input"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        activation = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if lookup:
                codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
                table = T.stage(T.load(codebook))
            quantize_rows(source, activation, M, K, "Q8_K" if format == "TQ2_0" else "Q8_0", partner.schema.elements, extreme_policy, inverse_policy, "rne")
            for m in T.serial(M):
                if lookup:
                    window_products(weights, activation[m, :], output[m, :], table, Scale, N, K, format, elements)
                else:
                    window_products(weights, activation[m, :], output[m, :], Codebook, Scale, N, K, format, elements)

    return main


@tilelang.jit(target="weft")
def ternary_lanes(M, K):
    """Retain sixteen exact I16 ternary partials across the five digit planes."""
    assert M > 0 and K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 54), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Powers: T.Tensor((5,), "uint32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=F.TQ1_0, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        powers = T.view(Powers, shape=(5,), axes=("power",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            table = T.load(powers)
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    packed, high, activation, d, ds = w.q, w.qh, x.q, w.d, x.ds
                    first = T.alloc_local((16,), "uint8")
                    second = T.alloc_local((16,), "uint8")
                    third = T.alloc_local((16,), "uint8")
                    partial = T.alloc_local((16,), "int16")
                    for k in T.Parallel(16):
                        lane = T.cast(k, "uint16")
                        first[k] = packed[lane]
                        second[k] = packed[T.uint16(16) + lane]
                        third[k] = packed[T.uint16(32) + lane]
                        power = T.cast(table[lane // T.uint16(4)], "uint8")
                        wrapped = power * T.cast(high[lane % T.uint16(4)], "uint8")
                        decoded = T.cast((T.cast(wrapped, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                        partial[k] = T.cast(activation[T.uint16(240) + lane], "int16") * T.cast(decoded, "int16")
                    for digit in T.unroll(5):
                        power = T.cast(table[digit], "uint8")
                        for k in T.Parallel(16):
                            lane = T.cast(k, "uint16")
                            wrapped0, wrapped1, wrapped2 = first[k] * power, second[k] * power, third[k] * power
                            code0 = T.cast((T.cast(wrapped0, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                            code1 = T.cast((T.cast(wrapped1, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                            code2 = T.cast((T.cast(wrapped2, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                            position = T.cast(digit, "uint16") * T.uint16(32) + lane
                            partial[k] += T.cast(activation[position], "int16") * T.cast(code0, "int16")
                            partial[k] += T.cast(activation[position + T.uint16(16)], "int16") * T.cast(code1, "int16")
                            position2 = T.uint16(160) + T.cast(digit, "uint16") * T.uint16(16) + lane
                            partial[k] += T.cast(activation[position2], "int16") * T.cast(code2, "int16")
                    widened = T.alloc_local((16,), "int32")
                    reduced = T.alloc_local((1,), "int32")
                    for k in T.Parallel(16):
                        widened[k] = T.cast(partial[k], "int32")
                    T.reduce_sum(widened, reduced, dim=0)
                    total += T.cast(reduced[0], "float32") * (T.cast(d[()], "float32") * ds[()])
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


@tilelang.jit(target="weft")
def ternary_planes(M, K):
    """Expose radix digit and payload axes as separate exact dot dimensions."""
    assert M > 0 and K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 54), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Powers: T.Tensor((5,), "uint32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=F.TQ1_0, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        powers = T.view(Powers, shape=(5,), axes=("power",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            table = T.stage(T.load(powers))
            indices = T.alloc_local((4,), "uint32")
            for digit in T.Parallel(4):
                indices[digit] = T.cast(digit, "uint32")
            four_powers = T.load(table, at=indices, bounds="in_bounds")
            with T.level.rows(M, group=1) as row:
                total = T.alloc_var("float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    w, x = T.load(weights[row, block][0, :]), T.load(values[block])
                    q, high, activation, d, ds = w.q, w.qh, x.q, w.d, x.ds
                    block_scale = T.cast(d[()], "float32") * ds[()]
                    first = T.alloc_local((4, 32), "int8")
                    first_inputs = T.alloc_local((4, 32), "int8")
                    for digit, k in T.Parallel(4, 32):
                        wrapped = T.cast(four_powers[digit], "uint8") * T.cast(q[T.cast(k, "uint16")], "uint8")
                        first[digit, k] = T.cast((T.cast(wrapped, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                        first_inputs[digit, k] = activation[T.cast(digit, "uint16") * T.uint16(32) + T.cast(k, "uint16")]
                    p0 = T.alloc_local((4,), "int32")
                    s0 = T.alloc_local((1,), "int32")
                    T.reduce_dot(first_inputs, first, p0, over="k", acc_dtype="int32", engine="vector")
                    T.reduce_sum(p0, s0, dim=0)
                    tail = T.alloc_local((32,), "int8")
                    tail_inputs = T.alloc_local((32,), "int8")
                    for k in T.Parallel(32):
                        wrapped = T.cast(q[T.cast(k, "uint16")], "uint8") * T.cast(table[4], "uint8")
                        tail[k] = T.cast((T.cast(wrapped, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                        tail_inputs[k] = activation[T.uint16(128) + T.cast(k, "uint16")]
                    st = T.alloc_local((1,), "int32")
                    T.reduce_dot(tail_inputs, tail, st, over="k", acc_dtype="int32", engine="vector")
                    second = T.alloc_local((5, 16), "int8")
                    second_inputs = T.alloc_local((5, 16), "int8")
                    for digit, k in T.Parallel(5, 16):
                        wrapped = T.cast(table[digit], "uint8") * T.cast(q[T.uint16(32) + T.cast(k, "uint16")], "uint8")
                        second[digit, k] = T.cast((T.cast(wrapped, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                        second_inputs[digit, k] = activation[T.uint16(160) + T.cast(digit, "uint16") * T.uint16(16) + T.cast(k, "uint16")]
                    p1 = T.alloc_local((5,), "int32")
                    s1 = T.alloc_local((1,), "int32")
                    T.reduce_dot(second_inputs, second, p1, over="k", acc_dtype="int32", engine="vector")
                    T.reduce_sum(p1, s1, dim=0)
                    high_codes = T.alloc_local((4, 4), "int8")
                    high_inputs = T.alloc_local((4, 4), "int8")
                    for digit, k in T.Parallel(4, 4):
                        wrapped = T.cast(four_powers[digit], "uint8") * T.cast(high[T.cast(k, "uint16")], "uint8")
                        high_codes[digit, k] = T.cast((T.cast(wrapped, "uint16") * T.uint16(3)) >> T.uint16(8), "int8") - T.int8(1)
                        high_inputs[digit, k] = activation[T.uint16(240) + T.cast(digit, "uint16") * T.uint16(4) + T.cast(k, "uint16")]
                    p2 = T.alloc_local((4,), "int32")
                    s2 = T.alloc_local((1,), "int32")
                    T.reduce_dot(high_inputs, high_codes, p2, over="k", acc_dtype="int32", engine="vector")
                    T.reduce_sum(p2, s2, dim=0)
                    integer = s0[0] + s1[0] + s2[0] + st[0]
                    total += T.cast(integer, "float32") * block_scale
                result = T.alloc_local((1,), "float32")
                result[0] = total
                T.copy(result, output[row])

    return main


@T.macro
def ternary_rows(weights, values, output, Powers, M, K, format, rows, records, axis):
    packed = format == "TQ1_0"
    with T.level.rows(M, group=rows, axis=axis) as row:
        first = T.state((rows, 16), "float32", init=0)
        second = T.state((rows, 16), "float32", init=0)
        result = T.alloc_local((rows,), "float32")
        with T.level.blocks(K, extent=256 * records) as block:
            for part in T.unroll(records):
                w = T.load(weights[row, block][:, part * 256 : (part + 1) * 256])
                x = T.load(values[block][part * 256 : (part + 1) * 256])
                q, d, activation, ds, sums = w.q, w.d, x.q, x.ds, x.bsum
                codes = T.alloc_local((rows, 256), "uint2")
                products = T.alloc_local((rows, 64), "int32")
                grouped = T.alloc_local((rows, 4, 16), "int32")
                partial = T.alloc_local((rows, 16), "int32")
                if packed:
                    ternary_digits(w, codes, rows)
                else:
                    for r, k in T.Parallel(rows, 256):
                        codes[r, k] = q[r, k]
                T.mac_groups(codes, activation, products, n=4, into="int32", over="k")
                for r, g, p in T.Parallel(rows, 4, 16):
                    grouped[r, g, p] = products[r, g * 16 + p]
                T.reduce_sum(grouped, partial, dim=1)
                for r, p in T.Parallel(rows, 16):
                    contribution = T.cast(partial[r, p] - T.cast(sums[p], "int32"), "float32") * (T.cast(d[r], "float32") * ds[()])
                    if part == 0:
                        first[r, p] += contribution
                    else:
                        second[r, p] += contribution
        for r, p in T.Parallel(rows, 16):
            first[r, p] += second[r, p]
        T.reduce_sum(first, result, dim=1)
        T.copy(result, output[row])


@tilelang.jit(target="weft")
def ternary(M, K, format="TQ1_0", rows=1, records=2):
    """Keep 16 integer lanes and distinct source sums across packed radix records."""
    assert format in ("TQ1_0", "TQ2_0") and records in (1, 2, 4)
    assert K > 0 and K % (256 * records) == 0
    encoding = getattr(F, format)
    size = encoding.schema.storage_bits // 8
    packed = format == "TQ1_0"

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * size), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Powers: T.Tensor((5 if packed else 0,), "uint32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        values = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            ternary_rows(weights, values, output, Powers, M, K, format, rows, records, "m")

    return main


@tilelang.jit(target="weft")
def projection(M, N, K, format="Q1_0", columns=1, records=1, partials=1,
               extreme_policy="first", inverse_policy="direct"):
    """Retain the normal packed dot's lane sums through a complete projection."""
    record32 = format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "MXFP4")
    assert record32 or format in ("Q1_0", "NVFP4", "TQ1_0", "TQ2_0", "IQ4_XS")
    assert M > 0 and N > 0 and N % columns == 0
    assert (partials in (8, 16) if format == "IQ4_XS" else
            partials in (1, 8) and (partials == 1 or record32))
    assert not (record32 or format == "IQ4_XS") or (columns == 1 and records in (1, 2))
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    assert K > 0 and K % (elements * records) == 0
    short = record32 or format in ("Q1_0", "NVFP4")
    affine = format in ("Q4_1", "Q5_1")
    lookup = format in ("IQ4_NL", "MXFP4", "NVFP4", "IQ4_XS")
    activation = "Q8_1" if affine else "Q8_0" if short else "Q8_K"
    partner = getattr(F, activation)
    weight_bytes = K // elements * (encoding.schema.storage_bits // 8)
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if format in ("NVFP4", "MXFP4") else 0,), "float32"),
        Powers: T.Tensor((5 if format == "TQ1_0" else 0,), "uint32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Codebook": "input", "Scale": "input", "Powers": "input",
                     "Quantized": "quantized", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, activation, partner.schema.elements,
                          extreme_policy, inverse_policy, "dynamic" if short else "rne", 64)
            for m in T.serial(M):
                if record32:
                    block32_rows(weights, prepared[m, :], output[m, :], Codebook, Scale,
                                 N, K, format, records, partials, affine, lookup)
                elif format == "IQ4_XS":
                    iq4_group_rows(weights, prepared[m, :], output[m, :], Codebook,
                                   N, K, partials, records, "n")
                elif short:
                    subblocks_rows(weights, prepared[m, :], output[m, :], Codebook, Scale,
                                   N, K, format, columns, elements, format == "NVFP4", 8, "n")
                else:
                    ternary_rows(weights, prepared[m, :], output[m, :], Powers,
                                 N, K, format, columns, records, "n")

    return main
