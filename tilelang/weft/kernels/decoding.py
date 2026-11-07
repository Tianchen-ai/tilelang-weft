"""Reusable author macros for decoding one explicitly admitted encoded block."""

from tilelang.weft import language as T


@T.macro
def decode_dequantize(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None, single=False, point=None):
    width = elements if width is None else width
    if format == "Q6_K":
        (q, high, scales, d) = (record.ql, record.qh, record.scales, record.d)
    elif format == "Q8_K":
        (q, d) = (record.q, record.ds)
    else:
        (q, d) = (record.q, record.d)
    if format in ("Q5_0", "Q5_1", "Q5_K"):
        high = record.qh
    if format in ("Q4_1", "Q5_1"):
        minimum = record.m
    if format == "Q2_K":
        (scales, dmin) = (record.scales, record.dmin)
    if format == "Q3_K":
        (high, scale_low, scale_high) = (record.hmask, record.scale_low, record.scale_high)
    if format in ("Q4_K", "Q5_K"):
        (scales, minimum, dmin) = (record.sc, record.m, record.dmin)
    if point is not None:
        q = T.load(q, at=point)
        if format in ("Q5_0", "Q5_1", "Q3_K", "Q5_K", "Q6_K"):
            high = T.load(high, at=point)
        if format in ("Q2_K", "Q6_K", "Q4_K", "Q5_K"):
            scales = T.load(scales, at=point)
        if format == "Q3_K":
            scale_low, scale_high = T.load(scale_low, at=point), T.load(scale_high, at=point)
        if format in ("Q4_K", "Q5_K"):
            minimum = T.load(minimum, at=point)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        if format == "Q1_0":
            value = T.cast(q[(k,) if single else (r, k)], "int32") * 2 - 1
        elif format in ("Q5_0", "Q5_1"):
            value = T.cast(q[(k,) if single else (r, k)], "int32") | T.cast(high[(k,) if single else (r, k)], "int32") << 4
            if format == "Q5_0":
                value = value - 16
        elif format == "Q3_K":
            value = T.cast(q[(k,) if single else (r, k)], "int32") - (1 - T.cast(high[(k,) if single else (r, k)], "int32")) * 4
        elif format == "Q5_K":
            value = T.cast(q[(k,) if single else (r, k)], "int32") | T.cast(high[(k,) if single else (r, k)], "int32") << 4
        elif format == "Q6_K":
            value = (T.cast(q[(k,) if single else (r, k)], "int32") | T.cast(high[(k,) if single else (r, k)], "int32") << 4) - 32
        else:
            value = T.cast(q[(k,) if single else (r, k)], "int32")
            if format == "Q4_0":
                value = value - 8
            elif format == "TQ2_0":
                value = value - 1
        if format == "Q2_K":
            metadata = scales[()] if point is not None else scales[(k // 16,) if single else (r, k // 16)]
            scale = T.cast(metadata, "int32") & 15
            minimum_value = T.cast(metadata, "int32") >> 4
            if point is not None:
                result[r, lane] = (T.cast(d[()], "float32") * T.cast(scale, "float32")) * T.cast(value, "float32") - T.cast(dmin[()], "float32") * T.cast(minimum_value, "float32")
            else:
                result[r, lane] = T.cast(d[() if single else (r,)], "float32") * T.cast(scale * value, "float32") - T.cast(dmin[() if single else (r,)], "float32") * T.cast(minimum_value, "float32")
        elif format == "Q3_K":
            low_scale = scale_low[()] if point is not None else scale_low[(k // 16,) if single else (r, k // 16)]
            high_scale = scale_high[()] if point is not None else scale_high[(k // 16,) if single else (r, k // 16)]
            scale = (T.cast(low_scale, "int32") | T.cast(high_scale, "int32") << 4) - 32
            if point is not None:
                result[r, lane] = (T.cast(d[()], "float32") * T.cast(scale, "float32")) * T.cast(value, "float32")
            else:
                result[r, lane] = T.cast(d[() if single else (r,)], "float32") * T.cast(scale * value, "float32")
        elif format in ("Q4_K", "Q5_K"):
            scale = scales[()] if point is not None else scales[(k // 32,) if single else (r, k // 32)]
            minimum_value = minimum[()] if point is not None else minimum[(k // 32,) if single else (r, k // 32)]
            result[r, lane] = T.cast(d[() if single else (r,)], "float32") * T.cast(scale, "float32") * T.cast(value, "float32") - T.cast(
                dmin[() if single else (r,)], "float32"
            ) * T.cast(minimum_value, "float32")
        elif format == "Q6_K":
            scale = scales[()] if point is not None else scales[(k // 16,) if single else (r, k // 16)]
            if point is not None:
                result[r, lane] = (T.cast(d[()], "float32") * T.cast(scale, "float32")) * T.cast(value, "float32")
            else:
                result[r, lane] = T.cast(d[() if single else (r,)], "float32") * T.cast(T.cast(scale, "int32") * value, "float32")
        else:
            result[r, lane] = T.cast(d[() if single else (r,)], "float32") * T.cast(value, "float32")
        if format in ("Q4_1", "Q5_1"):
            result[r, lane] += T.cast(minimum[() if single else (r,)], "float32")


@T.macro
def decode_dequantize_iq4_nl(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, d) = (record.q, record.d)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        result[r, lane] = T.cast(d[r], "float32") * T.cast(Codebook[T.cast(q[r, k], "int32")], "float32")


@T.macro
def decode_iq1_s(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, high, d) = (record.q, record.qh, record.d)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        (entry, payload) = (k // 8, k % 8)
        metadata = T.cast(high[r, entry // 4], "uint32")
        index = T.cast(q[r, entry], "uint32") | (metadata >> entry % 4 * 3 & 7) << 8
        delta = T.if_then_else(metadata & 32768 != 0, -0.125, 0.125)
        scale = T.cast(d[r], "float32") * T.cast((metadata >> 12 & 7) * 2 + 1, "float32")
        result[r, lane] = scale * (T.cast(Grid[index * 8 + payload], "float32") + delta)


@T.macro
def decode_iq1_m(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, high, scales) = (record.q, record.qh, record.scales)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        (entry, payload) = (k // 8, k % 8)
        sc0 = T.cast(scales[r, 0], "uint32") | T.cast(scales[r, 1], "uint32") << 8
        sc1 = T.cast(scales[r, 2], "uint32") | T.cast(scales[r, 3], "uint32") << 8
        sc2 = T.cast(scales[r, 4], "uint32") | T.cast(scales[r, 5], "uint32") << 8
        sc3 = T.cast(scales[r, 6], "uint32") | T.cast(scales[r, 7], "uint32") << 8
        bits = sc0 >> 12 | sc1 >> 8 & 240 | sc2 >> 4 & 3840 | sc3 & 61440
        metadata = T.cast(high[r, entry // 2], "uint32") >> entry % 2 * 4
        index = T.cast(q[r, entry], "uint32") | (metadata & 7) << 8
        group = entry // 4
        sc = T.cast(scales[r, group // 2 * 2], "uint32") | T.cast(scales[r, group // 2 * 2 + 1], "uint32") << 8
        shift = group % 2 * 6 + entry % 4 // 2 * 3
        scale = HalfBits[bits] * T.cast((sc >> shift & 7) * 2 + 1, "float32")
        delta = T.if_then_else(metadata & 8 != 0, -0.125, 0.125)
        result[r, lane] = scale * (T.cast(Grid[index * 8 + payload], "float32") + delta)


@T.macro
def decode_iq2_s(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, high, signs, scales, d) = (record.q, record.qh, record.signs, record.scales, record.d)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        (entry, payload) = (k // 8, k % 8)
        index = T.cast(q[r, entry], "uint32") | (T.cast(high[r, entry // 4], "uint32") >> entry % 4 * 2 & 3) << 8
        sign = 1 - T.cast(signs[r, k], "int32") * 2
        sc = T.cast(scales[r, entry // 4], "uint32") >> entry % 4 // 2 * 4 & 15
        scale = T.cast(d[r], "float32") * (0.5 + T.cast(sc, "float32")) * 0.25
        result[r, lane] = scale * T.cast(T.cast(Grid[index * 8 + payload], "int32") * sign, "float32")


@T.macro
def decode_iq2_indexed(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, d) = (record.q, record.d)
    if format == "IQ2_XS":
        scales = record.scales
    entries = width // 8
    grid_entries = T.alloc_local((rows, entries), "uint32")
    sign_entries = T.alloc_local((rows, entries), "uint32")
    grid_indices = T.alloc_local((rows, entries, 8), "uint32")
    sign_indices = T.alloc_local((rows, entries, 8), "uint32")
    factors = T.alloc_local((rows, entries), "float32")
    for r, local in T.Parallel(rows, entries):
        entry = T.uint32(offset // 8) + T.cast(local, "uint32")
        if format == "IQ2_XS":
            word = T.cast(q[r, entry], "uint32")
            (index, sign_index) = (word & 511, word >> 9)
            sc = T.cast(scales[r, entry // 4], "uint32") >> entry % 4 // 2 * 4 & 15
        else:
            group = entry // 4
            metadata = T.cast(q[r, group * 4 + 2], "uint32") | T.cast(q[r, group * 4 + 3], "uint32") << 16
            word = T.cast(q[r, group * 4 + entry % 4 // 2], "uint32")
            index = word >> entry % 2 * 8 & 255
            sign_index = metadata >> entry % 4 * 7 & 127
            sc = metadata >> 28
        grid_entries[r, local] = index
        sign_entries[r, local] = sign_index
        factors[r, local] = T.cast(d[r], "float32") * (0.5 + T.cast(sc, "float32")) * 0.25
    for r, entry, payload in T.Parallel(rows, entries, 8):
        grid_indices[r, entry, payload] = grid_entries[r, entry] * T.uint32(8) + T.cast(payload, "uint32")
        sign_indices[r, entry, payload] = sign_entries[r, entry] * T.uint32(8) + T.cast(payload, "uint32")
    values = T.load(Grid, at=grid_indices, bounds="in_bounds")
    signs = T.load(Signs, at=sign_indices, bounds="in_bounds")
    decoded = T.alloc_local((rows, entries, 8), "float32")
    for r, entry, payload in T.Parallel(rows, entries, 8):
        decoded[r, entry, payload] = factors[r, entry] * T.cast(
            T.cast(values[r, entry, payload], "int32") * T.cast(signs[r, entry, payload], "int32"), "float32")
    flat = T.reshape(decoded, (rows, width))
    T.copy(flat, result)


@T.macro
def decode_iq3_s(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, high, signs, scales, d) = (record.q, record.qh, record.signs, record.scales, record.d)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        (entry, payload) = (k // 4, k % 4)
        index = T.cast(q[r, entry], "uint32") | T.cast(high[r, entry], "uint32") << 8
        sign = 1 - T.cast(signs[r, k], "int32") * 2
        scale = T.cast(d[r], "float32") * T.cast(T.cast(scales[r, entry // 8], "int32") * 2 + 1, "float32")
        result[r, lane] = scale * T.cast(T.cast(Grid[index * 4 + payload], "int32") * sign, "float32")


@T.macro
def decode_iq3_xxs(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, meta, d) = (record.q, record.metadata, record.d)
    entries = width // 4
    grid_entries = T.alloc_local((rows, entries), "uint32")
    sign_entries = T.alloc_local((rows, entries), "uint32")
    factors = T.alloc_local((rows, entries), "float32")
    grid_indices = T.alloc_local((rows, entries, 4), "uint32")
    sign_indices = T.alloc_local((rows, entries, 4), "uint32")
    for r, local in T.Parallel(rows, entries):
        entry = T.uint32(offset // 4) + T.cast(local, "uint32")
        metadata = T.cast(meta[r, entry // T.uint32(8)], "uint32")
        grid_entries[r, local] = T.cast(q[r, entry], "uint32")
        sign_index = metadata >> (entry % T.uint32(8) // T.uint32(2) * T.uint32(7)) & T.uint32(127)
        sign_entries[r, local] = sign_index * T.uint32(8) + entry % T.uint32(2) * T.uint32(4)
        factors[r, local] = T.cast(d[r], "float32") * (0.5 + T.cast(metadata >> T.uint32(28), "float32")) * 0.5
    for r, entry, payload in T.Parallel(rows, entries, 4):
        grid_indices[r, entry, payload] = grid_entries[r, entry] * T.uint32(4) + T.cast(payload, "uint32")
        sign_indices[r, entry, payload] = sign_entries[r, entry] + T.cast(payload, "uint32")
    values = T.load(Grid, at=grid_indices, bounds="in_bounds")
    signs = T.load(Signs, at=sign_indices, bounds="in_bounds")
    decoded = T.alloc_local((rows, entries, 4), "float32")
    for r, entry, payload in T.Parallel(rows, entries, 4):
        decoded[r, entry, payload] = factors[r, entry] * T.cast(
            T.cast(values[r, entry, payload], "int32") * T.cast(signs[r, entry, payload], "int32"), "float32")
    flat = T.reshape(decoded, (rows, width))
    T.copy(flat, result)


@T.macro
def decode_iq4_xs(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, low, high, d) = (record.q, record.scales_l, record.scales_h, record.d)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        group = k // 32
        scale = T.cast(low[r, group // 2], "int32") >> group % 2 * 4 & 15 | (T.cast(high[r], "int32") >> group * 2 & 3) << 4
        result[r, lane] = T.cast(d[r], "float32") * T.cast(scale - 32, "float32") * T.cast(Codebook[T.cast(q[r, k], "int32")], "float32")


@T.macro
def decode_fp4(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    q = record.q
    if format == "MXFP4":
        exponent = record.e
    else:
        exponent = record.d
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        raw = exponent[r] if format == "MXFP4" else exponent[r, k // 16]
        result[r, lane] = Scale[T.cast(raw, "int32")] * T.cast(Codebook[T.cast(q[r, k], "int32")], "float32")


@T.macro
def ternary_digits(record, result, rows, width=256, offset=0, single=False):
    """Positive radix-3 digits with an explicit byte-width product."""
    q, high = record.q, record.qh
    for r, lane in T.Parallel(rows, width):
        k = T.int32(offset) + T.cast(lane, "int32")
        index = T.if_then_else(k < 160, k % 32, 32 + k % 16)
        packed = T.if_then_else(k < 240, q[(index,) if single else (r, index)],
                               high[(k % 4,) if single else (r, k % 4)])
        digit = T.if_then_else(k < 160, k // 32,
                              T.if_then_else(k < 240, (k - 160) // 16, (k - 240) // 4))
        power = T.if_then_else(digit == 0, T.uint8(1),
                              T.if_then_else(digit == 1, T.uint8(3),
                                             T.if_then_else(digit == 2, T.uint8(9),
                                                            T.if_then_else(digit == 3, T.uint8(27), T.uint8(81)))))
        wrapped = T.cast(power * packed, "uint8")
        result[r, lane] = T.cast(T.cast(wrapped, "uint16") * T.uint16(3) >> T.uint16(8), "uint2")


@T.macro
def decode_tq1_0(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    codes = T.alloc_local((rows, width), "uint2")
    ternary_digits(record, codes, rows, width, offset)
    d = record.d
    for r, lane in T.Parallel(rows, width):
        decoded = T.cast(codes[r, lane], "int32") - 1
        result[r, lane] = T.cast(d[r], "float32") * T.cast(decoded, "float32")
