"""Reusable author macros for decoding one explicitly admitted encoded block."""

from tilelang.weft import language as T


@T.macro
def decode_dequantize(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
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
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        if format == "Q1_0":
            value = T.cast(q[r, k], "int32") * 2 - 1
        elif format in ("Q5_0", "Q5_1"):
            value = T.cast(q[r, k], "int32") | T.cast(high[r, k], "int32") << 4
            if format == "Q5_0":
                value = value - 16
        elif format == "Q3_K":
            value = T.cast(q[r, k], "int32") - (1 - T.cast(high[r, k], "int32")) * 4
        elif format == "Q5_K":
            value = T.cast(q[r, k], "int32") | T.cast(high[r, k], "int32") << 4
        elif format == "Q6_K":
            value = (T.cast(q[r, k], "int32") | T.cast(high[r, k], "int32") << 4) - 32
        else:
            value = T.cast(q[r, k], "int32")
            if format == "Q4_0":
                value = value - 8
            elif format == "TQ2_0":
                value = value - 1
        if format == "Q2_K":
            scale = T.cast(scales[r, k // 16], "int32") & 15
            minimum_value = T.cast(scales[r, k // 16], "int32") >> 4
            result[r, lane] = T.cast(d[r], "float32") * T.cast(scale * value, "float32") - T.cast(dmin[r], "float32") * T.cast(
                minimum_value, "float32"
            )
        elif format == "Q3_K":
            scale = (T.cast(scale_low[r, k // 16], "int32") | T.cast(scale_high[r, k // 16], "int32") << 4) - 32
            result[r, lane] = T.cast(d[r], "float32") * T.cast(scale * value, "float32")
        elif format in ("Q4_K", "Q5_K"):
            result[r, lane] = T.cast(d[r], "float32") * T.cast(scales[r, k // 32], "float32") * T.cast(value, "float32") - T.cast(
                dmin[r], "float32"
            ) * T.cast(minimum[r, k // 32], "float32")
        elif format == "Q6_K":
            result[r, lane] = T.cast(d[r], "float32") * T.cast(T.cast(scales[r, k // 16], "int32") * value, "float32")
        else:
            result[r, lane] = T.cast(d[r], "float32") * T.cast(value, "float32")
        if format in ("Q4_1", "Q5_1"):
            result[r, lane] += T.cast(minimum[r], "float32")


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
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        (entry, payload) = (k // 8, k % 8)
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
        scale = T.cast(d[r], "float32") * (0.5 + T.cast(sc, "float32")) * 0.25
        result[r, lane] = scale * T.cast(
            T.cast(Grid[index * 8 + payload], "int32") * T.cast(Signs[sign_index * 8 + payload], "int32"), "float32"
        )


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
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        (entry, payload) = (k // 4, k % 4)
        metadata = meta[r, entry // 8]
        sign_index = metadata >> entry % 8 // 2 * 7 & 127
        value = Grid[T.cast(q[r, entry], "uint32") * 4 + payload]
        sign = Signs[sign_index * 8 + entry % 2 * 4 + payload]
        scale = T.cast(d[r], "float32") * (0.5 + T.cast(metadata >> 28, "float32")) * 0.5
        result[r, lane] = scale * T.cast(T.cast(value, "int32") * T.cast(sign, "int32"), "float32")


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
def decode_tq1_0(record, result, rows, format, elements, Grid, Signs, Codebook, Scale, HalfBits, Powers, offset=0, width=None):
    width = elements if width is None else width
    (q, high, d) = (record.q, record.qh, record.d)
    for r, lane in T.Parallel(rows, width):
        k = T.cast(offset, "uint8" if elements <= 256 else "uint16") + T.cast(lane, "uint8" if elements <= 256 else "uint16")
        digit = T.alloc_var("int32", 0)
        packed = T.alloc_var("uint8", 0)
        if k < 160:
            digit = k // 32
            packed = q[r, k % 32]
        elif k < 240:
            digit = (k - 160) // 16
            packed = q[r, 32 + (k - 160) % 16]
        else:
            digit = (k - 240) // 4
            packed = high[r, (k - 240) % 4]
        wrapped = T.cast(T.cast(Powers[digit], "uint8") * packed, "uint8")
        decoded = T.cast(T.cast(wrapped, "uint16") * 3 >> 8, "int32") - 1
        result[r, lane] = T.cast(d[r], "float32") * T.cast(decoded, "float32")
