"""Packed author programs with explicit integer partial and scale boundaries."""

from tilelang.weft import language as T


@T.macro
def block32(weight, activation, total, rows, format, Codebook, Scale):
    q, x, xd = weight.q, activation.q, activation.d
    d = weight.e if format == "MXFP4" else weight.d
    if format in ("Q5_0", "Q5_1"):
        high = weight.qh
    if format in ("Q4_1", "Q5_1"):
        minimum, activation_sum = weight.m, activation.s
    product = T.alloc_local((rows,), "int32")
    if format in ("IQ4_NL", "MXFP4"):
        codes = T.load(Codebook, at=q, bounds="in_bounds")
    else:
        codes = T.alloc_local((rows, 32), "int8")
        for r, k in T.Parallel(rows, 32):
            if format in ("Q5_0", "Q5_1"):
                value = T.cast(q[r, k], "int8") | (T.cast(high[r, k], "int8") << 4)
                codes[r, k] = value - (16 if format == "Q5_0" else 0)
            else:
                codes[r, k] = T.cast(q[r, k], "int8") - (8 if format == "Q4_0" else 0)
    T.reduce_dot(codes, x, product, over="k", acc_dtype="int32", engine="vector")
    for r in T.Parallel(rows):
        scale = Scale[T.cast(d[r], "int32")] if format == "MXFP4" else T.cast(d[r], "float32")
        total[r] += (scale * T.cast(xd[()], "float32")) * T.cast(product[r], "float32")
        if format in ("Q4_1", "Q5_1"):
            total[r] += T.cast(minimum[r], "float32") * T.cast(activation_sum[()], "float32")


@T.macro
def block256(weight, activation, total, rows, format):
    q, x, xd = weight.q, activation.q, activation.ds
    d = weight.ds if format == "Q8_K" else weight.d
    if format == "Q2_K":
        scales, dmin, sums = weight.scales, weight.dmin, activation.bsum
    elif format == "Q3_K":
        high, low_scale, high_scale = weight.hmask, weight.scale_low, weight.scale_high
    elif format == "Q5_K":
        high, scales, minimum, dmin, sums = weight.qh, weight.sc, weight.m, weight.dmin, activation.bsum
    group = 16 if format in ("Q2_K", "Q3_K") else 32 if format == "Q5_K" else 256
    codes = T.alloc_local((rows, 256 // group, group), "int8")
    values = T.alloc_local((256 // group, group), "int8")
    products = T.alloc_local((rows, 256 // group), "int32")
    weighted = T.alloc_local((rows, 256 // group), "int32")
    subtotal = T.alloc_local((rows,), "int32")
    for g, k in T.Parallel(256 // group, group):
        coordinate = T.cast(k, "uint8") if group == 256 else T.cast(g, "uint8") * T.uint8(group) + T.cast(k, "uint8")
        values[g, k] = x[coordinate]
    for r, g, k in T.Parallel(rows, 256 // group, group):
        coordinate = T.cast(k, "uint8") if group == 256 else T.cast(g, "uint8") * T.uint8(group) + T.cast(k, "uint8")
        if format == "Q3_K":
            codes[r, g, k] = T.cast(q[r, coordinate], "int8") - (1 - T.cast(high[r, coordinate], "int8")) * 4
        elif format == "Q5_K":
            codes[r, g, k] = T.cast(q[r, coordinate], "int8") | (T.cast(high[r, coordinate], "int8") << 4)
        else:
            codes[r, g, k] = T.cast(q[r, coordinate], "int8") - (1 if format == "TQ2_0" else 0)
    T.reduce_dot(codes, values, products, over="k", acc_dtype="int32", engine="vector")
    for r, g in T.Parallel(rows, 256 // group):
        if format == "Q2_K":
            weighted[r, g] = products[r, g] * (T.cast(scales[r, g], "int32") & 15)
        elif format == "Q3_K":
            scale = (T.cast(low_scale[r, g], "int32") | (T.cast(high_scale[r, g], "int32") << 4)) - 32
            weighted[r, g] = products[r, g] * scale
        elif format == "Q5_K":
            weighted[r, g] = products[r, g] * T.cast(scales[r, g], "int32")
        else:
            weighted[r, g] = products[r, g]
    T.reduce_sum(weighted, subtotal, dim=1)
    if format in ("Q2_K", "Q5_K"):
        corrections = T.alloc_local((rows, 256 // group), "int32")
        correction_sum = T.alloc_local((rows,), "int32")
        for r, g in T.Parallel(rows, 256 // group):
            if format == "Q2_K":
                corrections[r, g] = (T.cast(scales[r, g], "int32") >> 4) * T.cast(sums[g], "int32")
            else:
                corrections[r, g] = T.cast(minimum[r, g], "int32") * (T.cast(sums[2 * g], "int32") + T.cast(sums[2 * g + 1], "int32"))
        T.reduce_sum(corrections, correction_sum, dim=1)
        for r in T.Parallel(rows):
            total[r] -= (T.cast(xd[()], "float32") * T.cast(dmin[r], "float32")) * T.cast(correction_sum[r], "float32")
    for r in T.Parallel(rows):
        total[r] += (T.cast(xd[()], "float32") * T.cast(d[r], "float32")) * T.cast(subtotal[r], "float32")


@T.macro
def iq2_indexed(weight, activation, total, rows, format, Grid, Signs):
    q, d, x, xd = weight.q, weight.d, activation.q, activation.ds
    if format == "IQ2_XS":
        scales = weight.scales
    for r in T.unroll(rows):
        integer = T.alloc_var("int32", init=0)
        for group in T.unroll(16 if format == "IQ2_XS" else 8):
            subtotal = T.alloc_var("int32", init=0)
            for pair in T.unroll(1 if format == "IQ2_XS" else 2):
                values = T.alloc_local((16,), "int8")
                grid_indices = T.alloc_local((16,), "uint16")
                sign_indices = T.alloc_local((16,), "uint16")
                codes = T.alloc_local((16,), "int8")
                products = T.alloc_local((1,), "int32")
                if format == "IQ2_XS":
                    first, second = q[r, group * 2], q[r, group * 2 + 1]
                else:
                    metadata = T.cast(q[r, group * 4 + 2], "uint32") | (T.cast(q[r, group * 4 + 3], "uint32") << T.uint32(16))
                    word = q[r, group * 4 + pair]
                    first = T.cast(metadata >> T.uint32(pair * 14), "uint16") & T.uint16(127)
                    second = T.cast(metadata >> T.uint32(pair * 14 + 7), "uint16") & T.uint16(127)
                for k in T.Parallel(16):
                    coordinate = T.uint16(group * (16 if format == "IQ2_XS" else 32) + pair * 16) + T.cast(k, "uint16")
                    values[k] = x[coordinate]
                    half = T.cast(k, "uint16") // T.uint16(8)
                    if format == "IQ2_XS":
                        index = (first & T.uint16(511)) * (T.uint16(1) - half) + (second & T.uint16(511)) * half
                        sign_index = (first >> T.uint16(9)) * (T.uint16(1) - half) + (second >> T.uint16(9)) * half
                    else:
                        index = (word & T.uint16(255)) * (T.uint16(1) - half) + (word >> T.uint16(8)) * half
                        sign_index = first * (T.uint16(1) - half) + second * half
                    grid_indices[k] = index * T.uint16(8) + T.cast(k, "uint16") % T.uint16(8)
                    sign_indices[k] = sign_index * T.uint16(8) + T.cast(k, "uint16") % T.uint16(8)
                positive = T.load(Grid, at=grid_indices, bounds="in_bounds")
                sign = T.load(Signs, at=sign_indices, bounds="in_bounds")
                for k in T.Parallel(16):
                    codes[k] = positive[k] * sign[k]
                T.reduce_dot(codes, values, products, over="payload", acc_dtype="int32", engine="vector")
                subtotal += products[0]
            if format == "IQ2_XS":
                sc = (T.cast(scales[r, group // 2], "uint32") >> T.uint32(group % 2 * 4)) & T.uint32(15)
            else:
                metadata = T.cast(q[r, group * 4 + 2], "uint32") | (T.cast(q[r, group * 4 + 3], "uint32") << T.uint32(16))
                sc = metadata >> T.uint32(28)
            integer += subtotal * T.cast(sc * T.uint32(2) + T.uint32(1), "int32")
        total[r] += T.cast(integer, "float32") * (T.cast(d[r], "float32") * xd[()]) * 0.125


@T.macro
def iq1(weight, activation, total, rows, format, Grid, HalfBits):
    q, high, x, xd = weight.q, weight.qh, activation.q, activation.ds
    if format == "IQ1_M":
        scales = weight.scales
    else:
        d = weight.d
    group_size = 16 if format == "IQ1_M" else 32
    groups = 256 // group_size
    codes = T.alloc_local((rows, groups, group_size), "int8")
    values = T.alloc_local((groups, group_size), "int8")
    products = T.alloc_local((rows, groups), "int32")
    weighted = T.alloc_local((rows, groups), "int32")
    subtotal = T.alloc_local((rows,), "int32")
    for group, k in T.Parallel(groups, group_size):
        values[group, k] = x[T.cast(group, "uint8") * T.uint8(group_size) + T.cast(k, "uint8")]
    for r, group, k in T.Parallel(rows, groups, group_size):
        entry, payload = group * (group_size // 8) + k // 8, k % 8
        if format == "IQ1_M":
            metadata = T.cast(high[r, entry // 2], "uint32") >> (entry % 2 * 4)
            delta = 1 - T.cast((metadata >> 3) & 1, "int32") * 2
        else:
            metadata = T.cast(high[r, group], "uint32")
            delta = 1 - T.cast(metadata >> 15, "int32") * 2
        index = T.cast(q[r, entry], "uint32") | (((metadata >> (0 if format == "IQ1_M" else entry % 4 * 3)) & 7) << 8)
        value = T.cast(Grid[index * 8 + payload], "int32")
        codes[r, group, k] = T.cast(value * 8 + delta, "int8")
    T.reduce_dot(codes, values, products, over="lane", acc_dtype="int32", engine="vector")
    for r, group in T.Parallel(rows, groups):
        if format == "IQ1_M":
            sc = T.cast(scales[r, group // 4 * 2], "uint32") | (T.cast(scales[r, group // 4 * 2 + 1], "uint32") << 8)
            scale = (sc >> (group % 4 * 3)) & 7
            weighted[r, group] = products[r, group] * T.cast(scale * 2 + 1, "int32")
        else:
            scale = (T.cast(high[r, group], "uint32") >> 12) & 7
            weighted[r, group] = products[r, group] * T.cast(scale * 2 + 1, "int32")
    T.reduce_sum(weighted, subtotal, dim=1)
    for r in T.Parallel(rows):
        if format == "IQ1_M":
            sc0 = T.cast(scales[r, 0], "uint32") | (T.cast(scales[r, 1], "uint32") << 8)
            sc1 = T.cast(scales[r, 2], "uint32") | (T.cast(scales[r, 3], "uint32") << 8)
            sc2 = T.cast(scales[r, 4], "uint32") | (T.cast(scales[r, 5], "uint32") << 8)
            sc3 = T.cast(scales[r, 6], "uint32") | (T.cast(scales[r, 7], "uint32") << 8)
            bits = (sc0 >> 12) | ((sc1 >> 8) & 240) | ((sc2 >> 4) & 3840) | (sc3 & 61440)
            total[r] += T.cast(subtotal[r], "float32") * (HalfBits[bits] * xd[()]) * 0.125
        else:
            total[r] += T.cast(subtotal[r], "float32") * (T.cast(d[r], "float32") * xd[()]) * 0.125


@T.macro
def scaled_lookup(weight, activation, total, rows, format, Grid, Signs, Codebook):
    q, d, x, xd = weight.q, weight.d, activation.q, activation.ds
    if format in ("IQ2_S", "IQ3_S"):
        high, signs, scales = weight.qh, weight.signs, weight.scales
    elif format == "IQ3_XXS":
        metadata = weight.metadata
    else:
        low, high = weight.scales_l, weight.scales_h
    group = 16 if format == "IQ2_S" else 32
    codes = T.alloc_local((rows, 256 // group, group), "int8")
    values = T.alloc_local((256 // group, group), "int8")
    products = T.alloc_local((rows, 256 // group), "int32")
    weighted = T.alloc_local((rows, 256 // group), "int32")
    subtotal = T.alloc_local((rows,), "int32")
    for g, lane in T.Parallel(256 // group, group):
        values[g, lane] = x[T.cast(g, "uint8") * T.uint8(group) + T.cast(lane, "uint8")]
    for r, g, lane in T.Parallel(rows, 256 // group, group):
        k = g * group + lane
        if format == "IQ2_S":
            entry, payload = k // 8, k % 8
            index = T.cast(q[r, entry], "uint32") | (((T.cast(high[r, entry // 4], "uint32") >> (entry % 4 * 2)) & 3) << 8)
            value = T.cast(Grid[index * 8 + payload], "int32") * (1 - T.cast(signs[r, k], "int32") * 2)
        elif format == "IQ3_S":
            entry, payload = k // 4, k % 4
            index = T.cast(q[r, entry], "uint32") | (T.cast(high[r, entry], "uint32") << 8)
            value = T.cast(Grid[index * 4 + payload], "int32") * (1 - T.cast(signs[r, k], "int32") * 2)
        elif format == "IQ3_XXS":
            entry, payload = k // 4, k % 4
            meta = metadata[r, entry // 8]
            sign_index = (meta >> (entry % 8 // 2 * 7)) & 127
            value = T.cast(Grid[T.cast(q[r, entry], "uint32") * 4 + payload], "int32") * T.cast(
                Signs[sign_index * 8 + entry % 2 * 4 + payload], "int32"
            )
        else:
            value = T.cast(Codebook[T.cast(q[r, k], "int32")], "int32")
        codes[r, g, lane] = T.cast(value, "int8")
    T.reduce_dot(codes, values, products, over="lane", acc_dtype="int32", engine="vector")
    for r, g in T.Parallel(rows, 256 // group):
        if format == "IQ2_S":
            scale = ((T.cast(scales[r, g // 2], "uint32") >> (g % 2 * 4)) & 15) * 2 + 1
        elif format == "IQ3_S":
            scale = T.cast(scales[r, g], "uint32") * 2 + 1
        elif format == "IQ3_XXS":
            scale = (metadata[r, g] >> 28) * 2 + 1
        else:
            scale = ((T.cast(low[r, g // 2], "uint32") >> (g % 2 * 4)) & 15) | (((T.cast(high[r], "uint32") >> (g * 2)) & 3) << 4)
        weighted[r, g] = products[r, g] * (T.cast(scale, "int32") - (32 if format == "IQ4_XS" else 0))
    T.reduce_sum(weighted, subtotal, dim=1)
    for r in T.Parallel(rows):
        total[r] += (
            T.cast(subtotal[r], "float32")
            * (T.cast(d[r], "float32") * xd[()])
            * (0.125 if format == "IQ2_S" else 0.25 if format == "IQ3_XXS" else 1.0)
        )


@T.macro
def ternary(weight, activation, total, rows, Powers):
    q, high, d, x, xd = weight.q, weight.qh, weight.d, activation.q, activation.ds
    codes = T.alloc_local((rows, 256), "int8")
    products = T.alloc_local((rows,), "int32")
    for r, k in T.Parallel(rows, 256):
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
        codes[r, k] = T.cast(T.cast(wrapped, "uint16") * 3 >> 8, "int8") - 1
    T.reduce_dot(codes, x, products, over="k", acc_dtype="int32", engine="vector")
    for r in T.Parallel(rows):
        total[r] += T.cast(products[r], "float32") * (T.cast(d[r], "float32") * xd[()])
