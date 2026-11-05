"""Source-declared integer partials and separate scale/correction branches."""

from tilelang.weft import language as T


@T.macro
def q4_k_single(weight, activation, total):
    q, x, scale, minimum = weight.q, activation.q, weight.sc, weight.m
    d, dmin, ds, bsum = weight.d, weight.dmin, activation.ds, activation.bsum
    codes = T.alloc_local((8, 32), "uint4")
    values = T.alloc_local((8, 32), "int8")
    partial = T.alloc_local((8,), "int32")
    weighted = T.alloc_local((8,), "int32")
    corrections = T.alloc_local((8,), "int32")
    product_sum = T.alloc_local((1,), "int32")
    correction_sum = T.alloc_local((1,), "int32")
    for group, k in T.Parallel(8, 32):
        coordinate = T.cast(group, "uint8") * T.uint8(32) + T.cast(k, "uint8")
        codes[group, k] = q[coordinate]
        values[group, k] = x[coordinate]
    T.reduce_dot(codes, values, partial, over="k", acc_dtype="int32", engine="vector")
    for group in T.Parallel(8):
        weighted[group] = partial[group] * T.cast(scale[T.cast(group, "uint8")], "int32")
        corrections[group] = T.cast(minimum[T.cast(group, "uint32")], "int32") * (
            T.cast(bsum[T.cast(group, "uint32") * T.uint32(2)], "int32")
            + T.cast(bsum[T.cast(group, "uint32") * T.uint32(2) + T.uint32(1)], "int32")
        )
    T.reduce_sum(weighted, product_sum, dim=0)
    T.reduce_sum(corrections, correction_sum, dim=0)
    total[0] -= (ds[()] * T.cast(dmin[()], "float32")) * T.cast(correction_sum[0], "float32")
    total[0] += (ds[()] * T.cast(d[()], "float32")) * T.cast(product_sum[0], "float32")


@T.macro
def q4_k(weight, activation, total, rows, acc_dtype="float32"):
    q, x, scale, minimum = weight.q, activation.q, weight.sc, weight.m
    d, dmin, ds, bsum = weight.d, weight.dmin, activation.ds, activation.bsum
    partial = T.alloc_local((rows, 8), "int32")
    codes = T.alloc_local((rows, 8, 32), "uint4")
    values = T.alloc_local((8, 32), "int8")
    weighted = T.alloc_local((rows, 8), "int32")
    corrections = T.alloc_local((rows, 8), "int32")
    product_sum = T.alloc_local((rows,), "int32")
    correction_sum = T.alloc_local((rows,), "int32")
    for group, k in T.Parallel(8, 32):
        values[group, k] = x[T.cast(group, "uint8") * T.uint8(32) + T.cast(k, "uint8")]
    for r, group, k in T.Parallel(rows, 8, 32):
        coordinate = T.cast(group, "uint8") * T.uint8(32) + T.cast(k, "uint8")
        codes[r, group, k] = q[r, coordinate]
    T.reduce_dot(codes, values, partial, over="k", acc_dtype="int32", engine="vector")
    for r, group in T.Parallel(rows, 8):
        weighted[r, group] = partial[r, group] * T.cast(scale[r, group], "int32")
    for r, group in T.Parallel(rows, 8):
        corrections[r, group] = T.cast(minimum[r, group], "int32") * (
            T.cast(bsum[2 * group], "int32") + T.cast(bsum[2 * group + 1], "int32")
        )
    T.reduce_sum(weighted, product_sum, dim=1)
    T.reduce_sum(corrections, correction_sum, dim=1)
    for r in T.Parallel(rows):
        total[r] -= (ds[()] * T.cast(dmin[r], acc_dtype)) * T.cast(correction_sum[r], acc_dtype)
        total[r] += (ds[()] * T.cast(d[r], acc_dtype)) * T.cast(product_sum[r], acc_dtype)


@T.macro
def q6_k_single(weight, activation, total, scale_groups):
    """Finish exact 16-term scale groups before the block's float update."""
    low, high, scales, d = weight.ql, weight.qh, weight.scales, weight.d
    x, ds = activation.q, activation.ds
    integer = T.alloc_var("int32", init=0)
    for window in T.unroll(16 // scale_groups):
        decoded = T.alloc_local((scale_groups, 16), "int8")
        values = T.alloc_local((scale_groups, 16), "int8")
        partial = T.alloc_local((scale_groups,), "int32")
        weighted = T.alloc_local((scale_groups,), "int32")
        subtotal = T.alloc_local((1,), "int32")
        for group, k in T.Parallel(scale_groups, 16):
            coordinate = T.uint8(window * scale_groups * 16) + T.cast(group, "uint8") * T.uint8(16) + T.cast(k, "uint8")
            decoded[group, k] = (T.cast(low[coordinate], "int8") | (T.cast(high[coordinate], "int8") << T.int8(4))) - T.int8(32)
            values[group, k] = x[coordinate]
        T.reduce_dot(values, decoded, partial, over="k", acc_dtype="int32", engine="vector")
        for group in T.Parallel(scale_groups):
            weighted[group] = partial[group] * T.cast(scales[T.uint8(window * scale_groups) + T.cast(group, "uint8")], "int32")
        T.reduce_sum(weighted, subtotal, dim=0)
        integer += subtotal[0]
    total[0] += (T.cast(d[()], "float32") * ds[()]) * T.cast(integer, "float32")


@T.macro
def q6_k(weight, activation, total, rows, acc_dtype="float32"):
    low, high, scales, d = weight.ql, weight.qh, weight.scales, weight.d
    x, ds = activation.q, activation.ds
    decoded = T.alloc_local((rows, 16, 16), "int8")
    values = T.alloc_local((16, 16), "int8")
    partial = T.alloc_local((rows, 16), "int32")
    weighted = T.alloc_local((rows, 16), "int32")
    subtotal = T.alloc_local((rows,), "int32")
    for group, k in T.Parallel(16, 16):
        coordinate = T.cast(group, "uint8") * T.uint8(16) + T.cast(k, "uint8")
        values[group, k] = x[coordinate]
    for r, group, k in T.Parallel(rows, 16, 16):
        coordinate = T.cast(group, "uint8") * T.uint8(16) + T.cast(k, "uint8")
        decoded[r, group, k] = (T.cast(low[r, coordinate], "int8") | (T.cast(high[r, coordinate], "int8") << T.int8(4))) - T.int8(32)
    T.reduce_dot(decoded, values, partial, over="k", acc_dtype="int32", engine="vector")
    for r, group in T.Parallel(rows, 16):
        weighted[r, group] = partial[r, group] * T.cast(scales[r, group], "int32")
    T.reduce_sum(weighted, subtotal, dim=1)
    for r in T.Parallel(rows):
        total[r] += (T.cast(d[r], acc_dtype) * ds[()]) * T.cast(subtotal[r], acc_dtype)
