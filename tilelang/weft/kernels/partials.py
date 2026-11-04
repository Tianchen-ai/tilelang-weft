"""Source-declared integer partials and separate scale/correction branches."""

from tilelang.weft import language as T


@T.macro
def q4_k(weight, activation, total, rows, acc_dtype="float32"):
    q, x, scale, minimum = weight.q, activation.q, weight.sc, weight.m
    d, dmin, ds, bsum = weight.d, weight.dmin, activation.ds, activation.bsum
    partial = T.alloc_local((rows, 64), "int16")
    weighted = T.alloc_local((rows, 64), "int32")
    corrections = T.alloc_local((rows, 8), "int32")
    product_sum = T.alloc_local((rows,), "int32")
    correction_sum = T.alloc_local((rows,), "int32")
    T.mac_groups(q, x, partial, n=4, into="int16", over="k")
    for r, group in T.Parallel(rows, 64):
        weighted[r, group] = T.cast(partial[r, group], "int32") * T.cast(scale[r, group // 8], "int32")
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
def q6_k(weight, activation, total, rows, acc_dtype="float32"):
    low, high, scales, d = weight.ql, weight.qh, weight.scales, weight.d
    x, ds = activation.q, activation.ds
    decoded = T.alloc_local((rows, 256), "int8")
    partial = T.alloc_local((rows, 16), "int32")
    weighted = T.alloc_local((rows, 16), "int32")
    subtotal = T.alloc_local((rows,), "int32")
    for r, k in T.Parallel(rows, 256):
        decoded[r, k] = (T.cast(low[r, k], "int8") | (T.cast(high[r, k], "int8") << 4)) - 32
    T.mac_groups(decoded, x, partial, n=16, into="int32", over="k")
    for r, group in T.Parallel(rows, 16):
        weighted[r, group] = partial[r, group] * T.cast(scales[r, group], "int32")
    T.reduce_sum(weighted, subtotal, dim=1)
    for r in T.Parallel(rows):
        total[r] += (T.cast(d[r], acc_dtype) * ds[()]) * T.cast(subtotal[r], acc_dtype)
