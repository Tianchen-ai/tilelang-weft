"""Encoded author programs retain field layouts, typed partials and scale order."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


@tilelang.jit(target="weft")
def q4_k_q8_k(M, K, rows=4):
    assert K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 144), "uint8"), Activation: T.Tensor((K // 256 * 292,), "uint8"), Output: T.Tensor((M,), "float32")
    ):
        wv = T.view(Weights, shape=(M, K), encoding=F.Q4_K, axes=("m", "k"))
        xv = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",))
        yv = T.view(Output, shape=(M,), axes=("m",))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows,), "float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    weight = T.stage(T.load(wv[row, block]))
                    activation = T.stage(T.load(xv[block]))
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
                        total[r] -= (ds[()] * T.cast(dmin[r], "float32")) * T.cast(correction_sum[r], "float32")
                        total[r] += (ds[()] * T.cast(d[r], "float32")) * T.cast(product_sum[r], "float32")
                T.copy(total, yv[row])

    return main


@tilelang.jit(target="weft")
def q6_k_q8_k(M, K, rows=1):
    assert K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 210), "uint8"), Activation: T.Tensor((K // 256 * 292,), "uint8"), Output: T.Tensor((M,), "float32")
    ):
        wv = T.view(Weights, shape=(M, K), encoding=F.Q6_K, axes=("m", "k"))
        xv = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",))
        yv = T.view(Output, shape=(M,), axes=("m",))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows,), "float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    weight = T.stage(T.load(wv[row, block]))
                    activation = T.stage(T.load(xv[block]))
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
                        total[r] += (T.cast(d[r], "float32") * ds[()]) * T.cast(subtotal[r], "float32")
                T.copy(total, yv[row])

    return main


@tilelang.jit(target="weft")
def dequantize(M, K, format="Q4_0", rows=4):
    if format not in ("Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "TQ2_0"):
        raise ValueError(f"no integer-field author dequantizer for {format}")
    encoding = getattr(F, format)
    block = encoding.schema.elements
    record_bytes = encoding.schema.storage_bits // 8
    assert K > 0 and K % block == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // block * record_bytes), "uint8"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=block) as part:
                    record = T.load(wv[row, part])
                    if format == "Q6_K":
                        q, high, scales, d = record.ql, record.qh, record.scales, record.d
                    elif format == "Q8_K":
                        q, d = record.q, record.ds
                    else:
                        q, d = record.q, record.d
                    result = T.alloc_local((rows, block), "float32")
                    if format in ("Q5_0", "Q5_1", "Q5_K"):
                        high = record.qh
                    if format in ("Q4_1", "Q5_1"):
                        minimum = record.m
                    if format == "Q2_K":
                        scales, dmin = record.scales, record.dmin
                    if format == "Q3_K":
                        high, scale_low, scale_high = record.hmask, record.scale_low, record.scale_high
                    if format in ("Q4_K", "Q5_K"):
                        scales, minimum, dmin = record.sc, record.m, record.dmin
                    for r, k in T.Parallel(rows, block):
                        if format == "Q1_0":
                            value = T.cast(q[r, k], "int32") * 2 - 1
                        elif format in ("Q5_0", "Q5_1"):
                            value = T.cast(q[r, k], "int32") | (T.cast(high[r, k], "int32") << 4)
                            if format == "Q5_0":
                                value = value - 16
                        elif format == "Q3_K":
                            value = T.cast(q[r, k], "int32") - (1 - T.cast(high[r, k], "int32")) * 4
                        elif format == "Q5_K":
                            value = T.cast(q[r, k], "int32") | (T.cast(high[r, k], "int32") << 4)
                        elif format == "Q6_K":
                            value = (T.cast(q[r, k], "int32") | (T.cast(high[r, k], "int32") << 4)) - 32
                        else:
                            value = T.cast(q[r, k], "int32")
                            if format == "Q4_0":
                                value = value - 8
                            elif format == "TQ2_0":
                                value = value - 1
                        if format == "Q2_K":
                            scale = T.cast(scales[r, k // 16], "int32") & 15
                            minimum_value = T.cast(scales[r, k // 16], "int32") >> 4
                            result[r, k] = T.cast(d[r], "float32") * T.cast(scale * value, "float32") - T.cast(dmin[r], "float32") * T.cast(
                                minimum_value, "float32"
                            )
                        elif format == "Q3_K":
                            scale = (T.cast(scale_low[r, k // 16], "int32") | (T.cast(scale_high[r, k // 16], "int32") << 4)) - 32
                            result[r, k] = T.cast(d[r], "float32") * T.cast(scale * value, "float32")
                        elif format in ("Q4_K", "Q5_K"):
                            result[r, k] = T.cast(d[r], "float32") * T.cast(scales[r, k // 32], "float32") * T.cast(
                                value, "float32"
                            ) - T.cast(dmin[r], "float32") * T.cast(minimum[r, k // 32], "float32")
                        elif format == "Q6_K":
                            result[r, k] = T.cast(d[r], "float32") * T.cast(T.cast(scales[r, k // 16], "int32") * value, "float32")
                        else:
                            result[r, k] = T.cast(d[r], "float32") * T.cast(value, "float32")
                        if format in ("Q4_1", "Q5_1"):
                            result[r, k] += T.cast(minimum[r], "float32")
                    T.copy(result, yv[row, part])

    return main


@tilelang.jit(target="weft")
def dequantize_iq4_nl(M, K, rows=1):
    assert K > 0 and K % 32 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 32 * 18), "uint8"), Codebook: T.Tensor((16,), "int8"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ4_NL, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            table = T.alloc_local((16,), "int8")
            T.copy(Codebook, table)
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=32) as block:
                    record = T.load(wv[row, block])
                    q, d = record.q, record.d
                    result = T.alloc_local((rows, 32), "float32")
                    for r, k in T.Parallel(rows, 32):
                        result[r, k] = T.cast(d[r], "float32") * T.cast(table[T.cast(q[r, k], "int32")], "float32")
                    T.copy(result, yv[row, block])

    return main
