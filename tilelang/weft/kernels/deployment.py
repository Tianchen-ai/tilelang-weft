"""Dynamic row kernels with explicit serialized GGML entry contracts."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


@tilelang.jit(target="weft")
def vec_dot(format="Q4_K"):
    """One packed row, one activation row and one output; K remains runtime."""
    assert format in ("Q4_K", "Q6_K")
    encoding = getattr(F, format)
    record_bytes = encoding.schema.storage_bits // 8
    K = T.dynamic("K")

    @T.prim_func
    def main(
        Weights: T.Tensor((K // 256 * record_bytes,), "uint8"),
        Activation: T.Tensor((K // 256 * 292,), "uint8"),
        Output: T.Tensor((1,), "float32"),
        K: T.int64,
    ):
        wv = T.view(Weights, shape=(K,), encoding=encoding, axes=("k",), abi=True)
        xv = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",), abi=True)
        yv = T.view(Output, shape=(1,), axes=("out",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            total = T.state((1,), "float32", init=0)
            with T.level.blocks(K, extent=256) as block:
                weight = T.stage(T.load(wv[block]))
                activation = T.stage(T.load(xv[block]))
                x, ds = activation.q, activation.ds
                if format == "Q4_K":
                    q, scale, minimum, d, dmin, bsum = weight.q, weight.sc, weight.m, weight.d, weight.dmin, activation.bsum
                    partial = T.alloc_local((64,), "int16")
                    weighted = T.alloc_local((64,), "int32")
                    correction = T.alloc_local((8,), "int32")
                    product_sum = T.alloc_local((1,), "int32")
                    correction_sum = T.alloc_local((1,), "int32")
                    T.mac_groups(q, x, partial, n=4, into="int16", over="k")
                    for group in T.Parallel(64):
                        weighted[group] = T.cast(partial[group], "int32") * T.cast(scale[group // 8], "int32")
                    for group in T.Parallel(8):
                        correction[group] = T.cast(minimum[group], "int32") * (
                            T.cast(bsum[2 * group], "int32") + T.cast(bsum[2 * group + 1], "int32")
                        )
                    T.reduce_sum(weighted, product_sum, dim=0)
                    T.reduce_sum(correction, correction_sum, dim=0)
                    total[0] -= (ds[()] * T.cast(dmin[()], "float32")) * T.cast(correction_sum[0], "float32")
                    total[0] += (ds[()] * T.cast(d[()], "float32")) * T.cast(product_sum[0], "float32")
                else:
                    low, high, scales, d = weight.ql, weight.qh, weight.scales, weight.d
                    decoded = T.alloc_local((256,), "int8")
                    partial = T.alloc_local((16,), "int32")
                    weighted = T.alloc_local((16,), "int32")
                    subtotal = T.alloc_local((1,), "int32")
                    for k in T.Parallel(256):
                        decoded[k] = (T.cast(low[k], "int8") | (T.cast(high[k], "int8") << 4)) - 32
                    T.mac_groups(decoded, x, partial, n=16, into="int32", over="k")
                    for group in T.Parallel(16):
                        weighted[group] = partial[group] * T.cast(scales[group], "int32")
                    T.reduce_sum(weighted, subtotal, dim=0)
                    total[0] += (ds[()] * T.cast(d[()], "float32")) * T.cast(subtotal[0], "float32")
            T.copy(total, yv)

    return main
