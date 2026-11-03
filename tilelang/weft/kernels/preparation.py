"""Explicit caller-owned activation preparation with declared packed fields."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


@tilelang.jit(target="weft")
def quantize(M, K, format="Q8_0"):
    assert format in ("Q8_0", "Q8_1", "Q8_K")
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    record_bytes = encoding.schema.storage_bits // 8
    assert K > 0 and K % elements == 0

    @T.prim_func
    def main(X: T.Tensor((M, K), "float32"), Output: T.Tensor((M, K // elements * record_bytes), "uint8")):
        xv = T.view(X, shape=(M, K), axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), encoding=encoding, axes=("m", "k"))
        with T.Kernel(1):
            for row in T.serial(M):
                with T.level.blocks(K, extent=elements) as part:
                    value = T.load(xv[row, part])
                    absolute = T.alloc_local((elements,), "float32")
                    maximum = T.alloc_local((1,), "float32")
                    d = T.alloc_local((1,), "float32")
                    inverse = T.alloc_var("float32", init=0)
                    q = T.alloc_local((elements,), "int8")
                    for i in T.Parallel(elements):
                        absolute[i] = T.abs(value[i])
                    T.reduce_max(absolute, maximum, dim=0)
                    if format == "Q8_K":
                        extreme = T.alloc_var("float32", init=0)
                        found = T.alloc_var("int32", init=0)
                        for i in T.serial(elements):
                            if found == 0 and absolute[i] == maximum[0]:
                                extreme = value[i]
                                found = 1
                        d[0] = 0
                        if extreme != 0:
                            inverse = -127.0 / extreme
                            d[0] = 1.0 / inverse
                    else:
                        d[0] = maximum[0] / 127.0
                        if maximum[0] != 0:
                            inverse = 1.0 / d[0]
                    for i in T.Parallel(elements):
                        q[i] = T.narrow(
                            value[i] * inverse, "int8", rounding="rne" if format == "Q8_K" else "dynamic", saturation=format == "Q8_K"
                        )
                    T.copy(q, yv[row, part].q)
                    if format == "Q8_K":
                        sums = T.alloc_local((16,), "int16")
                        for g in T.Parallel(16):
                            subtotal = T.alloc_var("int16", init=0)
                            for i in T.serial(16):
                                subtotal = subtotal + T.cast(q[g * 16 + i], "int16")
                            sums[g] = subtotal
                        T.copy(sums, yv[row, part].bsum)
                        T.copy(d, yv[row, part].ds)
                    else:
                        T.copy(d, yv[row, part].d)
                        if format == "Q8_1":
                            wide = T.alloc_local((elements,), "int16")
                            total = T.alloc_local((1,), "int16")
                            scaled = T.alloc_local((1,), "float32")
                            for i in T.Parallel(elements):
                                wide[i] = T.cast(q[i], "int16")
                            T.reduce_sum(wide, total, dim=0)
                            scaled[0] = T.cast(total[0], "float32") * d[0]
                            T.copy(scaled, yv[row, part].s)

    return main
