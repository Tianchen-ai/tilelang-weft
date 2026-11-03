"""Explicit codebook, sign, exponent and radix decoding from serialized fields."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


@tilelang.jit(target="weft")
def iq1_s(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 50), "uint8"), Grid: T.Tensor((16384,), "int8"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ1_S, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, high, d = record.q, record.qh, record.d
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        entry, payload = k // 8, k % 8
                        metadata = T.cast(high[r, entry // 4], "uint32")
                        index = T.cast(q[r, entry], "uint32") | (((metadata >> (entry % 4 * 3)) & 7) << 8)
                        delta = T.if_then_else((metadata & 32768) != 0, -0.125, 0.125)
                        scale = T.cast(d[r], "float32") * T.cast(((metadata >> 12) & 7) * 2 + 1, "float32")
                        result[r, k] = scale * (T.cast(Grid[index * 8 + payload], "float32") + delta)
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def iq1_m(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 56), "uint8"),
        Grid: T.Tensor((16384,), "int8"),
        HalfBits: T.Tensor((65536,), "float32"),
        Output: T.Tensor((M, K), "float32"),
    ):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ1_M, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, high, scales = record.q, record.qh, record.scales
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        entry, payload = k // 8, k % 8
                        sc0 = T.cast(scales[r, 0], "uint32") | (T.cast(scales[r, 1], "uint32") << 8)
                        sc1 = T.cast(scales[r, 2], "uint32") | (T.cast(scales[r, 3], "uint32") << 8)
                        sc2 = T.cast(scales[r, 4], "uint32") | (T.cast(scales[r, 5], "uint32") << 8)
                        sc3 = T.cast(scales[r, 6], "uint32") | (T.cast(scales[r, 7], "uint32") << 8)
                        bits = (sc0 >> 12) | ((sc1 >> 8) & 240) | ((sc2 >> 4) & 3840) | (sc3 & 61440)
                        metadata = T.cast(high[r, entry // 2], "uint32") >> (entry % 2 * 4)
                        index = T.cast(q[r, entry], "uint32") | ((metadata & 7) << 8)
                        group = entry // 4
                        sc = T.cast(scales[r, group // 2 * 2], "uint32") | (T.cast(scales[r, group // 2 * 2 + 1], "uint32") << 8)
                        shift = group % 2 * 6 + entry % 4 // 2 * 3
                        scale = HalfBits[bits] * T.cast(((sc >> shift) & 7) * 2 + 1, "float32")
                        delta = T.if_then_else((metadata & 8) != 0, -0.125, 0.125)
                        result[r, k] = scale * (T.cast(Grid[index * 8 + payload], "float32") + delta)
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def iq2_s(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 82), "uint8"), Grid: T.Tensor((8192,), "int8"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ2_S, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, high, signs, scales, d = record.q, record.qh, record.signs, record.scales, record.d
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        entry, payload = k // 8, k % 8
                        index = T.cast(q[r, entry], "uint32") | (((T.cast(high[r, entry // 4], "uint32") >> (entry % 4 * 2)) & 3) << 8)
                        sign = 1 - T.cast(signs[r, k], "int32") * 2
                        sc = (T.cast(scales[r, entry // 4], "uint32") >> (entry % 4 // 2 * 4)) & 15
                        scale = T.cast(d[r], "float32") * (0.5 + T.cast(sc, "float32")) * 0.25
                        result[r, k] = scale * T.cast(T.cast(Grid[index * 8 + payload], "int32") * sign, "float32")
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def iq2_indexed(M, K, format="IQ2_XS", rows=1):
    assert format in ("IQ2_XS", "IQ2_XXS") and K % 256 == 0
    encoding = getattr(F, format)
    size = encoding.schema.storage_bits // 8
    entries = 4096 if format == "IQ2_XS" else 2048

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * size), "uint8"),
        Grid: T.Tensor((entries,), "int8"),
        Signs: T.Tensor((1024,), "int8"),
        Output: T.Tensor((M, K), "float32"),
    ):
        wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, d = record.q, record.d
                    if format == "IQ2_XS":
                        scales = record.scales
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        entry, payload = k // 8, k % 8
                        if format == "IQ2_XS":
                            word = T.cast(q[r, entry], "uint32")
                            index, sign_index = word & 511, word >> 9
                            sc = (T.cast(scales[r, entry // 4], "uint32") >> (entry % 4 // 2 * 4)) & 15
                        else:
                            group = entry // 4
                            metadata = T.cast(q[r, group * 4 + 2], "uint32") | (T.cast(q[r, group * 4 + 3], "uint32") << 16)
                            word = T.cast(q[r, group * 4 + entry % 4 // 2], "uint32")
                            index = (word >> (entry % 2 * 8)) & 255
                            sign_index = (metadata >> (entry % 4 * 7)) & 127
                            sc = metadata >> 28
                        scale = T.cast(d[r], "float32") * (0.5 + T.cast(sc, "float32")) * 0.25
                        result[r, k] = scale * T.cast(
                            T.cast(Grid[index * 8 + payload], "int32") * T.cast(Signs[sign_index * 8 + payload], "int32"), "float32"
                        )
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def iq3_s(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 110), "uint8"), Grid: T.Tensor((2048,), "int8"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ3_S, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, high, signs, scales, d = record.q, record.qh, record.signs, record.scales, record.d
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        entry, payload = k // 4, k % 4
                        index = T.cast(q[r, entry], "uint32") | (T.cast(high[r, entry], "uint32") << 8)
                        sign = 1 - T.cast(signs[r, k], "int32") * 2
                        scale = T.cast(d[r], "float32") * T.cast(T.cast(scales[r, entry // 8], "int32") * 2 + 1, "float32")
                        result[r, k] = scale * T.cast(T.cast(Grid[index * 4 + payload], "int32") * sign, "float32")
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def iq3_xxs(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 98), "uint8"),
        Grid: T.Tensor((1024,), "int8"),
        Signs: T.Tensor((1024,), "int8"),
        Output: T.Tensor((M, K), "float32"),
    ):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ3_XXS, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, meta, d = record.q, record.metadata, record.d
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        entry, payload = k // 4, k % 4
                        metadata = meta[r, entry // 8]
                        sign_index = (metadata >> (entry % 8 // 2 * 7)) & 127
                        value = Grid[T.cast(q[r, entry], "uint32") * 4 + payload]
                        sign = Signs[sign_index * 8 + entry % 2 * 4 + payload]
                        scale = T.cast(d[r], "float32") * (0.5 + T.cast(metadata >> 28, "float32")) * 0.5
                        result[r, k] = scale * T.cast(T.cast(value, "int32") * T.cast(sign, "int32"), "float32")
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def iq4_xs(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 136), "uint8"), Codebook: T.Tensor((16,), "int8"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ4_XS, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, low, high, d = record.q, record.scales_l, record.scales_h, record.d
                    result = T.alloc_local((rows, 256), "float32")
                    for r, k in T.Parallel(rows, 256):
                        group = k // 32
                        scale = ((T.cast(low[r, group // 2], "int32") >> (group % 2 * 4)) & 15) | (
                            ((T.cast(high[r], "int32") >> (group * 2)) & 3) << 4
                        )
                        result[r, k] = (
                            T.cast(d[r], "float32") * T.cast(scale - 32, "float32") * T.cast(Codebook[T.cast(q[r, k], "int32")], "float32")
                        )
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def fp4(M, K, format="MXFP4", rows=1):
    assert format in ("MXFP4", "NVFP4")
    encoding = getattr(F, format)
    elements, size = encoding.schema.elements, encoding.schema.storage_bits // 8
    assert K % elements == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // elements * size), "uint8"),
        Codebook: T.Tensor((16,), "int8"),
        Scale: T.Tensor((256,), "float32"),
        Output: T.Tensor((M, K), "float32"),
    ):
        wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=elements) as block:
                    record = T.load(wv[row, block])
                    q = record.q
                    if format == "MXFP4":
                        exponent = record.e
                    else:
                        exponent = record.d
                    result = T.alloc_local((rows, elements), "float32")
                    for r, k in T.Parallel(rows, elements):
                        raw = exponent[r] if format == "MXFP4" else exponent[r, k // 16]
                        result[r, k] = Scale[T.cast(raw, "int32")] * T.cast(Codebook[T.cast(q[r, k], "int32")], "float32")
                    T.copy(result, yv[row, block])

    return main


@tilelang.jit(target="weft")
def tq1_0(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 54), "uint8"), Powers: T.Tensor((5,), "uint32"), Output: T.Tensor((M, K), "float32")):
        wv = T.view(Weights, shape=(M, K), encoding=F.TQ1_0, axes=("m", "k"))
        yv = T.view(Output, shape=(M, K), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    q, high, d = record.q, record.qh, record.d
                    result = T.alloc_local((rows, 256), "float32")
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
                        decoded = T.cast((T.cast(wrapped, "uint16") * 3) >> 8, "int32") - 1
                        result[r, k] = T.cast(d[r], "float32") * T.cast(decoded, "float32")
                    T.copy(result, yv[row, block])

    return main
