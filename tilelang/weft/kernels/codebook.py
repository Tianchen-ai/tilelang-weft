"""Explicit codebook, sign, exponent and radix decoding from serialized fields."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .decoding import decode_iq1_s, decode_iq1_m, decode_iq2_s, decode_iq2_indexed, decode_iq3_s, decode_iq3_xxs, decode_iq4_xs, decode_fp4, decode_tq1_0


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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq1_s(record, result, rows, 'IQ1_S', 256, Grid, 0, 0, 0, 0, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq1_m(record, result, rows, 'IQ1_M', 256, Grid, 0, 0, 0, HalfBits, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq2_s(record, result, rows, 'IQ2_S', 256, Grid, 0, 0, 0, 0, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq2_indexed(record, result, rows, format, 256, Grid, Signs, 0, 0, 0, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq3_s(record, result, rows, 'IQ3_S', 256, Grid, 0, 0, 0, 0, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq3_xxs(record, result, rows, 'IQ3_XXS', 256, Grid, Signs, 0, 0, 0, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_iq4_xs(record, result, rows, 'IQ4_XS', 256, 0, 0, Codebook, 0, 0, 0)
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
                    result = T.alloc_local((rows, elements), 'float32')
                    decode_fp4(record, result, rows, format, elements, 0, 0, Codebook, Scale, 0, 0)
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
                    result = T.alloc_local((rows, 256), 'float32')
                    decode_tq1_0(record, result, rows, 'TQ1_0', 256, 0, 0, 0, 0, 0, Powers)
                    T.copy(result, yv[row, block])

    return main
