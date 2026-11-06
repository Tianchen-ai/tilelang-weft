"""Explicit codebook, sign, exponent and radix decoding from serialized fields."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .decoding import (
    decode_iq1_s,
    decode_iq1_m,
    decode_iq2_s,
    decode_iq2_indexed,
    decode_iq3_s,
    decode_iq3_xxs,
    decode_iq4_xs,
    decode_fp4,
    decode_tq1_0,
)


@tilelang.jit(target="weft")
def iq1_s(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 50), "uint8"), Grid: T.Tensor((16384,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ1_S, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_iq1_s(record, chunk, rows, "IQ1_S", 256, Grid, 0, 0, 0, 0, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

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
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "HalfBits": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ1_M, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_iq1_m(record, chunk, rows, "IQ1_M", 256, Grid, 0, 0, 0, HalfBits, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

    return main


@tilelang.jit(target="weft")
def iq2_s(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 82), "uint8"), Grid: T.Tensor((8192,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ2_S, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_iq2_s(record, chunk, rows, "IQ2_S", 256, Grid, 0, 0, 0, 0, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

    return main


@tilelang.jit(target="weft")
def iq2_indexed(M, K, format="IQ2_XS", rows=1, chunk=32):
    assert format in ("IQ2_XS", "IQ2_XXS") and K % 256 == 0
    assert chunk in (32, 64, 128, 256)
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
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "Signs": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        grid = T.view(Grid, shape=(entries // 8, 8), encoding=F.I8X8, axes=("entry", "payload"), abi=True)
        signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(256 // chunk):
                        values = T.alloc_local((rows, chunk), "float32")
                        decode_iq2_indexed(record, values, rows, format, 256, grid.values, signs.values, 0, 0, 0, 0, offset=segment * chunk, width=chunk)
                        T.copy(values, yv[row, block][:, segment * chunk : (segment + 1) * chunk])

    return main


@tilelang.jit(target="weft")
def iq3_s(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 110), "uint8"), Grid: T.Tensor((2048,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ3_S, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_iq3_s(record, chunk, rows, "IQ3_S", 256, Grid, 0, 0, 0, 0, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

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
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "Signs": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ3_XXS, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_iq3_xxs(record, chunk, rows, "IQ3_XXS", 256, Grid, Signs, 0, 0, 0, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

    return main


@tilelang.jit(target="weft")
def iq4_xs(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 136), "uint8"), Codebook: T.Tensor((16,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Codebook": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ4_XS, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_iq4_xs(record, chunk, rows, "IQ4_XS", 256, 0, 0, Codebook, 0, 0, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

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
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Codebook": "input", "Scale": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=elements) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(elements // 32):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_fp4(record, chunk, rows, format, elements, 0, 0, Codebook, Scale, 0, 0, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

    return main


@tilelang.jit(target="weft")
def tq1_0(M, K, rows=1):
    assert K % 256 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 54), "uint8"), Powers: T.Tensor((5,), "uint32"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Powers": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.TQ1_0, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(8):
                        chunk = T.alloc_local((rows, 32), "float32")
                        decode_tq1_0(record, chunk, rows, "TQ1_0", 256, 0, 0, 0, 0, 0, Powers, offset=segment * 32, width=32)
                        T.copy(chunk, yv[row, block][:, segment * 32 : (segment + 1) * 32])

    return main
