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
def iq1_s(M, K, rows=1, chunk=32):
    assert K % 256 == 0 and chunk in (32, 64, 128, 256)

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 50), "uint8"), Grid: T.Tensor((16384,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Grid": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ1_S, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        grid = T.view(Grid, shape=(2048, 8), encoding=F.I8X8,
                      axes=("code", "code_payload"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(256 // chunk):
                        values = T.alloc_local((rows, chunk), "float32")
                        decode_iq1_s(record, values, rows, "IQ1_S", 256, grid.values, 0, 0, 0, 0, 0, offset=segment * chunk, width=chunk)
                        T.copy(values, yv[row, block][:, segment * chunk : (segment + 1) * chunk])

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
def iq2_indexed(M, K, format="IQ2_XS", rows=1, chunk=32, cache_hint="normal"):
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
                        T.copy(values, yv[row, block][:, segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})

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
def iq3_xxs(M, K, rows=1, chunk=32, cache_hint="normal"):
    assert K % 256 == 0 and chunk in (32, 64, 128, 256)

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
        grid = T.view(Grid, shape=(256, 4), encoding=F.I8X4, axes=("entry", "payload"), abi=True)
        signs = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(256 // chunk):
                        values = T.alloc_local((rows, chunk), "float32")
                        decode_iq3_xxs(record, values, rows, "IQ3_XXS", 256, grid.values, signs.values, 0, 0, 0, 0, offset=segment * chunk, width=chunk)
                        T.copy(values, yv[row, block][:, segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})

    return main


@tilelang.jit(target="weft")
def short_records(M, K, format="IQ4_NL", records=2, cache_hint="normal"):
    """Decode an explicit cohort of short records using one saved codebook."""
    assert format in ("IQ4_NL", "MXFP4") and records in (1, 2, 4, 8)
    assert M > 0 and K > 0 and K % (32 * records) == 0
    encoding = getattr(F, format)
    size = encoding.schema.storage_bits // 8

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 32 * size), "uint8"),
        Codebook: T.Tensor((16,), "int8"),
        Scale: T.Tensor((256 if format == "MXFP4" else 0,), "float32"),
        Output: T.Tensor((M, K), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Codebook": "input", "Scale": "input", "Output": "output"}})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        if format == "MXFP4":
            scales = T.view(Scale, shape=(256,), axes=("exponent",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            table = T.stage(T.load(codebook))
            for row in T.serial(M):
                with T.level.blocks(K, extent=records * 32) as cohort:
                    if records == 1:
                        record = T.load(weights[row, cohort])
                    else:
                        record = T.reshape(T.load(weights[row, cohort]), (records, 32), axes=("record", "k"))
                    q = record.q
                    decoded = T.load(table, at=q, bounds="in_bounds")
                    if format == "MXFP4":
                        exponents = record.e
                        factors = T.load(scales, at=exponents, bounds="in_bounds")
                    else:
                        factors = record.d
                    if records == 1:
                        result = T.alloc_local((32,), "float32")
                        for k in T.Parallel(32):
                            result[k] = T.cast(decoded[k], "float32") * T.cast(factors[()], "float32")
                        T.copy(result, output[row, cohort], annotations={"cache_hint": cache_hint})
                    else:
                        result = T.alloc_local((records, 32), "float32")
                        for r, k in T.Parallel(records, 32):
                            result[r, k] = T.cast(decoded[r, k], "float32") * T.cast(factors[r], "float32")
                        flat = T.reshape(result, (records * 32,))
                        T.copy(flat, output[row, cohort], annotations={"cache_hint": cache_hint})

    return main


@tilelang.jit(target="weft")
def iq4_xs(M, K, rows=1, chunk=32, cache_hint="normal"):
    assert K % 256 == 0 and chunk in (32, 64, 128, 256)

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 136), "uint8"), Codebook: T.Tensor((16,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Codebook": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ4_XS, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        with T.Kernel(1):
            table = T.load(codebook)
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(256 // chunk):
                        decoded = T.alloc_local((rows, chunk), "float32")
                        decode_iq4_xs(record, decoded, rows, "IQ4_XS", 256, 0, 0, table, 0, 0, 0, offset=segment * chunk, width=chunk)
                        T.copy(decoded, yv[row, block][:, segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})

    return main


@tilelang.jit(target="weft")
def fp4(M, K, format="MXFP4", rows=1, chunk=32, cache_hint="normal"):
    assert format in ("MXFP4", "NVFP4")
    encoding = getattr(F, format)
    elements, size = encoding.schema.elements, encoding.schema.storage_bits // 8
    assert K % elements == 0 and chunk in (32, 64, 128, 256) and elements % chunk == 0

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
        codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        with T.Kernel(1):
            table = T.load(codebook)
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=elements) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(elements // chunk):
                        decoded = T.alloc_local((rows, chunk), "float32")
                        decode_fp4(record, decoded, rows, format, elements, 0, 0, table, Scale, 0, 0, offset=segment * chunk, width=chunk)
                        T.copy(decoded, yv[row, block][:, segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})

    return main


@tilelang.jit(target="weft")
def tq1_0(M, K, rows=1, chunk=32, cache_hint="normal"):
    assert K % 256 == 0 and chunk in (16, 32, 64, 128, 256)

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 256 * 54), "uint8"), Powers: T.Tensor((5,), "uint32"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Powers": "input", "Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.TQ1_0, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=256) as block:
                    record = T.load(wv[row, block])
                    for segment in T.unroll(256 // chunk):
                        decoded = T.alloc_local((rows, chunk), "float32")
                        decode_tq1_0(record, decoded, rows, "TQ1_0", 256, 0, 0, 0, 0, 0, Powers, offset=segment * chunk, width=chunk)
                        T.copy(decoded, yv[row, block][:, segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})

    return main
