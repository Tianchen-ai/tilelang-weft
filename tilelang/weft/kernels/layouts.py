"""Author-declared persistent vector and matrix operand layouts."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


@tilelang.derive
def Q4_0Rows(W: T.View[F.Q4_0], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@tilelang.derive
def Q4_1Rows(W: T.View[F.Q4_1], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "m": 2, "q": 4})


@tilelang.derive
def Q5_0Rows(W: T.View[F.Q5_0], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "qh": 4, "q": 4})


@tilelang.derive
def Q5_1Rows(W: T.View[F.Q5_1], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "m": 2, "qh": 4, "q": 4})


@tilelang.derive
def Q8_0Rows(W: T.View[F.Q8_0], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@tilelang.derive
def IQ4_NLRows(W: T.View[F.IQ4_NL], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@tilelang.derive
def MXFP4Rows(W: T.View[F.MXFP4], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"e": 1, "q": 4})


@tilelang.derive
def Q2KRows(W: T.View[F.Q2_K], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "scales": 4, "q": 4})


@tilelang.derive
def Q4KRows(W: T.View[F.Q4_K], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "sc": 1, "m": 1, "q": 4})


@tilelang.derive
def Q5KRows(W: T.View[F.Q5_K], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "sc": 6, "m": 6, "qh": 4, "q": 4})


@tilelang.derive
def IQ4XSRows(W: T.View[F.IQ4_XS], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "scales_h": 2, "scales_l": 4, "q": 4})


@tilelang.encoding
class BlockSum32:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 32
    value: T.Field("int16")


@tilelang.encoding
class IMEQ4Centered:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 32
    d: T.Field("float16")
    q: T.Field("uint4", (32,)) @ T.grouped(elements=16) @ T.bit_layers(elements=8, order=T.lo_first)


@tilelang.encoding
class IMEQ4Affine:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 1
    elements = 32
    d: T.Field("float16")
    zp: T.Field("uint8")
    q: T.Field("uint4", (32,)) @ T.grouped(elements=16) @ T.bit_layers(elements=8, order=T.lo_first)


@tilelang.encoding
class IMEQ8:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 4
    elements = 32
    d: T.Field("float32")
    q: T.Field("int8", (32,))


@tilelang.derive
def IMEQ4Centered_I(W: T.View[IMEQ4Centered], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "q": 8})


@tilelang.derive
def IMEQ4Affine_I(W: T.View[IMEQ4Affine], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "zp": 1, "q": 8})


@tilelang.derive
def IMEQ8_I(X: T.View[IMEQ8], *, rows: int):
    return T.interleave(X, rows=rows, field_chunks={"d": 4, "q": 8})


@tilelang.encoding
class IME2Weight8:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 32
    d: T.Field("float16")
    q: T.Field("int8", (32,))


@tilelang.encoding
class IME2Activation8:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 32
    d: T.Field("float32")
    neg_sum: T.Field("int16")
    q: T.Field("int8", (32,))


@tilelang.encoding
class IME2Weight4Affine:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 1
    elements = 32
    d: T.Field("float16")
    zp: T.Field("uint8")
    q: T.Field("uint4", (32,))


@tilelang.derive
def IME2Weight4AffineRows(W: T.View[IME2Weight4Affine], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "zp": 1, "q": 16})


@tilelang.derive
def IME2Weight8Rows(W: T.View[IME2Weight8], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"d": 2, "q": 32})


@tilelang.derive
def IME2Activation8Rows(X: T.View[IME2Activation8], *, rows: int):
    return T.interleave(X, rows=rows, field_chunks={"d": 4, "neg_sum": 2, "q": 32})


@tilelang.encoding
class AMXQ4_0:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 32
    q: T.Field("uint4", (32,)) @ T.grouped(elements=8) @ T.bit_layers(elements=4, order=T.lo_first)
    d: T.Field("float16")


@tilelang.derive
def AMXQ4_0Rows(W: T.View[AMXQ4_0], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"q": 4, "d": 2})


@tilelang.encoding
class BlockScale32:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 4
    elements = 32
    value: T.Field("float32")


@tilelang.derive
def BlockScale32Rows(S: T.View[BlockScale32], *, rows: int):
    return T.interleave(S, rows=rows, field_chunks={"value": 4})


@tilelang.encoding
class AMXQ4_K:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 256
    q: T.Field("uint4", (256,)) @ T.grouped(elements=8) @ T.bit_layers(elements=4, order=T.lo_first)
    scales: T.Field("uint8", (8,))
    mins: T.Field("uint8", (8,))
    d: T.Field("float16")
    dmin: T.Field("float16")


@tilelang.derive
def AMXQ4_KRows(W: T.View[AMXQ4_K], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"q": 4, "scales": 1, "mins": 2, "d": 2, "dmin": 2})


@tilelang.encoding
class AMXQ6_K:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 2
    elements = 256
    low: T.Field("uint4", (256,)) @ T.grouped(elements=8) @ T.bit_layers(elements=4, order=T.lo_first)
    high: T.Field("uint2", (256,)) @ T.grouped(elements=16) @ T.bit_layers(elements=4, order=T.lo_first)
    scales: T.Field("int8", (16,))
    d: T.Field("float16")


@tilelang.derive
def AMXQ6_KRows(W: T.View[AMXQ6_K], *, rows: int):
    return T.interleave(W, rows=rows, field_chunks={"low": 4, "high": 4, "scales": 1, "d": 2})
