"""The ordinary TileLang GEMM tile operation with an author engine constraint."""
from tilelang.language.gemm_op import gemm as _gemm
from tilelang.tileop.base import GemmWarpPolicy


def gemm(A, B, C, transpose_A=False, transpose_B=False, policy=GemmWarpPolicy.Square,
         clear_accum=False, annotations=None, *, engine="auto"):
    if engine not in ("auto", "vector", "matrix"):
        raise ValueError("engine must be auto, vector, or matrix")
    attributes = dict(annotations or {})
    attributes["weft_engine"] = engine
    return _gemm(A, B, C, transpose_A, transpose_B, policy, clear_accum, attributes)
