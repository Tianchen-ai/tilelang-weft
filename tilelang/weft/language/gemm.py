"""The ordinary TileLang GEMM tile operation with an author engine constraint."""

from tilelang.language.gemm_op import gemm as _gemm
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import to_tile_region
from tvm import tirx
from tvm.script.ir_builder import tirx as I
from .values import intrinsic


def gemm(
    A, B, C, transpose_A=False, transpose_B=False, policy=GemmWarpPolicy.Square, clear_accum=False, annotations=None, *, engine="auto"
):
    if engine not in ("auto", "vector", "matrix"):
        raise ValueError("engine must be auto, vector, or matrix")
    attributes = dict(annotations or {})
    attributes["weft_engine"] = engine
    return _gemm(A, B, C, transpose_A, transpose_B, policy, clear_accum, attributes)


def reduce_dot(A, B, C, *, over, acc_dtype=None, engine="auto"):
    if engine not in ("auto", "vector", "matrix"):
        raise ValueError("engine must be auto, vector, or matrix")
    axes = (over,) if isinstance(over, str) else tuple(over)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("reduce_dot requires distinct named reduction axes")
    I.evaluate(
        intrinsic(
            "reduce_dot",
            to_tile_region(A),
            to_tile_region(B),
            to_tile_region(C),
            axes=[tirx.StringImm(axis) for axis in axes],
            accumulator=tirx.StringImm(str(acc_dtype or C.dtype)),
            engine=tirx.StringImm(engine),
        )
    )
