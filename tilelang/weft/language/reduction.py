"""Tuple combiners use TileLang's scalar macro builder and shared IR regions."""

from tvm.script.ir_builder import tirx as I
from tilelang.language.eager.builder import Builder, Macro, macro
from tilelang.language.reduce_op import reduce as common_reduce
from tilelang.utils.language import _get_buffer, to_tile_region, retrieve_shape
from .values import bind, intrinsic


def _combined(kind, inputs, outputs, combine, dim, reverse=False):
    inputs = inputs if isinstance(inputs, tuple) else (inputs,)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("tuple reduction requires matching nonempty input/output tuples")
    rank = len(retrieve_shape(inputs[0]))
    dim = dim + rank if dim < 0 else dim
    if not 0 <= dim < rank:
        raise ValueError("tuple reduction dimension is outside the input rank")
    function = macro(combine.orig_func) if isinstance(combine, Macro) else macro(combine)
    builder = Builder.current()
    with builder.with_frame(I.sblock("weft_" + kind)):
        I.sblock_attr(
            {
                "weft.combine": kind,
                "weft.combine.axis": dim,
                "weft.combine.reverse": reverse,
                "weft.combine.inputs": [to_tile_region(v) for v in inputs],
                "weft.combine.outputs": [to_tile_region(v) for v in outputs],
            }
        )
        arguments = [bind(intrinsic("combine_argument", i, dtype=_get_buffer(v).dtype)) for i, v in enumerate((*inputs, *inputs))]
        results = function(*arguments)
        results = results if isinstance(results, tuple) else (results,)
        if len(results) != len(inputs):
            raise TypeError("combiner must return one scalar per input")
        I.evaluate(intrinsic("combine_yield", *results))


def reduce(buffer, out, reduce_type=None, dim=-1, clear=True, batch=1, annotations=None, *, combine=None):
    if combine is None:
        return common_reduce(buffer, out, reduce_type, dim, clear, batch, annotations)
    if reduce_type is not None or not clear or batch != 1 or annotations:
        raise ValueError("an explicit tuple combiner defines one complete reduction without a separate reduction kind")
    return _combined("reduce", buffer, out, combine, dim)


def scan(buffer, out, *, combine, dim=0, reverse=False):
    return _combined("scan", buffer, out, combine, dim, reverse)
