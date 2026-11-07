"""Native TileLang macros with typed return regions and optional call boundaries."""

from dataclasses import replace
import json

from tvm import tirx
from tvm.script.ir_builder import tirx as I
from tilelang.language.eager.builder import macro as common_macro
from tilelang.language.eager.ast import mutate
from tilelang.language.allocate import alloc_local
from tilelang.utils.language import to_tile_region, retrieve_shape, _get_buffer

from .values import intrinsic, bind, LogicalView, RecordValue


def _flatten(value):
    if isinstance(value, tuple):
        flat, children = [], []
        for item in value:
            values, structure = _flatten(item)
            flat.extend(values)
            children.append(structure)
        return flat, children
    return ([], "void") if value is None else ([value], "value")


def _restore(values, structure):
    if structure == "void":
        return None
    if structure == "value":
        return next(values)
    return tuple(_restore(values, child) for child in structure)


class HelperRegion:
    def __init__(self, noinline):
        self.noinline = noinline

    def __call__(self, macro, builder, args, kwargs):
        identity = getattr(builder, "source_helper_count", 0)
        builder.source_helper_count = identity + 1
        prototypes, structure, kinds = None, None, None

        def emit_return(value):
            nonlocal prototypes, structure, kinds
            values, tree = _flatten(value)
            operands, categories = [], []
            for item in values:
                if isinstance(item, (RecordValue, LogicalView)):
                    operands.append(item.var)
                    categories.append("logical")
                elif isinstance(item, (tirx.Buffer, tirx.BufferRegion)):
                    operands.append(to_tile_region(item))
                    categories.append("tile")
                else:
                    scalar = builder.unwrap_value(item)
                    operands.append(tirx.const(scalar) if isinstance(scalar, (bool, int, float)) else scalar)
                    categories.append("scalar")
            if prototypes is None:
                prototypes, structure, kinds = values, tree, categories
            elif tree != structure or categories != kinds:
                raise TypeError("helper return paths must preserve their tuple and value structure")
            I.evaluate(intrinsic("helper_return", *operands, kinds=categories))

        declaration = json.dumps({"name": macro.name, "id": identity, "noinline": self.noinline})
        with builder.with_frame(I.attr(tirx.IntImm("int32", 0), "weft.helper", declaration)):
            builder.return_handlers.append(emit_return)
            try:
                with builder.macro(macro.name, macro.annotations):
                    macro.ir_gen.gen(builder)(*args, **kwargs)
                if prototypes is None:
                    emit_return(None)
            finally:
                builder.return_handlers.pop()
        results = []
        for index, (prototype, kind) in enumerate(zip(prototypes, kinds)):
            if kind == "tile":
                output = alloc_local(retrieve_shape(prototype), _get_buffer(prototype).dtype)
                I.evaluate(intrinsic("helper_result_tile", to_tile_region(output), identity, index))
            else:
                scalar = builder.unwrap_value(prototype)
                if kind == "logical":
                    dtype = "handle"
                elif hasattr(scalar, "dtype"):
                    dtype = str(scalar.dtype)
                else:
                    dtype = "bool" if isinstance(scalar, bool) else "float32" if isinstance(scalar, float) else "int32"
                output = bind(intrinsic("helper_result", identity, index, dtype=dtype))
                if kind == "logical":
                    output = replace(prototype, var=output)
            results.append(output)
        return _restore(iter(results), structure)


def macro(function=None, *, noinline=False):
    def decorate(function):
        result = common_macro(function)
        result.ir_gen = mutate(function, structured_returns=True)
        result.region_builder = HelperRegion(noinline)
        return result

    return decorate(function) if function is not None else decorate
