"""TIRx construction for explicit Views, snapshots and finite-width values."""
from dataclasses import dataclass
import json

from tvm import tirx
from tvm.script.ir_builder import tirx as I
from tilelang.language.allocate import alloc_local
from tilelang.utils.language import to_tile_region

from .encoding import EncodingDefinition
from .levels import Point


def intrinsic(name, *args, dtype="handle", **annotations):
    return tirx.call_intrin(dtype, tirx.op.Op.get("tl.weft." + name), *args, annotations=annotations)


def bind(expression):
    return I.bind(expression)


@dataclass(frozen=True)
class LogicalView:
    var: tirx.Var
    shape: tuple
    axes: tuple
    encoding: object

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if len(indices) > len(self.shape):
            raise IndexError("View selectors exceed its logical rank")
        indices = (*indices, *(slice(None) for _ in range(len(self.shape) - len(indices))))
        selectors, operands, shape, axes = [], [], [], []
        for size, axis, index in zip(self.shape, self.axes, indices):
            if isinstance(index, Point):
                if index.axis != axis:
                    raise ValueError(f"Level axis {index.axis} cannot select View axis {axis}")
                selectors.append("domain")
                operands.append(index.var)
                shape.append(index.partition)
                axes.append(axis)
            elif isinstance(index, slice) and index == slice(None):
                selectors.append("all")
                shape.append(size)
                axes.append(axis)
            else:
                selectors.append("index")
                operands.append(index)
        variable = bind(intrinsic("slice", self.var, json.dumps(selectors), *operands))
        return LogicalView(variable, tuple(shape), tuple(axes), self.encoding)


def view(storage, *, shape, encoding=None, axes=None, strides=None):
    from .derived import DerivedEncodingInstance
    shape = tuple(shape)
    if strides is not None and len(strides) != len(shape):
        raise ValueError("View strides must contain one entry per logical axis")
    if isinstance(encoding, (EncodingDefinition, DerivedEncodingInstance)) and strides:
        raise ValueError("packed Views use the Encoding's record layout, not dense strides")
    axes = tuple(axes) if axes is not None else tuple(f"axis{i}" for i in range(len(shape)))
    if len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise ValueError("View axes must uniquely name each logical dimension")
    if isinstance(encoding, DerivedEncodingInstance):
        encoding.declare(shape, axes)
        description = encoding.description()
    else:
        description = encoding.schema.to_dict() if isinstance(encoding, EncodingDefinition) else {"dense": str(encoding or storage.dtype)}
    variable = bind(intrinsic("view", storage.data, json.dumps(description), *shape,
                              rank=len(shape), axes=list(axes), strides=list(strides or ())))
    return LogicalView(variable, shape, axes, encoding or str(storage.dtype))


@dataclass(frozen=True)
class RecordValue:
    var: tirx.Var
    shape: tuple
    axes: tuple
    encoding: EncodingDefinition
    __tilelang_immutable_value__: bool = False

    def __getattr__(self, name):
        fields = self.encoding.schema.fields
        field = next((field for field in fields if field.name == name), None)
        if field is None:
            raise AttributeError(name)
        from weft_core import dtypes
        scalar = getattr(dtypes, field.dtype)
        spelling = {"integer": "int" if scalar.signedness == "signed" else "uint", "float": "float", "bfloat": "bfloat"}[scalar.category.value] + str(scalar.bits)
        shape = self.shape[:-1] + tuple(field.shape)
        output = alloc_local(shape, spelling)
        I.evaluate(intrinsic("field", to_tile_region(output), self.var, name))
        return output


def load(source):
    from .derived import DerivedEncodingInstance
    if not isinstance(source, LogicalView):
        from tilelang.language.copy_op import copy
        from tilelang.utils.language import retrieve_shape, _get_buffer
        output = alloc_local(retrieve_shape(source), _get_buffer(source).dtype)
        copy(source, output)
        return output
    if isinstance(source.encoding, (EncodingDefinition, DerivedEncodingInstance)):
        variable = bind(intrinsic("load_record", source.var))
        return RecordValue(variable, source.shape, source.axes, source.encoding)
    output = alloc_local(source.shape, source.encoding)
    I.evaluate(intrinsic("load_value", to_tile_region(output), source.var))
    return output


def stage(value):
    if isinstance(value, RecordValue):
        return RecordValue(bind(intrinsic("stage", value.var)), value.shape, value.axes, value.encoding, True)
    from tilelang.utils.language import retrieve_shape, _get_buffer
    snapshot = alloc_local(retrieve_shape(value), _get_buffer(value).dtype)
    I.evaluate(intrinsic("stage", to_tile_region(snapshot), to_tile_region(value)))
    snapshot.__tilelang_immutable_value__ = True
    return snapshot


def state(shape, dtype, *, init):
    output = alloc_local(shape, dtype)
    I.evaluate(intrinsic("state", to_tile_region(output), init))
    return output


def mac_groups(left, right, destination, *, n, into, over):
    if type(n) is not int or n <= 0:
        raise ValueError("mac_groups n requires a positive compile-time group width")
    I.evaluate(intrinsic("mac_groups", to_tile_region(left), to_tile_region(right),
                         to_tile_region(destination), n, into, over))


stage.__tilelang_birth_kind__ = "staged"
state.__tilelang_birth_kind__ = "state"
