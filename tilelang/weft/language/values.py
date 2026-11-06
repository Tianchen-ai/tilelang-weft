"""TIRx construction for explicit Views, snapshots and finite-width values."""

from dataclasses import dataclass
import json
import math

from tvm import tirx
from tvm.script.ir_builder import tirx as I
from tilelang.language.allocate import alloc_local
from tilelang.language.utils import _normalize_annotations
from tilelang.utils.language import to_tile_region

from .encoding import EncodingDefinition
from .levels import Point


def intrinsic(name, *args, dtype="handle", **annotations):
    return tirx.call_intrin(dtype, tirx.op.Op.get("tl.weft." + name), *args, annotations=_normalize_annotations(annotations))


def bind(expression):
    return I.bind(expression)


def extent(view, axis):
    """Actual accessible extent of a logical View, including a Level tail."""
    if not isinstance(view, LogicalView) or type(axis) is not int or not 0 <= axis < len(view.shape):
        raise ValueError("extent requires a logical View and a valid compile-time axis")
    return intrinsic("extent", view.var, axis, dtype="int64")


@dataclass(frozen=True)
class LogicalView:
    var: tirx.Var
    shape: tuple
    axes: tuple
    encoding: object
    field_record_elements: int | None = None

    def __getattr__(self, name):
        from .derived import DerivedEncodingInstance

        if not isinstance(self.encoding, (EncodingDefinition, DerivedEncodingInstance)):
            raise AttributeError(name)
        field = next((item for item in self.encoding.schema.fields if item.name == name), None)
        if field is None:
            raise AttributeError(name)
        shape = self.shape[:-1] + tuple(field.shape)
        axes = self.axes[:-1] + tuple(f"{self.axes[-1]}.{name}.{i}" for i in range(max(0, len(field.shape) - 1)))
        if field.shape:
            axes += self.axes[-1:]
        from weft_core import dtypes

        scalar = getattr(dtypes, field.dtype)
        prefix = "float" if scalar.category == dtypes.DTypeCategory.FLOAT else "int" if scalar.signedness == "signed" else "uint"
        return LogicalView(
            bind(intrinsic("field_view", self.var, name)),
            shape,
            axes,
            prefix + str(scalar.bits),
            self.encoding.schema.elements if field.shape else None,
        )

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if len(indices) > len(self.shape):
            raise IndexError("View selectors exceed its logical rank")
        indices = (*indices, *(slice(None) for _ in range(len(self.shape) - len(indices))))
        if all(isinstance(index, slice) and index == slice(None) for index in indices):
            return self
        if any(isinstance(index, slice) and index != slice(None) for index in indices):
            from tvm import arith

            offsets, extents, remaining = [], [], []
            for size, index in zip(self.shape, indices):
                if isinstance(index, slice) and index != slice(None):
                    if index.step not in (None, 1):
                        raise ValueError("logical View ranges require unit stride")
                    start = 0 if index.start is None else index.start
                    end = size if index.stop is None else index.stop
                    extent = arith.Analyzer().simplify(tirx.const(0, "int64") + end - start)
                    if not isinstance(extent, tirx.IntImm) or int(extent.value) <= 0:
                        raise ValueError("logical View ranges require a positive compile-time extent")
                    offsets.append(start)
                    extents.append(int(extent.value))
                    remaining.append(slice(None))
                else:
                    offsets.append(0)
                    extents.append(size)
                    remaining.append(index)
            variable = bind(intrinsic("slice", self.var, "[]", *offsets, subview=True, extents=extents))
            window = LogicalView(variable, tuple(extents), self.axes, self.encoding)
            return window[tuple(remaining)]
        selectors, operands, shape, axes = [], [], [], []
        for size, axis, index in zip(self.shape, self.axes, indices):
            if isinstance(index, Point):
                if index.axis != axis:
                    raise ValueError(f"Level axis {index.axis} cannot select View axis {axis}")
                grouped = self.field_record_elements is not None and axis == self.axes[-1] and size < self.field_record_elements
                if grouped and index.partition * size != self.field_record_elements:
                    raise ValueError("compressed field projection requires one complete field group")
                selectors.append("group_index" if grouped else "domain")
                operands.append(index.var)
                if not grouped:
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


def view(storage, *, shape, encoding=None, axes=None, strides=None, abi=False):
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
    variable = bind(
        intrinsic(
            "view", storage.data, json.dumps(description), *shape, rank=len(shape), axes=list(axes), strides=list(strides or ()), abi=abi
        )
    )
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
        spelling = {"integer": "int" if scalar.signedness == "signed" else "uint", "float": "float", "bfloat": "bfloat"}[
            scalar.category.value
        ] + str(scalar.bits)
        shape = self.shape[:-1] + tuple(field.shape)
        output = alloc_local(shape, spelling)
        output.__tilelang_field_record_elements__ = self.encoding.schema.elements
        I.evaluate(intrinsic("field", to_tile_region(output), self.var, name))
        return output


def reshape(source, shape, *, axes=None):
    """Reshape a saved encoded value, or use TileLang's ordinary Tensor view."""
    if not isinstance(source, RecordValue):
        if axes is not None:
            raise TypeError("ordinary Tensor reshape retains positional axes")
        from tilelang.language.customize import reshape as tensor_reshape

        return tensor_reshape(source, shape)
    shape = tuple(shape)
    axes = tuple(axes) if axes is not None else None
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("encoded reshape requires a positive static shape")
    if axes is None or len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise ValueError("encoded reshape requires one distinct logical axis per dimension")
    if math.prod(source.shape) != math.prod(shape) or shape[-1] != source.encoding.schema.elements:
        raise ValueError("encoded reshape must preserve complete records and the logical element count")
    variable = bind(intrinsic("reshape", source.var, *shape, axes=list(axes)))
    return RecordValue(variable, shape, axes, source.encoding, source.__tilelang_immutable_value__)


def load(source, *, at=None, axis=-1, bounds=None):
    from .derived import DerivedEncodingInstance

    if at is not None and not isinstance(at, Point):
        from tilelang.utils.language import retrieve_shape, _get_buffer

        if bounds != "in_bounds":
            raise ValueError("indexed loads require the explicit bounds='in_bounds' contract")
        if isinstance(source, LogicalView):
            if isinstance(source.encoding, (EncodingDefinition, DerivedEncodingInstance)):
                raise TypeError("indexed loads require a numeric field, dense table or saved Value")
            source_region, scalar = source.var, source.encoding
        else:
            source_region, scalar = to_tile_region(source), _get_buffer(source).dtype
        indices = _get_buffer(at)
        if not str(indices.dtype).startswith(("int", "uint")):
            raise TypeError("indexed loads require an integer index tile")
        output = alloc_local(retrieve_shape(at), scalar)
        I.evaluate(intrinsic("load_value", to_tile_region(output), source_region, to_tile_region(at), bounds=bounds))
        return output
    if bounds is not None:
        raise ValueError("bounds is only valid for indexed loads")
    if at is not None:
        from tilelang.utils.language import retrieve_shape, _get_buffer

        if not isinstance(at, Point) or isinstance(source, LogicalView):
            raise TypeError("load(at=) projects a local Value with an explicit Level point")
        shape = list(retrieve_shape(source))
        axis = axis % len(shape)
        record_extent = getattr(_get_buffer(source), "__tilelang_field_record_elements__", None)
        if record_extent is not None and int(shape[axis]) < record_extent:
            if at.partition * int(shape[axis]) != record_extent:
                raise ValueError("compressed field projection requires one complete field group")
            del shape[axis]
        else:
            shape[axis] = at.partition
        output = alloc_local(tuple(shape), _get_buffer(source).dtype)
        I.evaluate(intrinsic("load_value", to_tile_region(output), to_tile_region(source), at.var, axis))
        return output
    if not isinstance(source, LogicalView):
        from tilelang.language.copy_op import copy
        from tilelang.utils.language import retrieve_shape, _get_buffer

        output = alloc_local(retrieve_shape(source), _get_buffer(source).dtype)
        I.evaluate(copy(source, output))
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
    I.evaluate(intrinsic("mac_groups", to_tile_region(left), to_tile_region(right), to_tile_region(destination), n, into, over))


def narrow(value, dtype, *, rounding="rne", saturation=False):
    """Explicit numerical narrowing; grouping and rounding stay in the source."""
    if rounding not in ("rne", "rtz", "rdn", "rup", "dynamic"):
        raise ValueError("unknown narrowing rounding mode")
    return intrinsic("narrow", value, rounding, int(saturation), dtype=dtype)


stage.__tilelang_birth_kind__ = "staged"
state.__tilelang_birth_kind__ = "state"
