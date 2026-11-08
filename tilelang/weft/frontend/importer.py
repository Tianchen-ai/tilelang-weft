"""Import native TIRx before tile operations lose their semantic boundaries."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math

from tvm import arith, tirx
from weft_core import compiler, dtypes as dt
from weft_core.diagnostics import FrontendError, SourceLocation
from weft_core.effects import argument_accesses
from weft_core.ir import IRBuilder, Value
from weft_core.types import (
    ScalarType,
    PointerType,
    LocalValueType,
    ViewType,
    EncodingType,
    DomainType,
    emit_type,
    element_type,
    shape_of,
    axes_of,
    value_type,
)
from .tiles import TileOperations
from .model import ModelOperations
from .atomics import AtomicOperations
from .helpers import HelperOperations
from .reduction import CombinedReductions


def dtype(name):
    name = str(name)
    if "x" in name:
        scalar, lanes = name.rsplit("x", 1)
        if lanes.isdecimal() and int(lanes) > 1:
            name = scalar
    aliases = {"bool": "i1", "int": "i32", "float": "f32", "bfloat16": "bf16"}
    name = aliases.get(name, name.replace("uint", "u").replace("int", "i").replace("float", "f"))
    result = getattr(dt, name, None)
    if not isinstance(result, dt.DType):
        raise NotImplementedError(f"Weft has no Canonical numerical type for {name}")
    return result


def integers(values):
    return "array<i64: " + ", ".join(map(str, values)) + ">" if values else "array<i64>"


class Importer(ModelOperations, TileOperations, AtomicOperations, HelperOperations, CombinedReductions):
    def __init__(self, function):
        self.function = function
        self.builder = IRBuilder()
        self.block = None
        self.env = {}
        self.locals = {}
        self.pointers = {}
        self.local_axes = {}
        self.operation_axes = {}
        self.vector_axes = {}
        self.next_axis = 1
        self.mask = None
        self.grid = []
        self.threads = 0
        self.implicit = []
        self.arguments = []
        self.analyzer = arith.Analyzer()
        self.current_node = function
        self.named_axes = {}
        self.declarations = []
        self.encodings = {}
        self.next_domain = 1
        self.next_local_shape = 1 << 32
        self.domain_partitions = {}
        self.level_domains = {}
        self.active_domain = None
        self.birth_phase = None
        self.birth_outputs = []
        self.readonly = set()
        self.parallel_domains = {}
        self.helper_stack = []
        self.helper_results = {}
        self.derivation = None
        self.derived_layouts = set()
        self.storage_contracts = {}
        self.addressed_locals = set()
        self.abi_views = {}
        self.shape_symbols = {}

        def address_use(node):
            if isinstance(node, tirx.Call) and node.op.name == "tl.access_ptr" and node.args and isinstance(node.args[0], tirx.BufferLoad):
                self.addressed_locals.add(node.args[0].buffer)

        tirx.stmt_functor.post_order_visit(function.body, address_use)

    def location(self, node=None):
        span = getattr(self.current_node if node is None else node, "span", None)
        if span is None:
            return SourceLocation("<tilelang>", 1, 0)
        return SourceLocation(str(span.source_name.name), max(1, int(span.line)), max(0, int(span.column) - 1))

    def error(self, message, node=None):
        raise FrontendError(message, self.location(node))

    def emit(self, name, operands=(), types=(), attrs=None, regions=()):
        return self.builder.emit(
            self.block, name, self.location(), operands=tuple(operands), result_types=tuple(types), attributes=attrs, regions=tuple(regions)
        )

    def op(self, name, operands=(), ty=None, attrs=None):
        result = self.emit("weft_kernel." + name, operands, () if ty is None else (ty,), attrs)
        return None if ty is None else result[0]

    def axis(self):
        result = self.next_axis
        self.next_axis += 1
        return result

    @contextmanager
    def insertion(self, block):
        previous = self.block
        self.block = block
        try:
            yield
        finally:
            self.block = previous

    def constant(self, number, typ=dt.i32):
        ty = ScalarType(typ)
        if typ == dt.i1:
            spelling = "true" if number else "false"
        elif typ.category in (dt.DTypeCategory.FLOAT, dt.DTypeCategory.BFLOAT):
            if not math.isfinite(float(number)):
                bits = {
                    dt.f16: ("0x7C00", "0xFC00", "0x7E00"),
                    dt.bf16: ("0x7F80", "0xFF80", "0x7FC0"),
                    dt.f32: ("0x7F800000", "0xFF800000", "0x7FC00000"),
                    dt.f64: ("0x7FF0000000000000", "0xFFF0000000000000", "0x7FF8000000000000"),
                }[typ]
                spelling = bits[2 if math.isnan(float(number)) else int(float(number) < 0)]
            else:
                spelling = repr(float(number))
            spelling += " : " + emit_type(ty)
        else:
            spelling = str(int(number)) + " : " + emit_type(ty)
        operation = "weft_kernel.constant" if typ.category == dt.DTypeCategory.INTEGER else "arith.constant"
        return self.emit(operation, types=(ty,), attrs={"value": spelling})[0]

    def cast(self, value, typ):
        source_type = element_type(value.type).dtype
        if source_type == typ:
            return value
        if typ.category == dt.DTypeCategory.INTEGER and typ.signedness == "unsigned":
            source = value
            widths = [typ.bits]
            while source.producer is not None and source.producer.name == "weft_kernel.cast":
                scalar = element_type(source.type).dtype
                if scalar.category != dt.DTypeCategory.INTEGER:
                    break
                widths.append(scalar.bits - (scalar.signedness == "signed"))
                source = source.producer.operands[0]
            producer = source.producer
            if producer is not None and producer.name == "weft_kernel.iota":
                start, end = int(producer.attributes["start"]), int(producer.attributes["end"])
                if 0 <= start < end <= 1 << min(widths):
                    return self.op(
                        "iota", ty=value_type(ScalarType(typ), shape_of(source.type), axes_of(source.type)), attrs=producer.attributes
                    )
        preserving_integer_widen = (
            source_type.category == typ.category == dt.DTypeCategory.INTEGER
            and typ.bits > source_type.bits
            and (source_type.signedness == typ.signedness or typ.signedness == "signed")
        )
        return self.op(
            "widen" if preserving_integer_widen else "cast",
            (value,),
            value_type(ScalarType(typ), shape_of(value.type), axes_of(value.type)),
        )

    def domain(self, *values):
        extents = {}
        for value in values:
            for axis, size in zip(axes_of(value.type), shape_of(value.type)):
                if axis in extents and extents[axis] not in (1, size) and size != 1:
                    self.error("pointwise operands disagree on logical axis extent")
                extents[axis] = max(size, extents.get(axis, 1))
        return tuple(extents.values()), tuple(extents)

    def binary(self, kind, left, right, *, predicate=False):
        if element_type(left.type) != element_type(right.type):
            self.error("TIRx binary operands must carry the same dtype")
        shape, axes = self.domain(left, right)
        ty = value_type(ScalarType(dt.i1) if predicate else element_type(left.type), shape, axes)
        return self.op("compare" if predicate else "binary", (left, right), ty, {"predicate" if predicate else "kind": json.dumps(kind)})

    def broadcast(self, value, shape, axes):
        if (shape_of(value.type), axes_of(value.type)) == (tuple(shape), tuple(axes)):
            return value
        if dict(zip(axes_of(value.type), shape_of(value.type))) == dict(zip(axes, shape)):
            # A different traversal order of the same coordinates is a
            # transpose, not a broadcast. Preserve the saved payload and its
            # validity instead of introducing a predicate with another order.
            return self.rename(value, shape, axes)
        target = value_type(element_type(value.type), tuple(shape), tuple(axes))
        if not shape_of(value.type):
            # A numerical broadcast is not a source state birth. Build its
            # explicit coordinate domain from ordinary pure primitives so it
            # can travel through a physical pipeline without moving a birth.
            predicate = self.constant(True, dt.i1)
            for size, axis in zip(shape, axes):
                coordinate = self.iota(size, axis)
                predicate = self.binary("and", predicate, self.binary("eq", coordinate, coordinate, predicate=True))
            return self.op("select", (predicate, value, value), target)
        # The predicate supplies coordinates; both arms retain the same saved
        # value and its active domain, including signed zero and partial tiles.
        condition = self.broadcast(self.constant(True, dt.i1), shape, axes)
        domain_shape, domain_axes = self.domain(value, condition)
        condition = self.rename(condition, domain_shape, domain_axes)
        selected = self.op("select", (condition, value, value), value_type(element_type(value.type), domain_shape, domain_axes))
        return self.rename(selected, shape, axes)

    def rename(self, value, shape, axes, order=None):
        target = value_type(element_type(value.type), tuple(shape), tuple(axes))
        if value.type == target and order is None:
            return value
        if not shape:
            indices = (self.constant(0, dt.index),) * len(shape_of(value.type))
            return self.op("extract", (value, *indices), target, {"selectors": json.dumps(["index"] * len(indices))})
        if not shape_of(value.type):
            return self.broadcast(value, shape, axes)
        source_axes = axes_of(value.type)
        # T.Parallel can write a tile in a different loop order from the
        # expression's pointer loads. Preserve those coordinate identities
        # before assigning the destination's positional storage axes.
        if order is None:
            order = tuple(axes) if set(axes) == set(source_axes) else source_axes
        return self.op("reshape", (value,), target, {"order": integers(order)})

    def expr(self, node):
        previous = self.current_node
        if getattr(node, "span", None) is not None:
            self.current_node = node
        try:
            return self._expr(node)
        finally:
            self.current_node = previous

    def _expr(self, node):
        if isinstance(node, Value):
            return node
        if isinstance(node, (int, float, bool)):
            return self.constant(node, dt.f32 if isinstance(node, float) else dt.i32)
        if isinstance(node, (tirx.IntImm, tirx.FloatImm)):
            return self.constant(node.value, dtype(node.dtype))
        if isinstance(node, tirx.Var):
            if node not in self.env:
                self.error(f"unbound TIRx variable {node.name}", node)
            value = self.env[node]
            if value is None:
                self.error("CPU kernels cannot use GPU thread ownership; use the Kernel grid or T.Parallel", node)
            return value
        if isinstance(node, tirx.Cast):
            return self.cast(self.expr(node.value), dtype(node.dtype))
        if isinstance(node, (tirx.Ramp, tirx.Broadcast)):
            lanes = self.static(node.lanes)
            if lanes <= 0:
                self.error("TIRx vectors require a positive fixed lane count", node)
            if lanes not in self.vector_axes:
                self.vector_axes[lanes] = self.axis()
            axis = self.vector_axes[lanes]
            if isinstance(node, tirx.Broadcast):
                return self.broadcast(self.expr(node.value), (lanes,), (axis,))
            value = self.expr(node.base)
            indices = self.cast(self.iota(lanes, axis), element_type(value.type).dtype)
            return self.binary("add", value, self.binary("mul", indices, self.expr(node.stride)))
        if isinstance(node, tirx.BufferLoad):
            axes = self.complete_local_domain(node.buffer, node.indices)
            if node.buffer in getattr(self, "field_record_extents", {}) and axes != axes_of(self.locals[node.buffer].type):
                axes = None
            if axes is not None:
                value = self.local_value(node.buffer)
                return self.rename(value, shape_of(value.type), axes)
            return self.load(node.buffer, [self.expr(i) for i in node.indices])
        if isinstance(node, tirx.Select):
            condition, yes, no = self.expr(node.condition), self.expr(node.true_value), self.expr(node.false_value)
            shape, axes = self.domain(yes, no, condition)
            return self.op("select", (condition, yes, no), value_type(element_type(yes.type), shape, axes))
        names = {
            "Add": "add",
            "Sub": "sub",
            "Mul": "mul",
            "Div": "div",
            "Mod": "mod",
            "Min": "min",
            "Max": "max",
            "And": "and",
            "Or": "or",
            "EQ": "eq",
            "NE": "ne",
            "LT": "lt",
            "LE": "le",
            "GT": "gt",
            "GE": "ge",
        }
        kind = type(node).__name__
        if kind in names:
            left, right = self.expr(node.a), self.expr(node.b)
            if kind in ("Min", "Max") and element_type(left.type).dtype.category in (dt.DTypeCategory.FLOAT, dt.DTypeCategory.BFLOAT):
                # Native CPU TIRx uses ordered compare/select, including the
                # right operand for a NaN comparison or equal signed zeros.
                choose = self.binary("lt" if kind == "Min" else "gt", left, right, predicate=True)
                return self.op("select", (choose, left, right), value_type(element_type(left.type), *self.domain(left, right)))
            return self.binary(names[kind], left, right, predicate=kind in ("EQ", "NE", "LT", "LE", "GT", "GE"))
        if kind in ("FloorDiv", "FloorMod"):
            # TIR floor division differs from signed truncation. Keep its
            # quotient/remainder correction explicit in the Canonical program.
            left, right = self.expr(node.a), self.expr(node.b)
            if self.analyzer.can_prove(node.a >= 0) and self.analyzer.can_prove(node.b > 0):
                return self.binary("div" if kind == "FloorDiv" else "mod", left, right)
            q = self.binary("div", left, right)
            r = self.binary("mod", left, right)
            zero = self.constant(0, element_type(left.type).dtype)
            nonzero = self.binary("ne", r, zero, predicate=True)
            signs = self.binary(
                "ne", self.binary("lt", left, zero, predicate=True), self.binary("lt", right, zero, predicate=True), predicate=True
            )
            adjust = self.binary("and", nonzero, signs)
            corrected = self.binary("sub", q, self.cast(adjust, element_type(q.type).dtype))
            return corrected if kind == "FloorDiv" else self.binary("sub", left, self.binary("mul", corrected, right))
        if isinstance(node, tirx.Not):
            return self.binary("xor", self.expr(node.a), self.constant(True, dt.i1))
        if isinstance(node, tirx.Call):
            atomic = self.atomic_expression(node)
            if atomic is not None:
                return atomic
            if node.op.name.startswith("tl.weft."):
                value = self.model_expr(node)
                if value is not None:
                    return value
            name = node.op.name if hasattr(node.op, "name") else str(node.op)
            binary = {
                "tirx.bitwise_and": "and",
                "tirx.bitwise_or": "or",
                "tirx.bitwise_xor": "xor",
                "tirx.shift_left": "shl",
                "tirx.shift_right": "shr",
            }
            if name in binary:
                return self.binary(binary[name], self.expr(node.args[0]), self.expr(node.args[1]))
            unary = {"exp", "exp2", "log", "log2", "sqrt", "rsqrt", "abs", "sin", "cos", "tanh", "floor", "ceil", "fabs"}
            leaf = name.rsplit(".", 1)[-1]
            if leaf in unary:
                value = self.expr(node.args[0])
                return self.op("unary", (value,), value.type, {"kind": json.dumps("abs" if leaf == "fabs" else leaf)})
            if leaf == "if_then_else":
                selected = self.static_condition(node.args[0])
                if selected is not None:
                    return self.expr(node.args[1] if selected else node.args[2])
                condition, yes, no = map(self.expr, node.args)
                shape, axes = self.domain(yes, no, condition)
                return self.op("select", (condition, yes, no), value_type(element_type(yes.type), shape, axes))
            if leaf == "reinterpret":
                value = self.expr(node.args[0])
                return self.op("bitcast", (value,), value_type(ScalarType(dtype(node.dtype)), shape_of(value.type), axes_of(value.type)))
        self.error(f"TIRx expression {type(node).__name__} is not yet imported: {node}", node)

    def static_condition(self, expression):
        simplified = self.analyzer.simplify(expression)
        return bool(simplified.value) if isinstance(simplified, tirx.IntImm) else None

    def static(self, expression):
        simplified = self.analyzer.simplify(expression)
        if not isinstance(simplified, tirx.IntImm):
            self.error(f"this local tile requires a static extent, got {expression}")
        return int(simplified.value)

    def iota(self, extent, axis):
        raw = self.op("iota", ty=LocalValueType(ScalarType(dt.u32), (extent,), (axis,)), attrs={"start": "0", "end": str(extent)})
        return self.cast(raw, dt.i32)

    def offsets(self, buffer, indices):
        result = self.constant(0, dt.i64)
        strides = list(buffer.strides)
        if not strides:
            stride = tirx.IntImm("int64", 1)
            strides = []
            for size in reversed(buffer.shape):
                strides.insert(0, stride)
                stride *= size
        for index, stride in zip(indices, strides):
            offset = self.cast(index, dt.i64)
            if not self.analyzer.can_prove_equal(stride, 1):
                offset = self.binary("mul", offset, self.cast(self.expr(stride), dt.i64))
            result = self.binary("add", result, offset)
        if not isinstance(buffer.elem_offset, tirx.IntImm) or int(buffer.elem_offset.value):
            result = self.binary("add", result, self.cast(self.expr(buffer.elem_offset), dt.i64))
        return result

    def memory_domain(self, offsets, value=None):
        values = [offsets] + ([] if value is None else [value]) + ([] if self.mask is None else [self.mask])
        return self.domain(*values)

    def memory_mask(self, shape, axes):
        value = self.constant(True, dt.i1) if self.mask is None else self.mask
        return self.broadcast(value, shape, axes)

    def access_mask(self, buffer, indices, shape, axes):
        mask = self.memory_mask(shape, axes)
        for index, extent in zip(indices, buffer.shape):
            coordinate = self.cast(index, dt.i64)
            nonnegative = self.binary("ge", coordinate, self.constant(0, dt.i64), predicate=True)
            below = self.binary("lt", coordinate, self.cast(self.expr(extent), dt.i64), predicate=True)
            mask = self.binary("and", mask, self.binary("and", nonnegative, below))
        return self.broadcast(mask, shape, axes)

    def buffer_owner(self, buffer):
        if buffer in self.locals or buffer in self.pointers:
            return buffer
        owners = [owner for owner in (*self.locals, *self.pointers) if owner.data.same_as(buffer.data)]
        if not owners:
            return None
        if len(owners) != 1:
            self.error(f"buffer view {buffer.name} has no unique storage owner")
        owner = owners[0]
        if str(buffer.dtype) != str(owner.dtype):
            self.error("CPU buffer views currently require the owner's element type")
        for selected in (owner, buffer):
            if not self.analyzer.can_prove_equal(selected.elem_offset, 0):
                self.error("CPU buffer reshape requires zero element offsets")
            stride = 1
            for position in reversed(range(len(selected.shape))):
                if selected.strides and not self.analyzer.can_prove_equal(selected.strides[position], stride):
                    self.error("CPU buffer reshape requires contiguous storage")
                stride *= selected.shape[position]
        if not self.analyzer.can_prove_equal(math.prod(buffer.shape), math.prod(owner.shape)):
            self.error("CPU buffer reshape must preserve the number of elements")
        return owner

    def local_value(self, buffer):
        owner = self.buffer_owner(buffer)
        if owner not in self.locals:
            self.error(f"read of unallocated local buffer {buffer.name}")
        value = self.locals[owner]
        if owner is buffer or owner.same_as(buffer):
            return value
        shape = tuple(self.static(n) for n in buffer.shape)
        if math.prod(shape) != math.prod(shape_of(value.type)):
            self.error("CPU buffer reshape cannot change an implicit Parallel domain")
        axes = self.local_axes_for(buffer)
        return self.rename(value, shape, axes)

    def local_axes_for(self, buffer):
        owner = self.buffer_owner(buffer)
        if owner not in self.locals:
            self.error(f"local axes require allocated storage for {buffer.name}")
        if buffer not in self.local_axes:
            shape = tuple(self.static(n) for n in buffer.shape)
            if shape == shape_of(self.locals[owner].type):
                self.local_axes[buffer] = axes_of(self.locals[owner].type)
            else:
                self.local_axes[buffer] = tuple(
                    self.operation_axes[(buffer, i)] if (buffer, i) in self.operation_axes else self.axis()
                    for i in range(len(shape)))
        return self.local_axes[buffer]

    def set_local_value(self, buffer, value):
        owner = self.buffer_owner(buffer)
        if owner in self.readonly:
            self.error(f"staged buffer {owner.name} is an immutable snapshot")
        if owner is buffer or (owner is not None and owner.same_as(buffer)):
            self.locals[owner] = value
            self.local_axes[owner] = axes_of(value.type)
        else:
            original = self.locals[owner]
            self.locals[owner] = self.rename(value, shape_of(original.type), axes_of(original.type))

    def load(self, buffer, indices):
        owner = self.buffer_owner(buffer)
        if owner in self.pointers:
            offsets = self.offsets(buffer, indices)
            shape, axes = self.memory_domain(offsets)
            offsets = self.broadcast(offsets, shape, axes)
            ty = value_type(ScalarType(dtype(buffer.dtype)), shape, axes)
            other = self.broadcast(self.constant(0, dtype(buffer.dtype)), shape, axes)
            return self.op("pointer_load", (self.pointers[owner], offsets, self.access_mask(buffer, indices, shape, axes), other), ty)
        source = self.local_value(buffer)
        if not buffer.shape or buffer.scope() == "local.var":
            # A scalar allocated inside Parallel has one value per logical
            # iteration, already recorded by its defining assignment. A scalar
            # defined outside the Parallel remains scalar; broadcasting every
            # read would turn loop bases and invariant scale arithmetic into
            # unrelated multidimensional vector work before its actual use.
            axes = tuple(axis for _, _, axis in self.parallel_domains.values())
            if not set(axes_of(source.type)) <= set(axes):
                source = self.rename(source, (), ())
            return source
        selectors, operands = [], []
        shape, axes = (), ()
        source_shape, source_axes = shape_of(source.type), axes_of(source.type)
        for position, index in enumerate(indices):
            full = any(
                self.env.get(var) is index
                and axis == source_axes[position]
                and self.analyzer.can_prove_equal(start, 0)
                and self.analyzer.can_prove_equal(extent, source_shape[position])
                for var, (start, extent, axis) in self.parallel_domains.items()
            )
            if full:
                selectors.append("all")
                shape, axes = (*shape, source_shape[position]), (*axes, source_axes[position])
                continue
            if shape_of(index.type):
                selectors.append("gather")
                index = self.local_index(index)
                shape, axes = self.domain_types(shape, axes, index)
            else:
                selectors.append("index")
                index = self.cast(index, dt.index)
            operands.append(index)
        if all(selector == "all" for selector in selectors):
            return source
        return self.op(
            "extract", (source, *operands), value_type(element_type(source.type), shape, axes), {"selectors": json.dumps(selectors)}
        )

    def domain_types(self, shape, axes, value):
        dimensions = dict(zip(axes, shape))
        for axis, size in zip(axes_of(value.type), shape_of(value.type)):
            dimensions[axis] = size
        return tuple(dimensions.values()), tuple(dimensions)

    def local_index(self, value):
        scalar = element_type(value.type).dtype
        return value if scalar.category == dt.DTypeCategory.INTEGER and scalar.signedness == "unsigned" else self.cast(value, dt.u32)

    def store(self, buffer, indices, value):
        owner = self.buffer_owner(buffer)
        if owner in self.readonly:
            self.error(f"staged buffer {owner.name} is an immutable snapshot")
        value = self.cast(value, dtype(buffer.dtype))
        if owner in self.pointers:
            offsets = self.offsets(buffer, indices)
            shape, axes = self.memory_domain(offsets, value)
            offsets, value = (self.broadcast(v, shape, axes) for v in (offsets, value))
            self.op("pointer_store", (self.pointers[owner], offsets, self.access_mask(buffer, indices, shape, axes), value))
            return
        if owner not in self.locals:
            self.error(f"write of unallocated local buffer {buffer.name}")
        if not buffer.shape or buffer.scope() == "local.var":
            original = self.local_value(buffer)
            shape = tuple(self.static(extent) for _, extent, _ in self.parallel_domains.values())
            axes = tuple(axis for _, _, axis in self.parallel_domains.values())
            if not set(axes_of(original.type)) <= set(axes):
                original = self.rename(original, (), ())
            original = self.broadcast(original, shape, axes)
            value = self.broadcast(value, shape, axes)
            if self.mask is not None:
                mask = self.broadcast(self.mask, shape, axes)
                value = self.op("select", (mask, value, original), original.type)
            self.set_local_value(buffer, value)
            return
        if self.mask is not None:
            old = self.load(buffer, indices)
            shape, axes = self.domain(value, old, self.mask)
            operands = tuple(self.broadcast(v, shape, axes) for v in (self.mask, value, old))
            value = self.op("select", operands, value_type(element_type(value.type), shape, axes))
        converted = [self.cast(i, dt.u32 if shape_of(i.type) else dt.index) for i in indices]
        selectors = ["gather" if shape_of(i.type) else "index" for i in converted]
        shape, axes = (), ()
        for index in converted:
            shape, axes = self.domain_types(shape, axes, index)
        # A masked expression can discover column axes before row axes. The
        # update still uses the destination selectors' ordered domain.
        value = self.broadcast(value, shape, axes)
        original = self.local_value(buffer)
        self.set_local_value(buffer, self.op("update", (original, value, *converted), original.type, {"selectors": json.dumps(selectors)}))

    def complete_local_domain(self, buffer, indices):
        if self.buffer_owner(buffer) not in self.locals or len(indices) != len(buffer.shape):
            return None
        axes = []
        for index, size in zip(indices, buffer.shape):
            matches = [
                axis for variable, (start, extent, axis) in self.parallel_domains.items()
                if self.analyzer.can_prove_equal(start, 0)
                and self.analyzer.can_prove_equal(extent, size)
                and self.analyzer.can_prove_equal(index, variable)
            ]
            if len(matches) != 1:
                return None
            axes.append(matches[0])
        return tuple(axes) if len(set(axes)) == len(axes) else None

    def allocate(self, buffer):
        if buffer in self.locals or buffer in self.pointers:
            self.error(f"local buffer {buffer.name} was allocated twice")
        if buffer.scope() not in ("local", "local.var", "local.fragment", "shared"):
            self.error(f"CPU local allocation does not implement storage scope {buffer.scope()}")
        shape = tuple(self.static(n) for n in buffer.shape)
        if buffer in self.addressed_locals:
            scalar = dtype(buffer.dtype)
            if scalar.bits < 8:
                self.error("address-observable local storage requires a byte-addressable element type")
            pointer = self.op(
                "local_alloc",
                ty=PointerType(ScalarType(scalar)),
                attrs={"elements": str(math.prod(shape)) + " : i64", "alignment": str(max(1, scalar.bits // 8)) + " : i64"},
            )
            self.pointers[buffer] = pointer
            self.env[buffer.data] = pointer
            return
        axes = tuple(self.operation_axes[(buffer, i)] if (buffer, i) in self.operation_axes else self.axis() for i in range(len(shape)))
        if not shape:
            shape = tuple(self.static(extent) for _, extent, _ in self.parallel_domains.values())
            axes = tuple(axis for _, _, axis in self.parallel_domains.values())
        self.local_axes[buffer] = axes
        ty = LocalValueType(ScalarType(dtype(buffer.dtype)), shape, axes)
        value = self.op("empty", tuple(self.constant(n, dt.index) for n in shape), ty)
        # Empty storage has a shaped result even at rank zero. A scalar local
        # carries the ordinary scalar numerical type through SCF and helpers,
        # just like its eventual field or arithmetic assignment.
        self.locals[buffer] = self.rename(value, (), ()) if not shape else value

    def stmt(self, node):
        previous = self.current_node
        if getattr(node, "span", None) is not None:
            self.current_node = node
        try:
            self._stmt(node)
        finally:
            self.current_node = previous

    def _stmt(self, node):
        name = type(node).__name__
        if isinstance(node, tirx.SeqStmt):
            for statement in node.seq:
                self.stmt(statement)
            return
        if name == "Bind":
            if isinstance(node.value, tirx.Call) and node.value.op.name.rsplit(".", 1)[-1] == "launch_thread_idx":
                self.env[node.var] = None
            else:
                self.env[node.var] = self.expr(node.value)
                if str(node.var.dtype).startswith(("int", "uint")):
                    self.analyzer.bind(node.var, node.value)
            return
        if isinstance(node, tirx.BufferStore):
            axes = self.complete_local_domain(node.buffer, node.indices)
            value = self.expr(node.value)
            if axes is not None:
                if self.buffer_owner(node.buffer) in self.readonly:
                    self.error(f"staged buffer {node.buffer.name} is an immutable snapshot")
                original = self.local_value(node.buffer)
                shape = shape_of(original.type)
                value = self.broadcast(self.cast(value, dtype(node.buffer.dtype)), shape, axes)
                if self.mask is not None:
                    mask = self.broadcast(self.mask, shape, axes)
                    previous = self.rename(original, shape, axes)
                    value = self.op("select", (mask, value, previous), previous.type)
                self.set_local_value(node.buffer, value)
            else:
                self.store(node.buffer, [self.expr(i) for i in node.indices], value)
            return
        if name == "AllocBuffer":
            self.allocate(node.buffer)
            return
        if isinstance(node, tirx.SBlockRealize):
            if not isinstance(node.predicate, tirx.IntImm) or not bool(node.predicate.value):
                self.error("a predicated structured block needs an explicit conditional")
            for iv, expression in zip(node.block.iter_vars, node.iter_values):
                self.env[iv.var] = self.expr(expression)
            self.stmt(node.block)
            return
        if isinstance(node, tirx.SBlock):
            if "weft.combine" in node.annotations:
                self.combined_reduction(node)
                return
            if "weft.derive" in node.annotations:
                self.layout_builder(node)
                return
            if "weft_cpu_num_threads" in node.annotations:
                self.threads = int(node.annotations["weft_cpu_num_threads"])
            for buffer in node.alloc_buffers:
                self.allocate(buffer)
            if node.init is not None:
                self.error("reduction block init must be expressed by tile reduction or explicit local initialization")
            self.stmt(node.body)
            return
        if isinstance(node, tirx.For):
            if "weft.level.relation" in node.annotations:
                self.semantic_level(node)
                return
            if node.thread_binding is not None:
                tag = node.thread_binding.thread_tag
                if tag not in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
                    self.error(f"CPU importer has no GPU thread binding {tag}")
                axis = ("blockIdx.x", "blockIdx.y", "blockIdx.z").index(tag)
                if axis != len(self.grid):
                    self.error("Kernel grid must be a contiguous outer grid nest")
                self.grid.append(node.extent)
                self.env[node.loop_var] = self.cast(self.launch[axis], dtype(node.loop_var.dtype))
                self.stmt(node.body)
                return
            if node.kind in (tirx.ForKind.PARALLEL, tirx.ForKind.VECTORIZED):
                extent = self.static(node.extent)
                if extent == 0:
                    return
                environment = dict(self.env)
                private = {buffer: value for buffer, value in self.locals.items() if buffer.scope() == "local.var"}
                private_axes = {buffer: self.local_axes[buffer] for buffer in private}
                axis = self.parallel_axis(node)
                value = self.iota(extent, axis)
                self.env[node.loop_var] = (
                    value if self.analyzer.can_prove_equal(node.min, 0) else self.binary("add", self.expr(node.min), value)
                )
                self.parallel_domains[node.loop_var] = (node.min, node.extent, axis)
                try:
                    with (
                        self.analyzer.constraint_scope(node.loop_var >= node.min),
                        self.analyzer.constraint_scope(node.loop_var < node.min + node.extent),
                    ):
                        self.stmt(node.body)
                finally:
                    self.parallel_domains.pop(node.loop_var)
                    self.env = environment
                    self.locals.update(private)
                    self.local_axes.update(private_axes)
                return
            self.serial(node)
            return
        if isinstance(node, tirx.While):
            self.while_loop(node)
            return
        if isinstance(node, tirx.IfThenElse):
            selected = self.static_condition(node.condition)
            if selected is not None:
                branch = node.then_case if selected else node.else_case
                if branch is not None:
                    self.stmt(branch)
                return
            condition = self.expr(node.condition)
            if shape_of(condition.type):
                saved = self.mask
                self.mask = condition if saved is None else self.binary("and", saved, condition)
                self.stmt(node.then_case)
                if node.else_case is not None:
                    opposite = self.binary("xor", condition, self.constant(True, dt.i1))
                    self.mask = opposite if saved is None else self.binary("and", saved, opposite)
                    self.stmt(node.else_case)
                self.mask = saved
                return
            self.conditional(node, condition)
            return
        if isinstance(node, tirx.Evaluate):
            if isinstance(node.value, tirx.IntImm):
                return
            if isinstance(node.value, tirx.Call) and (
                self.model_tile(node.value) or self.tile_operation(node.value) or self.atomic_tile(node.value)
            ):
                return
            self.expr(node.value)
            return
        if isinstance(node, tirx.AttrStmt):
            if node.attr_key == "weft.helper":
                self.source_helper(node)
                return
            self.error(f"unhandled TIRx region attribute {node.attr_key}")
        self.error(f"TIRx statement {name} has no Canonical import: {node}")

    def parallel_axis(self, loop):
        candidates = set()
        destinations = set()
        sources = set()
        definitions = {}

        def definition(node):
            if type(node).__name__ == "Bind" and str(node.var.dtype).startswith(("int", "uint")):
                definitions[node.var] = tirx.stmt_functor.substitute(node.value, definitions)

        tirx.stmt_functor.post_order_visit(loop.body, definition)

        def access(node):
            if not isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
                return
            owner = self.buffer_owner(node.buffer)
            if owner not in self.locals:
                return
            axes = self.local_axes_for(node.buffer)
            # TileLang can wrap a scalar local in a unit-sized Buffer. Its
            # storage index does not introduce a numerical Parallel axis.
            if not axes:
                return
            record = getattr(self, "field_record_extents", {}).get(node.buffer)
            for dimension, (index, extent) in enumerate(zip(node.indices, node.buffer.shape)):
                index = tirx.stmt_functor.substitute(index, definitions)
                if not self.analyzer.can_prove_equal(index, loop.loop_var):
                    continue
                if record is not None and dimension + 1 == len(axes) and not self.analyzer.can_prove_equal(extent, record):
                    continue
                if self.analyzer.can_prove_equal(loop.min, 0) and self.analyzer.can_prove_equal(loop.extent, extent):
                    candidates.add(axes[dimension])
                    if isinstance(node, tirx.BufferStore):
                        destinations.add(axes[dimension])
                    elif self.locals[owner].producer is None or self.locals[owner].producer.name != "weft_kernel.empty":
                        sources.add(axes[dimension])

        with (
            self.analyzer.constraint_scope(loop.loop_var >= loop.min),
            self.analyzer.constraint_scope(loop.loop_var < loop.min + loop.extent),
        ):
            tirx.stmt_functor.post_order_visit(loop.body, access)
        named = candidates.intersection(self.named_axes.values())
        if len(named) == 1:
            return next(iter(named))
        # A newly written positional tile can retain the coordinate of its
        # saved input. In particular, scaling a contraction's free axes does
        # not create new logical coordinates merely because its destination
        # was allocated before the contraction result existed.
        if not named and len(sources) == 1:
            return next(iter(sources))
        if not named and len(destinations) == 1:
            return next(iter(destinations))
        if not named and len(candidates) == 1:
            return next(iter(candidates))
        return self.axis()

    def serial(self, node):
        request = node.annotations.get("weft.pipeline.depth", node.annotations.get("num_stages"))
        if "tl_pipeline_order" in node.annotations or "tl_pipeline_stage" in node.annotations:
            self.error("manual T.Pipelined order/stage requires a closed source-operation schedule", node)
        attributes = None
        if node.kind == tirx.ForKind.UNROLLED:
            self.static(node.extent)
            attributes = {"weft.unroll": "unit"}
        if request is not None:
            depth = self.static(request)
            if depth not in (0, 1, 2, 3, 4):
                self.error("T.Pipelined depth must be one of 0, 1, 2, 3, 4", node)
            attributes = {**(attributes or {}), "weft.pipeline.depth": str(depth) + " : i64"}
        environment = dict(self.env)
        bounds = []
        for expression in (node.min, node.min + node.extent):
            simplified = self.analyzer.simplify(expression)
            bounds.append(
                self.constant(int(simplified.value), dt.index)
                if isinstance(simplified, tirx.IntImm)
                else self.cast(self.expr(expression), dt.index)
            )
        lower, upper = bounds
        step = self.constant(1, dt.index)
        incoming = dict(self.locals)
        incoming_axes = dict(self.local_axes)
        keys = tuple(incoming)
        region = self.builder.region((ScalarType(dt.index), *(incoming[k].type for k in keys)))
        with self.insertion(region):
            self.locals = {k: value for k, value in zip(keys, region.arguments[1:])}
            self.env[node.loop_var] = self.cast(region.arguments[0], dtype(node.loop_var.dtype))
            with (
                self.analyzer.constraint_scope(node.loop_var >= node.min),
                self.analyzer.constraint_scope(node.loop_var < node.min + node.extent),
            ):
                self.stmt(node.body)
            final = [self.rename(self.locals[k], shape_of(incoming[k].type), axes_of(incoming[k].type)) for k in keys]
            self.emit("scf.yield", final)
        changed = tuple(i for i, value in enumerate(final) if value is not region.arguments[i + 1] and value is not incoming[keys[i]])
        for i, key in enumerate(keys):
            if i not in changed:
                region.arguments[i + 1].replace_all_uses_with(incoming[key])
        region.operations[-1].operands = tuple(final[i] for i in changed)
        region.arguments = (region.arguments[0], *(region.arguments[i + 1] for i in changed))
        results = self.emit(
            "scf.for",
            (lower, upper, step, *(incoming[keys[i]] for i in changed)),
            tuple(incoming[keys[i]].type for i in changed),
            attrs=attributes,
            regions=(region,),
        )
        self.locals = dict(incoming)
        self.locals.update(zip((keys[i] for i in changed), results))
        self.local_axes = incoming_axes
        self.env = environment

    def conditional(self, node, condition):
        incoming, env = dict(self.locals), dict(self.env)
        incoming_axes = dict(self.local_axes)
        keys = tuple(incoming)
        branches = []
        for statement in (node.then_case, node.else_case):
            region = self.builder.region(())
            self.locals, self.env, self.local_axes = dict(incoming), dict(env), dict(incoming_axes)
            with self.insertion(region):
                if statement is not None:
                    self.stmt(statement)
                self.emit(
                    "scf.yield", tuple(self.rename(self.locals[k], shape_of(incoming[k].type), axes_of(incoming[k].type)) for k in keys)
                )
            branches.append(region)
        changed = tuple(
            i for i, key in enumerate(keys) if any(branch.operations[-1].operands[i] is not incoming[key] for branch in branches)
        )
        for branch in branches:
            branch.operations[-1].operands = tuple(branch.operations[-1].operands[i] for i in changed)
        results = self.emit("scf.if", (condition,), tuple(incoming[keys[i]].type for i in changed), regions=branches)
        self.locals, self.env = dict(incoming), env
        self.locals.update(zip((keys[i] for i in changed), results))
        self.local_axes = incoming_axes

    def while_loop(self, node):
        incoming, environment = dict(self.locals), dict(self.env)
        incoming_axes = dict(self.local_axes)
        keys = tuple(incoming)
        types = tuple(incoming[k].type for k in keys)
        before, after = self.builder.region(types), self.builder.region(types)
        with self.insertion(before):
            self.locals = dict(zip(keys, before.arguments))
            condition = self.expr(node.condition)
            if shape_of(condition.type):
                self.error("ordinary while condition must be scalar; a parallel domain needs explicit per-element state")
            self.emit("scf.condition", (condition, *before.arguments))
        with self.insertion(after):
            self.locals = dict(zip(keys, after.arguments))
            self.stmt(node.body)
            self.emit("scf.yield", tuple(self.rename(self.locals[k], shape_of(incoming[k].type), axes_of(incoming[k].type)) for k in keys))
        results = self.emit("scf.while", tuple(incoming[k] for k in keys), types, regions=(before, after))
        self.locals, self.env = dict(zip(keys, results)), environment
        self.local_axes = incoming_axes

    def compile(self):
        function = self.function
        parameters, types, names = [], [], []
        abi_declarations = {}
        strided_arguments = set()

        def abi_view(node):
            if not isinstance(node, tirx.Call) or node.op.name != "tl.weft.view" or not bool(node.annotations.get("abi", False)):
                return
            source = next((buffer for buffer in function.buffer_map.values() if buffer.data.same_as(node.args[0])), None)
            if source is None or source in abi_declarations:
                self.error("each ABI View must declare one unique entry buffer", node)
            from weft_core.encoding import Encoding

            description = json.loads(str(node.args[1].value))
            if "dense" in description:
                element = dtype(description["dense"])
                encoding = EncodingType(element.name, "dense", "dense." + element.name)
            elif "derived" in description:
                derived = description["derived"]
                schema = Encoding.from_dict(derived["source"])
                self.declare_encoding(schema)
                self.encodings[derived["family"]] = schema
                encoding = EncodingType(derived["family"], "derived_instance", derived["identity"], tuple(derived["parameters"]))
            else:
                schema = Encoding.from_dict(description)
                self.declare_encoding(schema)
                encoding = EncodingType(schema.name, "base", schema.name)
            if node.annotations["strides"]:
                if encoding.kind != "dense":
                    self.error("serialized Encoding ABI uses its declared record layout, not dense strides", node)
                declared = tuple(node.annotations["strides"])
                actual = tuple(source.strides)
                if not actual:
                    actual = tuple(math.prod(tuple(self.static(n) for n in source.shape)[i + 1 :]) for i in range(len(source.shape)))
                if len(declared) != len(actual) or any(not self.analyzer.can_prove(a == b) for a, b in zip(declared, actual)):
                    self.error("ABI View strides differ from the source Tensor storage", node)
                strided_arguments.add(source.name)
            shape = []
            for extent in node.args[2:]:
                extent = self.analyzer.simplify(extent)
                if isinstance(extent, tirx.IntImm) and extent.value > 0:
                    shape.append(int(extent.value))
                elif isinstance(extent, tirx.Var):
                    if extent.name not in self.shape_symbols:
                        self.shape_symbols[extent.name] = (extent, -len(self.shape_symbols) - 1)
                    shape.append(self.shape_symbols[extent.name][1])
                else:
                    self.error("ABI View extents must be positive constants or named runtime dimensions", node)
            axes = tuple(self.named_axis(str(axis)) for axis in node.annotations["axes"])
            abi_declarations[source] = (node, ViewType(encoding, tuple(shape), axes))

        tirx.stmt_functor.post_order_visit(function.body, abi_view)
        axis_requests = {}

        def contraction_axes(node):
            if not isinstance(node, tirx.Call):
                return
            if node.op.name == "tl.weft.mac_groups":
                names = (str(node.args[5].value),)
            elif node.op.name == "tl.weft.reduce_dot":
                names = tuple(str(axis.value) for axis in node.annotations["axes"])
            else:
                return
            identities = tuple(self.named_axis(name) for name in names)
            for argument in node.args[:2]:
                buffer, _, shape = self.region(argument)
                if not shape or not buffer.scope().startswith("local"):
                    continue
                if len(shape) < len(identities):
                    self.error("reduction axes exceed the positional tile rank", node)
                for position, identity in enumerate(identities, start=len(shape) - len(identities)):
                    axis_requests.setdefault((buffer, position), set()).add(identity)

        # Explicit named contractions constrain the positional tiles that
        # feed them. Resolve these declarations before importing Parallel
        # producers, as Triton does when constructing a named arange Value.
        tirx.stmt_functor.post_order_visit(function.body, contraction_axes)
        self.operation_axes = {key: next(iter(identities)) for key, identities in axis_requests.items() if len(identities) == 1}
        seen = set(function.params)
        for param in function.params:
            buffer = function.buffer_map.get(param)
            if buffer is None and param.name in self.shape_symbols:
                continue
            names.append(buffer.name if buffer is not None else param.name)
            types.append(
                abi_declarations[buffer][1]
                if buffer in abi_declarations
                else PointerType(ScalarType(dtype(buffer.dtype)))
                if buffer is not None
                else ScalarType(dtype(param.dtype))
            )
            parameters.append((param, buffer))
        for _param, buffer in parameters:
            if buffer is None:
                continue
            for expression in (*buffer.shape, *buffer.strides, buffer.elem_offset):
                for var in tirx.analysis.undefined_vars(expression):
                    if var not in seen and var.name not in self.shape_symbols:
                        seen.add(var)
                        self.implicit.append(var)
                        types.append(ScalarType(dtype(var.dtype)))
                        names.append(var.name)
        types += [ScalarType(dt.index)] * 3
        names += [f"__triton_pid_{i}" for i in range(3)]
        body = self.builder.region(tuple(types), tuple(names))
        self.block = body
        for (param, buffer), value in zip(parameters, body.arguments):
            if buffer is None:
                self.env[param] = value
            else:
                self.pointers[buffer] = value
                self.env[param] = value
                self.env[buffer.data] = value
                if buffer in abi_declarations:
                    self.abi_views[abi_declarations[buffer][0]] = value
        for name, (var, _identity) in self.shape_symbols.items():
            symbol = self.op(
                "symbol", ty=ScalarType(dt.index), attrs={"name": json.dumps(name), "kind": '"shape"', "choices": integers(())}
            )
            self.env[var] = self.cast(symbol, dtype(var.dtype))
            for param in function.params:
                if param.name == name:
                    self.env[param] = self.cast(symbol, dtype(param.dtype))
        for var, value in zip(self.implicit, body.arguments[len(parameters) :]):
            self.env[var] = value
        self.launch = body.arguments[-3:]
        self.active_domain = self.op("root_domain", ty=DomainType("root", 0, -1, 0, "root", "exact"))
        self.level_domains[0] = self.active_domain.type
        self.stmt(function.body)
        self.emit("weft_kernel.return")
        accesses = argument_accesses(body)
        used = body.used_values()
        indices = [i for i, value in enumerate(body.arguments) if i < len(types) - 3 or value in used]
        body.arguments = tuple(body.arguments[i] for i in indices)
        names = [names[i] for i in indices]
        access_names = ["readwrite" if accesses[i] == {"read", "write"} else next(iter(accesses[i]), "none") for i in indices]
        noalias = bool(function.attrs.get("tir.noalias", False))
        alias_groups = {str(name): str(group) for name, group in function.attrs.get("weft.alias_groups", {}).items()}
        memory_names = {name for name, value in zip(names, body.arguments) if isinstance(value.type, (PointerType, ViewType))}
        if set(alias_groups) - memory_names or any(not group for group in alias_groups.values()):
            self.error("weft.alias_groups requires named memory arguments and nonempty alias groups")
        if noalias and alias_groups:
            self.error("tir.noalias and explicit weft.alias_groups cannot be combined")
        identities = {group: i + 1 for i, group in enumerate(sorted(set(alias_groups.values())))}
        alias = [
            i + 1
            if noalias and isinstance(body.arguments[j].type, (PointerType, ViewType))
            else identities.get(alias_groups.get(names[j]), 0)
            if isinstance(body.arguments[j].type, (PointerType, ViewType))
            else -1
            for j, i in enumerate(indices)
        ]
        symbol = "tilelang_" + str(function.attrs["global_symbol"])
        kernel = self.builder.operation(
            "weft_kernel.kernel",
            self.location(function),
            attributes={
                "sym_name": json.dumps(symbol),
                "arg_names": json.dumps(names),
                "arg_access": json.dumps(access_names),
                "arg_alias_sets": integers(alias),
                "strided_args": json.dumps(sorted(strided_arguments)),
                "shape_symbols": json.dumps(list(self.shape_symbols)),
                "source": json.dumps(self.location(function).format()),
            },
            regions=(body,),
        )
        module = compiler.create_module(compiler.Context(), symbol, (*self.declarations, kernel), type_encoder=emit_type)
        return module, self


def import_module(module):
    from tvm import transform

    # T.unroll is an author request, like Triton's static_range. Expand it
    # with the native TIRx transform before inventing loop-carried SSA for
    # local tiles. Ordinary serial/Level and pipelined loops stay structured.
    with transform.PassContext(config={"tirx.UnrollLoop": {
        "auto_max_step": 0, "auto_max_extent": 0, "explicit_unroll": True,
        "unroll_local_access": False,
    }}):
        module = tirx.transform.UnrollLoop()(module)
    functions = [function for function in module.functions.values() if isinstance(function, tirx.PrimFunc)]
    if len(functions) != 1:
        raise NotImplementedError("one TileLang JITKernel must contain one entry PrimFunc; helpers are imported within that entry")
    return Importer(functions[0]).compile()
