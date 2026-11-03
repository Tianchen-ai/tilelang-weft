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
    ScalarType, PointerType, LocalValueType, DomainType,
    emit_type, element_type, shape_of, axes_of, value_type,
)
from .tiles import TileOperations
from .model import ModelOperations
from .atomics import AtomicOperations
from .helpers import HelperOperations


def dtype(name):
    name = str(name)
    aliases = {"bool": "i1", "int": "i32", "float": "f32", "bfloat16": "bf16"}
    name = aliases.get(name, name.replace("uint", "u").replace("int", "i").replace("float", "f"))
    result = getattr(dt, name, None)
    if not isinstance(result, dt.DType):
        raise NotImplementedError(f"Weft has no Canonical numerical type for {name}")
    return result


def integers(values):
    return "array<i64: " + ", ".join(map(str, values)) + ">" if values else "array<i64>"


class Importer(ModelOperations, TileOperations, AtomicOperations, HelperOperations):
    def __init__(self, function):
        self.function = function
        self.builder = IRBuilder()
        self.block = None
        self.env = {}
        self.locals = {}
        self.pointers = {}
        self.local_axes = {}
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
        self.active_domain = None
        self.birth_phase = None
        self.birth_outputs = []
        self.readonly = set()
        self.parallel_domains = {}
        self.helper_stack = []
        self.helper_results = {}
        self.derivation = None
        self.derived_layouts = set()

    def location(self, node=None):
        span = getattr(self.current_node if node is None else node, "span", None)
        if span is None:
            return SourceLocation("<tilelang>", 1, 0)
        return SourceLocation(str(span.source_name.name), max(1, int(span.line)), max(0, int(span.column) - 1))

    def error(self, message, node=None):
        raise FrontendError(message, self.location(node))

    def emit(self, name, operands=(), types=(), attrs=None, regions=()):
        return self.builder.emit(self.block, name, self.location(), operands=tuple(operands),
                                 result_types=tuple(types), attributes=attrs, regions=tuple(regions))

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
            if math.isinf(float(number)):
                bits = {16: ("0x7C00", "0xFC00"), 32: ("0x7F800000", "0xFF800000"),
                        64: ("0x7FF0000000000000", "0xFFF0000000000000")}[typ.bits]
                spelling = bits[float(number) < 0]
            else:
                spelling = repr(float(number))
            spelling += " : " + emit_type(ty)
        else:
            spelling = str(int(number)) + " : " + emit_type(ty)
        operation = "weft_kernel.constant" if typ.category == dt.DTypeCategory.INTEGER else "arith.constant"
        return self.emit(operation, types=(ty,), attrs={"value": spelling})[0]

    def cast(self, value, typ):
        if element_type(value.type) == ScalarType(typ):
            return value
        return self.op("cast", (value,), value_type(ScalarType(typ), shape_of(value.type), axes_of(value.type)))

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
        return self.op("compare" if predicate else "binary", (left, right), ty,
                       {"predicate" if predicate else "kind": json.dumps(kind)})

    def broadcast(self, value, shape, axes):
        if (shape_of(value.type), axes_of(value.type)) == (tuple(shape), tuple(axes)):
            return value
        target = value_type(element_type(value.type), tuple(shape), tuple(axes))
        if not shape_of(value.type):
            return self.op("new", (value,), target, {"initialized": "true"})
        # Select broadcasts its scalar true branch over the false value and
        # explicit domain mask without arithmetic that could alter signed zero.
        condition = self.op("new", (self.constant(True, dt.i1),),
                            value_type(ScalarType(dt.i1), tuple(shape), tuple(axes)), {"initialized": "true"})
        zero = self.constant(0, element_type(value.type).dtype)
        selected = self.op("select", (condition, value, zero),
                           value_type(element_type(value.type), *self.domain(value, condition)))
        return self.rename(selected, shape, axes)

    def rename(self, value, shape, axes, order=None):
        target = value_type(element_type(value.type), tuple(shape), tuple(axes))
        if value.type == target and order is None:
            return value
        if not shape_of(value.type):
            return self.broadcast(value, shape, axes)
        if not shape:
            indices = (self.constant(0, dt.index),) * len(shape_of(value.type))
            return self.op("extract", (value, *indices), target,
                           {"selectors": json.dumps(["index"] * len(indices))})
        return self.op("reshape", (value,), target, {"order": integers(order or axes_of(value.type))})

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
        if isinstance(node, tirx.BufferLoad):
            axes = self.complete_local_domain(node.buffer, node.indices)
            if axes is not None:
                return self.rename(self.locals[node.buffer],
                                   shape_of(self.locals[node.buffer].type), axes)
            return self.load(node.buffer, [self.expr(i) for i in node.indices])
        if isinstance(node, tirx.Select):
            condition, yes, no = self.expr(node.condition), self.expr(node.true_value), self.expr(node.false_value)
            shape, axes = self.domain(yes, no, condition)
            return self.op("select", (condition, yes, no), value_type(element_type(yes.type), shape, axes))
        names = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Mod": "mod",
                 "Min": "min", "Max": "max", "And": "and", "Or": "or",
                 "EQ": "eq", "NE": "ne", "LT": "lt", "LE": "le", "GT": "gt", "GE": "ge"}
        kind = type(node).__name__
        if kind in names:
            return self.binary(names[kind], self.expr(node.a), self.expr(node.b), predicate=kind in ("EQ", "NE", "LT", "LE", "GT", "GE"))
        if kind in ("FloorDiv", "FloorMod"):
            # TIR floor division differs from signed truncation. Keep its
            # quotient/remainder correction explicit in the Canonical program.
            left, right = self.expr(node.a), self.expr(node.b)
            q = self.binary("div", left, right)
            r = self.binary("mod", left, right)
            zero = self.constant(0, element_type(left.type).dtype)
            nonzero = self.binary("ne", r, zero, predicate=True)
            signs = self.binary("ne", self.binary("lt", left, zero, predicate=True),
                                self.binary("lt", right, zero, predicate=True), predicate=True)
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
            binary = {"tirx.bitwise_and": "and", "tirx.bitwise_or": "or", "tirx.bitwise_xor": "xor",
                      "tirx.shift_left": "shl", "tirx.shift_right": "shr"}
            if name in binary:
                return self.binary(binary[name], self.expr(node.args[0]), self.expr(node.args[1]))
            unary = {"exp", "exp2", "log", "log2", "sqrt", "rsqrt", "abs", "sin", "cos", "tanh", "floor", "ceil"}
            leaf = name.rsplit(".", 1)[-1]
            if leaf in unary:
                value = self.expr(node.args[0])
                return self.op("unary", (value,), value.type, {"kind": json.dumps(leaf)})
            if leaf == "if_then_else":
                condition, yes, no = map(self.expr, node.args)
                shape, axes = self.domain(yes, no, condition)
                return self.op("select", (condition, yes, no), value_type(element_type(yes.type), shape, axes))
            if leaf == "reinterpret":
                value = self.expr(node.args[0])
                return self.op("bitcast", (value,), value_type(ScalarType(dtype(node.dtype)), shape_of(value.type), axes_of(value.type)))
        self.error(f"TIRx expression {type(node).__name__} is not yet imported: {node}", node)

    def static(self, expression):
        simplified = self.analyzer.simplify(expression)
        if not isinstance(simplified, tirx.IntImm):
            self.error(f"this local tile requires a static extent, got {expression}")
        return int(simplified.value)

    def iota(self, extent, axis):
        raw = self.op("iota", ty=LocalValueType(ScalarType(dt.u32), (extent,), (axis,)),
                      attrs={"start": "0", "end": str(extent)})
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
            result = self.binary("add", result, self.binary("mul", self.cast(index, dt.i64), self.cast(self.expr(stride), dt.i64)))
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

    def load(self, buffer, indices):
        if buffer in self.pointers:
            offsets = self.offsets(buffer, indices)
            shape, axes = self.memory_domain(offsets)
            offsets = self.broadcast(offsets, shape, axes)
            ty = value_type(ScalarType(dtype(buffer.dtype)), shape, axes)
            other = self.broadcast(self.constant(0, dtype(buffer.dtype)), shape, axes)
            return self.op("pointer_load", (self.pointers[buffer], offsets, self.access_mask(buffer, indices, shape, axes), other), ty)
        if buffer not in self.locals:
            self.error(f"read of unallocated local buffer {buffer.name}")
        source = self.locals[buffer]
        selectors, operands = [], []
        shape, axes = (), ()
        for index in indices:
            if shape_of(index.type):
                selectors.append("gather")
                index = self.cast(index, dt.u32)
            else:
                selectors.append("index")
                index = self.cast(index, dt.index)
            operands.append(index)
        for index in operands:
            shape, axes = self.domain_types(shape, axes, index)
        return self.op("extract", (source, *operands), value_type(element_type(source.type), shape, axes),
                       {"selectors": json.dumps(selectors)})

    def domain_types(self, shape, axes, value):
        dimensions = dict(zip(axes, shape))
        for axis, size in zip(axes_of(value.type), shape_of(value.type)):
            dimensions[axis] = size
        return tuple(dimensions.values()), tuple(dimensions)

    def store(self, buffer, indices, value):
        if buffer in self.readonly:
            self.error(f"staged buffer {buffer.name} is an immutable snapshot")
        value = self.cast(value, dtype(buffer.dtype))
        if buffer in self.pointers:
            offsets = self.offsets(buffer, indices)
            shape, axes = self.memory_domain(offsets, value)
            offsets, value = (self.broadcast(v, shape, axes) for v in (offsets, value))
            self.op("pointer_store", (self.pointers[buffer], offsets, self.access_mask(buffer, indices, shape, axes), value))
            return
        if buffer not in self.locals:
            self.error(f"write of unallocated local buffer {buffer.name}")
        if self.mask is not None:
            old = self.load(buffer, indices)
            shape, axes = self.domain(value, old, self.mask)
            value = self.op("select", (self.mask, value, old), value_type(element_type(value.type), shape, axes))
        converted = [self.cast(i, dt.u32 if shape_of(i.type) else dt.index) for i in indices]
        selectors = ["gather" if shape_of(i.type) else "index" for i in converted]
        original = self.locals[buffer]
        self.locals[buffer] = self.op("update", (original, value, *converted), original.type,
                                      {"selectors": json.dumps(selectors)})

    def complete_local_domain(self, buffer, indices):
        if buffer not in self.locals or self.mask is not None or len(indices) != len(buffer.shape):
            return None
        axes = []
        for index, size in zip(indices, buffer.shape):
            if not isinstance(index, tirx.Var) or index not in self.parallel_domains:
                return None
            start, extent, axis = self.parallel_domains[index]
            if not self.analyzer.can_prove_equal(start, 0) or not self.analyzer.can_prove_equal(extent, size):
                return None
            axes.append(axis)
        return tuple(axes) if len(set(axes)) == len(axes) else None

    def allocate(self, buffer):
        if buffer in self.locals:
            self.error(f"local buffer {buffer.name} was allocated twice")
        if buffer.scope() not in ("local", "local.var", "local.fragment", "shared"):
            self.error(f"CPU local allocation does not implement storage scope {buffer.scope()}")
        shape = tuple(self.static(n) for n in buffer.shape)
        axes = tuple(self.axis() for _ in shape)
        self.local_axes[buffer] = axes
        ty = LocalValueType(ScalarType(dtype(buffer.dtype)), shape, axes)
        self.locals[buffer] = self.op("empty", tuple(self.constant(n, dt.index) for n in shape), ty)

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
            return
        if isinstance(node, tirx.BufferStore):
            axes = self.complete_local_domain(node.buffer, node.indices)
            value = self.expr(node.value)
            if axes is not None:
                if node.buffer in self.readonly:
                    self.error(f"staged buffer {node.buffer.name} is an immutable snapshot")
                value = self.broadcast(value, shape_of(self.locals[node.buffer].type), axes)
                self.locals[node.buffer] = self.rename(self.cast(value, dtype(node.buffer.dtype)),
                                                      shape_of(self.locals[node.buffer].type),
                                                      self.local_axes[node.buffer])
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
            if "weft.derive" in node.annotations:
                self.layout_builder(node)
                return
            if "weft.helper" in node.annotations:
                self.source_helper(node)
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
                axis = self.axis()
                value = self.iota(extent, axis)
                self.env[node.loop_var] = self.binary("add", self.expr(node.min), value)
                self.parallel_domains[node.loop_var] = (node.min, node.extent, axis)
                try:
                    self.stmt(node.body)
                finally:
                    self.parallel_domains.pop(node.loop_var)
                    self.env = environment
                return
            self.serial(node)
            return
        if isinstance(node, tirx.While):
            self.while_loop(node)
            return
        if isinstance(node, tirx.IfThenElse):
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
            if isinstance(node.value, tirx.Call) and (self.model_tile(node.value) or self.tile_operation(node.value) or self.atomic_tile(node.value)):
                return
            self.expr(node.value)
            return
        if isinstance(node, tirx.AttrStmt):
            self.error(f"unhandled TIRx region attribute {node.attr_key}")
        self.error(f"TIRx statement {name} has no Canonical import: {node}")

    def serial(self, node):
        environment = dict(self.env)
        lower, upper, step = self.cast(self.expr(node.min), dt.index), self.cast(self.expr(node.min + node.extent), dt.index), self.constant(1, dt.index)
        incoming = dict(self.locals)
        keys = tuple(incoming)
        region = self.builder.region((ScalarType(dt.index), *(incoming[k].type for k in keys)))
        with self.insertion(region):
            self.locals = {k: value for k, value in zip(keys, region.arguments[1:])}
            self.env[node.loop_var] = self.cast(region.arguments[0], dtype(node.loop_var.dtype))
            self.stmt(node.body)
            final = [self.locals[k] for k in keys]
            self.emit("scf.yield", final)
        results = self.emit("scf.for", (lower, upper, step, *(incoming[k] for k in keys)),
                            tuple(incoming[k].type for k in keys), regions=(region,))
        self.locals = dict(zip(keys, results))
        self.env = environment

    def conditional(self, node, condition):
        incoming, env = dict(self.locals), dict(self.env)
        keys = tuple(incoming)
        branches = []
        for statement in (node.then_case, node.else_case):
            region = self.builder.region(())
            self.locals, self.env = dict(incoming), dict(env)
            with self.insertion(region):
                if statement is not None:
                    self.stmt(statement)
                self.emit("scf.yield", tuple(self.locals[k] for k in keys))
            branches.append(region)
        results = self.emit("scf.if", (condition,), tuple(incoming[k].type for k in keys), regions=branches)
        self.locals, self.env = dict(zip(keys, results)), env

    def while_loop(self, node):
        incoming, environment = dict(self.locals), dict(self.env)
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
            self.emit("scf.yield", tuple(self.locals[k] for k in keys))
        results = self.emit("scf.while", tuple(incoming[k] for k in keys), types, regions=(before, after))
        self.locals, self.env = dict(zip(keys, results)), environment

    def compile(self):
        function = self.function
        parameters, types, names = [], [], []
        seen = set(function.params)
        for param in function.params:
            buffer = function.buffer_map.get(param)
            names.append(buffer.name if buffer is not None else param.name)
            types.append(PointerType(ScalarType(dtype(buffer.dtype))) if buffer is not None else ScalarType(dtype(param.dtype)))
            parameters.append((param, buffer))
        for param, buffer in parameters:
            if buffer is None:
                continue
            for expression in (*buffer.shape, *buffer.strides, buffer.elem_offset):
                for var in tirx.analysis.undefined_vars(expression):
                    if var not in seen:
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
        for var, value in zip(self.implicit, body.arguments[len(parameters):]):
            self.env[var] = value
        self.launch = body.arguments[-3:]
        self.active_domain = self.op("root_domain", ty=DomainType("root", 0, -1, 0, "root", "exact"))
        self.stmt(function.body)
        self.emit("weft_kernel.return")
        accesses = argument_accesses(body)
        used = body.used_values()
        indices = [i for i, value in enumerate(body.arguments) if i < len(types) - 3 or value in used]
        body.arguments = tuple(body.arguments[i] for i in indices)
        names = [names[i] for i in indices]
        access_names = ["readwrite" if accesses[i] == {"read", "write"} else next(iter(accesses[i]), "none") for i in indices]
        noalias = bool(function.attrs.get("tir.noalias", False))
        alias = [i + 1 if noalias and isinstance(body.arguments[j].type, PointerType) else
                 0 if isinstance(body.arguments[j].type, PointerType) else -1 for j, i in enumerate(indices)]
        symbol = "tilelang_" + str(function.attrs["global_symbol"])
        kernel = self.builder.operation("weft_kernel.kernel", self.location(function), attributes={
            "sym_name": json.dumps(symbol), "arg_names": json.dumps(names), "arg_access": json.dumps(access_names),
            "arg_alias_sets": integers(alias), "strided_args": "[]", "shape_symbols": "[]",
            "source": json.dumps(self.location(function).format()),
        }, regions=(body,))
        module = compiler.create_module(compiler.Context(), symbol, (*self.declarations, kernel), type_encoder=emit_type)
        return module, self


def import_module(module):
    functions = [function for function in module.functions.values() if isinstance(function, tirx.PrimFunc)]
    if len(functions) != 1:
        raise NotImplementedError("one TileLang JITKernel must contain one entry PrimFunc; helpers are imported within that entry")
    return Importer(functions[0]).compile()
