"""Semantic tile operations retained by the native TileLang builder."""
import json

from tvm import tirx
from weft_core import dtypes as dt
from weft_core.types import ScalarType, LocalValueType, axes_of, shape_of, element_type, value_type, emit_type


class TileOperations:
    def region(self, argument):
        if isinstance(argument, tirx.BufferRegion):
            return argument.buffer, tuple(r.min for r in argument.region), tuple(self.static(r.extent) for r in argument.region)
        if isinstance(argument, tirx.Buffer):
            return argument, (tirx.IntImm("int32", 0),) * len(argument.shape), tuple(self.static(n) for n in argument.shape)
        if isinstance(argument, tirx.Call) and argument.op.name == "tl.region":
            load = argument.args[0]
            return load.buffer, tuple(load.indices), tuple(self.static(n) for n in argument.args[2:])
        if isinstance(argument, tirx.BufferLoad):
            return argument.buffer, tuple(argument.indices), (1,) * len(argument.indices)
        self.error(f"tile operand does not carry a BufferRegion: {argument}")

    def region_indices(self, region, axes):
        buffer, starts, shape = region
        return [self.binary("add", self.expr(start), self.iota(size, axis))
                for start, size, axis in zip(starts, shape, axes)]

    def read_region(self, region, axes):
        buffer, starts, shape = region
        if buffer in self.locals and all(self.static(start) == 0 for start in starts) and shape == shape_of(self.locals[buffer].type):
            return self.rename(self.locals[buffer], shape, axes)
        return self.load(buffer, self.region_indices(region, axes))

    def write_region(self, region, value, axes):
        buffer, starts, shape = region
        if buffer in self.readonly:
            self.error(f"staged buffer {buffer.name} is an immutable snapshot")
        from .importer import dtype
        value = self.cast(value, dtype(buffer.dtype))
        value = self.rename(value, shape, axes)
        if buffer in self.locals and all(self.static(start) == 0 for start in starts) and shape == shape_of(self.locals[buffer].type) and self.mask is None:
            self.locals[buffer] = self.rename(value, shape, self.local_axes[buffer])
            return
        self.store(buffer, self.region_indices(region, axes), value)

    def tile_operation(self, call):
        name = call.op.name
        if name == "tl.tileop.copy":
            left, right = self.region(call.args[0]), self.region(call.args[1])
            if left[2] != right[2]:
                self.error("CPU tile copy requires explicitly matching logical extents")
            axes = tuple(self.axis() for _ in left[2])
            self.write_region(right, self.read_region(left, axes), axes)
            return True
        if name == "tl.tileop.fill":
            region = self.region(call.args[0])
            from .importer import dtype
            value = self.cast(self.expr(call.args[1]), dtype(region[0].dtype))
            axes = tuple(self.axis() for _ in region[2])
            self.write_region(region, self.broadcast(value, region[2], axes), axes)
            return True
        if name in ("tl.tileop.cumsum", "tl.tileop.cummax"):
            source, output = self.region(call.args[0]), self.region(call.args[1])
            axis, reverse = self.static(call.args[2]), bool(self.static(call.args[3]))
            axes = tuple(self.axis() for _ in source[2])
            value = self.read_region(source, axes)
            combine = self.builder.region((element_type(value.type),) * 2)
            with self.insertion(combine):
                result = self.binary("add" if name.endswith("cumsum") else "max", *combine.arguments)
                self.emit("weft_kernel.scan_yield", (result,))
            scanned = self.emit("weft_kernel.scan", (value,), (value.type,),
                                {"axis": str(axis), "reverse": "true" if reverse else "false"}, (combine,))[0]
            self.write_region(output, scanned, axes)
            return True
        if name == "tl.tileop.gemm":
            left, right, output = [self.region(a) for a in call.args[:3]]
            transpose_left, transpose_right = (bool(int(a.value)) for a in call.args[3:5])
            m, n, k = (self.static(a) for a in call.args[5:8])
            clear = bool(int(call.args[9].value))
            row, column, reduction = self.axis(), self.axis(), self.axis()
            left_axes = ((reduction, row) if transpose_left else (row, reduction))
            right_axes = ((column, reduction) if transpose_right else (reduction, column))
            if len(left[2]) != 2 or len(right[2]) != 2 or len(output[2]) != 2:
                self.error("gemm tile operands require explicit rank-two regions")
            a, b = self.read_region(left, left_axes), self.read_region(right, right_axes)
            if transpose_left:
                a = self.rename(a, (m, k), (row, reduction), (row, reduction))
            if transpose_right:
                b = self.rename(b, (k, n), (reduction, column), (reduction, column))
            from .importer import dtype, integers
            accum = ScalarType(dtype(output[0].dtype))
            engine = call.annotations.get("weft_engine", "auto")
            engine = str(engine.value) if isinstance(engine, tirx.StringImm) else str(engine)
            product = self.op("outer_contract", (a, b), LocalValueType(accum, (m, n), (row, column)),
                              {"over": integers((reduction,)), "acc_type": emit_type(accum),
                               "engine": json.dumps(engine)})
            if not clear:
                product = self.binary("add", self.read_region(output, (row, column)), product)
            self.write_region(output, product, (row, column))
            return True
        if name == "tl.tileop.reduce":
            source, output = self.region(call.args[0]), self.region(call.args[1])
            kind = str(call.args[2].value)
            axis = self.static(call.args[3])
            if axis < 0:
                axis += len(source[2])
            clear = bool(self.static(call.args[4]))
            names = {"sum": "add", "max": "max", "min": "min", "bitand": "and", "bitor": "or", "bitxor": "xor"}
            if kind not in names and kind not in ("abssum", "absmax"):
                self.error(f"unknown tile reduction kind {kind}")
            axes = tuple(self.axis() for _ in source[2])
            value = self.read_region(source, axes)
            if kind in ("abssum", "absmax"):
                value = self.op("unary", (value,), value.type, {"kind": json.dumps("abs")})
                kind = "sum" if kind == "abssum" else "max"
            shape = source[2][:axis] + source[2][axis + 1:]
            result_axes = axes[:axis] + axes[axis + 1:]
            output_type = value_type(element_type(value.type), shape, result_axes)
            if kind.startswith("bit"):
                region = self.builder.region((element_type(value.type),) * 2)
                with self.insertion(region):
                    combined = self.binary(names[kind], *region.arguments)
                    self.emit("weft_kernel.reduce_yield", (combined,))
                result = self.emit("weft_kernel.reduce_region", (value,), (output_type,),
                                   {"axis": str(axis)}, (region,))[0]
            else:
                result = self.op("reduce", (value,), output_type,
                                 {"kind": json.dumps(names[kind]), "axis": str(axis)})
            if len(output[2]) == len(source[2]):
                shape = shape[:axis] + (1,) + shape[axis:]
                result_axes = result_axes[:axis] + (axes[axis],) + result_axes[axis:]
                result = self.rename(result, shape, result_axes)
            if not clear:
                result = self.binary(names[kind], self.read_region(output, result_axes), result)
            self.write_region(output, result, result_axes)
            return True
        return False
