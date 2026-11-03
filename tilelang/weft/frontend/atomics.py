"""Import native TileLang atomic operations without changing their memory scope."""

import json

from tvm import tirx
from weft_core import dtypes as dt
from weft_core.types import ScalarType, value_type


class AtomicOperations:
    def atomic(self, buffer, indices, value, ordering, kind, compare=None):
        from .importer import dtype

        if buffer not in self.pointers:
            self.error("atomic storage must have an addressable origin")
        scalar = dtype(buffer.dtype)
        if kind == "add" and scalar.category in (dt.DTypeCategory.FLOAT, dt.DTypeCategory.BFLOAT):
            kind = "fadd"
        elif kind in ("min", "max") and scalar.signedness == "unsigned":
            kind = "u" + kind
        offsets = self.offsets(buffer, indices)
        value = self.cast(value, scalar)
        shape, axes = self.memory_domain(offsets, value)
        offsets = self.broadcast(offsets, shape, axes)
        orders = {0: "relaxed", 1: "acquire", 2: "acquire", 3: "release", 4: "acq_rel", 5: "seq_cst"}
        if ordering not in orders:
            self.error(f"unknown TileLang atomic memory ordering {ordering}")
        return self.op(
            "pointer_atomic",
            (
                self.pointers[buffer],
                offsets,
                self.access_mask(buffer, indices, shape, axes),
                value,
                self.constant(0, scalar) if compare is None else self.cast(compare, scalar),
            ),
            value_type(ScalarType(scalar), shape, axes),
            {
                "kind": json.dumps(kind),
                "ordering": json.dumps(orders[ordering]),
                "scope": json.dumps("sys"),
            },
        )

    def atomic_expression(self, call):
        name = call.op.name
        kinds = {f"tl.atomic_{kind}{suffix}_elem_op": kind for kind in ("add", "min", "max", "or") for suffix in ("", "_ret")}
        kinds.update({"tl.atomic_load_elem_op": "load", "tl.atomic_store_elem_op": "store"})
        if name == "tl.weft.atomic_rmw":
            kinds[name] = str(call.annotations["kind"].value)
        if name not in kinds:
            return None
        pointer = call.args[0]
        if not isinstance(pointer, tirx.Call) or pointer.op.name != "tl.access_ptr":
            self.error("an atomic element requires a typed TileLang access_ptr")
        address = pointer.args[0]
        if not isinstance(address, tirx.BufferLoad):
            self.error("atomic access_ptr must preserve its Buffer and element coordinates")
        from .importer import dtype

        if kinds[name] == "load":
            value, order = self.constant(0, dtype(address.buffer.dtype)), self.static(call.args[1])
        else:
            value = self.expr(call.args[1])
            order = self.static(call.args[3]) if name == "tl.weft.atomic_rmw" else self.static(call.args[2]) if len(call.args) > 2 else 0
        compare = self.expr(call.args[2]) if name == "tl.weft.atomic_rmw" else None
        return self.atomic(address.buffer, [self.expr(i) for i in address.indices], value, order, kinds[name], compare)

    def atomic_tile(self, call):
        kinds = {"tl.tileop.atomicadd": "add", "tl.tileop.atomicmin": "min", "tl.tileop.atomicmax": "max"}
        if call.op.name not in kinds:
            return False
        destination = self.region(call.args[1])
        axes = tuple(self.axis() for _ in destination[2])
        argument = call.args[0]
        value = (
            self.read_region(self.region(argument), axes)
            if isinstance(argument, (tirx.BufferRegion, tirx.BufferLoad))
            else self.expr(argument)
        )
        self.atomic(
            destination[0],
            self.region_indices(destination, axes),
            value,
            self.static(call.annotations.get("memory_order", tirx.IntImm("int32", 0))),
            kinds[call.op.name],
        )
        return True
