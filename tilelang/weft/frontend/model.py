"""Import explicit source-model regions and operations into Canonical IR."""

import json

from tvm import tirx
from weft_core import dtypes as dt
from weft_core.encoding import Encoding
from weft_core.types import (
    DomainType,
    DomainPointType,
    EncodingType,
    PointerType,
    ViewType,
    SliceType,
    LocalValueType,
    ScalarType,
    shape_of,
    axes_of,
    element_type,
    value_type,
)


def string(value):
    return str(value.value) if isinstance(value, tirx.StringImm) else str(value)


class ModelOperations:
    def declare_encoding(self, schema):
        if schema.name in self.encodings and self.encodings[schema.name] != schema:
            self.error(f"two different Encoding declarations share {schema.name}")
        if schema.name not in self.encodings:
            self.encodings[schema.name] = schema
            self.declarations.append(self.builder.operation("weft_kernel.encoding_decl", self.location(), attributes=schema.attributes()))

    def named_axis(self, name):
        if name not in self.named_axes:
            self.named_axes[name] = self.axis()
        return self.named_axes[name]

    def align_anonymous_batches(self, left, right, *, matrix=False):
        left_shape, right_shape = shape_of(left.type), shape_of(right.type)
        left_axes, right_axes = axes_of(left.type), list(axes_of(right.type))
        named = set(self.named_axes.values())
        # Matrix operands have independent free row/column axes immediately
        # before K. Only their leading batch axes align positionally.
        for offset in range(3 if matrix else 2, min(len(left_shape), len(right_shape)) + 1):
            if left_shape[-offset] != right_shape[-offset] and 1 not in (left_shape[-offset], right_shape[-offset]):
                continue
            if right_axes[-offset] not in named:
                right_axes[-offset] = left_axes[-offset]
        return left, self.rename(right, right_shape, tuple(right_axes))

    def align_contraction_batches(self, left, right, reduction, output_shape):
        """Use the declared destination to distinguish matrix and grouped tiles."""
        def fits(a, b):
            shape, axes = self.domain(a, b)
            free = tuple(size for size, axis in zip(shape, axes) if axis not in reduction)
            return free == output_shape or (not free and output_shape == (1,))

        if fits(left, right):
            return left, right
        # Ordinary matrix tiles share leading batches; grouped elementwise
        # partials can also share the coordinate immediately before K. Neither
        # rule replaces an explicit named axis or changes a tile's extent.
        for matrix in (True, False):
            a, b = self.align_anonymous_batches(left, right, matrix=matrix)
            if fits(a, b):
                return a, b
        return left, right

    def bind_group_axis(self, left, right, name):
        named = set(self.named_axes.values())
        left_axes, right_axes = axes_of(left.type), axes_of(right.type)
        if not left_axes or not right_axes:
            self.error("grouped operands require a shaped reduction axis")
        if name not in self.named_axes and left_axes[-1] not in named and right_axes[-1] not in named:
            self.named_axes[name] = left_axes[-1]
        axis = self.named_axis(name)
        values = []
        for value, axes in ((left, left_axes), (right, right_axes)):
            if axes[-1] != axis:
                if axes[-1] in named:
                    self.error("grouped operands cannot replace a different explicit logical axis")
                value = self.rename(value, shape_of(value.type), (*axes[:-1], axis))
            values.append(value)
        return axis, *values

    def model_expr(self, call):
        name = call.op.name
        if name == "tl.weft.narrow":
            from .importer import dtype

            value = self.expr(call.args[0])
            output = value_type(ScalarType(dtype(call.dtype)), shape_of(value.type), axes_of(value.type))
            return self.op(
                "narrow",
                (value,),
                output,
                {"rounding": json.dumps(string(call.args[1])), "saturate": "true" if self.static(call.args[2]) else "false"},
            )
        if name == "tl.weft.field_view":
            owner = self.expr(call.args[0])
            field_name = string(call.args[1])
            schema = self.encodings[owner.type.encoding.family]
            field = next(field for field in schema.fields if field.name == field_name)
            shape = shape_of(owner.type)[:-1] + field.shape
            axes = (
                axes_of(owner.type)[:-1] + tuple(self.axis() for _ in field.shape[:-1]) + (axes_of(owner.type)[-1:] if field.shape else ())
            )
            scalar = getattr(dt, field.dtype)
            encoding = EncodingType(scalar.name, "dense", "dense." + scalar.name)
            return self.op("field", (owner,), SliceType(encoding, shape, axes), {"name": json.dumps(field_name)})
        if name == "tl.weft.extent":
            view = self.expr(call.args[0])
            bound = self.op("extent", (view,), ScalarType(dt.index), {"axis": str(self.static(call.args[1]))})
            return self.cast(bound, dt.i64)
        if name == "tl.weft.combine_argument":
            if getattr(self, "combiner", None) is None:
                self.error("combine_argument requires a closed reduction region")
            return self.combiner[0][self.static(call.args[0])]
        if name == "tl.weft.derive_argument":
            if self.derivation is None:
                self.error("derive_argument requires a layout-builder region")
            return self.derivation["argument"]
        if name == "tl.weft.interleave":
            if self.derivation is None:
                self.error("interleave is a persistent layout declaration used inside derive")
            source = self.expr(call.args[0])
            rows = self.static(call.args[1])
            from .importer import integers

            return self.op(
                "interleave",
                (source,),
                self.derivation["result"],
                {"rows": str(rows), "field_chunks": integers(tuple(int(x) for x in call.annotations["field_chunks"]))},
            )
        if name == "tl.weft.helper_result":
            return self.helper_result(call)
        if name == "tl.weft.view":
            from .importer import dtype, integers

            if bool(call.annotations.get("abi", False)):
                if call not in self.abi_views:
                    self.error("an ABI View must be an entry declaration with static or named runtime extents")
                return self.abi_views[call]
            description = json.loads(string(call.args[1]))
            schema = None
            rank = int(call.annotations["rank"])
            extents = tuple(self.cast(self.expr(extent), dt.index) for extent in call.args[2 : 2 + rank])
            shape = []
            for extent, value in zip(call.args[2 : 2 + rank], extents):
                static = self.analyzer.simplify(extent)
                if isinstance(static, tirx.IntImm):
                    shape.append(int(static.value))
                else:
                    identity = self.next_local_shape
                    self.next_local_shape += 1
                    self.op("local_shape", (value,), ScalarType(dt.index), {"id": str(identity)})
                    shape.append(-identity)
            axes = tuple(self.named_axis(str(name)) for name in call.annotations["axes"])
            if "dense" in description:
                scalar = dtype(description["dense"])
                encoding = EncodingType(scalar.name, "dense", "dense." + scalar.name)
            elif "derived" in description:
                derived = description["derived"]
                schema = Encoding.from_dict(derived["source"])
                self.declare_encoding(schema)
                self.encodings[derived["family"]] = schema
                encoding = EncodingType(derived["family"], "derived_instance", derived["identity"], tuple(derived["parameters"]))
            else:
                schema = Encoding.from_dict(description)
                encoding = EncodingType(schema.name, "base", schema.name)
                self.declare_encoding(schema)
            strides = tuple(self.cast(self.expr(s), dt.index) for s in call.annotations["strides"])
            storage = self.expr(call.args[0])
            if not isinstance(storage.type, PointerType):
                self.error("T.view requires pointer storage; use slices of an existing ABI View rather than rebinding its storage")
            if schema is not None and encoding.kind != "dense" and storage.name_hint:
                contract = {
                    "encoding": encoding.layout_identity,
                    "shape": list(shape),
                    "record_elements": schema.elements,
                    "storage_bytes": schema.storage_bits // 8,
                }
                contracts = self.storage_contracts.setdefault(storage.name_hint, [])
                if contract not in contracts:
                    contracts.append(contract)
            return self.op(
                "bind_view",
                (storage, *extents, *strides),
                ViewType(encoding, tuple(shape), axes),
                {"operandSegmentSizes": "array<i32: 1, " + str(len(extents)) + ", " + str(len(strides)) + ">"},
            )
        if name == "tl.weft.slice":
            base = self.expr(call.args[0])
            if bool(call.annotations.get("subview", False)):
                from .importer import integers

                offsets, extents, shape, dynamic = [], [], [], []
                for offset in call.args[2:]:
                    simplified = self.analyzer.simplify(offset)
                    if isinstance(simplified, tirx.IntImm):
                        offsets.append(int(simplified.value))
                    else:
                        offsets.append(-1)
                        dynamic.append(self.cast(self.expr(offset), dt.index))
                for axis, extent in enumerate(call.annotations["extents"]):
                    simplified = self.analyzer.simplify(extent)
                    if isinstance(simplified, tirx.IntImm):
                        extents.append(int(simplified.value))
                        shape.append(int(simplified.value))
                    else:
                        extents.append(-1)
                        shape.append(shape_of(base.type)[axis])
                        dynamic.append(self.cast(self.expr(extent), dt.index))
                return self.op(
                    "subview", (base, *dynamic), SliceType(base.type.encoding, tuple(shape), axes_of(base.type)),
                    {"offsets": integers(offsets), "extents": integers(extents)},
                )
            selectors = json.loads(string(call.args[1]))
            values, shape, axes = [], [], []
            cursor = 2
            for size, axis, selector in zip(shape_of(base.type), axes_of(base.type), selectors):
                if selector == "all":
                    shape.append(size)
                    axes.append(axis)
                    continue
                index = self.expr(call.args[cursor])
                cursor += 1
                if selector in ("domain", "group_index"):
                    if not isinstance(index.type, DomainPointType) or index.type.domain.axis_id != axis:
                        self.error("View domain selector must match the explicit logical axis")
                    if selector == "domain":
                        shape.append(self.domain_partitions[index.type.domain.domain_id])
                        axes.append(axis)
                else:
                    index = self.cast(index, dt.index)
                values.append(index)
            return self.op(
                "slice", (base, *values), SliceType(base.type.encoding, tuple(shape), tuple(axes)), {"selectors": json.dumps(selectors)}
            )
        if name == "tl.weft.load_record":
            view = self.expr(call.args[0])
            return self.op("admit", (view,), LocalValueType(view.type.encoding, shape_of(view.type), axes_of(view.type)))
        if name == "tl.weft.stage":
            value = self.expr(call.args[0])
            result = self.op("materialize", (value,), value.type)
            if self.birth_phase == "staged":
                self.birth_outputs.append(("value", None, result))
            return result
        return None

    def model_tile(self, call):
        name = call.op.name
        if name == "tl.weft.combine_yield":
            if getattr(self, "combiner", None) is None:
                self.error("combine_yield requires a closed reduction region")
            self.emit("weft_kernel." + self.combiner[1] + "_yield", tuple(self.expr(arg) for arg in call.args))
            return True
        if name == "tl.weft.derive_yield":
            if self.derivation is None:
                self.error("derive_yield requires a layout-builder region")
            self.emit("weft_kernel.derive_yield", (self.expr(call.args[0]),))
            return True
        if name == "tl.weft.helper_return":
            self.source_return(call)
            return True
        if name == "tl.weft.helper_result_tile":
            destination = self.region(call.args[0])[0]
            identity, index = self.static(call.args[1]), self.static(call.args[2])
            value = self.helper_results[identity][index]
            self.locals[destination], self.local_axes[destination] = value, axes_of(value.type)
            return True
        if name == "tl.weft.load_value":
            if len(call.args) == 3:
                table = self.expr(call.args[1])
                indices = self.region(call.args[2])
                index = self.read_region(indices, self.local_axes[indices[0]])
                bounds = string(call.annotations["bounds"])
                if bounds != "in_bounds" or not isinstance(table.type, (ViewType, SliceType)) or table.type.encoding.kind != "dense":
                    self.error("indexed View loads require a numeric table and the in_bounds contract")
                scalar = ScalarType(getattr(dt, table.type.encoding.family))
                numeric = value_type(scalar, shape_of(table.type), axes_of(table.type))
                table = self.op("admit", (table,), numeric)
                value = self.op(
                    "lookup", (table, index), value_type(scalar, shape_of(index.type), axes_of(index.type)), {"bounds": json.dumps(bounds)}
                )
                destination = self.region(call.args[0])
                self.write_region(destination, value, axes_of(value.type))
                return True
            if len(call.args) == 4:
                source = self.region(call.args[1])[0]
                value = self.locals[source]
                point = self.expr(call.args[2])
                axis = self.static(call.args[3])
                if not isinstance(point.type, DomainPointType):
                    self.error("local Value projection requires an explicit Level point")
                shape, axes = list(shape_of(value.type)), list(axes_of(value.type))
                identity = point.type.domain.axis_id
                if axes[axis] != identity:
                    self.error(
                        "Level projection requires the Value's existing logical axis; use a positional tile slice for an anonymous local tile"
                    )
                extent = self.domain_partitions[point.type.domain.domain_id]
                record_extent = getattr(self, "field_record_extents", {}).get(source)
                grouped = record_extent is not None and shape[axis] < record_extent
                if grouped:
                    if extent * shape[axis] != record_extent:
                        self.error("field projection must select complete declared field groups")
                    del shape[axis]
                    del axes[axis]
                else:
                    shape[axis] = extent
                selector = "group_index" if grouped else "domain"
                selectors = [selector if i == axis else "all" for i in range(len(shape_of(value.type)))]
                result = self.op(
                    "extract",
                    (value, point),
                    value_type(element_type(value.type), tuple(shape), tuple(axes)),
                    {"selectors": json.dumps(selectors)},
                )
                destination = self.region(call.args[0])[0]
                self.locals[destination], self.local_axes[destination] = result, tuple(axes)
                return True
            view = self.expr(call.args[1])
            from .importer import dtype

            typ = value_type(ScalarType(dtype(view.type.encoding.family)), shape_of(view.type), axes_of(view.type))
            value = self.op("admit", (view,), typ)
            destination = self.region(call.args[0])[0]
            self.locals[destination] = value
            self.local_axes[destination] = axes_of(value.type)
            return True
        if name == "tl.weft.field":
            owner = self.expr(call.args[1])
            field_name = string(call.args[2])
            schema = self.encodings[element_type(owner.type).family]
            field = next(field for field in schema.fields if field.name == field_name)
            shape = shape_of(owner.type)[:-1] + field.shape
            axes = (
                axes_of(owner.type)[:-1] + tuple(self.axis() for _ in field.shape[:-1]) + (axes_of(owner.type)[-1:] if field.shape else ())
            )
            value = self.op(
                "field", (owner,), value_type(ScalarType(getattr(dt, field.dtype)), shape, axes), {"name": json.dumps(field_name)}
            )
            destination = self.region(call.args[0])[0]
            self.locals[destination] = value
            self.local_axes[destination] = axes
            if not hasattr(self, "field_record_extents"):
                self.field_record_extents = {}
            self.field_record_extents[destination] = schema.elements
            return True
        if name == "tl.weft.state":
            from .importer import dtype

            destination, _, shape = self.region(call.args[0])
            initial = self.cast(self.expr(call.args[1]), dtype(destination.dtype))
            axes = self.local_axes[destination]
            if (len(shape) == 1 and self.active_domain.type.domain_id
                    and shape[0] == self.domain_partitions[self.active_domain.type.domain_id]):
                axes = (self.active_domain.type.axis_id,)
            ty = value_type(ScalarType(dtype(destination.dtype)), shape, axes)
            result = self.op("new", (initial,), ty, {"initialized": "true"})
            self.locals[destination], self.local_axes[destination] = result, axes
            if self.birth_phase == "state":
                self.birth_outputs.append(("buffer", destination, result))
            return True
        if name == "tl.weft.stage":
            destination = self.region(call.args[0])[0]
            source = self.region(call.args[1])
            value = self.read_region(source, self.local_axes_for(source[0]))
            result = self.op("materialize", (value,), value.type)
            self.locals[destination] = result
            self.local_axes[destination] = axes_of(result.type)
            self.readonly.add(destination)
            if self.birth_phase == "staged":
                self.birth_outputs.append(("staged_buffer", destination, result))
            return True
        if name == "tl.weft.mac_groups":
            from .importer import dtype

            left_region, right_region = (self.region(arg) for arg in call.args[:2])
            left = self.read_region(left_region, self.local_axes_for(left_region[0]))
            right = self.read_region(right_region, self.local_axes_for(right_region[0]))
            output = self.region(call.args[2])
            group = self.static(call.args[3])
            into = dtype(string(call.args[4]))
            axis, left, right = self.bind_group_axis(left, right, string(call.args[5]))
            left, right = self.align_contraction_batches(left, right, (axis,), output[2][:-1])
            shape, axes = self.domain(left, right)
            position = axes.index(axis)
            if shape[position] <= 0 or shape[position] % group:
                self.error("mac_groups requires a complete group domain")
            # As in Triton's mac_groups, independent free axes precede the
            # grouped axis. A pointwise union would place the right free axis
            # after the left operand's group and change the partial layout.
            shape = shape[:position] + shape[position + 1 :] + (shape[position] // group,)
            axes = axes[:position] + axes[position + 1 :] + (axis,)
            result = self.op(
                "mac_groups",
                (left, right),
                value_type(ScalarType(into), shape, axes),
                {"group": str(group), "overflow": json.dumps("wrap")},
            )
            if shape != output[2]:
                self.error("mac_groups destination shape must match its explicit partial groups")
            self.write_region(output, result, axes)
            return True
        if name == "tl.weft.reduce_dot":
            from .importer import dtype, integers
            from weft_core.types import emit_type

            left_region, right_region, output = (self.region(arg) for arg in call.args)
            left = self.read_region(left_region, self.local_axes_for(left_region[0]))
            right = self.read_region(right_region, self.local_axes_for(right_region[0]))
            names = tuple(string(axis) for axis in call.annotations["axes"])
            if len(names) == 1:
                axis, left, right = self.bind_group_axis(left, right, names[0])
                reduction = (axis,)
                left, right = self.align_contraction_batches(left, right, reduction, output[2])
            else:
                reduction = tuple(self.named_axis(name) for name in names)
            if any(axis not in axes_of(value.type) for value in (left, right) for axis in reduction):
                self.error("reduce_dot operands must retain every declared named reduction axis")
            shape, axes = self.domain(left, right)
            free_shape = tuple(n for n, axis in zip(shape, axes) if axis not in reduction)
            free_axes = tuple(axis for axis in axes if axis not in reduction)
            accumulator = ScalarType(dtype(string(call.annotations["accumulator"])))
            result = self.op(
                "contract",
                (left, right),
                value_type(accumulator, free_shape, free_axes),
                {"over": integers(reduction), "acc_type": emit_type(accumulator), "engine": json.dumps(string(call.annotations["engine"]))},
            )
            output_axes = free_axes
            if output[2] != free_shape:
                if free_shape or output[2] != (1,):
                    self.error("reduce_dot destination must match its free logical axes")
                output_axes = self.local_axes[output[0]]
                result = self.broadcast(result, output[2], output_axes)
            self.write_region(output, result, output_axes)
            self.local_axes[output[0]] = output_axes
            return True
        if name == "tl.weft.store_value":
            source = self.region(call.args[0])
            destination = self.expr(call.args[1])
            from .importer import dtype

            value = self.read_region(source, self.local_axes_for(source[0]))
            if destination.type.encoding.kind == "dense":
                value = self.cast(value, dtype(destination.type.encoding.family))
            value = self.rename(value, shape_of(destination.type), axes_of(destination.type))
            self.op("commit", (value, destination))
            return True
        return False

    def layout_builder(self, block):
        from .importer import integers

        declaration = json.loads(str(block.annotations["weft.derive"]))["derived"]
        if declaration["identity"] in self.derived_layouts:
            return
        schema = Encoding.from_dict(declaration["source"])
        self.declare_encoding(schema)
        shape = tuple(self.static(n) for n in block.annotations["weft.derive.shape"])
        axes = tuple(self.named_axis(str(name)) for name in block.annotations["weft.derive.axes"])
        base = ViewType(EncodingType(schema.name, "base", schema.name), shape, axes)
        result = ViewType(
            EncodingType(declaration["family"], "derived_instance", declaration["identity"], tuple(declaration["parameters"])), shape, axes
        )
        region = self.builder.region((base,))
        environment, previous = self.env, self.derivation
        self.env = {}
        self.derivation = {"argument": region.arguments[0], "result": result}
        try:
            with self.insertion(region):
                self.stmt(block.body)
        finally:
            self.env, self.derivation = environment, previous
        self.declarations.append(
            self.builder.operation(
                "weft_kernel.derive",
                self.location(),
                attributes={
                    "sym_name": json.dumps(declaration["family"] + "$" + "$".join(map(str, declaration["parameters"]))),
                    "source_family": json.dumps(schema.name),
                    "result_family": json.dumps(declaration["family"]),
                    "layout_identity": json.dumps(declaration["identity"]),
                    "parameter_names": json.dumps([str(name) for name in block.annotations["weft.derive.parameters"]]),
                    "parameter_values": integers(declaration["parameters"]),
                },
                regions=(region,),
            )
        )
        self.encodings[declaration["family"]] = schema
        self.derived_layouts.add(declaration["identity"])

    def semantic_level(self, node):
        relation = str(node.annotations["weft.level.relation"])
        name = str(node.annotations["weft.level.axis"])
        partition = self.static(node.annotations["weft.level.partition"])
        extent_expr = node.annotations["weft.level.extent"]
        extent = self.cast(self.expr(extent_expr), dt.index)
        part = self.constant(partition, dt.index)
        multiplicity = self.emit("arith.ceildivui", (extent, part), (ScalarType(dt.index),))[0]
        serial = self.next_domain
        self.next_domain += 1
        static = self.analyzer.simplify(extent_expr)
        tail = "exact" if isinstance(static, tirx.IntImm) and int(static.value) % partition == 0 else "tail"
        domain_type = DomainType(f"{name}.{serial}", serial, self.active_domain.type.domain_id, self.named_axis(name), relation, tail)
        domain = self.op("domain", (self.active_domain, extent, part, multiplicity), domain_type)
        self.domain_partitions[serial] = partition
        point_type = DomainPointType(domain_type)
        blocks = {}

        def collect(statement):
            if isinstance(statement, tirx.SeqStmt):
                for item in statement.seq:
                    collect(item)
            elif isinstance(statement, tirx.SBlockRealize) and "weft.level.region" in statement.block.annotations:
                blocks[str(statement.block.annotations["weft.level.region"])] = statement
            else:
                self.error("a semantic Level must retain its three explicit declaration/body regions", statement)

        collect(node.body)
        if set(blocks) != {"state", "staged", "body"}:
            self.error("Level requires distinct state, staged and body regions")
        incoming, environment = dict(self.locals), dict(self.env)
        incoming_axes, incoming_readonly = dict(self.local_axes), set(self.readonly)
        outer_domain, outer_phase, outer_outputs = self.active_domain, self.birth_phase, self.birth_outputs
        self.active_domain = domain
        regions, births = [], []
        for phase in ("state", "staged"):
            region = self.builder.region((point_type,))
            self.locals, self.env = dict(incoming), dict(environment)
            self.local_axes, self.readonly = dict(incoming_axes), set(incoming_readonly)
            self.env[node.loop_var] = region.arguments[0]
            self.birth_phase, self.birth_outputs = phase, []
            with self.insertion(region):
                self.stmt(blocks[phase])
                outputs = []
                for kind, key, value in self.birth_outputs:
                    if kind == "value":
                        matches = [var for var, candidate in self.env.items() if candidate is value]
                        if len(matches) != 1:
                            self.error("a staged record must bind one SSA value")
                        key = matches[0]
                    outputs.append((kind, key, value))
                self.emit("weft_kernel.births_yield", tuple(value for _, _, value in outputs))
            regions.append(region)
            births.extend(outputs)
        keys = tuple(incoming)
        carried = tuple(incoming[k] for k in keys)
        body = self.builder.region((point_type, *(v.type for v in carried), *(v.type for _, _, v in births)))
        with self.insertion(body):
            self.locals, self.env = dict(zip(keys, body.arguments[1 : 1 + len(keys)])), dict(environment)
            self.local_axes, self.readonly = dict(incoming_axes), set(incoming_readonly)
            self.env[node.loop_var] = body.arguments[0]
            for (kind, key, _), value in zip(births, body.arguments[1 + len(keys) :]):
                if kind in ("buffer", "staged_buffer"):
                    self.locals[key] = value
                    self.local_axes[key] = axes_of(value.type)
                    if kind == "staged_buffer":
                        self.readonly.add(key)
                else:
                    self.env[key] = value
            self.birth_phase, self.birth_outputs = "body", []
            self.stmt(blocks["body"])
            final = tuple(self.rename(self.locals[k], shape_of(incoming[k].type), axes_of(incoming[k].type)) for k in keys)
            self.emit("weft_kernel.handoff", final)
        changed = tuple(i for i, value in enumerate(final) if value is not body.arguments[i + 1] and value is not carried[i])
        for i, value in enumerate(carried):
            if i not in changed:
                body.arguments[i + 1].replace_all_uses_with(value)
        body.operations[-1].operands = tuple(final[i] for i in changed)
        body.arguments = (body.arguments[0], *(body.arguments[i + 1] for i in changed), *body.arguments[1 + len(keys) :])
        results = self.emit(
            "weft_kernel.level", (domain, *(carried[i] for i in changed)), tuple(carried[i].type for i in changed), regions=(*regions, body)
        )
        self.locals, self.env = dict(incoming), environment
        self.locals.update(zip((keys[i] for i in changed), results))
        self.local_axes, self.readonly = incoming_axes, incoming_readonly
        self.active_domain, self.birth_phase, self.birth_outputs = outer_domain, outer_phase, outer_outputs
