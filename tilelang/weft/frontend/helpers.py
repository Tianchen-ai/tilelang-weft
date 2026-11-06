"""Typed native helper regions, including exits through ordinary control/Level."""

import json

from weft_core.types import shape_of, axes_of, element_type


class HelperOperations:
    def source_helper(self, block):
        incoming, environment = dict(self.locals), dict(self.env)
        incoming_axes, readonly = dict(self.local_axes), set(self.readonly)
        keys = tuple(incoming)
        region = self.builder.region(())
        context = {"keys": keys, "carried_types": tuple(incoming[key].type for key in keys), "types": None, "returns": 0}
        self.helper_stack.append(context)
        with self.insertion(region):
            try:
                for buffer in block.alloc_buffers:
                    self.allocate(buffer)
                self.stmt(block.body)
            finally:
                self.helper_stack.pop()
        if context["types"] is None:
            self.error("helper has no typed return path", block)
        count = len(context["types"])

        def returns(body):
            for operation in body.operations:
                if operation.name == "weft_kernel.helper_return":
                    yield operation
                elif operation.name != "weft_kernel.helper":
                    for nested in operation.regions:
                        yield from returns(nested)

        exits = tuple(returns(region))
        changed = tuple(i for i, key in enumerate(keys) if any(exit.operands[count + i] is not incoming[key] for exit in exits))
        for exit in exits:
            exit.operands = (*exit.operands[:count], *(exit.operands[count + i] for i in changed))
        result_types = (*context["types"], *(incoming[keys[i]].type for i in changed))
        results = self.emit(
            "weft_kernel.helper",
            types=result_types,
            attrs={
                "name": json.dumps(str(block.annotations["weft.helper"])),
                "noinline": "true" if bool(block.annotations["weft.helper.noinline"]) else "false",
            },
            regions=(region,),
        )
        self.helper_results[int(block.annotations["weft.helper.id"])] = results[:count]
        self.locals, self.env = incoming, environment
        self.locals.update(zip((keys[i] for i in changed), results[count:]))
        self.local_axes, self.readonly = incoming_axes, readonly

    def source_return(self, call):
        if not self.helper_stack:
            self.error("helper_return has no enclosing helper region")
        context = self.helper_stack[-1]
        values = []
        kinds = tuple(str(kind) for kind in call.annotations["kinds"])
        for argument, kind in zip(call.args, kinds):
            if str(kind) == "tile":
                region = self.region(argument)
                axes = self.local_axes.get(region[0]) or tuple(self.axis() for _ in region[2])
                value = self.read_region(region, axes)
            else:
                value = self.expr(argument)
            values.append(value)
        if context["types"] is None:
            context["types"] = tuple(v.type for v in values)
        if len(values) != len(context["types"]):
            self.error("helper return paths have different result counts")
        normalized = []
        for value, expected, kind in zip(values, context["types"], kinds):
            if value.type != expected:
                if kind == "logical":
                    self.error("helper return paths must preserve Encoding and logical axis identity")
                if element_type(value.type) != element_type(expected) or shape_of(value.type) != shape_of(expected):
                    self.error("helper return paths must preserve dtype and logical shape")
                value = self.rename(value, shape_of(expected), axes_of(expected))
            normalized.append(value)
        context["returns"] += 1
        carried = []
        for key, expected in zip(context["keys"], context["carried_types"]):
            value = self.locals[key]
            if value.type != expected:
                if element_type(value.type) != element_type(expected) or shape_of(value.type) != shape_of(expected):
                    self.error("helper mutation must preserve its caller buffer dtype and shape")
                value = self.rename(value, shape_of(expected), axes_of(expected))
            carried.append(value)
        self.emit("weft_kernel.helper_return", (*normalized, *carried))

    def helper_result(self, call):
        identity, index = self.static(call.args[0]), self.static(call.args[1])
        if identity not in self.helper_results or index >= len(self.helper_results[identity]):
            self.error("helper result has no dominating typed function region")
        return self.helper_results[identity][index]
