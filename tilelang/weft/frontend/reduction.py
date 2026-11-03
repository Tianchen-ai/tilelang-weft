"""Import the complete scalar combiner without routing through a JIT helper."""

from weft_core.types import element_type, value_type


class CombinedReductions:
    def combined_reduction(self, block):
        kind = str(block.annotations["weft.combine"])
        sources = [self.region(v) for v in block.annotations["weft.combine.inputs"]]
        outputs = [self.region(v) for v in block.annotations["weft.combine.outputs"]]
        axis = int(block.annotations["weft.combine.axis"])
        shape = sources[0][2]
        if any(source[2] != shape for source in sources):
            self.error("tuple reduction inputs must share one logical shape", block)
        axes = tuple(self.axis() for _ in shape)
        inputs = tuple(self.read_region(source, axes) for source in sources)
        result_shape = shape if kind == "scan" else shape[:axis] + shape[axis + 1 :]
        result_axes = axes if kind == "scan" else axes[:axis] + axes[axis + 1 :]
        types = tuple(element_type(value.type) for value in inputs)
        region = self.builder.region((*types, *types))
        environment, previous = self.env, getattr(self, "combiner", None)
        self.env = {}
        self.combiner = (region.arguments, kind)
        try:
            with self.insertion(region):
                self.stmt(block.body)
        finally:
            self.env, self.combiner = environment, previous
        attributes = {"axis": str(axis)}
        if kind == "scan":
            attributes["reverse"] = "true" if bool(block.annotations["weft.combine.reverse"]) else "false"
        results = self.emit(
            "weft_kernel.scan" if kind == "scan" else "weft_kernel.reduce_region",
            inputs,
            tuple(value_type(ty, result_shape, result_axes) for ty in types),
            attributes,
            (region,),
        )
        for output, result in zip(outputs, results):
            if output[2] != result_shape:
                if output[2] == result_shape[:axis] + (1,) + result_shape[axis:]:
                    expanded_axes = result_axes[:axis] + (axes[axis],) + result_axes[axis:]
                    result = self.rename(result, output[2], expanded_axes)
                    self.write_region(output, result, expanded_axes)
                    continue
                self.error("tuple reduction output has the wrong logical shape", block)
            self.write_region(output, result, result_axes)
