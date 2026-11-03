"""TileLang argument/output conventions over the shared native launcher."""
from __future__ import annotations

import inspect
from pathlib import Path
import operator

import torch
from tvm import tirx
from tilelang import tvm
from weft_core.artifact import NativeArtifact

from .base import BaseKernelAdapter


def expression_functor(expression):
    if isinstance(expression, tirx.IntImm):
        number = int(expression.value)
        return lambda values: number
    if isinstance(expression, tirx.Var):
        name = expression.name
        return lambda values: values[name]
    if isinstance(expression, tirx.Cast):
        return expression_functor(expression.value)
    operators = {"Add": operator.add, "Sub": operator.sub, "Mul": operator.mul,
                 "FloorDiv": operator.floordiv, "FloorMod": operator.mod, "Div": operator.floordiv,
                 "Min": min, "Max": max}
    name = type(expression).__name__
    if name not in operators:
        raise NotImplementedError(f"native launch shape expression {name} is not implemented")
    left, right = expression_functor(expression.a), expression_functor(expression.b)
    operation = operators[name]
    return lambda values: operation(left(values), right(values))


class WeftNativeKernelAdapter(BaseKernelAdapter):
    input_device = "cpu"
    benchmark_backend = "wall"
    def __init__(self, *, native_artifact, params, result_idx, **unused):
        self.native = native_artifact
        self.libpath = str(native_artifact.library)
        description = native_artifact.metadata.tilelang
        self.function = tvm.ir.load_json(description["function"])
        self.names = description["parameter_names"]
        self.implicit_names = description["implicit_names"]
        self.signature = inspect.Signature(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in self.names)
        self.grid = tuple(expression_functor(dim) for dim in tvm.ir.load_json(description["grid"]))
        self.geometry = []
        for position, param in enumerate(self.function.params):
            buffer = self.function.buffer_map.get(param)
            if buffer is None:
                continue
            self.geometry.append((self.names[position], tuple(buffer.shape), tuple(buffer.strides)))
        super().__init__(None, params, result_idx)

    @classmethod
    def from_database(cls, *, kernel_lib_path, params, result_idx, **kwargs):
        path = Path(kernel_lib_path)
        return cls(native_artifact=NativeArtifact.load(path.parent, library=path.name),
                   params=params, result_idx=result_idx)

    def bind_arguments(self, *arguments, **kwargs):
        outputs = set(self.result_idx)
        input_names = [name for index, name in enumerate(self.names) if index not in outputs]
        input_signature = inspect.Signature(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in input_names)

        def bind():
            supplied = dict((input_signature if outputs else self.signature).bind(*arguments, **kwargs).arguments)
            symbols = {}
            for name, value in supplied.items():
                if isinstance(value, int):
                    symbols[name] = value
            for name, shapes, strides in self.geometry:
                if name not in supplied:
                    continue
                value = supplied[name]
                if getattr(value, "device", torch.device("cpu")).type != "cpu":
                    raise TypeError(f"{name} must reside on the CPU selected for this kernel")
                actual_shape = tuple(value.shape)
                actual_stride = tuple(value.stride()) if isinstance(value, torch.Tensor) else tuple(
                    s // value.itemsize for s in value.strides)
                if len(actual_shape) != len(shapes):
                    raise ValueError(f"{name}: expected rank {len(shapes)}, got {len(actual_shape)}")
                actual_geometry = (*actual_shape, *(actual_stride if strides else ()))
                for expected, actual in zip((*shapes, *strides), actual_geometry):
                    if isinstance(expected, tirx.Var):
                        if expected.name in symbols and symbols[expected.name] != actual:
                            raise ValueError(f"inconsistent runtime extent/stride for {expected.name}")
                        symbols[expected.name] = actual
            for position in self.result_idx:
                shape = tuple(expression_functor(dim)(symbols) for dim in self.params[position].shape)
                supplied[self.names[position]] = torch.empty(shape, dtype=self.params[position].torch_dtype(), device="cpu")
            for name in self.implicit_names:
                if name not in symbols:
                    raise ValueError(f"runtime shape/stride {name} could not be inferred from the input tensors")
                supplied[name] = symbols[name]
            for name, shapes, strides in self.geometry:
                value = supplied[name]
                expected_shape = tuple(expression_functor(dim)(symbols) for dim in shapes)
                if tuple(value.shape) != expected_shape:
                    raise ValueError(f"{name}: shape {tuple(value.shape)} does not match {expected_shape}")
                actual_stride = tuple(value.stride()) if isinstance(value, torch.Tensor) else tuple(s // value.itemsize for s in value.strides)
                if strides:
                    expected_stride = tuple(expression_functor(dim)(symbols) for dim in strides)
                else:
                    expected_stride, stride = [], 1
                    for extent in reversed(expected_shape):
                        expected_stride.insert(0, stride)
                        stride *= extent
                    expected_stride = tuple(expected_stride)
                if actual_stride != expected_stride:
                    raise ValueError(f"{name}: stride {actual_stride} does not match {expected_stride}; use T.StridedTensor for runtime strides")
            return supplied, symbols
        return bind()

    def _convert_torch_func(self):
        def invoke(*arguments, **kwargs):
            supplied, symbols = self.bind_arguments(*arguments, **kwargs)
            grid = tuple(int(dim(symbols)) for dim in self.grid) or (1,)
            self.native.launch(grid, tuple(supplied[name] for name in self.native.metadata.launch_args))
            results = [supplied[self.names[index]] for index in self.result_idx]
            return results[0] if len(results) == 1 else tuple(results) if results else None
        return invoke

    def trial_inputs(self, arguments):
        from weft_core.tuning_state import TrialCall, TuningState
        supplied, _ = self.bind_arguments(*arguments)
        state = TuningState(TrialCall(self.native, supplied))
        names = [name for i, name in enumerate(self.names) if i not in self.result_idx]
        state.arguments = [state.values[name] for name in names]
        abi = self.native.metadata.kernel_abi
        writable = {arg["name"] for arg in abi["arguments"] if arg.get("access") in ("write", "readwrite")}
        state.mutations = [state.values[name] for name in names if name in writable]
        return state

    def get_kernel_source(self, kernel_only=True):
        extension = "llir" if self.native.metadata.target.arch == "x86_64" else "c"
        return (self.native.directory / ("kernel." + extension)).read_text()

    def get_host_source(self):
        from weft_core.launcher import launch_source
        return launch_source(self.native.metadata.kernel_abi)

    def export(self, directory):
        return self.native.export(directory)
