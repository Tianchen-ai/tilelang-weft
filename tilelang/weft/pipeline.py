"""Early semantic import and the single shared physical compilation route."""

from dataclasses import asdict, fields
import json
import tempfile
from types import SimpleNamespace
from tvm_ffi.container import Map, Array

from tilelang import tvm
from tvm import tirx
from tilelang.transform import get_pass_context, PassConfigKey
from tilelang.engine.param import CompiledArtifact
from weft_core import compiler
from weft_core.artifact import NativeArtifact
from weft_core.launcher import single_instance_abi
from weft_core.options import WeftOptions, X86Options
from weft_core.toolchain import Toolchain

from .frontend import import_module
from .target import compilation_identity, descriptor


def compile_artifact(mod, context, *, params, runtime_only, enable_device_compile):
    if str(context.target.attrs["core"]) != compilation_identity():
        raise RuntimeError("the resolved TileLang target requires a different frontend or weft-core compiler build")
    canonical, imported = import_module(mod)
    target = descriptor(context.target)
    option_type = X86Options if target.arch == "x86_64" else WeftOptions
    physical = json.loads(str(context.target.attrs["options"]))
    pass_config = get_pass_context().config

    def static_value(value):
        if isinstance(value, (Map, dict, tvm.ir.DictAttrs)):
            return {str(k): static_value(v) for k, v in value.items()}
        if isinstance(value, (Array, list, tuple)):
            return [static_value(v) for v in value]
        return value.value if isinstance(value, (tvm.tirx.IntImm, tvm.tirx.FloatImm)) else value

    backend_options = static_value(pass_config.get("tl.weft.backend_options", {}))
    if not isinstance(backend_options, dict):
        raise TypeError("tl.weft.backend_options must be a mapping of finite compiler parameters")
    physical.update(backend_options)
    unknown = set(physical) - {field.name for field in fields(option_type)}
    if unknown:
        raise ValueError(f"unknown Weft physical options: {sorted(unknown)}")
    if target.arch == "riscv64" and physical.get("matrix_extension", "auto") == "auto":
        physical["matrix_extension"] = target.matrix_extension
    options = option_type(**physical)
    if options.pipeline_depth not in (1, 2, 3, 4):
        raise ValueError("pipeline_depth must be one of 1, 2, 3, 4")
    bindings = {**asdict(options), **asdict(target)}
    lowered, report = compiler.lower(canonical, bindings)
    emitted = compiler.emit(lowered)
    report["pipeline"].extend(emitted["compilation"]["pipeline"])
    abi = emitted["kernel"] if target.arch == "x86_64" else emitted["kernels"][0]
    source = emitted["llvm_ir"] if target.arch == "x86_64" else emitted["intrinsic_c"]
    metadata = {
        **asdict(options),
        "target": target,
        "producer": "tilelang",
        "kernel_abi": abi,
        "single_instance_abi": single_instance_abi(abi),
        "compilation": report,
        "source_bindings": options.meta,
        "source_storage": imported.storage_contracts,
        "source_tensors": {
            buffer.name: {
                "shape": [int(n.value) if isinstance(n, tirx.IntImm) else str(n) for n in buffer.shape],
                "dtype": str(buffer.dtype),
            }
            for buffer in imported.function.buffer_map.values()
        },
        "launch_args": [a["name"] for a in abi["arguments"] if not a["name"].startswith("__triton_")],
        "num_cpu_threads": imported.threads,
        "tilelang": {
            "grid": tvm.ir.save_json(tvm.runtime.convert(imported.grid)),
            "parameter_names": [
                imported.function.buffer_map[p].name if p in imported.function.buffer_map else p.name for p in imported.function.params
            ],
            "implicit_names": [p.name for p in imported.implicit],
            "function": tvm.ir.save_json(imported.function),
        },
    }
    native = None
    if enable_device_compile:
        toolchain_config = json.loads(str(context.target.attrs["toolchain"]))
        flags = pass_config.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS.value, ())
        if flags:
            toolchain_config["cflags"] = [*toolchain_config["cflags"], *(str(flag) for flag in flags)]
        toolchain = Toolchain.resolve(toolchain_config, target)
        with tempfile.TemporaryDirectory(prefix="tilelang-weft-compile-") as directory:
            binary = toolchain.compile(source, metadata, target, directory)
            native = NativeArtifact.create(
                binary,
                SimpleNamespace(**metadata),
                {
                    "weftir": str(canonical),
                    "x86ir" if target.arch == "x86_64" else "riscvir": str(lowered),
                    "llir" if target.arch == "x86_64" else "c": source,
                },
            )
    return CompiledArtifact(
        None,
        None,
        params,
        source,
        target=context.target,
        target_host=context.target_host,
        native_artifact=native,
        semantic_ir=str(canonical),
        physical_ir=str(lowered),
    )
