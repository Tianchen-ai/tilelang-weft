"""Resolve TileLang CPU requests through the shared SDK target discovery."""

from dataclasses import asdict
import json
import os

from tilelang.backend.target import register_target_normalizer
from tvm.target import Target


def target(profile="native", *, toolchain=None, backend_options=None):
    from weft_core.compiler import identity
    from weft_core.cpu import target as cpu_target
    from weft_core.toolchain import Toolchain

    descriptor = cpu_target(profile)
    configuration = Toolchain.resolve(toolchain or {}, descriptor).to_dict()
    values = asdict(descriptor)
    values.pop("backend")
    return Target(
        {
            "kind": "weft",
            **values,
            "toolchain": json.dumps(configuration, sort_keys=True),
            "options": json.dumps(backend_options or {}, sort_keys=True),
            "core": json.dumps(identity(), sort_keys=True),
        }
    )


def descriptor(target):
    from weft_core.target import CPUTarget

    return CPUTarget(
        "weft",
        str(target.attrs["arch"]),
        str(target.attrs["march"]),
        str(target.attrs["abi"]),
        int(target.attrs["vlen_bits"]),
        str(target.attrs["matrix_extension"]),
    )


def normalize(value):
    if isinstance(value, str) and value.strip() == "weft":
        from weft_core.cpu import _execution_target
        from weft_core.compiler import native_target

        facts = _execution_target(native_target(), os.environ.get("WEFT_CPU_PROFILE"))
        return target_from_descriptor(facts)
    if isinstance(value, dict) and value.get("kind") == "weft" and "arch" not in value:
        return target(value.get("profile", "native"), toolchain=value.get("toolchain"), backend_options=value.get("backend_options"))
    return None


def target_from_descriptor(value, *, toolchain=None, backend_options=None):
    from weft_core.compiler import identity
    from weft_core.toolchain import Toolchain

    values = asdict(value)
    values.pop("backend")
    return Target(
        {
            "kind": "weft",
            **values,
            "toolchain": json.dumps(Toolchain.resolve(toolchain or {}, value).to_dict(), sort_keys=True),
            "options": json.dumps(backend_options or {}, sort_keys=True),
            "core": json.dumps(identity(), sort_keys=True),
        }
    )


register_target_normalizer("weft", normalize)
