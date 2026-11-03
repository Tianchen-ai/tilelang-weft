"""Native TIRx to shared Canonical compilation manifest."""

from tilelang.backend.module import BackendModule, register_backend
from .execution_backend import EXECUTION_BACKENDS
from functools import cache
from hashlib import sha256
from pathlib import Path


def compile_artifact(*args, **kwargs):
    from .pipeline import compile_artifact as compile_impl

    return compile_impl(*args, **kwargs)


@cache
def compiler_identity():
    # Native library stamps do not cover an importer implemented in Python.
    # Include that implementation in the existing frontend cache identity.
    root = Path(__file__).parent
    digest = sha256()
    for path in sorted((*root.glob("*.py"), *root.joinpath("frontend").glob("*.py"))):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


BACKEND = register_backend(
    BackendModule(
        name="weft",
        target_kinds=("weft",),
        pipelines={},
        device_codegens={},
        execution_backends=EXECUTION_BACKENDS,
        artifact_compiler=compile_artifact,
        compiler_identity=compiler_identity,
        parallel_iteration_model="independent_values",
    )
)
