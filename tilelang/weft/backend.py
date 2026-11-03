"""Native TIRx to shared Canonical compilation manifest."""
from tilelang.backend.module import BackendModule, register_backend
from .execution_backend import EXECUTION_BACKENDS


def compile_artifact(*args, **kwargs):
    from .pipeline import compile_artifact as compile_impl
    return compile_impl(*args, **kwargs)


BACKEND = register_backend(BackendModule(
    name="weft", target_kinds=("weft",), pipelines={}, device_codegens={},
    execution_backends=EXECUTION_BACKENDS, artifact_compiler=compile_artifact,
    parallel_iteration_model="independent_values",
))
