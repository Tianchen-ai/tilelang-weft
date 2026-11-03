"""GGML role validation and packaging without a Triton dependency."""

from weft_core.integrations.llama_cpp import install_source as install_source, validate_package as validate_package
from weft_core.integrations.llama_cpp import create_package as _create_package


def create_package(directory, kernels, *, bindings):
    """Bind selected TileLang kernels to the shared, explicit GGML contracts."""
    from tilelang.jit.kernel import JITKernel
    from weft_core.artifact import NativeArtifact

    native = {}
    for name, kernel in kernels.items():
        if isinstance(kernel, JITKernel) and kernel.execution_backend == "weft_native":
            native[name] = kernel.adapter.native
        elif isinstance(kernel, NativeArtifact):
            native[name] = kernel
        else:
            raise TypeError("GGML package entries must be compiled TileLang Weft kernels or native artifacts")
    return _create_package(directory, native, bindings=bindings)
