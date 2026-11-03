"""The common native artifact/ABI execution path."""

from tilelang.backend.execution_backend import ExecutionBackendSpec

EXECUTION_BACKENDS = (ExecutionBackendSpec("weft_native", enable_device_compile=True),)
