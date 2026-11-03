"""Persist native files through TileLang's existing atomic cache lifecycle."""

from weft_core.artifact import NativeArtifact

from tilelang.cache.kernel_cache import KernelCache


class WeftNativeKernelCache(KernelCache):
    execution_backend = "weft_native"
    device_kernel_path = "kernel.source"
    host_kernel_path = "kernel.launch"

    def _save_so_cubin_to_disk(self, kernel, cache_path, verbose=False):
        super()._save_so_cubin_to_disk(kernel, cache_path, verbose)
        kernel.adapter.native.write_metadata(cache_path)

    def _get_required_files(self, cache_path):
        return super()._get_required_files(cache_path) + [str(path) for path in NativeArtifact.required_files(cache_path)]
