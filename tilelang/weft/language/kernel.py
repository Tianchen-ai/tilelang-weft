"""CPU grid declaration; programs execute through the shared native launcher."""
from tvm import tirx
from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel


@kernel_launch_factory
def Kernel(*blocks: int | tirx.PrimExpr, cpu_num_threads: int = 0) -> KernelLaunchFrame:
    if type(cpu_num_threads) is not int or cpu_num_threads < 0:
        raise ValueError("cpu_num_threads must be a nonnegative compile-time integer")
    return launch_kernel(blocks, weft_cpu_num_threads=cpu_num_threads)
