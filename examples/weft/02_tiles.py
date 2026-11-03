"""Tile copies, local values, reduction and contraction on a CPU."""
import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def matmul_factory(M=16, N=16, K=32):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(1):
            a = T.alloc_local((M, K), "float32")
            b = T.alloc_local((K, N), "float32")
            c = T.alloc_local((M, N), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True, engine="vector")
            T.copy(c, C)
    return main


@tilelang.jit(target="weft")
def sum_factory(M=4, N=32):
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            x = T.alloc_local((M, N), "float32")
            y = T.alloc_local((M,), "float32")
            T.copy(X, x)
            T.reduce_sum(x, y, dim=1)
            T.copy(y, Y)
    return main
