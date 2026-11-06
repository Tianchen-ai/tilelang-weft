"""Tile copies, shared-storage reshaping, reduction and contraction on a CPU."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def matmul_factory(M=16, N=16, K=32):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"), C: T.Tensor((M, N), "float32")):
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
            flat = T.alloc_local((M * N,), "float32")
            x = T.reshape(flat, (M, N))
            y = T.alloc_local((M,), "float32")
            T.fill(flat, 0)
            for row in T.serial(M):
                for column in T.Parallel(N):
                    x[row, column] = X[row, column]
            T.reduce_sum(x, y, dim=1)
            T.copy(y, Y)

    return main


if __name__ == "__main__":
    import torch

    torch.manual_seed(2)
    a, b, c = torch.randn(16, 32), torch.randn(32, 16), torch.empty(16, 16)
    matmul_factory()(a, b, c)
    torch.testing.assert_close(c, a @ b)
    rows, sums = torch.randn(4, 32), torch.empty(4)
    sum_factory()(rows, sums)
    torch.testing.assert_close(sums, rows.sum(1))
    print("tile GEMM and reduction: correct")
