"""One row snapshot shared by its reductions and affine epilogue."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def rms_norm(M, N, cpu_num_threads=0):
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Weight: T.Tensor((N,), "float32"), Y: T.Tensor((M, N), "float32"), epsilon: T.float32):
        rows = T.view(X, shape=(M, N), axes=("m", "k"))
        with T.Kernel(M, cpu_num_threads=cpu_num_threads) as row:
            value = T.stage(T.load(rows[row, :]))
            square = T.alloc_local((N,), "float32")
            total = T.alloc_local((1,), "float32")
            for k in T.Parallel(N):
                square[k] = value[k] * value[k]
            T.reduce_sum(square, total, dim=0)
            for k in T.Parallel(N):
                Y[row, k] = value[k] * T.rsqrt(total[0] / N + epsilon) * Weight[k]

    return main


@tilelang.jit(target="weft")
def layer_norm(M, N, cpu_num_threads=0):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Weight: T.Tensor((N,), "float32"),
        Bias: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
        epsilon: T.float32,
    ):
        rows = T.view(X, shape=(M, N), axes=("m", "k"))
        with T.Kernel(M, cpu_num_threads=cpu_num_threads) as row:
            value = T.stage(T.load(rows[row, :]))
            mean = T.alloc_local((1,), "float32")
            centered = T.alloc_local((N,), "float32")
            square = T.alloc_local((N,), "float32")
            variance = T.alloc_local((1,), "float32")
            T.reduce_sum(value, mean, dim=0)
            for k in T.Parallel(N):
                centered[k] = value[k] - mean[0] / N
                square[k] = centered[k] * centered[k]
            T.reduce_sum(square, variance, dim=0)
            for k in T.Parallel(N):
                Y[row, k] = centered[k] * T.rsqrt(variance[0] / N + epsilon) * Weight[k] + Bias[k]

    return main


@tilelang.jit(target="weft")
def residual_rms_norm(M, N, cpu_num_threads=0):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Residual: T.Tensor((M, N), "float32"),
        Weight: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
        epsilon: T.float32,
    ):
        with T.Kernel(M, cpu_num_threads=cpu_num_threads) as row:
            value = T.alloc_local((N,), "float32")
            square = T.alloc_local((N,), "float32")
            total = T.alloc_local((1,), "float32")
            for k in T.Parallel(N):
                value[k] = X[row, k] + Residual[row, k]
                square[k] = value[k] * value[k]
            T.reduce_sum(square, total, dim=0)
            for k in T.Parallel(N):
                Y[row, k] = value[k] * T.rsqrt(total[0] / N + epsilon) * Weight[k]

    return main
