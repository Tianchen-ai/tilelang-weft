"""One row snapshot shared by its reductions and affine epilogue."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def rms_norm(M, N, cpu_num_threads=0, block=128):
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Weight: T.Tensor((N,), "float32"), Y: T.Tensor((M, N), "float32"), epsilon: T.float32):
        rows = T.view(X, shape=(M, N), axes=("m", "k"), abi=True)
        weights = T.view(Weight, shape=(N,), axes=("k",), abi=True)
        output = T.view(Y, shape=(M, N), axes=("m", "k"), abi=True)
        with T.Kernel(M, cpu_num_threads=cpu_num_threads) as row:
            total = T.alloc_local((1,), "float32")
            T.fill(total, 0)
            with T.level.blocks(N, extent=block) as part:
                saved = T.stage(T.load(rows[row, part]))
                square = T.alloc_local((block,), "float32")
                subtotal = T.alloc_local((1,), "float32")
                for k in T.Parallel(block):
                    square[k] = saved[k] * saved[k]
                T.reduce_sum(square, subtotal, dim=0)
                total[0] += subtotal[0]
            with T.level.blocks(N, extent=block) as part:
                saved = T.stage(T.load(rows[row, part]))
                weight = T.load(weights[part])
                result = T.alloc_local((block,), "float32")
                for k in T.Parallel(block):
                    result[k] = saved[k] * T.rsqrt(total[0] / N + epsilon) * weight[k]
                T.copy(result, output[row, part])

    return main


@tilelang.jit(target="weft")
def layer_norm(M, N, cpu_num_threads=0, block=128):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Weight: T.Tensor((N,), "float32"),
        Bias: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
        epsilon: T.float32,
    ):
        rows = T.view(X, shape=(M, N), axes=("m", "k"), abi=True)
        weights = T.view(Weight, shape=(N,), axes=("k",), abi=True)
        biases = T.view(Bias, shape=(N,), axes=("k",), abi=True)
        output = T.view(Y, shape=(M, N), axes=("m", "k"), abi=True)
        with T.Kernel(M, cpu_num_threads=cpu_num_threads) as row:
            mean = T.alloc_local((1,), "float32")
            variance = T.alloc_local((1,), "float32")
            T.fill(mean, 0)
            T.fill(variance, 0)
            with T.level.blocks(N, extent=block) as part:
                saved = T.stage(T.load(rows[row, part]))
                subtotal = T.alloc_local((1,), "float32")
                T.reduce_sum(saved, subtotal, dim=0)
                mean[0] += subtotal[0]
            with T.level.blocks(N, extent=block) as part:
                saved = T.stage(T.load(rows[row, part]))
                square = T.alloc_local((block,), "float32")
                subtotal = T.alloc_local((1,), "float32")
                for k in T.Parallel(block):
                    centered = saved[k] - mean[0] / N
                    square[k] = centered * centered
                T.reduce_sum(square, subtotal, dim=0)
                variance[0] += subtotal[0]
            with T.level.blocks(N, extent=block) as part:
                saved = T.stage(T.load(rows[row, part]))
                weight, bias = T.load(weights[part]), T.load(biases[part])
                result = T.alloc_local((block,), "float32")
                for k in T.Parallel(block):
                    result[k] = (saved[k] - mean[0] / N) * T.rsqrt(variance[0] / N + epsilon) * weight[k] + bias[k]
                T.copy(result, output[row, part])

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
