"""Explicit row reductions and bounded affine epilogues."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def rms_norm(M, N, cpu_num_threads=0, block=128, rows_per_program=1, reduction_width=None):
    reduction_width = block if reduction_width is None else reduction_width
    assert rows_per_program > 0 and reduction_width > 0

    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Weight: T.Tensor((N,), "float32"), Y: T.Tensor((M, N), "float32"), epsilon: T.float32):
        T.func_attr({"weft.alias_groups": {"X": "activation", "Y": "activation", "Weight": "parameters"}})
        rows = T.view(X, shape=(M, N), axes=("m", "k"), abi=True)
        weights = T.view(Weight, shape=(N,), axes=("k",), abi=True)
        output = T.view(Y, shape=(M, N), axes=("m", "k"), abi=True)
        with T.Kernel(T.ceildiv(M, rows_per_program), cpu_num_threads=cpu_num_threads) as program:
            base = 0 if rows_per_program == M else program * rows_per_program
            count = rows_per_program if M % rows_per_program == 0 else T.min(rows_per_program, M - base)
            for local_row in T.serial(count):
                row = base + local_row
                total = T.alloc_local((1,), "float32")
                if reduction_width == N:
                    full_values = T.load(rows[row, :])
                    square = T.alloc_local((N,), "float32")
                    for k in T.Parallel(N):
                        square[k] = full_values[k] * full_values[k]
                    T.reduce_sum(square, total, dim=0)
                else:
                    partial = T.alloc_local((reduction_width,), "float32")
                    T.fill(partial, 0)
                    with T.level.blocks(N, extent=reduction_width) as part:
                        value = T.load(rows[row, part])
                        active = T.extent(rows[row, part], axis=0)
                        for k in T.Parallel(reduction_width):
                            if k < active:
                                partial[k] += value[k] * value[k]
                    T.reduce_sum(partial, total, dim=0)
                factor = T.alloc_local((1,), "float32")
                factor[0] = T.rsqrt(total[0] / N + epsilon)
                with T.level.blocks(N, extent=block) as part:
                    saved = T.load(rows[row, part])
                    weight = T.load(weights[part])
                    result = T.alloc_local((block,), "float32")
                    for k in T.Parallel(block):
                        result[k] = saved[k] * factor[0] * weight[k]
                    T.copy(result, output[row, part])

    return main


@tilelang.jit(target="weft")
def layer_norm(M, N, cpu_num_threads=0, block=128, rows_per_program=1, reduction_width=None):
    reduction_width = block if reduction_width is None else reduction_width
    assert rows_per_program > 0 and reduction_width > 0

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Weight: T.Tensor((N,), "float32"),
        Bias: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
        epsilon: T.float32,
    ):
        T.func_attr({"weft.alias_groups": {"X": "activation", "Y": "activation", "Weight": "parameters", "Bias": "parameters"}})
        rows = T.view(X, shape=(M, N), axes=("m", "k"), abi=True)
        weights = T.view(Weight, shape=(N,), axes=("k",), abi=True)
        biases = T.view(Bias, shape=(N,), axes=("k",), abi=True)
        output = T.view(Y, shape=(M, N), axes=("m", "k"), abi=True)
        with T.Kernel(T.ceildiv(M, rows_per_program), cpu_num_threads=cpu_num_threads) as program:
            base = 0 if rows_per_program == M else program * rows_per_program
            count = rows_per_program if M % rows_per_program == 0 else T.min(rows_per_program, M - base)
            for local_row in T.serial(count):
                row = base + local_row
                total = T.alloc_local((1,), "float32")
                variance = T.alloc_local((1,), "float32")
                mean = T.alloc_local((1,), "float32")
                factor = T.alloc_local((1,), "float32")
                if reduction_width == N:
                    values = T.load(rows[row, :])
                    T.reduce_sum(values, total, dim=0)
                else:
                    partial = T.alloc_local((reduction_width,), "float32")
                    T.fill(partial, 0)
                    with T.level.blocks(N, extent=reduction_width) as part:
                        value = T.load(rows[row, part])
                        active = T.extent(rows[row, part], axis=0)
                        for k in T.Parallel(reduction_width):
                            if k < active:
                                partial[k] += value[k]
                    T.reduce_sum(partial, total, dim=0)
                mean[0] = total[0] / N
                square = T.alloc_local((reduction_width,), "float32")
                if reduction_width == N:
                    for k in T.Parallel(N):
                        centered = values[k] - mean[0]
                        square[k] = centered * centered
                else:
                    T.fill(square, 0)
                    with T.level.blocks(N, extent=reduction_width) as part:
                        value = T.load(rows[row, part])
                        active = T.extent(rows[row, part], axis=0)
                        for k in T.Parallel(reduction_width):
                            if k < active:
                                centered = value[k] - mean[0]
                                square[k] += centered * centered
                T.reduce_sum(square, variance, dim=0)
                factor[0] = T.rsqrt(variance[0] / N + epsilon)
                with T.level.blocks(N, extent=block) as part:
                    value = T.load(rows[row, part])
                    weight, bias = T.load(weights[part]), T.load(biases[part])
                    result = T.alloc_local((block,), "float32")
                    for k in T.Parallel(block):
                        result[k] = (value[k] - mean[0]) * factor[0] * weight[k] + bias[k]
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
