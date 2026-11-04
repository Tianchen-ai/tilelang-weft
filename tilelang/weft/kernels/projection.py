"""Complete-call projections with explicit activation preparation and closed GEMM."""

import tilelang
from tilelang.weft import language as T


@T.macro
def project(weights, activation, output, M, N, K, rows, block_k=None):
    with T.level.rows(M, group=1) as row:
        if block_k is None:
            value = T.stage(T.load(activation[row, :]))
            with T.level.cols(N, group=rows) as col:
                weight = T.load(weights[col, :])
                result = T.alloc_local((1, rows), "float32")
                T.gemm(value, weight, result, transpose_B=True, clear_accum=True, engine="vector")
                T.copy(result, output[row, col])
        else:
            with T.level.cols(N, group=rows) as col:
                total = T.state((1, rows), "float32", init=0)
                with T.level.blocks(K, extent=block_k) as part:
                    value = T.stage(T.load(activation[row, part]))
                    weight = T.load(weights[col, part])
                    result = T.alloc_local((1, rows), "float32")
                    T.gemm(value, weight, result, transpose_B=True, clear_accum=True, engine="vector")
                    for r, c in T.Parallel(1, rows):
                        total[r, c] += result[r, c]
                T.copy(total, output[row, col])


@tilelang.jit(target="weft")
def f16_projection(M, N, K, rows=4, block_k=None):
    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), "float16"), Activation: T.Tensor((M, K), "float32"),
        Half: T.Tensor((M, K), "float16"), Output: T.Tensor((M, N), "float32"),
    ):
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        half = T.view(Half, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            for m in T.serial(M):
                with T.level.blocks(K, extent=128) as part:
                    value = T.load(source[m, part])
                    converted = T.alloc_local((128,), "float16")
                    for k in T.Parallel(128):
                        converted[k] = T.cast(value[k], "float16")
                    T.copy(converted, half[m, part])
            project(weights, half, output, M, N, K, rows, block_k)

    return main


@tilelang.jit(target="weft")
def f16_f32_projection(M, N, K, rows=4, block_k=None):
    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), "float16"), Activation: T.Tensor((M, K), "float32"),
        Output: T.Tensor((M, N), "float32"),
    ):
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            project(weights, source, output, M, N, K, rows, block_k)

    return main
