"""Complete-call projections with explicit activation preparation and closed GEMM."""

import tilelang
from tilelang.weft import language as T


@T.macro
def prepare_half(source, half, M, K):
    for row in T.serial(M):
        with T.level.blocks(K, extent=128) as part:
            value = T.load(source[row, part])
            converted = T.alloc_local((128,), "float16")
            for k in T.Parallel(128):
                converted[k] = T.cast(value[k], "float16")
            T.copy(converted, half[row, part])


@T.macro
def project(weights, activation, output, M, N, K, rows, block_k=None, batch_rows=1):
    with T.level.rows(M, group=batch_rows) as row:
        if block_k is None:
            value = T.stage(T.load(activation[row, :]))
            with T.level.cols(N, group=rows) as col:
                weight = T.load(weights[col, :])
                result = T.alloc_local((batch_rows, rows), "float32")
                T.gemm(value, weight, result, transpose_B=True, clear_accum=True, engine="vector")
                T.copy(result, output[row, col])
        else:
            with T.level.cols(N, group=rows) as col:
                total = T.state((batch_rows, rows), "float32", init=0)
                with T.level.blocks(K, extent=block_k) as part:
                    value = T.stage(T.load(activation[row, part]))
                    weight = T.load(weights[col, part])
                    T.gemm(value, weight, total, transpose_B=True, engine="vector")
                T.copy(total, output[row, col])


@tilelang.jit(target="weft")
def f16_projection(M, N, K, rows=4, block_k=None, batch_rows=1):
    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), "float16"),
        Activation: T.Tensor((M, K), "float32"),
        Half: T.Tensor((M, K), "float16"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Half": "workspace", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        half = T.view(Half, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            prepare_half(source, half, M, K)
            project(weights, half, output, M, N, K, rows, block_k, batch_rows)

    return main


@tilelang.jit(target="weft")
def staged_f16_projection(M, N, K, staged_columns=32, batch_rows=16, rows=4, columns=2):
    """Reuse declared weight/activation stages around full-K output tiles."""
    assert min(M, N, K, staged_columns, batch_rows, rows, columns) > 0
    assert staged_columns % columns == 0 and batch_rows % rows == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), "float16"),
        Activation: T.Tensor((M, K), "float32"),
        Half: T.Tensor((M, K), "float16"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Half": "workspace", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        activation = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Half, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            prepare_half(activation, prepared, M, K)
            with T.level.cols(N, group=staged_columns) as weight_columns:
                weight = T.stage(T.load(weights[weight_columns, :]))
                with T.level.rows(M, group=batch_rows) as batch:
                    values = T.stage(T.load(prepared[batch, :]))
                    with T.level.rows(batch, group=rows) as row:
                        with T.level.cols(weight_columns, group=columns) as col:
                            left = T.load(values, at=row, axis=0)
                            right = T.load(weight, at=col, axis=0)
                            result = T.alloc_local((rows, columns), "float32")
                            T.gemm(left, right, result, transpose_B=True, clear_accum=True, engine="vector")
                            T.copy(result, output[row, col])

    return main


@tilelang.jit(target="weft")
def f16_f32_projection(M, N, K, rows=4, block_k=None, batch_rows=1):
    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), "float16"),
        Activation: T.Tensor((M, K), "float32"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            project(weights, source, output, M, N, K, rows, block_k, batch_rows)

    return main


@tilelang.jit(target="weft")
def row_projection(M, N, K, lanes=128):
    """Retain F32 numerical lane sums across K before the final row reduction."""
    assert M > 0 and N > 0 and K > 0 and K % lanes == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), "float16"),
        Activation: T.Tensor((M, K), "float32"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        activation = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            for m in T.serial(M):
                with T.level.cols(N, group=1) as col:
                    partial = T.state((lanes,), "float32", init=0)
                    with T.level.blocks(K, extent=lanes) as block:
                        x = T.load(activation[m, block])
                        w = T.load(weights[col, block][0, :])
                        for k in T.Parallel(lanes):
                            partial[k] += x[k] * T.cast(w[k], "float32")
                    result = T.alloc_local((1,), "float32")
                    T.reduce_sum(partial, result, dim=0)
                    T.copy(result, output[m, col])

    return main
