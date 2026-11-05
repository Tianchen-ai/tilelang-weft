"""Explicitly blocked projections; engine applies to each local GEMM."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def matmul(
    M,
    N,
    K,
    block_m=2,
    block_n=4,
    block_k=8,
    dtype="float32",
    acc_dtype="float32",
    engine="auto",
    pipeline_depth=0,
    cpu_num_threads=0,
    transpose_weight=False,
    output_dtype=None,
    operand_dtype=None,
):
    output_dtype = acc_dtype if output_dtype is None else output_dtype
    operand_dtype = dtype if operand_dtype is None else operand_dtype

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K) if transpose_weight else (K, N), dtype), C: T.Tensor((M, N), output_dtype)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(N, block_n), cpu_num_threads=cpu_num_threads) as (row, column):
            accumulator = T.alloc_local((block_m, block_n), acc_dtype)
            T.fill(accumulator, 0)
            for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=pipeline_depth):
                left = T.alloc_local((block_m, block_k), operand_dtype)
                right = T.alloc_local((block_n, block_k) if transpose_weight else (block_k, block_n), operand_dtype)
                T.copy(A[row * block_m, k * block_k], left)
                if transpose_weight:
                    T.copy(B[column * block_n, k * block_k], right)
                else:
                    T.copy(B[k * block_k, column * block_n], right)
                T.gemm(left, right, accumulator, transpose_B=transpose_weight, engine=engine)
            T.copy(accumulator, C[row * block_m, column * block_n])

    return main


@tilelang.jit(target="weft")
def full_reduction(M, N, K, dtype="bfloat16", acc_dtype="float32", rows=16, columns=16, engine="auto"):
    """Keep each K reduction closed; share an activation tile across columns."""

    @T.prim_func
    def main(X: T.Tensor((M, K), dtype), W: T.Tensor((N, K), dtype), Y: T.Tensor((M, N), acc_dtype)):
        T.func_attr({"tir.noalias": True})
        xv = T.view(X, shape=(M, K), axes=("m", "k"), abi=True)
        wv = T.view(W, shape=(N, K), axes=("n", "k"), abi=True)
        yv = T.view(Y, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=rows) as row:
                activation = T.load(xv[row, :])
                with T.level.cols(N, group=columns) as col:
                    weights = T.load(wv[col, :])
                    result = T.alloc_local((rows, columns), acc_dtype)
                    T.reduce_dot(activation, weights, result, over="k", acc_dtype=acc_dtype, engine=engine)
                    T.copy(result, yv[row, col])

    return main


@tilelang.jit(target="weft")
def projection(Batch, M, N, K, block_m=4, block_n=16, block_k=32, transpose_weight=True, bias=True, activation="none", engine="auto"):
    """Batched dense projection, optional bias and explicit epilogue."""
    assert activation in ("none", "silu")
    from .activation import sigmoid

    @T.prim_func
    def main(
        X: T.Tensor((Batch, M, K), "float32"),
        Weight: T.Tensor((Batch, N, K) if transpose_weight else (Batch, K, N), "float32"),
        Bias: T.Tensor((N,), "float32"),
        Y: T.Tensor((Batch, M, N), "float32"),
    ):
        with T.Kernel(Batch, T.ceildiv(M, block_m), T.ceildiv(N, block_n)) as (batch, row, column):
            accumulator = T.alloc_local((block_m, block_n), "float32")
            T.fill(accumulator, 0)
            for chunk in T.serial(T.ceildiv(K, block_k)):
                left = T.alloc_local((block_m, block_k), "float32")
                right = T.alloc_local((block_k, block_n), "float32")
                for m, k in T.Parallel(block_m, block_k):
                    left[m, k] = X[batch, row * block_m + m, chunk * block_k + k]
                for k, n in T.Parallel(block_k, block_n):
                    if transpose_weight:
                        right[k, n] = Weight[batch, column * block_n + n, chunk * block_k + k]
                    else:
                        right[k, n] = Weight[batch, chunk * block_k + k, column * block_n + n]
                T.gemm(left, right, accumulator, engine=engine)
            for m, n in T.Parallel(block_m, block_n):
                value = accumulator[m, n]
                if bias:
                    value += Bias[column * block_n + n]
                if activation == "silu":
                    value = value * sigmoid(value)
                Y[batch, row * block_m + m, column * block_n + n] = value

    return main
