"""Stable routing and explicit ragged expert execution, using caller workspaces."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def router(Tokens, Experts, K):
    assert 0 < K <= Experts

    @T.prim_func
    def main(
        Logits: T.Tensor((Tokens, Experts), "float32"), Indices: T.Tensor((Tokens, K), "int32"), Weights: T.Tensor((Tokens, K), "float32")
    ):
        with T.Kernel(Tokens) as token:
            for slot in T.serial(K):
                best = T.alloc_var("float32", float("-inf"))
                chosen = T.alloc_var("int32", 0)
                for expert in T.serial(Experts):
                    taken = T.alloc_var("bool", False)
                    for previous in T.serial(slot):
                        taken = taken | (Indices[token, previous] == expert)
                    if not taken and Logits[token, expert] > best:
                        best = Logits[token, expert]
                        chosen = expert
                Indices[token, slot] = chosen
                Weights[token, slot] = best
            maximum = Weights[token, 0]
            total = T.alloc_var("float32", 0)
            for slot in T.serial(K):
                value = T.exp(Weights[token, slot] - maximum)
                Weights[token, slot] = value
                total += value
            for slot in T.serial(K):
                Weights[token, slot] = Weights[token, slot] / total

    return main


@tilelang.jit(target="weft")
def group_tokens(Tokens, D, Experts, K):
    @T.prim_func
    def main(
        X: T.Tensor((Tokens, D), "float32"),
        Indices: T.Tensor((Tokens, K), "int32"),
        Offsets: T.Tensor((Experts + 1,), "int32"),
        Cursor: T.Tensor((Experts,), "int32"),
        Positions: T.Tensor((Tokens, K), "int32"),
        Grouped: T.Tensor((Tokens * K, D), "float32"),
    ):
        with T.Kernel(1):
            for expert in T.Parallel(Experts + 1):
                Offsets[expert] = 0
            for token in T.serial(Tokens):
                for slot in T.serial(K):
                    expert = Indices[token, slot]
                    Offsets[expert + 1] += 1
            for expert in T.serial(Experts):
                Offsets[expert + 1] += Offsets[expert]
                Cursor[expert] = Offsets[expert]
            for token in T.serial(Tokens):
                for slot in T.serial(K):
                    expert = Indices[token, slot]
                    position = Cursor[expert]
                    Positions[token, slot] = position
                    Cursor[expert] = position + 1
                    for channel in T.Parallel(D):
                        Grouped[position, channel] = X[token, channel]

    return main


@tilelang.jit(target="weft")
def expert_linear(P, K, N, Experts, block_n=16, block_k=32, engine="auto"):
    @T.prim_func
    def main(
        X: T.Tensor((P, K), "float32"),
        Weight: T.Tensor((Experts, N, K), "float32"),
        Offsets: T.Tensor((Experts + 1,), "int32"),
        Y: T.Tensor((P, N), "float32"),
    ):
        with T.Kernel(Experts, T.ceildiv(N, block_n)) as (expert, column):
            for row in T.serial(Offsets[expert], Offsets[expert + 1]):
                accumulator = T.alloc_local((1, block_n), "float32")
                T.fill(accumulator, 0)
                for chunk in T.serial(T.ceildiv(K, block_k)):
                    left = T.alloc_local((1, block_k), "float32")
                    right = T.alloc_local((block_k, block_n), "float32")
                    for k in T.Parallel(block_k):
                        left[0, k] = X[row, chunk * block_k + k]
                    for k, n in T.Parallel(block_k, block_n):
                        right[k, n] = Weight[expert, column * block_n + n, chunk * block_k + k]
                    T.gemm(left, right, accumulator, engine=engine)
                for n in T.Parallel(block_n):
                    Y[row, column * block_n + n] = accumulator[0, n]

    return main


@tilelang.jit(target="weft")
def combine(Tokens, D, K):
    @T.prim_func
    def main(
        Grouped: T.Tensor((Tokens * K, D), "float32"),
        Positions: T.Tensor((Tokens, K), "int32"),
        Weights: T.Tensor((Tokens, K), "float32"),
        Y: T.Tensor((Tokens, D), "float32"),
    ):
        with T.Kernel(Tokens) as token:
            local = T.alloc_local((D,), "float32")
            T.fill(local, 0)
            for slot in T.serial(K):
                position = Positions[token, slot]
                weight = Weights[token, slot]
                for channel in T.Parallel(D):
                    local[channel] += Grouped[position, channel] * weight
            for channel in T.Parallel(D):
                Y[token, channel] = local[channel]

    return main
