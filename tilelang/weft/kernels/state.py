"""Ordinary control and explicit recurrence for statistics, routing and SSMs."""

import tilelang
from tilelang.weft import language as T
from .activation import sigmoid, softplus_value


@tilelang.jit(target="weft")
def welford(M, N):
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Moments: T.Tensor((M, 3), "float32")):
        with T.Kernel(M) as row:
            count = T.alloc_var("float32", 0)
            mean = T.alloc_var("float32", 0)
            m2 = T.alloc_var("float32", 0)
            for k in T.serial(N):
                count += 1
                delta = X[row, k] - mean
                mean += delta / count
                m2 += delta * (X[row, k] - mean)
            Moments[row, 0] = count
            Moments[row, 1] = mean
            Moments[row, 2] = m2

    return main


@tilelang.jit(target="weft")
def top_k(M, N, K):
    assert 0 < K <= N

    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Values: T.Tensor((M, K), "float32"), Indices: T.Tensor((M, K), "int32")):
        with T.Kernel(M) as row:
            for slot in T.serial(K):
                best = T.alloc_var("float32", float("-inf"))
                chosen = T.alloc_var("int32", 0)
                for column in T.serial(N):
                    taken = T.alloc_var("bool", False)
                    for previous in T.serial(slot):
                        taken = taken | (Indices[row, previous] == column)
                    if not taken and X[row, column] > best:
                        best = X[row, column]
                        chosen = column
                Values[row, slot] = best
                Indices[row, slot] = chosen

    return main


@tilelang.jit(target="weft")
def selective_scan(B, L, D, N):
    @T.prim_func
    def main(
        X: T.Tensor((B, L, D), "float32"),
        Delta: T.Tensor((B, L, D), "float32"),
        Bcoef: T.Tensor((B, L, N), "float32"),
        Ccoef: T.Tensor((B, L, N), "float32"),
        Alog: T.Tensor((D, N), "float32"),
        Skip: T.Tensor((D,), "float32"),
        Gate: T.Tensor((B, L, D), "float32"),
        Bias: T.Tensor((D,), "float32"),
        State: T.Tensor((B, D, N), "float32"),
        Y: T.Tensor((B, L, D), "float32"),
    ):
        with T.Kernel(B, D) as (batch, channel):
            local = T.alloc_local((N,), "float32")
            product = T.alloc_local((N,), "float32")
            total = T.alloc_local((1,), "float32")
            for n in T.Parallel(N):
                local[n] = State[batch, channel, n]
            for token in T.serial(L):
                dt = softplus_value(Delta[batch, token, channel] + Bias[channel])
                value = X[batch, token, channel]
                for n in T.Parallel(N):
                    decay = T.exp(-T.exp(Alog[channel, n]) * dt)
                    local[n] = local[n] * decay + Bcoef[batch, token, n] * (dt * value)
                    product[n] = local[n] * Ccoef[batch, token, n]
                T.reduce_sum(product, total, dim=0)
                Y[batch, token, channel] = (
                    (total[0] + Skip[channel] * value) * Gate[batch, token, channel] * sigmoid(Gate[batch, token, channel])
                )
            for n in T.Parallel(N):
                State[batch, channel, n] = local[n]

    return main


@tilelang.jit(target="weft")
def causal_conv1d(B, L, D, W):
    assert W > 0

    @T.prim_func
    def main(
        X: T.Tensor((B, L, D), "float32"),
        Weight: T.Tensor((D, W), "float32"),
        Bias: T.Tensor((D,), "float32"),
        History: T.Tensor((B, W, D), "float32"),
        Y: T.Tensor((B, L, D), "float32"),
    ):
        with T.Kernel(B) as batch:
            for token in T.serial(L):
                for lag in T.serial(W - 1):
                    for channel in T.Parallel(D):
                        History[batch, lag, channel] = History[batch, lag + 1, channel]
                for channel in T.Parallel(D):
                    History[batch, W - 1, channel] = X[batch, token, channel]
                local = T.alloc_local((D,), "float32")
                for channel in T.Parallel(D):
                    local[channel] = Bias[channel]
                for lag in T.serial(W):
                    for channel in T.Parallel(D):
                        local[channel] += History[batch, lag, channel] * Weight[channel, lag]
                for channel in T.Parallel(D):
                    Y[batch, token, channel] = local[channel] * sigmoid(local[channel])

    return main
