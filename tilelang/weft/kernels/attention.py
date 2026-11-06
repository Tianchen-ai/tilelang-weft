"""Softmax and rotary layouts expressed by native tile/scalar operations."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def softmax(M, N, cpu_num_threads=0, rows_per_program=1):
    assert rows_per_program > 0
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32")):
        rows = T.view(X, shape=(M, N), axes=("m", "k"))
        with T.Kernel(T.ceildiv(M, rows_per_program), cpu_num_threads=cpu_num_threads) as program:
            base = 0 if rows_per_program >= M else program * rows_per_program
            count = rows_per_program if M % rows_per_program == 0 else T.min(rows_per_program, M - base)
            for r in T.serial(count):
                row = base + r
                value = T.stage(T.load(rows[row, :]))
                maximum = T.alloc_local((1,), "float32")
                probability = T.alloc_local((N,), "float32")
                denominator = T.alloc_local((1,), "float32")
                T.reduce_max(value, maximum, dim=0)
                for k in T.Parallel(N):
                    probability[k] = T.exp(value[k] - maximum[0])
                T.reduce_sum(probability, denominator, dim=0)
                for k in T.Parallel(N):
                    Y[row, k] = probability[k] / denominator[0]

    return main


@tilelang.jit(target="weft")
def kv_update(B, Tokens, Groups, D, Capacity, block=64):
    """Append a contiguous token prefix to caller-owned key/value caches."""

    @T.prim_func
    def main(
        Key: T.Tensor((B, Tokens, Groups, D), "float32"),
        Value: T.Tensor((B, Tokens, Groups, D), "float32"),
        KeyCache: T.Tensor((B, Capacity, Groups, D), "float32"),
        ValueCache: T.Tensor((B, Capacity, Groups, D), "float32"),
        start: T.int32,
    ):
        with T.Kernel(B, Tokens, Groups) as (batch, token, head):
            for chunk in T.serial(T.ceildiv(D, block)):
                for lane in T.Parallel(block):
                    d = chunk * block + lane
                    if d < D:
                        KeyCache[batch, start + token, head, d] = Key[batch, token, head, d]
                        ValueCache[batch, start + token, head, d] = Value[batch, token, head, d]

    return main


@tilelang.jit(target="weft")
def causal_attention(B, Tokens, Heads, Groups, D, Capacity):
    """Grouped-query online attention over the populated causal KV prefix."""
    assert Heads % Groups == 0

    @T.prim_func
    def main(
        Query: T.Tensor((B, Tokens, Heads, D), "float32"),
        Key: T.Tensor((B, Capacity, Groups, D), "float32"),
        Value: T.Tensor((B, Capacity, Groups, D), "float32"),
        Output: T.Tensor((B, Tokens, Heads, D), "float32"),
        start: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(B, Tokens, Heads) as (batch, token, head):
            group = head // (Heads // Groups)
            query = T.alloc_local((D,), "float32")
            numerator = T.alloc_local((D,), "float32")
            product = T.alloc_local((D,), "float32")
            score = T.alloc_local((1,), "float32")
            maximum = T.alloc_var("float32", init=float("-inf"))
            denominator = T.alloc_var("float32", init=0)
            T.fill(numerator, 0)
            for d in T.Parallel(D):
                query[d] = Query[batch, token, head, d]
            for key in T.serial(start + token + 1):
                for d in T.Parallel(D):
                    product[d] = query[d] * Key[batch, key, group, d]
                T.reduce_sum(product, score, dim=0)
                value = score[0] * scale
                next_max = T.max(maximum, value)
                keep = T.exp(maximum - next_max)
                probability = T.exp(value - next_max)
                for d in T.Parallel(D):
                    numerator[d] = numerator[d] * keep + Value[batch, key, group, d] * probability
                denominator = denominator * keep + probability
                maximum = next_max
            for d in T.Parallel(D):
                Output[batch, token, head, d] = numerator[d] / denominator

    return main


@tilelang.jit(target="weft")
def rope(M, H, P, interleaved=False, cpu_num_threads=0, heads_per_program=1, pair_tile=None, rows_per_program=1):
    pair_tile = P if pair_tile is None else pair_tile
    assert heads_per_program > 0 and pair_tile > 0 and rows_per_program > 0
    @T.prim_func
    def main(
        X: T.Tensor((M, H, 2 * P), "float32"),
        Cos: T.Tensor((M, P), "float32"),
        Sin: T.Tensor((M, P), "float32"),
        Y: T.Tensor((M, H, 2 * P), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"X": "activation", "Y": "activation", "Cos": "coefficients", "Sin": "coefficients"}})
        cosine_view = T.view(Cos, shape=(M, P), axes=("m", "k"))
        sine_view = T.view(Sin, shape=(M, P), axes=("m", "k"))
        if interleaved:
            inputs = T.view(X, shape=(M, H, P, 2), axes=("m", "h", "k", "pair"))
            outputs = T.view(Y, shape=(M, H, P, 2), axes=("m", "h", "k", "pair"))
        else:
            inputs = T.view(X, shape=(M, H, 2, P), axes=("m", "h", "pair", "k"))
            outputs = T.view(Y, shape=(M, H, 2, P), axes=("m", "h", "pair", "k"))
        with T.Kernel(T.ceildiv(M, rows_per_program), T.ceildiv(H, heads_per_program), cpu_num_threads=cpu_num_threads) as (tile, cohort):
            for token in T.serial(rows_per_program):
                row = token if rows_per_program >= M else tile * rows_per_program + token
                if row < M:
                    with T.level.blocks(P, extent=pair_tile) as part:
                        cosine = T.stage(T.load(cosine_view[row, part]))
                        sine = T.stage(T.load(sine_view[row, part]))
                        for offset in T.serial(heads_per_program):
                            head = offset if heads_per_program >= H else cohort * heads_per_program + offset
                            if head < H:
                                if interleaved:
                                    first, second = inputs[row, head, part, 0], inputs[row, head, part, 1]
                                    first_output, second_output = outputs[row, head, part, 0], outputs[row, head, part, 1]
                                else:
                                    first, second = inputs[row, head, 0, part], inputs[row, head, 1, part]
                                    first_output, second_output = outputs[row, head, 0, part], outputs[row, head, 1, part]
                                a, b = T.load(first), T.load(second)
                                left, right = T.alloc_local((pair_tile,), "float32"), T.alloc_local((pair_tile,), "float32")
                                for k in T.Parallel(pair_tile):
                                    left[k] = a[k] * cosine[k] - b[k] * sine[k]
                                    right[k] = a[k] * sine[k] + b[k] * cosine[k]
                                T.copy(left, first_output)
                                T.copy(right, second_output)

    return main


@tilelang.jit(target="weft")
def online_attention(Q, S, D, rows=2, keys=16, engine="auto"):
    @T.prim_func
    def main(
        Query: T.Tensor((Q, D), "float32"),
        Key: T.Tensor((S, D), "float32"),
        Value: T.Tensor((S, D), "float32"),
        Output: T.Tensor((Q, D), "float32"),
        scale: T.float32,
    ):
        qv = T.view(Query, shape=(Q, D), axes=("m", "d"))
        kv = T.view(Key, shape=(S, D), axes=("k", "d"))
        vv = T.view(Value, shape=(S, D), axes=("k", "d"))
        yv = T.view(Output, shape=(Q, D), axes=("m", "d"))
        with T.Kernel(1):
            with T.level.rows(Q, group=rows) as row:
                query = T.stage(T.load(qv[row, :]))
                maximum = T.state((rows,), "float32", init=float("-inf"))
                denominator = T.state((rows,), "float32", init=0)
                numerator = T.state((rows, D), "float32", init=0)
                with T.level.blocks(S, extent=keys) as part:
                    key_region = kv[part, :]
                    key = T.load(key_region)
                    value = T.load(vv[part, :])
                    score = T.alloc_local((rows, keys), "float32")
                    block_max = T.alloc_local((rows,), "float32")
                    next_max = T.alloc_local((rows,), "float32")
                    keep = T.alloc_local((rows,), "float32")
                    probability = T.alloc_local((rows, keys), "float32")
                    block_sum = T.alloc_local((rows,), "float32")
                    contribution = T.alloc_local((rows, D), "float32")
                    T.gemm(query, key, score, transpose_B=True, clear_accum=True, engine=engine)
                    for r, k in T.Parallel(rows, keys):
                        score[r, k] = T.if_then_else(k < T.extent(key_region, 0), score[r, k] * scale, float("-inf"))
                    T.reduce_max(score, block_max, dim=1)
                    for r in T.Parallel(rows):
                        next_max[r] = T.max(maximum[r], block_max[r])
                        keep[r] = T.exp(maximum[r] - next_max[r])
                    for r, k in T.Parallel(rows, keys):
                        probability[r, k] = T.exp(score[r, k] - next_max[r])
                    T.reduce_sum(probability, block_sum, dim=1)
                    T.gemm(probability, value, contribution, clear_accum=True, engine=engine)
                    for r, d in T.Parallel(rows, D):
                        numerator[r, d] = numerator[r, d] * keep[r] + contribution[r, d]
                    for r in T.Parallel(rows):
                        denominator[r] = denominator[r] * keep[r] + block_sum[r]
                        maximum[r] = next_max[r]
                for r, d in T.Parallel(rows, D):
                    numerator[r, d] = numerator[r, d] / denominator[r]
                T.copy(numerator, yv[row, :])

    return main
