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
def causal_attention(B, Tokens, Heads, Groups, D, Capacity, tokens_per_program=1, heads_per_program=1):
    """Grouped-query online attention over the populated causal KV prefix."""
    assert Heads % Groups == 0
    assert tokens_per_program > 0 and heads_per_program > 0

    @T.prim_func
    def main(
        Query: T.Tensor((B, Tokens, Heads, D), "float32"),
        Key: T.Tensor((B, Capacity, Groups, D), "float32"),
        Value: T.Tensor((B, Capacity, Groups, D), "float32"),
        Output: T.Tensor((B, Tokens, Heads, D), "float32"),
        start: T.int32,
        scale: T.float32,
    ):
        queries = T.view(Query, shape=(B, Tokens, Heads, D), axes=("b", "t", "h", "d"))
        keys = T.view(Key, shape=(B, Capacity, Groups, D), axes=("b", "s", "g", "d"))
        values = T.view(Value, shape=(B, Capacity, Groups, D), axes=("b", "s", "g", "d"))
        outputs = T.view(Output, shape=(B, Tokens, Heads, D), axes=("b", "t", "h", "d"))
        with T.Kernel(B, T.ceildiv(Tokens, tokens_per_program), T.ceildiv(Heads, heads_per_program)) as (batch, token_group, head_group):
            for t in T.serial(tokens_per_program):
                token = token_group * tokens_per_program + t
                if token < Tokens:
                    for h in T.serial(heads_per_program):
                        head = head_group * heads_per_program + h
                        if head < Heads:
                            causal_row(queries, keys, values, outputs, batch, token, head, start, scale, Heads, Groups, D)

    return main


@T.macro
def causal_row(queries, keys, values, outputs, batch, token, head, start, scale, Heads, Groups, D):
    group = head // (Heads // Groups)
    query = T.stage(T.load(queries[batch, token, head, :]))
    numerator = T.state((D,), "float32", init=0)
    score = T.alloc_local((1,), "float32")
    maximum = T.alloc_var("float32", init=float("-inf"))
    denominator = T.alloc_var("float32", init=0)
    for key in T.serial(start + token + 1):
        key_value = T.load(keys[batch, key, group, :])
        T.reduce_dot(query, key_value, score, over="d", acc_dtype="float32", engine="vector")
        value = score[0] * scale
        next_max = T.max(maximum, value)
        keep = T.exp(maximum - next_max)
        probability = T.exp(value - next_max)
        payload = T.load(values[batch, key, group, :])
        for d in T.Parallel(D):
            numerator[d] = numerator[d] * keep + payload[d] * probability
        denominator = denominator * keep + probability
        maximum = next_max
    for d in T.Parallel(D):
        numerator[d] = numerator[d] / denominator
    T.copy(numerator, outputs[batch, token, head, :])


@tilelang.jit(target="weft")
def rope(M, H, P, interleaved=False, cpu_num_threads=0, heads_per_program=1, pair_tile=None, rows_per_program=1, head_major=False):
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
        cosine_view = T.view(Cos, shape=(M, P), axes=("m", "k"), abi=True)
        sine_view = T.view(Sin, shape=(M, P), axes=("m", "k"), abi=True)
        if interleaved:
            inputs = T.view(X, shape=(M, H, P, 2), axes=("m", "h", "k", "pair"), abi=True)
            outputs = T.view(Y, shape=(M, H, P, 2), axes=("m", "h", "k", "pair"), abi=True)
        else:
            inputs = T.view(X, shape=(M, H, 2, P), axes=("m", "h", "pair", "k"), abi=True)
            outputs = T.view(Y, shape=(M, H, 2, P), axes=("m", "h", "pair", "k"), abi=True)
        with T.Kernel(T.ceildiv(M, rows_per_program), T.ceildiv(H, heads_per_program), cpu_num_threads=cpu_num_threads) as (tile, cohort):
            for token in T.serial(rows_per_program):
                row = token if rows_per_program >= M else tile * rows_per_program + token
                if row < M:
                    if head_major:
                        for offset in T.serial(heads_per_program):
                            head = offset if heads_per_program >= H else cohort * heads_per_program + offset
                            if head < H:
                                with T.level.blocks(P, extent=pair_tile) as part:
                                    cosine, sine = T.load(cosine_view[row, part]), T.load(sine_view[row, part])
                                    rope_apply(inputs, outputs, row, head, part, cosine, sine, pair_tile, interleaved)
                    else:
                        with T.level.blocks(P, extent=pair_tile) as part:
                            cosine = T.stage(T.load(cosine_view[row, part]))
                            sine = T.stage(T.load(sine_view[row, part]))
                            for offset in T.serial(heads_per_program):
                                head = offset if heads_per_program >= H else cohort * heads_per_program + offset
                                if head < H:
                                    rope_apply(inputs, outputs, row, head, part, cosine, sine, pair_tile, interleaved)

    return main


@tilelang.jit(target="weft")
def rope_runtime(pair_tile=32, cpu_num_threads=1):
    """Reuse one shape-polymorphic rotary program for dense CPU tensors."""
    M, H, P = T.dynamic("T, H, P")

    @T.prim_func
    def main(
        X: T.Tensor((M, H, 2 * P), "float32"),
        Cos: T.Tensor((M, P), "float32"),
        Sin: T.Tensor((M, P), "float32"),
        Y: T.Tensor((M, H, 2 * P), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"X": "activation", "Y": "activation", "Cos": "coefficients", "Sin": "coefficients"}})
        inputs = T.view(X, shape=(M, H, 2, P), axes=("m", "h", "pair", "k"), abi=True)
        outputs = T.view(Y, shape=(M, H, 2, P), axes=("m", "h", "pair", "k"), abi=True)
        cosine_view = T.view(Cos, shape=(M, P), axes=("m", "k"), abi=True)
        sine_view = T.view(Sin, shape=(M, P), axes=("m", "k"), abi=True)
        with T.Kernel(1, cpu_num_threads=cpu_num_threads):
            for row in T.serial(M):
                for head in T.serial(H):
                    with T.level.blocks(P, extent=pair_tile) as part:
                        cosine, sine = T.load(cosine_view[row, part]), T.load(sine_view[row, part])
                        rope_apply(inputs, outputs, row, head, part, cosine, sine, pair_tile, False)

    return main


@T.macro
def rope_apply(inputs, outputs, row, head, part, cosine, sine, pair_tile, interleaved):
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
    T.copy(left, first_output)
    for k in T.Parallel(pair_tile):
        right[k] = a[k] * sine[k] + b[k] * cosine[k]
    T.copy(right, second_output)


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
