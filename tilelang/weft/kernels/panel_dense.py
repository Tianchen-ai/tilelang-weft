"""Author-owned panel packing and outer products for row-major CPU inputs."""

import tilelang
from tilelang.weft import language as T


@T.macro
def transpose_panel(source, destination, rows, columns, step, round_f16):
    for start in T.serial(columns // step):
        value = T.load(source[:, start * step : (start + 1) * step])
        packed = T.alloc_local((step, rows), "float32")
        for r, k in T.Parallel(rows, step):
            packed[k, r] = T.cast(T.cast(value[r, k], "float16"), "float32") if round_f16 else T.cast(value[r, k], "float32")
        T.copy(packed, destination[start * step : (start + 1) * step, :])


@T.macro
def product(weights, activation, output, weight_panel, activation_panel,
            M, N, K, rows, columns, block_k, round_f16, unroll_k=1, pack_activation=True):
    first = T.alloc_local((1,), "bool")
    first[0] = True
    with T.level.blocks(K, extent=block_k) as block:
        if pack_activation:
            with T.level.rows(M, group=rows) as row:
                transpose_panel(activation[row, block], activation_panel[:, row], rows, block_k, 16, round_f16)
        with T.level.cols(N, group=columns) as col:
            transpose_panel(weights[col, block], weight_panel, columns, block_k, 8, False)
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows, columns), "float32", init=0)
                for start in T.serial(block_k // unroll_k):
                    for offset in T.unroll(unroll_k):
                        k = start * unroll_k + offset
                        if pack_activation:
                            left = T.load(activation_panel[k, row])
                            right = T.load(weight_panel[k, :])
                            for r, c in T.Parallel(rows, columns):
                                total[r, c] += left[r] * right[c]
                        else:
                            left = T.alloc_local((rows,), "float32")
                            for r in T.unroll(rows):
                                value = T.load(activation[row, block][r, k])
                                left[r] = T.cast(T.cast(value[()], "float16"), "float32") if round_f16 else value[()]
                            right = T.load(weight_panel[k, :])
                            for r, c in T.Parallel(rows, columns):
                                total[r, c] += left[r] * right[c]
                if not first[0]:
                    previous = T.load(output[row, col])
                    for r, c in T.Parallel(rows, columns):
                        total[r, c] += previous[r, c]
                T.copy(total, output[row, col])
        first[0] = False


@tilelang.jit(target="weft")
def projection(M, N, K, rows=8, columns=32, block_k=256,
               weight_dtype="float16", round_f16=True, unroll_k=1):
    """Include both invocation-local transposes and optional F16 activation rounding."""
    assert weight_dtype in ("float16", "float32", "bfloat16")
    assert M > 0 and N > 0 and K > 0
    assert M % rows == N % columns == K % block_k == 0 and block_k % 16 == 0
    assert unroll_k in (1, 2, 4, 8) and block_k % unroll_k == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), weight_dtype),
        Activation: T.Tensor((M, K), "float32"),
        Output: T.Tensor((M, N), "float32"),
        Scratch: T.Tensor((block_k, columns), "float32"),
        PackedActivation: T.Tensor((block_k, M), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Output": "output", "Scratch": "weights", "PackedActivation": "activation"}})
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        activation = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        weight_panel = T.view(Scratch, shape=(block_k, columns), axes=("bk", "bn"), abi=True)
        activation_panel = T.view(PackedActivation, shape=(block_k, M), axes=("bk", "m"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            product(weights, activation, output, weight_panel, activation_panel,
                    M, N, K, rows, columns, block_k, round_f16, unroll_k)

    return main


@tilelang.jit(target="weft")
def direct_activation_projection(M, N, K, rows=4, columns=32, block_k=64,
                                 weight_dtype="float32", round_f16=False, unroll_k=1):
    """Read activation rows directly and prepare only the bounded weight panel."""
    assert weight_dtype in ("float16", "float32", "bfloat16")
    assert min(M, N, K, rows, columns, block_k) > 0
    assert M % rows == N % columns == K % block_k == 0 and block_k % 8 == 0
    assert unroll_k in (1, 2, 4, 8) and block_k % unroll_k == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, K), weight_dtype),
        Activation: T.Tensor((M, K), "float32"),
        Output: T.Tensor((M, N), "float32"),
        Scratch: T.Tensor((block_k, columns), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Output": "output", "Scratch": "weights"}})
        weights = T.view(Weights, shape=(N, K), axes=("n", "k"), abi=True)
        activation = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        weight_panel = T.view(Scratch, shape=(block_k, columns), axes=("bk", "bn"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            product(weights, activation, output, weight_panel, activation,
                    M, N, K, rows, columns, block_k, round_f16, unroll_k, False)

    return main
