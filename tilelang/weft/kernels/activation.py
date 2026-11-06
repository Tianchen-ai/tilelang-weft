"""Elementwise activation programs with explicit independent tiles."""

import tilelang
from tilelang.weft import language as T


@T.macro
def sigmoid(value):
    exponential = T.exp(-T.abs(value))
    return T.if_then_else(value >= 0, 1 / (1 + exponential), exponential / (1 + exponential))


@T.macro
def softplus_value(value):
    return T.max(value, 0) + T.log(1 + T.exp(-T.abs(value)))


@tilelang.jit(target="weft")
def elementwise(N, operation="silu", block=128, cpu_num_threads=0):
    if operation not in ("silu", "softplus", "exp", "log", "copy"):
        raise ValueError(f"unknown activation program {operation!r}")

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, block), cpu_num_threads=cpu_num_threads) as tile:
            for lane in T.Parallel(block):
                i = tile * block + lane
                if i < N:
                    value = X[i]
                    if operation == "silu":
                        Y[i] = value * sigmoid(value)
                    elif operation == "softplus":
                        Y[i] = softplus_value(value)
                    elif operation == "exp":
                        Y[i] = T.exp(value)
                    elif operation == "log":
                        Y[i] = T.log(value)
                    else:
                        Y[i] = value

    return main


@tilelang.jit(target="weft")
def swiglu(N, block=128, cpu_num_threads=0):
    @T.prim_func
    def main(Gate: T.Tensor((N,), "float32"), Up: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, block), cpu_num_threads=cpu_num_threads) as tile:
            for lane in T.Parallel(block):
                i = tile * block + lane
                if i < N:
                    Y[i] = Gate[i] * sigmoid(Gate[i]) * Up[i]

    return main


@tilelang.jit(target="weft")
def affine(N, block=128, cpu_num_threads=0, blocks_per_program=1):
    assert block > 0 and blocks_per_program > 0
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32"), scale: T.float32, bias: T.float32):
        xv = T.view(X, shape=(N,), axes=("k",), abi=True)
        yv = T.view(Y, shape=(N,), axes=("k",), abi=True)
        with T.Kernel(T.ceildiv(N, block * blocks_per_program), cpu_num_threads=cpu_num_threads) as tile:
            begin = 0 if N <= block * blocks_per_program else tile * block * blocks_per_program
            source = xv[begin : begin + block * blocks_per_program]
            output = yv[begin : begin + block * blocks_per_program]
            with T.level.blocks(T.extent(source, 0), extent=block) as part:
                value = T.load(source[part])
                result = T.alloc_local((block,), "float32")
                for lane in T.Parallel(block):
                    result[lane] = value[lane] * scale + bias
                T.copy(result, output[part])

    return main


@tilelang.jit(target="weft")
def affine_eager(X, Y, scale: T.float32, bias: T.float32, block=128, cpu_num_threads=0):
    N = T.dynamic("N")
    X: T.Tensor((N,), "float32")
    Y: T.Tensor((N,), "float32")
    with T.Kernel(T.ceildiv(N, block), cpu_num_threads=cpu_num_threads) as tile:
        for lane in T.Parallel(block):
            i = tile * block + lane
            if i < N:
                Y[i] = X[i] * scale + bias


@tilelang.jit(target="weft")
def binary(N, operation="add", block=128, cpu_num_threads=0):
    if operation not in ("add", "mul", "maximum"):
        raise ValueError(f"unknown binary program {operation!r}")

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Other: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, block), cpu_num_threads=cpu_num_threads) as tile:
            for lane in T.Parallel(block):
                i = tile * block + lane
                if i < N:
                    if operation == "add":
                        Y[i] = X[i] + Other[i]
                    elif operation == "mul":
                        Y[i] = X[i] * Other[i]
                    else:
                        Y[i] = T.max(X[i], Other[i])

    return main


@tilelang.jit(target="weft")
def where(N, block=128):
    @T.prim_func
    def main(
        Condition: T.Tensor((N,), "bool"), X: T.Tensor((N,), "float32"), Other: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")
    ):
        with T.Kernel(T.ceildiv(N, block)) as tile:
            for lane in T.Parallel(block):
                i = tile * block + lane
                if i < N:
                    Y[i] = T.if_then_else(Condition[i], X[i], Other[i])

    return main


@tilelang.jit(target="weft")
def fill(N, block=128):
    @T.prim_func
    def main(Y: T.Tensor((N,), "float32"), value: T.float32):
        with T.Kernel(T.ceildiv(N, block)) as tile:
            for lane in T.Parallel(block):
                i = tile * block + lane
                if i < N:
                    Y[i] = value

    return main
