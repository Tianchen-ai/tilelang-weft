"""Runtime extents/strides, multiple outputs and ordered scalar control."""
import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft", out_idx=[1, 2])
def split_factory(B=64):
    N, stride = T.symbolic("N"), T.symbolic("stride")

    @T.prim_func
    def main(X: T.StridedTensor((N,), (stride,), "float32"),
             Positive: T.Tensor((N,), "float32"),
             Negative: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, B)) as block:
            for lane in T.Parallel(B):
                i = block * B + lane
                if i < N:
                    Positive[i] = T.max(X[i], 0)
                    Negative[i] = T.min(X[i], 0)
    return main


@tilelang.jit(target="weft")
def iterative_factory():
    @T.prim_func
    def main(X: T.Tensor((1,), "int32"), N: T.int32):
        with T.Kernel(1):
            count = T.alloc_var("int32", init=0)
            value = T.alloc_var("int32", init=X[0])
            while count < N:
                if value > 0:
                    value = value - 1
                else:
                    value = value + 2
                count = count + 1
            X[0] = value
    return main
