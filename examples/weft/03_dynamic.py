"""Runtime extents/strides, multiple outputs and ordered scalar control."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft", out_idx=[1, 2])
def split_factory(B=64):
    N, stride = T.dynamic("N"), T.dynamic("stride")

    @T.prim_func
    def main(X: T.StridedTensor((N,), (stride,), "float32"), Positive: T.Tensor((N,), "float32"), Negative: T.Tensor((N,), "float32")):
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


if __name__ == "__main__":
    import torch

    kernel = split_factory()
    for length in (17, 65):
        x = torch.linspace(-1, 1, length * 2)[::2]
        positive, negative = kernel(x)
        torch.testing.assert_close(positive, x.clamp_min(0))
        torch.testing.assert_close(negative, x.clamp_max(0))
    state = torch.tensor([3], dtype=torch.int32)
    iterative_factory()(state, 5)
    assert state.item() == 1
    print("runtime shape/stride, multiple outputs and state: correct")
