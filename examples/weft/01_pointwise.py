"""A normal TileLang CPU grid compiled through the shared Weft core."""

import tilelang
from tilelang.weft import language as T


@tilelang.jit(target="weft")
def add_factory(N, B=128):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32"), Z: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, B), cpu_num_threads=2) as block:
            for lane in T.Parallel(B):
                i = block * B + lane
                if i < N:
                    Z[i] = X[i] + Y[i]

    return main


if __name__ == "__main__":
    import torch

    x = torch.linspace(-1, 1, 257)
    y = torch.linspace(2, 3, 257)
    z = torch.empty_like(x)
    kernel = add_factory(x.numel())
    kernel(x, y, z)
    torch.testing.assert_close(z, x + y)
    print("pointwise: correct")
