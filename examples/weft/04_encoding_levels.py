"""Packed storage, typed partial sums and explicit state/stage lifetimes."""

import tilelang
from tilelang.weft import language as T


@tilelang.encoding
class Packed32:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 1
    elements = 32
    q: T.Field("uint4", (32,)) @ T.grouped(elements=32) @ T.bit_layers(elements=16, order=T.lo_first)


@tilelang.jit(target="weft")
def packed_dot_factory(M=8, K=64):
    @T.prim_func
    def main(Wbytes: T.Tensor((M, K // 2), "uint8"), X: T.Tensor((K,), "int8"), Y: T.Tensor((M,), "int32")):
        W = T.view(Wbytes, shape=(M, K), encoding=Packed32, axes=("m", "k"))
        xv = T.view(X, shape=(K,), axes=("k",))
        yv = T.view(Y, shape=(M,), axes=("m",))
        with T.Kernel(1):
            with T.level.rows(M, group=4) as rows:
                total = T.state((4,), "int32", init=0)
                with T.level.blocks(K, extent=32) as block:
                    weights = T.stage(T.load(W[rows, block]))
                    activation = T.stage(T.load(xv[block]))
                    partial = T.alloc_local((4, 8), "int16")
                    wide = T.alloc_local((4, 8), "int32")
                    contribution = T.alloc_local((4,), "int32")
                    T.mac_groups(weights.q, activation, partial, n=4, into="int16", over="k")
                    for r, g in T.Parallel(4, 8):
                        wide[r, g] = T.cast(partial[r, g], "int32")
                    T.reduce_sum(wide, contribution, dim=1)
                    for r in T.Parallel(4):
                        total[r] += contribution[r]
                T.copy(total, yv[rows])

    return main


if __name__ == "__main__":
    import torch

    m, k = 8, 64
    q = torch.randint(0, 16, (m, k // 32, 32), dtype=torch.uint8)
    w = (q[..., :16] | (q[..., 16:] << 4)).reshape(m, k // 2).contiguous()
    x = torch.randint(-128, 128, (k,), dtype=torch.int8)
    y = torch.empty(m, dtype=torch.int32)
    packed_dot_factory(m, k)(w, x, y)
    expected = (q.reshape(m, k).to(torch.int32) * x.to(torch.int32)).sum(1).to(torch.int32)
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    print("Encoding + typed partials + nested Level/state/stage: correct")
