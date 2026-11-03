"""Explicit preparation creates a persistent layout consumed by another kernel."""

import tilelang
from tilelang.weft import language as T


@tilelang.encoding
class Bytes16:
    layout = (T.bitorder.lsb_first, T.byteorder.little)
    alignment = 1
    elements = 16
    q: T.Field("uint8", (16,))


@tilelang.derive
def InterleavedBytes(source: T.View[Bytes16], *, rows: int):
    return T.interleave(source, rows=rows)


@tilelang.jit(target="weft")
def prepare(M=8, R=4):
    @T.prim_func
    def main(Source: T.Tensor((M, 16), "uint8"), Prepared: T.Tensor((M * 16,), "uint8")):
        with T.Kernel(1):
            for row, byte in T.Parallel(M, 16):
                Prepared[(row // R) * R * 16 + byte * R + row % R] = Source[row, byte]

    return main


@tilelang.jit(target="weft")
def consume(M=8, R=4):
    @T.prim_func
    def main(Prepared: T.Tensor((M * 16,), "uint8"), Output: T.Tensor((M, 16), "uint8")):
        W = T.view(Prepared, shape=(M, 16), encoding=InterleavedBytes[R], axes=("m", "k"))
        output = T.view(Output, shape=(M, 16), axes=("m", "k"))
        with T.Kernel(1):
            with T.level.rows(M, group=R) as row:
                with T.level.blocks(16, extent=16) as block:
                    values = T.load(W[row, block])
                    T.copy(values.q, output[row, block])

    return main


if __name__ == "__main__":
    import torch

    source = torch.arange(128, dtype=torch.uint8).reshape(8, 16)
    prepared, output = torch.empty(128, dtype=torch.uint8), torch.empty_like(source)
    prepare()(source, prepared)
    consume()(prepared, output)
    torch.testing.assert_close(output, source, atol=0, rtol=0)
    print("Persistent interleave: separate preparation + typed derived consumer, correct")
