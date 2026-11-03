"""Native TileLang tuning, followed by an explicit deployable artifact export."""

from pathlib import Path
import argparse

import tilelang
from tilelang.weft import language as T


@tilelang.autotune(configs=[{"block": 16}, {"block": 32}], warmup=1, rep=3)
@tilelang.jit(target="weft")
def tuned_affine(N=16, block=8):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32"), scale: T.float32, bias: T.float32):
        with T.Kernel(T.ceildiv(N, block)) as tile:
            for lane in T.Parallel(block):
                i = tile * block + lane
                if i < N:
                    Y[i] = X[i] * scale + bias

    return main


if __name__ == "__main__":
    import torch
    from weft_core.artifact import NativeArtifact

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    x, y = torch.arange(16, dtype=torch.float32), torch.empty(16)
    from tilelang.autotuner import set_autotune_inputs

    with set_autotune_inputs(x, y, 2.0, 1.0):
        kernel = tuned_affine()
    kernel(x, y, 2.0, 1.0)
    torch.testing.assert_close(y, x * 2 + 1)
    kernel.export_native(args.output)
    loaded = NativeArtifact.load(args.output)
    loaded.launch((1,), [x, y, 2.0, 1.0])
    torch.testing.assert_close(y, x * 2 + 1)
    print(f"autotune and native export: correct; {args.output}")
