"""Explicit host composition of normal kernels with caller-owned workspace."""

from dataclasses import dataclass
from .catalog import get
from .linear import matmul


@dataclass(frozen=True)
class PackedProjection:
    preparation: object
    projection: object

    def prepare(self, weights, workspace, **tables):
        self.preparation(Weights=weights, Output=workspace, **tables)

    def compute(self, activation, workspace, output):
        self.projection(activation, workspace, output)

    def __call__(self, weights, activation, workspace, output, **tables):
        """Both preparation and compute execute within this explicit call."""
        self.prepare(weights, workspace, **tables)
        self.compute(activation, workspace, output)


def packed_projection(M, N, K, *, format="Q4_K", rows=1, block_m=2, block_n=4, block_k=32, engine="auto"):
    """Decode [N,K] into the supplied F32 workspace, then compute X @ W.T.

    A caller choosing persistent preparation invokes prepare/compute separately
    and owns workspace invalidation. This is two ordinary compiled programs.
    """
    program = get(format.lower() + "_dequantize")
    return PackedProjection(
        program.compile(M=N, K=K, rows=rows),
        matmul(M, N, K, block_m=block_m, block_n=block_n, block_k=block_k, engine=engine, transpose_weight=True),
    )
