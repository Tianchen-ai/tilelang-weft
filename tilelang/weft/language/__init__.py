"""Common TileLang syntax with explicit Weft CPU model extensions."""

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from .kernel import Kernel
from .gemm import gemm, reduce_dot
from .encoding import Field, grouped, bit_layers, pack_fields, bitorder, byteorder, lo_first, hi_first, padding
from .levels import level
from .values import view, load, state, stage, mac_groups, extent, narrow, reshape
from .copy import copy
from .helpers import macro
from .derived import View, interleave
from .loop import Pipelined
from .atomic import atomic_cas, atomic_xchg, atomic_and, atomic_xor
from .reduction import reduce, scan, sum_pairs

__tilelang_dialect__ = "weft"
__all__ = [
    *_COMMON_ALL,
    "Kernel",
    "gemm",
    "reduce_dot",
    "Field",
    "grouped",
    "bit_layers",
    "pack_fields",
    "bitorder",
    "byteorder",
    "lo_first",
    "hi_first",
    "padding",
    "level",
    "view",
    "reshape",
    "extent",
    "narrow",
    "load",
    "state",
    "stage",
    "mac_groups",
    "copy",
    "macro",
    "View",
    "interleave",
    "Pipelined",
    "atomic_cas",
    "atomic_xchg",
    "atomic_and",
    "atomic_xor",
    "reduce",
    "scan",
    "sum_pairs",
]
