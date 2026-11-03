"""Additional scalar CPU atomics use the same typed pointer and ordering ABI."""

from tilelang.language import access_ptr
from tilelang.language.atomic import _MEMORY_ORDER_ID_MAP
from tilelang.language.utils import get_extent
from tvm import tirx
from .values import intrinsic


def _atomic(kind, destination, value, compare=0, *, memory_order):
    if get_extent(destination) is not None:
        raise TypeError("CPU atomic extensions require an explicitly indexed scalar address")
    if memory_order not in _MEMORY_ORDER_ID_MAP:
        raise ValueError(f"unsupported atomic ordering {memory_order!r}")
    return intrinsic(
        "atomic_rmw",
        access_ptr(destination, "rw"),
        value,
        compare,
        _MEMORY_ORDER_ID_MAP[memory_order],
        dtype=destination.dtype,
        kind=tirx.StringImm(kind),
    )


def atomic_cas(destination, compare, value, *, memory_order="acq_rel"):
    return _atomic("cas", destination, value, compare, memory_order=memory_order)


def atomic_xchg(destination, value, *, memory_order="acq_rel"):
    return _atomic("xchg", destination, value, memory_order=memory_order)


def atomic_and(destination, value, *, memory_order="acq_rel"):
    return _atomic("and", destination, value, memory_order=memory_order)


def atomic_xor(destination, value, *, memory_order="acq_rel"):
    return _atomic("xor", destination, value, memory_order=memory_order)
