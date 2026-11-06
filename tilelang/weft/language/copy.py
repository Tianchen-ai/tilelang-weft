"""TileLang copies combined with explicit logical storage Views."""

from tilelang.language.copy_op import copy as _copy
from tilelang.utils.language import to_tile_region
from tvm.script.ir_builder import tirx as I

from .values import LogicalView, load, intrinsic


def copy(source, destination, **kwargs):
    if isinstance(source, LogicalView):
        source = load(source)
    if isinstance(destination, LogicalView):
        annotations = kwargs.pop("annotations", {}) or {}
        if kwargs:
            raise TypeError("logical View copies do not accept GPU transfer hints")
        if set(annotations) - {"cache_hint"} or annotations.get("cache_hint", "normal") not in ("normal", "streaming"):
            raise ValueError("logical View copy annotations support cache_hint='normal' or 'streaming'")
        return I.evaluate(intrinsic("store_value", to_tile_region(source), destination.var, **annotations))
    return _copy(source, destination, **kwargs)
