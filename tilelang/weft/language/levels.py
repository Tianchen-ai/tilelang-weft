"""Explicit semantic Level regions, with separate state/stage initialization."""
from contextlib import contextmanager
from dataclasses import dataclass

from tvm import tirx
from tvm.script.ir_builder import tirx as I


@dataclass(frozen=True)
class Point:
    var: tirx.Var
    axis: str
    partition: int


class LevelFrame:
    def __init__(self, domain, partition, relation, axis):
        self.domain, self.partition, self.relation, self.axis = domain, partition, relation, axis
        self.phase, self.forbidden = None, set()
        self.incoming = {}

    @contextmanager
    def __tilelang_with__(self, builder):
        from tilelang.language.loop import serial
        domain = self.domain.partition if isinstance(self.domain, Point) else self.domain
        frame = serial(0, tirx.ceildiv(domain, self.partition), annotations={
            "weft.level.relation": self.relation, "weft.level.axis": self.axis,
            "weft.level.extent": domain, "weft.level.partition": self.partition,
        })
        builder.declaration_regions.append(self)
        try:
            with builder.with_frame(frame) as variable:
                yield Point(variable, self.axis, self.partition)
        finally:
            builder.declaration_regions.pop()

    @contextmanager
    def declarations(self, builder, phase, forbidden):
        previous = self.phase, self.forbidden
        self.phase, self.forbidden = phase, set(forbidden)
        try:
            with builder.with_frame(I.sblock("weft_" + phase)):
                I.sblock_attr({"weft.level.region": phase})
                yield
        finally:
            self.phase, self.forbidden = previous


class LevelConstructor:
    __tilelang_region_declarations__ = True

    def __init__(self, relation, axis):
        self.relation, self.axis = relation, axis

    def __call__(self, domain, *, group=None, extent=None, axis=None):
        partition = group if self.relation in ("rows", "cols") else extent
        if type(partition) is not int or partition <= 0:
            raise ValueError("Level partition requires a positive compile-time integer")
        name = domain.axis if isinstance(domain, Point) else (self.axis if axis is None else axis)
        return LevelFrame(domain, partition, self.relation, name)


class _Levels:
    rows = LevelConstructor("rows", "m")
    cols = LevelConstructor("cols", "n")
    blocks = LevelConstructor("blocks", "k")
    tiles = LevelConstructor("tiles", "k")
    subtiles = LevelConstructor("subs", "k")


level = _Levels()
