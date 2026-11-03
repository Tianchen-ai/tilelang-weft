"""Layout builders use native TileLang macros; instances describe stored bytes."""

from dataclasses import dataclass
import inspect
import json

from tvm.script.ir_builder import tirx as I
from tilelang.language.eager.builder import Builder, macro
from .encoding import EncodingDefinition


@dataclass(frozen=True)
class ViewAnnotation:
    encoding: object


class View:
    @classmethod
    def __class_getitem__(cls, encoding):
        return ViewAnnotation(encoding)


class DerivedEncodingDefinition:
    def __init__(self, function):
        self.builder = macro(function)
        parameters = tuple(inspect.signature(function).parameters.values())
        if not parameters or not isinstance(self.builder.annotations.get(parameters[0].name), ViewAnnotation):
            raise TypeError("derive requires a source View[BaseEncoding] annotation")
        self.base = self.builder.annotations[parameters[0].name].encoding
        if not isinstance(self.base, EncodingDefinition):
            raise TypeError("derive source must identify a base Encoding")
        if any(p.kind != inspect.Parameter.KEYWORD_ONLY for p in parameters[1:]):
            raise TypeError("derive layout parameters are keyword-only static integers")
        self.parameters = tuple(p.name for p in parameters[1:])
        self.__name__ = function.__name__

    def __getitem__(self, values):
        values = values if isinstance(values, tuple) else (values,)
        if len(values) != len(self.parameters) or any(type(value) is not int for value in values):
            raise TypeError("derived Encoding instance must bind every static integer parameter")
        return DerivedEncodingInstance(self, values)


@dataclass(frozen=True)
class DerivedEncodingInstance:
    family: DerivedEncodingDefinition
    parameters: tuple

    @property
    def schema(self):
        return self.family.base.schema

    @property
    def identity(self):
        return self.family.__name__ + "[" + ",".join(map(str, self.parameters)) + "]"

    def description(self):
        return {
            "derived": {
                "family": self.family.__name__,
                "identity": self.identity,
                "parameters": self.parameters,
                "source": self.schema.to_dict(),
            }
        }

    def declare(self, shape, axes):
        from .values import intrinsic, bind, LogicalView

        builder = Builder.current()
        with builder.with_frame(I.sblock("weft_derive")):
            I.sblock_attr(
                {
                    "weft.derive": json.dumps(self.description()),
                    "weft.derive.shape": shape,
                    "weft.derive.axes": axes,
                    "weft.derive.parameters": self.family.parameters,
                }
            )
            argument = LogicalView(bind(intrinsic("derive_argument")), shape, axes, self.family.base)
            result = self.family.builder(argument, **dict(zip(self.family.parameters, self.parameters)))
            if not isinstance(result, LogicalView):
                raise TypeError("derive builder must return a logical View")
            I.evaluate(intrinsic("derive_yield", result.var))


def derive(function):
    return DerivedEncodingDefinition(function)


def interleave(view, *, rows, field_chunks=None):
    from .values import intrinsic, bind, LogicalView

    if type(rows) is not int or rows <= 0:
        raise TypeError("interleave rows is a positive static integer")
    fields = view.encoding.schema.fields
    if field_chunks is not None and set(field_chunks) != {field.name for field in fields}:
        raise ValueError("interleave field_chunks must name every Encoding field")
    chunks = [1 if field_chunks is None else field_chunks[field.name] for field in fields]
    if any(type(chunk) is not int or chunk <= 0 for chunk in chunks):
        raise ValueError("interleave field chunks must be positive byte counts")
    result = bind(intrinsic("interleave", view.var, rows, field_chunks=chunks))
    return LogicalView(result, view.shape, view.axes, view.encoding)
