"""Host declarations become explicit Encoding schema in native TIRx."""

import ast
import inspect
import textwrap

from weft_core.encoding import storage_declaration
from weft_core.schema import ArraySpec, PaddingSpec, grouped, bit_layers, pack_fields, bitorder, byteorder, lo_first, hi_first, padding

__all__ = (
    "Field",
    "EncodingDefinition",
    "encoding",
    "grouped",
    "bit_layers",
    "pack_fields",
    "bitorder",
    "byteorder",
    "lo_first",
    "hi_first",
    "padding",
)


def Field(dtype, shape=()):
    from tilelang.weft.frontend.importer import dtype as resolve_dtype

    return ArraySpec(resolve_dtype(dtype), tuple(shape))


class EncodingDefinition:
    @classmethod
    def from_schema(cls, schema):
        result = cls.__new__(cls)
        result.schema, result.__name__ = schema, schema.name
        return result

    def __init__(self, declaration, bindings):
        annotations = inspect.get_annotations(declaration, globals=bindings, locals=bindings, eval_str=True)
        tree = ast.parse(textwrap.dedent(inspect.getsource(declaration)))
        body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef))
        items = []
        for statement in body:
            if isinstance(statement, ast.AnnAssign):
                items.append((statement.target.id, annotations[statement.target.id]))
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = vars(declaration).get(name)
                if isinstance(value, PaddingSpec):
                    items.append((name, value))
        self.schema = storage_declaration(
            declaration.__name__, layout=declaration.layout, alignment=declaration.alignment, elements=declaration.elements, items=items
        )
        self.__name__ = declaration.__name__


def encoding(declaration):
    frame = inspect.currentframe()
    try:
        bindings = {**frame.f_back.f_globals, **frame.f_back.f_locals}
        return EncodingDefinition(declaration, bindings)
    finally:
        del frame
