"""GGML storage declarations from the single shared core schema catalog."""

from weft_core.formats.ggml import CATALOG
from tilelang.weft.language.encoding import EncodingDefinition

globals().update({name: EncodingDefinition.from_schema(schema) for name, schema in CATALOG.items()})
__all__ = tuple(CATALOG)
