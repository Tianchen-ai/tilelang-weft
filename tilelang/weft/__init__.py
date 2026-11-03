"""TileLang frontend for the shared Weft CPU compiler."""
from . import target as _target
from . import backend as backend
from .target import target

__all__ = ["target"]
