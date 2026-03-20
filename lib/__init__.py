"""Core library package for extraction and chunking."""

from .extract import DoclingExtractor
from .chunk import Chunker
from .dispatch import Dispatch
from .work import Work
from .merge import Merge
from .config import cfg
__all__ = ["DoclingExtractor", "Chunker", "Dispatch", "Work", "Merge", "cfg"]
