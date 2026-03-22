"""Core library package for extraction and chunking."""

from .chunking_pipeline import ChunkingPipeline
from .config import get_config


__all__ = ["ChunkingPipeline", "get_config"]
