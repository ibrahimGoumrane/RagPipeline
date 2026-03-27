"""Pydantic models for pipeline stage outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from docling_core.types.doc.document import DoclingDocument


class ExtractRunOutput(BaseModel):
    """Output of the extraction stage."""

    document: DoclingDocument = Field(
        ...,
        description="Parsed document object produced by the extraction pipeline."
    )


class ChunkRunOutput(BaseModel):
    """Output of the chunking stage."""

    chunks_vector: list[dict[str, Any]] = Field(
        ...,
        description="List of vector-ready chunks formatted for downstream vector storage (e.g., embeddings)."
    )
    chunks_parent: list[dict[str, Any]] = Field(
        ...,
        description="List of parent-level chunks containing metadata or hierarchical structure."
    )


class DispatchRunOutput(BaseModel):
    """Output of the dispatch stage."""

    status: str = Field(
        default="success",
        description="Dispatch operation status (e.g., 'success', 'failed')."
    )
    message: str = Field(
        default="Dispatch completed",
        description="Human-readable message describing the dispatch result."
    )
    chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="List of assigned chunk index ranges for workers (start_index, end_index)."
    )


class WorkRunOutput(BaseModel):
    """Output of an individual worker processing a page range end-to-end."""

    worker_id: str = Field(
        ...,
        description="Unique identifier of the worker handling this task."
    )
    status: str = Field(
        default="success",
        description="Execution status of the worker (e.g., 'success', 'failed')."
    )
