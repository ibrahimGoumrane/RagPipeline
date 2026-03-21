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
    output_file: str = Field(
        ...,
        description="Absolute or relative path to the exported document file."
    )
    output_dir: str = Field(
        ...,
        description="Directory containing all generated extraction outputs."
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
    chunks_vector_path: str = Field(
        ...,
        description="File path where vector chunks JSON is stored."
    )
    chunks_parent_path: str = Field(
        ...,
        description="File path where parent chunks JSON is stored."
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


class WorkRunOutput(ChunkRunOutput):
    """Output of an individual worker processing chunks."""

    worker_id: str = Field(
        ...,
        description="Unique identifier of the worker handling this task."
    )
    status: str = Field(
        default="success",
        description="Execution status of the worker (e.g., 'success', 'failed')."
    )


class MergeRunOutput(BaseModel):
    """Output of the merge stage."""

    status: str = Field(
        default="success",
        description="Merge operation status (e.g., 'success', 'failed')."
    )
    chunks_vector: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Combined list of all vector chunks after merging worker outputs."
    )
    chunks_parent: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Combined list of all parent chunks after merging worker outputs."
    )
    chunks_vector_path: str | None = Field(
        default=None,
        description="Path to the final merged vector chunks JSON file, if applicable."
    )
    chunks_parent_path: str | None = Field(
		default=None,
		description="Path to the final merged parent chunks JSON file, if applicable."
	)