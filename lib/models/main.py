"""Pydantic models for pipeline stage outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from docling_core.types.doc.document import DoclingDocument

class ExtractRunOutput(BaseModel):
	document: DoclingDocument
	output_file: str
	output_dir: str


class ChunkRunOutput(BaseModel):
	chunks_vector: list[dict[str, Any]]
	chunks_parent: list[dict[str, Any]]
	chunks_vector_path: str
	chunks_parent_path: str


class DispatchRunOutput(BaseModel):
	status: str = "success"
	message: str = "Dispatch completed"
	chunks: list[tuple[int, int]] = Field(default_factory=list)


class WorkRunOutput(ChunkRunOutput):
	worker_id: int
	status: str = "success"


class MergeRunOutput(BaseModel):
	status: str = "success"
	merged_file_path: str | None = None
