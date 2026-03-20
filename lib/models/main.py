"""Pydantic models for pipeline stage outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractRunOutput(BaseModel):
	markdown_content: str
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


class WorkRunOutput(BaseModel):
	status: str = "success"
	result_data: dict[str, Any] = Field(default_factory=dict)


class MergeRunOutput(BaseModel):
	status: str = "success"
	merged_file_path: str | None = None
