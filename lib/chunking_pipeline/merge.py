"""Merge stage for worker outputs."""

import json
import os

from lib.models.main import MergeRunOutput, WorkRunOutput


class Merge:
    def __init__(self, vector_path: str, parent_path: str):
        self.vector_path = vector_path
        self.parent_path = parent_path

    def _combine_chunks(self, outputs: list[WorkRunOutput]) -> tuple[list[dict], list[dict]]:
        merged_vector = []
        merged_parent = []

        for output in outputs:
            merged_vector.extend(output.chunks_vector)
            merged_parent.extend(output.chunks_parent)

        return merged_vector, merged_parent

    def _save_to_json(self, data: list[dict], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run(self, worker_outputs: list[WorkRunOutput]) -> MergeRunOutput:
        chunks_vector, chunks_parent = self._combine_chunks(worker_outputs)
        self._save_to_json(chunks_vector, self.vector_path)
        self._save_to_json(chunks_parent, self.parent_path)

        return MergeRunOutput(
            status="success",
            chunks_vector=chunks_vector,
            chunks_parent=chunks_parent,
            chunks_vector_path=self.vector_path,
            chunks_parent_path=self.parent_path,
        )
