"""Dispatch module for orchestrating pipeline stages."""

from __future__ import annotations

import pymupdf

from lib.models.main import DispatchRunOutput


class Dispatch:
    def __init__(self, pdf_path: str, num_workers: int):
        self.pdf_path = pdf_path
        self.num_workers = num_workers

    def _page_chunks(self) -> list[tuple[int, int]]:
        doc = pymupdf.open(self.pdf_path)
        n = doc.page_count
        doc.close()

        size = (n + self.num_workers - 1) // self.num_workers
        chunks: list[tuple[int, int]] = []
        for i in range(1, n + 1, size):
            start = i
            end = min(i + size - 1, n)
            chunks.append((start, end))

        return chunks

    def run(self) -> DispatchRunOutput:
        chunks = self._page_chunks()
        return DispatchRunOutput(
            status="success",
            message=f"Dispatched {len(chunks)} chunks",
            chunks=chunks,
        )
