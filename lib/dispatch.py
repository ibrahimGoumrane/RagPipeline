"""Dispatch module for orchestrating the pipeline stages.
Description:
    Take the pdf path and then subdivide the document based on the NUM_WORKERS
    So each worker will receive a chunk of the document to process in parallel.
Usage:
    dispatch = Dispatch("path/to/document.pdf", num_workers=4)
    result = dispatch.run()


"""

from __future__ import annotations

import pymupdf
from lib.models.main import DispatchRunOutput

class Dispatch:
    def __init__(self, pdf_path: str, num_workers: int):
        self.pdf_path = pdf_path
        self.num_workers = num_workers

    def _page_ranges(self) -> list[tuple[int, int]]:
        doc = pymupdf.open(self.pdf_path)
        n   = doc.page_count
        doc.close()

        size   = (n + self.num_workers - 1) // self.num_workers
        return [(i, min(i + size, n)) for i in range(0, n, size)]

    def run(self) -> DispatchRunOutput:
        ranges = self._page_ranges()
        return DispatchRunOutput(
            status="success",
            message=f"Dispatched {len(ranges)} chunks",
            chunks=ranges,
        )
    
if __name__ == "__main__":    
    dispatch = Dispatch("RFA2024-pages.pdf", num_workers=4)
    result = dispatch.run()
    print(result)    