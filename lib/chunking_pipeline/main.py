"""Class-based entrypoint for the chunking pipeline."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

from lib.config.main import Config
from lib.utils.logger import get_logger
from lib.utils.timer import Timer

from .dispatch import Dispatch
from .merge import Merge
from .work import Work


def _run_worker(worker_data: dict) -> Any:
    """Helper to instantiate and run a Work instance synchronously."""
    worker = Work(**worker_data)
    return worker.run()


class ChunkingPipeline:
    """Orchestrates dispatch, parallel work execution, and merge stages."""

    def __init__(self, config: Config):
        self.cfg = config
        self.logger = get_logger(name="Pipeline", log_level=logging.INFO)

    async def _map_in_processes(
        self,
        fn: Callable[[Any], Any],
        items: list[Any],
        max_workers: int,
    ) -> list[Any]:
        """Apply fn to each item concurrently in processes; keeps input order."""
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            tasks = [loop.run_in_executor(pool, fn, item) for item in items]
            return list(await asyncio.gather(*tasks))

    async def _run_async(self) -> None:
        num_workers = self.cfg.num_workers
        with Timer("pipeline_total"):
            self.logger.info("Dispatching document chunks for %s", self.cfg.pdf_path)
            with Timer("dispatch_stage"):
                dispatch = Dispatch(pdf_path=self.cfg.pdf_path, num_workers=num_workers)
                dispatch_out = dispatch.run()

            worker_args = [
                {
                    "pdf_path": self.cfg.pdf_path,
                    "output_dir": self.cfg.output_dir,
                    "doc_id": self.cfg.doc_id,
                    "max_words_per_chunk": self.cfg.max_words_per_chunk,
                    "min_words_per_chunk": self.cfg.min_words_per_chunk,
                    "use_image_processor": self.cfg.use_image_processor,
                    "model_api_url": self.cfg.model_api_url,
                    "model_api_model": self.cfg.model_api_model,
                    "accelerator_device": self.cfg.accelerator_device,
                    "accelerator_num_threads": self.cfg.accelerator_num_threads,
                    "description_api_url": self.cfg.description_api_url,
                    "description_api_key": self.cfg.description_api_key,
                    "description_api_model": self.cfg.description_api_model,
                    "start_page": start,
                    "end_page": end,
                }
                for start, end in dispatch_out.chunks
            ]

            self.logger.info("Running %d workers in parallel processes", len(worker_args))
            with Timer("work_stage_parallel"):
                worker_outputs = await self._map_in_processes(
                    _run_worker,
                    worker_args,
                    max_workers=num_workers,
                )

            self.logger.info("Merging outputs")
            vector_path = str(Path(self.cfg.output_dir) / "chunks" / "chunks_vector.json")
            parent_path = str(Path(self.cfg.output_dir) / "chunks" / "chunks_parent.json")

            with Timer("merge_stage"):
                merge = Merge(vector_path=vector_path, parent_path=parent_path)
                merge.run(worker_outputs)

            self.logger.info("Pipeline complete. Output saved to %s and %s", vector_path, parent_path)

    def run(self) -> None:
        """Public synchronous pipeline entrypoint."""
        asyncio.run(self._run_async())
