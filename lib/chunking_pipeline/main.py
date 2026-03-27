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
from .store import Store
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
                    "doc_id": self.cfg.doc_id,
                    "max_words_per_chunk": self.cfg.max_words_per_chunk,
                    "tokenizer_model": self.cfg.tokenizer_model,
                    "use_image_processor": self.cfg.use_image_processor,
                    "model_api_url": self.cfg.model_api_url,
                    "model_api_model": self.cfg.model_api_model,
                    "model_api_key": self.cfg.api_key,
                    "accelerator_device": self.cfg.accelerator_device,
                    "accelerator_num_threads": self.cfg.accelerator_num_threads,
                    "description_api_url": self.cfg.description_api_url,
                    "description_api_key": self.cfg.description_api_key,
                    "description_api_model": self.cfg.description_api_model,
                    "embedding_api_url": self.cfg.embedding_api_url,
                    "embedding_api_key": self.cfg.embedding_api_key,
                    "embedding_model": self.cfg.embedding_model,
                    "embedding_dim": self.cfg.embedding_dim,
                    "embedding_batch_size": self.cfg.embedding_batch_size,
                    "milvus_host": self.cfg.milvus_host,
                    "milvus_port": self.cfg.milvus_port,
                    "milvus_children_collection": self.cfg.milvus_children_collection,
                    "milvus_parent_collection": self.cfg.milvus_parent_collection,
                    "milvus_metric_type": self.cfg.milvus_metric_type,
                    "milvus_hnsw_m": self.cfg.milvus_hnsw_m,
                    "milvus_hnsw_ef_construction": self.cfg.milvus_hnsw_ef_construction,
                    "milvus_search_ef": self.cfg.milvus_search_ef,
                    "start_page": start,
                    "end_page": end,
                }
                for start, end in dispatch_out.chunks
            ]

            retrieval_store = Store(
                host=self.cfg.milvus_host,
                port=self.cfg.milvus_port,
                children_collection=self.cfg.milvus_children_collection,
                parent_collection=self.cfg.milvus_parent_collection,
                vector_dim=self.cfg.embedding_dim,
                metric_type=self.cfg.milvus_metric_type,
                hnsw_m=self.cfg.milvus_hnsw_m,
                hnsw_ef_construction=self.cfg.milvus_hnsw_ef_construction,
                search_ef=self.cfg.milvus_search_ef,
            )
            # Clear once per run; workers append in parallel.
            retrieval_store.clear_doc(self.cfg.doc_id)

            self.logger.info("Running %d workers in parallel processes", len(worker_args))
            with Timer("work_stage_parallel"):
                worker_outputs = await self._map_in_processes(
                    _run_worker,
                    worker_args,
                    max_workers=num_workers,
                )

            self.logger.info("Workers complete: %d/%d succeeded", len(worker_outputs), len(worker_args))

            self.logger.info("Pipeline complete.")

    def run(self) -> None:
        """Public synchronous pipeline entrypoint."""
        asyncio.run(self._run_async())
