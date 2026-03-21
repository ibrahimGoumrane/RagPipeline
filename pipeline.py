"""Pipeline entrypoint."""

import asyncio
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ProcessPoolExecutor

from lib.config.main import cfg
from lib.dispatch import Dispatch
from lib.work import Work
from lib.merge import Merge


async def map_in_processes(
    fn: Callable[[Any], Any],
    items: list[Any],
    max_workers: int = 8,
) -> list[Any]:
    """Apply fn to each item concurrently in processes; keeps input order."""
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        tasks = [loop.run_in_executor(pool, fn, item) for item in items]
        return list(await asyncio.gather(*tasks))


def _run_worker(worker_data: dict) -> Any:
    """Helper to instantiate and run a Work instance synchronously."""
    worker = Work(**worker_data)
    return worker.run()


async def amain() -> None:
    # Set workers default in case it isn't explicitly in cfg yet
    num_workers = getattr(cfg, "num_workers", 4)

    # 1. Dispatch stage: determine page chunks for the PDF based on workers
    print(f"Dispatching document chunks for {cfg.pdf_path}...")
    dispatch = Dispatch(pdf_path=cfg.pdf_path, num_workers=num_workers)
    dispatch_out = dispatch.run()

    # Prepare arguments to send to each worker process
    worker_args = [
        {
            "pdf_path": cfg.pdf_path,
            "output_dir": cfg.output_dir,
            "doc_id": cfg.doc_id,
            "max_words_per_chunk": cfg.max_words_per_chunk,
            "min_words_per_chunk": cfg.min_words_per_chunk,
            "use_image_processor": cfg.use_image_processor,
            "use_hierarchical_headings": cfg.use_hierarchical_headings,
            "model_api_url": cfg.model_api_url,
            "model_api_model": cfg.model_api_model,
            "description_api_url": cfg.description_api_url,
            "description_api_key": cfg.description_api_key,
            "description_api_model": cfg.description_api_model,
            "start_page": start,
            "end_page": end,
        }
        for start, end in dispatch_out.chunks
    ]

    # 2. Work stage: execute extraction and chunking concurrently using threading
    print(f"Running {len(worker_args)} workers in parallel processes...")
    worker_outputs = await map_in_processes(_run_worker, worker_args, max_workers=num_workers)

    # 3. Merge stage: combine the individual worker outputs into the final unified files
    print("Merging outputs...")
    vector_path = str(Path(cfg.output_dir) / "chunks" / "chunks_vector.json")
    parent_path = str(Path(cfg.output_dir) / "chunks" / "chunks_parent.json")

    merge = Merge(vector_path=vector_path, parent_path=parent_path)
    merge.run(worker_outputs)
    
    print(f"Pipeline complete! Output saved to:\n- {vector_path}\n- {parent_path}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()