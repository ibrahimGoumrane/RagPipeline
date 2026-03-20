"""Minimal async API examples for thread and process worker handling.

This file is intentionally small and focused on the core patterns:
- Run blocking I/O in threads
- Run CPU-heavy work in processes
- Fan out jobs concurrently and collect ordered results
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable


async def run_in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
	"""Run a sync function in the default thread pool."""
	loop = asyncio.get_running_loop()
	call = lambda: fn(*args, **kwargs)
	return await loop.run_in_executor(None, call)


async def run_in_process(
	pool: ProcessPoolExecutor,
	fn: Callable[..., Any],
	*args: Any,
	**kwargs: Any,
) -> Any:
	"""Run a sync function in a process pool (good for CPU-heavy tasks)."""
	loop = asyncio.get_running_loop()
	call = lambda: fn(*args, **kwargs)
	return await loop.run_in_executor(pool, call)


async def map_in_threads(
	fn: Callable[[Any], Any],
	items: list[Any],
	max_workers: int = 8,
) -> list[Any]:
	"""Apply fn to each item concurrently in threads; keeps input order."""
	loop = asyncio.get_running_loop()
	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		tasks = [loop.run_in_executor(pool, fn, item) for item in items]
		return list(await asyncio.gather(*tasks))


async def map_in_processes(
	fn: Callable[[Any], Any],
	items: list[Any],
	max_workers: int = 4,
) -> list[Any]:
	"""Apply fn to each item concurrently in processes; keeps input order."""
	loop = asyncio.get_running_loop()
	with ProcessPoolExecutor(max_workers=max_workers) as pool:
		tasks = [loop.run_in_executor(pool, fn, item) for item in items]
		return list(await asyncio.gather(*tasks))


# --- Tiny usage example ---
def cpu_heavy_square(x: int) -> int:
	return x * x


def io_like_upper(text: str) -> str:
	return text.upper()


async def demo() -> None:
	# Thread example (I/O-like)
	words = ["a", "b", "c"]
	thread_results = await map_in_threads(io_like_upper, words, max_workers=3)

	# Process example (CPU-like)
	nums = [1, 2, 3, 4]
	process_results = await map_in_processes(cpu_heavy_square, nums, max_workers=2)

	print("threads:", thread_results)
	print("processes:", process_results)


if __name__ == "__main__":
	asyncio.run(demo())
