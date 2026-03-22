# PDF Ingestion — Parallel Worker Strategy

---

## What we are building

A standalone Python utility that takes a single PDF and processes it in parallel using N workers.
Each worker is responsible for a slice of pages and operates fully independently.
Once all workers finish, their outputs are concatenated into two final files — one for parent chunks
and one for child chunks — ready for vectorisation.

---

## What each worker does

Each worker receives a page range `[n → n+m]` and performs three operations on that slice, in sequence, completely on its own.

| Step        | Operation                                                                                                                                                             |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EXTRACT** | Run Docling on its page range. Produce a structured representation of every text block, table, and heading found in those pages.                                      |
| **CHUNK**   | Apply the semantic chunking strategy to the extracted content. Split into parent chunks (section level) and child chunks (leaf level, ~512 tokens, 64-token overlap). |
| **SAVE**    | Write two temporary JSON files: one for its parent chunks, one for its child chunks. Files are named by worker index to avoid collisions.                             |

---

## Concatenation — final output

After all N workers complete, a single merge step concatenates all temporary files in page order.
Page-order sorting is critical: child chunks carry a `page_ref` field and the parent-child
relationship must be consistent across the boundary between adjacent workers.

```
worker_0_parents.json  ─┐
worker_1_parents.json   ├─► sort by page range → concatenate → parents.json
worker_N_parents.json  ─┘

worker_0_children.json ─┐
worker_1_children.json  ├─► sort by page range → concatenate → children.json
worker_N_children.json ─┘
```

---

## Output file structure

The two final JSON files are linked by `parent_id`.
The children file is what gets embedded into ChromaDB.
The parents file is retrieved at query time to provide broader context around a matched child chunk.

**`parents.json`**

```json
[
  {
    "chunk_id":   "uuid4",
    "doc_id":     "uuid4",
    "doc_title":  "...",
    "page_range": [n, m],
    "token_count": 0,
    "children":   ["uuid4"]
  }
]
```

**`children.json`**

```json
[
  {
    "chunk_id": "uuid4",
    "parent_id": "uuid4",
    "doc_id": "uuid4",
    "doc_title": "...",
    "text": "...",
    "token_count": 0,
    "page_ref": 0
  }
]
```

---

## Concurrency approach

Workers run in a `ProcessPoolExecutor` — separate OS processes, not threads. This is the right
choice because Docling extraction is CPU-heavy and Python's GIL would prevent real parallelism
with threads. The orchestrator dispatches all workers at once and waits for all of them to finish
before merging. The utility is written as a plain async coroutine so it can be plugged into
FastAPI later without any changes — the route simply calls it via `loop.run_in_executor()`
and returns a job ID immediately.

---

## Accelerator options

Docling extraction can be run with explicit accelerator settings per worker.

- `ACCELERATOR_DEVICE`: `AUTO`, `CPU`, `MPS`, `CUDA`, or `XPU`
- `ACCELERATOR_NUM_THREADS`: Number of worker threads for accelerator execution

These values are passed into `PdfPipelineOptions.accelerator_options` via:

```python
AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU)
```

Notes:

- `CUDA` and `XPU` require compatible hardware and runtime.
- `CPU` mode works everywhere.

---

## Design rules

| DO                                                          | DON'T                                                       |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| One `ProcessPoolExecutor` shared across all requests        | No `ThreadPoolExecutor` for Docling — GIL kills parallelism |
| Workers communicate only via tmp files — no shared memory   | No blocking calls on the async event loop                   |
| Merge strictly in page order after all workers finish       | No shared mutable state between worker processes            |
| Utility has zero FastAPI imports — framework-agnostic       | No partial merges — wait for all N workers before concat    |
| Each worker writes to its own tmp file (no write conflicts) | No framework coupling in the utility layer                  |

---

Runtime progress is emitted through the project logger rather than `print` statements.
