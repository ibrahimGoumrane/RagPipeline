# Worker Architecture — End-to-End Processing

---

## Overview

The pipeline processes a single PDF in parallel using N workers. Each worker is responsible for a page range and operates fully independently, executing the complete ingestion lifecycle for its slice.

---

## What Each Worker Does

Each worker receives a page range `[start → end]` and performs the complete pipeline in sequence:

| Stage       | Operation                                                            | Input                             | Output                                                |
| ----------- | -------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------- |
| **EXTRACT** | Run Docling on page range, extract text, tables, figures             | PDF pages `[start → end]`         | Structured document representation                    |
| **CHUNK**   | Apply semantic chunking strategy; split into parent and child chunks | Extracted content                 | `ChunkRunOutput` with parent and child chunk lists    |
| **EMBED**   | Generate embeddings for all child chunks using embedding API         | Child chunks                      | Chunks with embeddings (2560-dim vectors)             |
| **STORE**   | Insert chunked data and embeddings into Milvus collections           | Embedded chunks + parent metadata | Persisted in Milvus (children and parent collections) |

---

## Parallel Execution

All N workers run concurrently in separate OS processes using `ProcessPoolExecutor`. This approach is essential because PDF extraction (Docling) is CPU-intensive and Python's GIL prevents real parallelism with threads.

**Concurrency model:**

- Orchestrator (main process) dispatches all workers at once
- Workers process independently with no shared mutable state
- No inter-worker communication — each worker writes directly to Milvus
- Orchestrator waits for all workers to complete before proceeding

---

## Why Processes, Not Threads?

| Aspect                     | ProcessPoolExecutor   | ThreadPoolExecutor       |
| -------------------------- | --------------------- | ------------------------ |
| GIL constraint             | ❌ Not limited by GIL | ✅ Blocked by GIL        |
| PDF extraction parallelism | ✅ True parallelism   | ❌ Serialized extraction |
| Memory overhead            | ⚠️ Higher per-process | ✅ Shared memory         |
| Use case fit               | ✅ CPU-intensive work | ❌ I/O-bound only        |

Docling is CPU-heavy, making process-based parallelism the correct choice.

---

## Shared Store (Singleton)

All workers share a single `Store` instance (singleton pattern) that maintains one connection to Milvus. This ensures:

- Efficient connection pooling
- Single data source of truth in Milvus
- No duplicate or conflicting data writes
- Clean append-only semantics per worker

Before workers start, the orchestrator clears all records for the current `doc_id` in Milvus. Workers then append independently without coordination.

---

## Data Flow

```
PDF file
   ↓
Dispatch → Page ranges [0-50], [50-100], ..., [N-m, N]
   ↓
Worker 1          Worker 2          Worker N
EXTRACT ─→        EXTRACT ─→        EXTRACT ─→
CHUNK ──→         CHUNK ──→         CHUNK ──→
EMBED ──→         EMBED ──→         EMBED ──→
└─→ STORE ────────┬──→ STORE ────────┴──→ STORE ────→ Milvus (single connection)
                  ↓
            All chunks indexed
```

---

## Configuration

Each worker receives the same configuration for embedding and storage:

- **Embedding**: API endpoint URL, model name, batch size, API key
- **Storage**: Milvus host/port, collection names, vector dimension, HNSW parameters
- **Processing**: Tokenizer model, max words per chunk, accelerator settings

---

## Worker Independence

Workers are fully independent with no shared state except the Milvus connection (via singleton Store):

| Aspect                | Isolation                              |
| --------------------- | -------------------------------------- |
| Memory heap           | ✅ Separate per process                |
| File handles          | ✅ Independent                         |
| Extracted document    | ✅ Local to worker                     |
| Chunk generation      | ✅ No synchronization needed           |
| Embedding computation | ✅ Batch per worker                    |
| Milvus writes         | ✅ Append-only (cleared once at start) |

---

## Error Handling

Each worker processes independently; if one worker fails:

- That worker's page range is incomplete in Milvus
- Other workers continue unaffected
- Partial data is retrievable (chunks from successful workers remain indexed)
- Orchestrator logs success count (`M/N succeeded`)

---

## Design Rules

| DO                                               | DON'T                                                      |
| ------------------------------------------------ | ---------------------------------------------------------- |
| Use `ProcessPoolExecutor` for CPU-intensive work | Use `ThreadPoolExecutor` for Docling extraction            |
| Share one Milvus connection (singleton Store)    | Create duplicate Store connections (connection waste)      |
| Clear doc_id once before all workers start       | Clear per-worker (leaves gaps, overwrites)                 |
| Append-only writes from workers to Milvus        | Per-worker conditional deletes (synchronization nightmare) |
| Log progress from each worker independently      | Expect deterministic worker completion order               |
| Configure workers from centralized config        | Scatter config across worker code                          |

---

## Runtime Flow in Main

1. Load configuration
2. Initialize singleton Store (Milvus connection established once)
3. Clear `doc_id` from Milvus (fresh start for this ingestion)
4. Dispatch PDF into N page ranges
5. Launch N workers in ProcessPoolExecutor
6. Wait for all workers to complete
7. Log aggregate success count
8. (Optionally) Run retrieval examples for validation

---

## Scalability

At the expected project scale of 500 documents × 300+ pages each:

- **Total chunks**: 2–3 million
- **Collection size**: Manageable in Milvus RAM
- **Worker count**: Typically 8–16 (one per CPU core)
- **Query latency**: Sub-400ms (retrieval + reranking)

Adding more documents or workers requires no changes to the worker code; the singleton Store handles all writes efficiently.
