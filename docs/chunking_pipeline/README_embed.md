# RAG Pipeline — Technical Decisions & Architecture

---

## 1. Pipeline Overview

The RAG chunking and ingestion pipeline is organized in two phases:

### Phase 1: Ingestion (Per-Document)

Each document is processed through a worker-parallel pipeline that extracts, chunks, embeds, and stores content:

| Stage   | Role                               | Component          | Output                                                  |
| ------- | ---------------------------------- | ------------------ | ------------------------------------------------------- |
| Extract | Parse PDF and structure content    | Docling            | Structured document with text blocks, tables, figures   |
| Chunk   | Split into parent/child chunks     | HybridChunker      | Parent chunks (sections) and child chunks (~512 tokens) |
| Embed   | Convert chunks to semantic vectors | Qwen3-Embedding-4B | 2560-dim vectors per child chunk                        |
| Store   | Persist vectors and metadata       | Milvus HNSW        | Indexed child chunks + parent metadata                  |

### Phase 2: Retrieval & Reranking (Per-Query)

Once a document is indexed, queries use a two-stage funnel to retrieve and score relevant chunks:

| Stage     | Role                           | Component          | Output                 |
| --------- | ------------------------------ | ------------------ | ---------------------- |
| Retrieval | Fast ANN search for candidates | Milvus HNSW        | Top 100 similar chunks |
| Reranking | Precise cross-encoder scoring  | BGE-Reranker-v2-M3 | Top 10 chunks for LLM  |

---

## 2. Ingestion Pipeline

### Worker-Parallel Extraction & Chunking

All N workers extract and chunk independently using separate OS processes. This is critical because PDF extraction (Docling) is CPU-heavy and benefits from true parallelism.

**Worker flow (per page range):**

1. **Extract**: Docling processes pages `[start → end]` → structured markdown
2. **Chunk**: Split into parent (heading-anchored) and child (leaf-level) chunks
3. **Embed**: Generate embeddings in batches
4. **Store**: Append embeddings and metadata to Milvus (singleton connection)

Workers operate independently with no synchronization except the shared Milvus connection (singleton `Store` instance).

### Why Embed in Workers, Not After Merge?

**Before** (file-based): Extract → Chunk → Write temp files → (merge all files) → Embed → Store

- Creates intermediate files on disk for every worker
- Merge stage required for concatenation logic
- Embedding happens centrally after all files written
- I/O bound and scales poorly with worker count

**After** (in-memory): Extract → Chunk → Embed → Store (parallel, per-worker)

- No intermediate files; data stays in memory
- Each worker completes full lifecycle independently
- Embedding parallelized across N workers
- Milvus handles concurrent appends efficiently
- Lower disk footprint, faster wall-clock time

This change eliminated the merge stage entirely while improving throughput.

---

## 3. Embedding — Qwen3-Embedding-4B

Embedding is the process of converting a text chunk into a vector of numbers that represents its meaning. Two chunks that discuss the same concept will have vectors that are geometrically close to each other, which is what makes semantic search possible.

### Why Qwen3-Embedding-4B

Qwen3-Embedding-4B is a state-of-the-art open embedding model that ranks among the top performers on multilingual retrieval benchmarks. It is self-hosted on the project infrastructure, which means there is no dependency on an external API, no per-token cost, and full control over availability. It produces 2560-dimensional vectors — a dimensionality that offers excellent semantic resolution without excessive storage overhead. The model handles Arabic, French, and English content natively, which is directly relevant for the document corpus this system will process.

> **Deployment** — The model is served at `http://102.222.177.32:8002/v1/` via an OpenAI-compatible API. This means it can be called with any standard HTTP client using the same interface as commercial embedding APIs, with no vendor lock-in and no external data leaving the infrastructure.

### Configuration

| Parameter        | Value                                                |
| ---------------- | ---------------------------------------------------- |
| Model            | Qwen3-Embedding-4B                                   |
| Endpoint         | http://102.222.177.32:8002/v1/                       |
| Vector dimension | 2560                                                 |
| Normalisation    | All vectors normalised to unit length before storage |
| Batch size       | Configurable per worker (default: 32)                |
| Retry logic      | Exponential backoff on API failures                  |

---

## 4. Storage — Milvus with HNSW

Storage persists both embedding vectors and hierarchical parent/child metadata for rapid retrieval at query time.

### Milvus Collections

**Children Collection** — Stores all leaf-level chunks with embeddings

- `chunk_id` (primary key): Unique identifier per chunk
- `parent_id`: Reference to parent chunk
- `doc_id`: Document identifier
- `content`: Chunk text (max 65K chars)
- `embedding`: Vector (2560 dims, FLOAT_VECTOR)
- `page_ref`: Source page number
- `token_estimate`: Approximate token count
- `element_type`: "paragraph", "table_row", or "figure"

**Parent Collection** — Stores section-level context chunks

- `parent_id` (primary key): Unique identifier per parent
- `doc_id`: Document identifier
- `full_content`: Complete section text
- `heading`: Section heading
- `page_no`: Starting page number
- `embedding`: Placeholder vector (not used for search, required by Milvus schema)

### Why HNSW

HNSW (Hierarchical Navigable Small World) is the indexation algorithm chosen for this project. It builds a layered navigation graph over all stored vectors, allowing the search to jump quickly to the right neighbourhood and then refine locally. It consistently delivers 97–99% recall — meaning it finds the true nearest neighbours almost every time — while keeping query latency under 10 milliseconds at the scale of 2 to 3 million chunks that this project will reach.

> **Why not other algorithms** — IVF-based algorithms partition the space into clusters and can miss relevant chunks sitting near cluster boundaries. DiskANN moves data to disk and introduces I/O latency that is unnecessary at our scale. HNSW keeps everything in memory, has no boundary artefacts, and achieves the highest recall of all options — making it the natural choice when the collection fits comfortably in RAM.

### HNSW Configuration

| Parameter        | Value                                                     |
| ---------------- | --------------------------------------------------------- |
| Index algorithm  | HNSW                                                      |
| Distance metric  | Cosine similarity — consistent with normalised vectors    |
| M                | 16 — connections per node, balances recall and memory     |
| efConstruction   | 200 — thoroughness of graph build                         |
| ef (search)      | 100 — exploration width at query time                     |
| Similarity floor | 0.45 — chunks below this score discarded before reranking |

---

## 5. Retrieval & Reranking — Query Time

Once a document is indexed, retrieval uses a two-stage funnel:

### Stage 1 — HNSW Retrieval

Retrieval scans the indexed vectors and returns the chunks whose embeddings are closest to the query embedding. The query is embedded using the same Qwen3-Embedding-4B model.

### Stage 2 — Reranking with BGE-Reranker-v2-M3

The retrieval stage is fast but approximate. It compares vectors that were encoded independently — the query never directly interacts with each document during that step. Reranking corrects this by reading the query and each candidate together and producing a precise relevance score for each pair. The top 10 by that score are passed to generation or application endpoints.

#### Why BGE-Reranker-v2-M3

BGE-Reranker-v2-M3, released by the Beijing Academy of AI, is one of the most reliable open reranking models available. It is multilingual, handling French, Arabic, and English without any language-specific configuration. It was trained on large-scale retrieval datasets with hard-negative pairs — examples specifically chosen to be misleadingly similar to the query — which makes it robust at distinguishing genuinely relevant passages from ones that merely share vocabulary. It runs efficiently on CPU for the batches of 100 pairs this pipeline produces, adding under 300 milliseconds of latency per query.

> **The two-stage logic** — Retrieval and reranking solve different problems. Retrieval handles scale — it must search millions of chunks in milliseconds, so it uses fast approximate methods. Reranking handles precision — it only ever sees 100 candidates, so it can afford a thorough joint analysis of query and document together. Neither stage alone is sufficient: retrieval without reranking has too many false positives; reranking without retrieval would be too slow to run on the full collection.

### Reranking Configuration

| Parameter | Value                                                     |
| --------- | --------------------------------------------------------- |
| Model     | BAAI/bge-reranker-v2-m3                                   |
| Input     | 100 (query, chunk) pairs from HNSW retrieval              |
| Output    | Relevance score per pair; top 10 selected                 |
| Languages | Multilingual — Arabic, French, English natively supported |
| Latency   | Under 300ms for batch of 100 pairs on CPU                 |

---

## 6. End-to-End Retrieval Flow (Query Time)

| Step | Operation                             | From                                | To                                   |
| ---- | ------------------------------------- | ----------------------------------- | ------------------------------------ |
| 1    | Embed query with Qwen3-Embedding-4B   | Natural language question           | 1 vector of 2560 dimensions          |
| 2    | HNSW cosine search in Milvus (ef=100) | Query vector vs. all indexed chunks | Top 100 candidates                   |
| 3    | Apply similarity floor (>= 0.45)      | 100 candidates                      | Variable — low-signal chunks removed |
| 4    | BGE-Reranker-v2-M3 cross-scoring      | Remaining (query, chunk) pairs      | Relevance score per pair             |
| 5    | Select top 10 by reranker score       | Scored candidates                   | 10 chunks for downstream use         |

---

## 7. Testing & Validation

The pipeline includes optional retrieval example tests that can be run after ingestion:

```bash
python pipeline.py --test
```

This runs 10 predefined test queries and logs the retrieved chunks and reranking scores. Outputs are persisted as JSON for offline analysis.

---

## 8. Why These Choices Work Together

Each component was chosen to complement the others. Qwen3-Embedding-4B produces high-quality multilingual vectors that give HNSW accurate distances to navigate. HNSW's near-perfect recall ensures the reranker receives a candidate set that almost certainly contains all truly relevant chunks. BGE-Reranker-v2-M3 then applies precise joint scoring to surface the 10 passages that best answer the query, filtering out the approximate false positives that the retrieval stage cannot avoid by design.

The result is a pipeline that is fast enough for production use — total ingestion time measured per worker × pages (parallelized), and retrieval + reranking latency under 400 milliseconds per query — while being accurate enough to support citation-quality sourcing in generated text. Every affirmation in the final output can be traced back to one of the 10 retrieved passages, and those passages were selected by two successive stages of relevance filtering.

> **Scale note** — At the expected project scale of 500 documents of 300+ pages each, the collection will contain approximately 2 to 3 million chunks. HNSW handles this comfortably in memory. Should the system be extended to index significantly larger repositories in the future, the migration path is to DiskANN, which moves graph storage to SSD while preserving comparable recall. No changes to the embedding model or reranker would be required for such a migration.

---

## 9. Architecture Decisions Log

### Latest Changes (Phase 2)

**Removed:**

- Intermediate file writes from extract/chunk/merge stages
- `merge.py` orchestration stage entirely
- `enable_embedding_store` feature flag
- `replace_existing_by_doc_id` parameter logic
- Dead config variables (`md_input_path`)

**Added:**

- In-worker embedding and storage (workers now end-to-end)
- Singleton `Store` pattern for centralized Milvus connection
- Optional `--test` flag in `pipeline.py` for retrieval examples
- Dedicated `test.py` module for test orchestration

**Result:** Cleaner, faster pipeline with no intermediate state on disk and true parallel ingest-to-index flow per worker.
