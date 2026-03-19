# RAG Pipeline (WIP)

This repository is being refocused as a Retrieval-Augmented Generation (RAG) pipeline project.

## Current status

Only the chunking stage is considered implemented for now.

Implemented now:

- PDF ingestion and parsing with Docling.
- Ordered Markdown reconstruction from extracted document elements.
- Chunking-oriented output generation to prepare retrieval-ready content.

Planned next:

- Embedding generation.
- Vector store indexing.
- Retrieval and answer generation.
- Evaluation and quality checks.

## Project goal

Build an end-to-end RAG pipeline that transforms raw PDF documents into searchable chunks and then into answerable knowledge for downstream applications.

## Requirements

- Python 3.10+
- `uv` installed
- Optional GPU for better OCR/performance

## Setup

```bash
uv sync
```

## Run

```bash
uv run python pipeline.py
```

## Environment variables

Configure runtime values in `.env`:

- `PDF_PATH`: Input PDF path.
- `OUTPUT_DIR`: Output directory (default: `output/docling`).
- `USE_IMAGE_PROCESSOR`: Enable image/figure description calls.
- `MODEL_API_URL`: API endpoint for your model provider.
- `MODEL_API_MODEL`: Model name used by your provider.
- `API_KEY`: API key for remote model calls.
- `USE_HIERARCHICAL_HEADINGS`: Enable heading hierarchy postprocessing.

## Output

Current output is written under `output/docling/`.

Primary artifact:

- `full_document.md`

## Notes

- The model API integration is provider-agnostic through `MODEL_API_URL` and `MODEL_API_MODEL`.
- If you do not need figure/image processing, set `USE_IMAGE_PROCESSOR=False`.
- This README will evolve as embedding, retrieval, and generation stages are added.
