# Chunking Strategy (lib/chunking_pipeline/chunk.py)

This document summarizes the chunking logic implemented in lib/chunking_pipeline/chunk.py.

## Core Principles

- Chunk creation is chunk-first.
- Parent reference is the closest previous markdown heading line that starts with #.
- No content section is dropped.
- Small paragraph chunks are merged to avoid overly short fragments.
- HTML tables are split row-by-row into separate chunks.

## Processing Flow

1. Load and extract markdown from PDF via Docling (extract stage).
2. Build parent containers and content blocks in one pass:
   - Parent changes when a heading line is found.
   - Paragraph content accumulates into paragraph blocks.
   - Table HTML between `<table ...>` and `</table>` becomes a table block.
   - Figure markdown `![...](...) becomes a figure block.
3. Convert blocks into child chunks:
   - Paragraph blocks use sentence-window splitting (configured tokenizer).
   - Table blocks produce one chunk per row.
   - Figure blocks produce one chunk with optional figure_ref.
4. Merge small paragraph chunks:
   - Prefer forward merge with the next paragraph chunk in same parent.
   - If last chunk is still too small, merge backward.
5. Return outputs in-memory:
   - `ChunkRunOutput` model with `chunks_vector` (retrieval chunks) and `chunks_parent` (parent containers).
   - No intermediate file writes; data flows directly to embedding and storage stages.

## Chunk Types

- paragraph
- table_row
- figure

## Parent Output Rules

- Every parent is tied to one heading_path (closest heading at capture time).
- A parent is included in output only if it has non-empty content_lines.

## Configuration

Configuration is centralized in `lib/config/main.py` and loaded from environment variables:

- `MAX_WORDS_PER_CHUNK`: Target chunk size in words
- `TOKENIZER_MODEL`: HuggingFace tokenizer model for accurate word counting
- `PDF_PATH`: Input PDF file to process
- `DOC_ID`: Document identifier for all chunks in this batch

## Output Model

Chunks are returned as `ChunkRunOutput` dataclass with:

- `chunks_vector`: List of child chunks with embeddings (to be embedded and stored)
- `chunks_parent`: List of parent chunks (for hierarchical retrieval context)

## Logging

The chunker uses `lib/utils/logger.py` and logs:

- run start/end
- document extraction counts
- block extraction counts
- chunk splitting counts
- merge statistics
