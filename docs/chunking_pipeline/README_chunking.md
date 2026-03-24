# Chunking Strategy (utils/chunk.py)

This document summarizes the chunking logic implemented in utils/chunk.py.

## Core Principles

- Chunk creation is chunk-first.
- Parent reference is the closest previous markdown heading line that starts with #.
- No content section is dropped.
- Small paragraph chunks are merged to avoid overly short fragments.
- HTML tables are split row-by-row into separate chunks.

## Processing Flow

1. Load markdown from MD_INPUT_PATH.
2. Build parent containers and content blocks in one pass:
   - Parent changes when a heading line is found.
   - Paragraph content accumulates into paragraph blocks.
   - Table HTML between <table ...> and </table> becomes a table block.
   - Figure markdown ![...](...) becomes a figure block.
3. Convert blocks into child chunks:
   - Paragraph blocks use sentence-window splitting.
   - Table blocks produce one chunk per row.
   - Figure blocks produce one chunk with optional figure_ref.
4. Merge small paragraph chunks:
   - Prefer forward merge with the next paragraph chunk in same parent.
   - If last chunk is still too small, merge backward.
5. Write outputs:
   - chunks_vector.json: retrieval chunks.
   - chunks_parent.json: documented parent containers.

## Chunk Types

- paragraph
- table_row
- figure

## Parent Output Rules

- Every parent is tied to one heading_path (closest heading at capture time).
- A parent is written to chunks_parent.json only if it has non-empty content_lines.

## Config Keys Used

- MD_INPUT_PATH
- OUTPUT_DIR
- DOC_ID
- MAX_WORDS_PER_CHUNK
- TOKENIZER_MODEL
- OVERLAP_SENTENCES

## Logging

The chunker uses utils/logger.py via get_logger and logs:

- run start/end
- markdown load
- block extraction counts
- table chunking counts
- output write summary
