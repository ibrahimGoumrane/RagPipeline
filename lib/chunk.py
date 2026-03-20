from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument, TextItem , SectionHeaderItem, TableItem, PictureItem

from lib.models.main import ChunkRunOutput
from lib.utils.logger import get_logger


class Chunker:
    def __init__(
        self,
        doc_id: str | None = None,
        max_words: int | None = None,
        min_words: int | None = None,
    ):
        self.doc_id = doc_id or "default-doc"
        self.logger = get_logger(name="Chunker", log_level=logging.INFO)
        self.max_words = max_words
        self.min_words = min_words
        self._init_buffer()

    def _init_buffer(self):
        """Initializes or resets the chunking buffer."""
        self.chunk_buffer = []
        self.buffer_word_count = 0
        self.buffer_parent_id = None
        self.buffer_page_no = None

    def _flush_buffer(self, children_list: list) -> None:
        """Flushes the current buffer contents into a new child chunk and appends it."""
        if self.chunk_buffer and self.buffer_parent_id is not None and self.buffer_page_no is not None:
            merged_text = "\n\n".join(self.chunk_buffer)
            child = self._create_child_chunk(
                self.buffer_parent_id, self.buffer_page_no, merged_text, "text"
            )
            children_list.append(child)
        self._init_buffer()

    def _add_to_buffer_and_flush(
        self, 
        content: str, 
        parent_id: str, 
        page_no: int, 
        children_list: list
    ) -> None:
        item_word_count = len(content.split())

        # Flush if adding this exceeds max_words and buffer has content
        if self.max_words and self.buffer_word_count + item_word_count > self.max_words and self.chunk_buffer:
            self._flush_buffer(children_list)
            self.buffer_parent_id = parent_id
            self.buffer_page_no = page_no

        self.chunk_buffer.append(content)
        self.buffer_word_count += item_word_count

        # Flush if min_words is reached
        if self.min_words and self.buffer_word_count >= self.min_words:
            self._flush_buffer(children_list)
            self.buffer_parent_id = parent_id
            self.buffer_page_no = page_no

    def _create_child_chunk(
        self, parent_id: str, page_no: int, content: str, element_type: str
    ) -> dict[str, Any]:
        return {
            "chunk_id": str(uuid.uuid4()),
            "parent_id": parent_id,
            "element_type": element_type,
            "page_ref": page_no,
            "doc_id": self.doc_id,
            "content": content.strip(),
            "content_for_embedding": f"Page: {page_no}\n{content.strip()}",
            "token_estimate": len(content.split()),
        }

    def run_from_docling_document(
        self,
        doc: DoclingDocument,
        exclude_pages: set[int] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Chunk from a DoclingDocument, grouping items by page."""
        exclude_pages = exclude_pages or set()
        pages_content = {}

        for item, _ in doc.iterate_items():  
            try:
                page_no = None
                if hasattr(item, "prov") and item.prov:
                    page_no = item.prov[0].page_no

                if page_no is None:
                    self.logger.warning(f"Could not determine page number for item type '{type(item).__name__}'")
                    continue

                if page_no in exclude_pages:
                    continue

                if page_no not in pages_content:
                    pages_content[page_no] = []

                if isinstance(item, SectionHeaderItem):
                    pages_content[page_no].append({"type": "header", "content": item.text})
                elif isinstance(item, TextItem):
                    pages_content[page_no].append({"type": "text", "content": item.text})
                elif isinstance(item, TableItem):
                    try:
                        html = item.export_to_html(doc=doc)
                        pages_content[page_no].append({"type": "table", "content": html})
                    except Exception as e:
                        self.logger.warning(f"Failed to export table on page {page_no} to HTML: {e}")
                        pages_content[page_no].append({"type": "table", "content": "Unparsed Table"})
                elif isinstance(item, PictureItem):
                    caption = "Figure"
                    if hasattr(item, "captions") and item.captions:
                        caption = item.captions[0].text
                    pages_content[page_no].append({"type": "figure", "content": f"Figure : {caption}"})
            except Exception as e:
                self.logger.error(f"Error processing item on doc iteration: {e}", exc_info=True)

        parents = []
        children = []
        
        current_header = ""
        self._init_buffer()

        for page_no, items in sorted(pages_content.items()):
            parent_id = str(uuid.uuid4())
            page_text_lines = []
            
            if self.buffer_parent_id is None:
                self.buffer_parent_id = parent_id
                self.buffer_page_no = page_no

            for item in items:
                page_text_lines.append(item["content"])
                if not item["content"].strip():
                    continue

                if item["type"] == "header":
                    current_header = item["content"]
                    continue
                
                content = item["content"]
                if current_header:
                    content = f"{current_header}\n{content}"

                self._add_to_buffer_and_flush(content, parent_id, page_no, children)

            parents.append({
                "parent_id": parent_id,
                "doc_id": self.doc_id,
                "page_no": page_no,
                "full_content": "\n".join(page_text_lines),
            })

        # Flush any remaining content in the buffer
        self._flush_buffer(children)

        self.logger.info("Chunking done — %d page parents, %d children", len(parents), len(children))
        return parents, children

    def run_to_output(
        self,
        document: DoclingDocument,
        output_dir: str | None = None,
        exclude_pages: set[int] | None = None,
    ) -> ChunkRunOutput:
        """Process DoclingDocument directly and write outputs."""
        out_dir = Path(output_dir) / "chunks" if output_dir else Path("output/chunks")
        out_dir.mkdir(parents=True, exist_ok=True)

        parents, children = self.run_from_docling_document(document, exclude_pages=exclude_pages)

        vector_path = out_dir / "chunks_vector.json"
        parent_path = out_dir / "chunks_parent.json"

        vector_path.write_text(json.dumps(children, ensure_ascii=False, indent=2), encoding="utf-8")
        parent_path.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info("Outputs written to %s", out_dir)

        return ChunkRunOutput(
            chunks_vector=children,
            chunks_parent=parents,
            chunks_vector_path=str(vector_path),
            chunks_parent_path=str(parent_path),
        )
