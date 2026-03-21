"""Chunker — splits a DoclingDocument into parent + child chunks for RAG."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import (
    DoclingDocument, PictureItem, SectionHeaderItem, TableItem, TextItem,
)

from lib.models.main import ChunkRunOutput
from lib.utils.logger import get_logger


class Chunker:
    def __init__(
        self,
        doc_id: str | None = None,
        max_words: int | None = None,
        min_words: int | None = None,
    ):
        self.doc_id    = doc_id or "default-doc"
        self.max_words = max_words
        self.min_words = min_words
        self.logger    = get_logger(name="Chunker", log_level=logging.INFO)

    # ── Buffer ────────────────────────────────────────────────────────────────

    def _empty_buffer(self) -> dict[str, Any]:
        return {"lines": [], "word_count": 0, "parent_id": None, "page_no": None, "header": None}

    def _flush(self, buf: dict, children: list) -> dict:
        """Emit a child chunk from the buffer, return a fresh empty buffer."""
        if buf["lines"] and buf["parent_id"] is not None:
            text = "\n\n".join(buf["lines"])
            if buf["header"]:
                text = f"{buf['header']}\n{text}"
            children.append(self._child(buf["parent_id"], buf["page_no"], text))
        return self._empty_buffer()

    def _child(self, parent_id: str, page_no: int, content: str) -> dict[str, Any]:
        content = content.strip()
        return {
            "chunk_id":              str(uuid.uuid4()),
            "parent_id":             parent_id,
            "element_type":          "text",
            "page_ref":              page_no,
            "doc_id":                self.doc_id,
            "content":               content,
            "content_for_embedding": f"Page: {page_no}\n{content}",
            "token_estimate":        len(content.split()),
        }

    # ── Page content extraction ───────────────────────────────────────────────

    def _collect_pages(self, doc: DoclingDocument) -> dict[int, list[dict]]:
        """Walk the DoclingDocument and group items by page number."""
        pages: dict[int, list[dict]] = {}

        for item, _ in doc.iterate_items():
            try:
                page_no = item.prov[0].page_no if getattr(item, "prov", None) else None
                if page_no is None:
                    self.logger.warning("No page number for '%s' — skipped", type(item).__name__)
                    continue

                if page_no not in pages:
                    pages[page_no] = []

                if isinstance(item, SectionHeaderItem):
                    pages[page_no].append({"type": "header", "content": item.text})

                elif isinstance(item, TextItem):
                    pages[page_no].append({"type": "text", "content": item.text})

                elif isinstance(item, TableItem):
                    try:
                        html = item.export_to_html(doc=doc)
                    except Exception:
                        html = "Unparsed Table"
                        self.logger.warning("Table export failed on page %d", page_no)
                    pages[page_no].append({"type": "table", "content": html})

                elif isinstance(item, PictureItem):
                    caption = (
                        item.captions[0].text
                        if getattr(item, "captions", None)
                        else "Figure"
                    )
                    pages[page_no].append({"type": "figure", "content": f"Figure : {caption}"})

            except Exception as exc:
                self.logger.error("Item iteration error: %s", exc, exc_info=True)

        return pages

    # ── Chunking ──────────────────────────────────────────────────────────────

    def run_from_docling_document(
        self, doc: DoclingDocument,
    ) -> tuple[list[dict], list[dict]]:
        pages    = self._collect_pages(doc)
        parents  : list[dict] = []
        children : list[dict] = []
        buf      = self._empty_buffer()
        current_header = ""

        for page_no, items in sorted(pages.items()):
            parent_id       = str(uuid.uuid4())
            page_text_lines : list[str] = []

            if buf["parent_id"] is None:
                buf["parent_id"] = parent_id
                buf["page_no"]   = page_no

            for item in items:
                page_text_lines.append(item["content"])
                if not item["content"].strip():
                    continue

                if item["type"] == "header":
                    current_header = item["content"]
                    if buf["lines"]:               # flush on section change
                        buf = self._flush(buf, children)
                        buf["parent_id"] = parent_id
                        buf["page_no"]   = page_no
                    continue

                content    = item["content"]
                word_count = len(content.split())

                # Flush if this item would push us over max_words
                if self.max_words and buf["word_count"] + word_count > self.max_words and buf["lines"]:
                    buf = self._flush(buf, children)
                    buf["parent_id"] = parent_id
                    buf["page_no"]   = page_no

                if not buf["lines"]:               # register header on fresh buffer
                    buf["header"]      = current_header
                    buf["word_count"] += len(current_header.split()) if current_header else 0

                buf["lines"].append(content)
                buf["word_count"] += word_count

                # Flush once min_words reached
                if self.min_words and buf["word_count"] >= self.min_words:
                    buf = self._flush(buf, children)
                    buf["parent_id"] = parent_id
                    buf["page_no"]   = page_no

            parents.append({
                "parent_id":    parent_id,
                "doc_id":       self.doc_id,
                "page_no":      page_no,
                "full_content": "\n".join(page_text_lines),
            })

        self._flush(buf, children)  # flush remainder
        self.logger.info("Chunking done — %d parents, %d children", len(parents), len(children))
        return parents, children

    # ── Output ────────────────────────────────────────────────────────────────

    def run_to_output(
        self,
        document: DoclingDocument,
        output_dir: str | None = None,
    ) -> ChunkRunOutput:
        out_dir = Path(output_dir) / "chunks" if output_dir else Path("output/chunks")
        out_dir.mkdir(parents=True, exist_ok=True)

        parents, children = self.run_from_docling_document(document)

        vector_path = out_dir / "chunks_vector.json"
        parent_path = out_dir / "chunks_parent.json"

        vector_path.write_text(json.dumps(children, ensure_ascii=False, indent=2), encoding="utf-8")
        parent_path.write_text(json.dumps(parents,  ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info("Outputs written to %s", out_dir)

        return ChunkRunOutput(
            chunks_vector=children,
            chunks_parent=parents,
            chunks_vector_path=str(vector_path),
            chunks_parent_path=str(parent_path),
        )