"""Chunker — splits a DoclingDocument into parent + child chunks for RAG.

Parent = one per section header.
Children = content chunks under that section, split by word budget.
"""

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

        # Seed a root parent for any content before the first header
        current_parent = {
            "parent_id":    str(uuid.uuid4()),
            "doc_id":       self.doc_id,
            "heading":      "[Document root]",
            "page_no":      None,
            "full_content": [],
        }
        parents.append(current_parent)
        buf["parent_id"] = current_parent["parent_id"]

        for page_no, items in sorted(pages.items()):
            if current_parent["page_no"] is None:
                current_parent["page_no"] = page_no  # set root parent page on first page

            for item in items:
                content = item["content"]
                if not content.strip():
                    continue

                # ── New section header → new parent ───────────────────────────
                if item["type"] == "header":
                    # flush whatever was buffered under the previous parent
                    buf = self._flush(buf, children)

                    current_parent = {
                        "parent_id":    str(uuid.uuid4()),
                        "doc_id":       self.doc_id,
                        "heading":      content,
                        "page_no":      page_no,
                        "full_content": [],
                    }
                    parents.append(current_parent)

                    buf["parent_id"] = current_parent["parent_id"]
                    buf["page_no"]   = page_no
                    buf["header"]    = content
                    continue

                # ── Accumulate content under current parent ───────────────────
                current_parent["full_content"].append(content)
                word_count = len(content.split())

                # Flush if this item would push us over max_words
                if self.max_words and buf["word_count"] + word_count > self.max_words and buf["lines"]:
                    buf = self._flush(buf, children)
                    buf["parent_id"] = current_parent["parent_id"]
                    buf["page_no"]   = page_no
                    buf["header"]    = current_parent["heading"]

                if not buf["lines"]:
                    buf["page_no"] = page_no

                buf["lines"].append(content)
                buf["word_count"] += word_count

                # Flush once min_words reached
                if self.min_words and buf["word_count"] >= self.min_words:
                    buf = self._flush(buf, children)
                    buf["parent_id"] = current_parent["parent_id"]
                    buf["page_no"]   = page_no
                    buf["header"]    = current_parent["heading"]

        # Flush any remaining buffer content
        self._flush(buf, children)

        # Serialise full_content from list to string and drop empty parents
        parents_out = [
            {
                "parent_id":    p["parent_id"],
                "doc_id":       p["doc_id"],
                "heading":      p["heading"],
                "page_no":      p["page_no"],
                "full_content": "\n".join(p["full_content"]),
            }
            for p in parents
            if p["full_content"]   # drop parents that had no content (bare headers)
        ]

        self.logger.info(
            "Chunking done — %d parents, %d children", len(parents_out), len(children)
        )
        return parents_out, children

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