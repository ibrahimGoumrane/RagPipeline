"""
Markdown chunker — splits a markdown string into parent + child chunks for RAG.

Usage:
    chunker           = Chunker(doc_id="my-doc")
    parents, children = chunker.run(markdown_text)
    # or get a typed output:
    result = chunker.run_to_output(markdown_text, output_dir="output/chunks")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from lib.models.main import ChunkRunOutput
from lib.utils.logger import get_logger


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FIGURE_RE  = re.compile(r"!\[(.*?)\]\((.*?)\)")


class Chunker:
    def __init__(
        self,
        doc_id: str | None = None,
        max_words: int | None = None,
        min_words: int | None = None,
        overlap_sentences: int | None = None,
    ):
        self.doc_id           = doc_id
        self.max_words        = max_words      
        self.min_words        = min_words    
        self.overlap_sentences= overlap_sentences if overlap_sentences is not None else 2
        self.logger = get_logger(name="Chunker", log_level=logging.INFO)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]

    def _new_parent(self, heading: str) -> dict[str, Any]:
        return {
            "parent_id":     str(uuid.uuid4()),
            "heading_path":  [heading],
            "doc_id":        self.doc_id,
            "content_lines": [],
        }

    def _child(self, text: str, kind: str, parent: dict[str, Any]) -> dict[str, Any]:
        section = " > ".join(parent["heading_path"])
        return {
            "chunk_id":              str(uuid.uuid4()),
            "parent_id":             parent["parent_id"],
            "element_type":          kind,
            "heading_path":          parent["heading_path"],
            "doc_id":                self.doc_id,
            "content":               text,
            "content_for_embedding": f"Section : {section}\n{text}",
            "token_estimate":        self._word_count(text),
        }

    # ── Block extraction ──────────────────────────────────────────────────────

    def _extract_blocks(self, markdown: str) -> tuple[list[dict], list[dict]]:
        parents: list[dict] = []
        blocks:  list[dict] = []

        current   = self._new_parent("[Racine]")
        para_buf:  list[str] = []
        table_buf: list[str] = []
        parents.append(current)

        def flush_para():
            text = " ".join(para_buf).strip()
            if text:
                blocks.append({"kind": "paragraph", "content": text, "parent": current})
            para_buf.clear()

        def flush_table():
            html = "\n".join(table_buf).strip()
            if html:
                blocks.append({"kind": "table", "content": html, "parent": current})
            table_buf.clear()

        for raw in markdown.splitlines():
            line = raw.strip()

            m = HEADING_RE.match(line)
            if m and not table_buf:
                flush_para()
                current = self._new_parent(m.group(2).strip())
                parents.append(current)
                continue

            current["content_lines"].append(raw)

            if table_buf:
                table_buf.append(raw)
                if "</table>" in line.lower():
                    flush_table()
                continue

            if "<table" in line.lower():
                flush_para()
                table_buf.append(raw)
                if "</table>" in line.lower():
                    flush_table()
                continue

            if FIGURE_RE.search(line):
                flush_para()
                blocks.append({"kind": "figure", "content": line, "parent": current})
                continue

            if line:
                para_buf.append(line)
            else:
                flush_para()

        flush_para()
        flush_table()
        return parents, blocks

    # ── Chunk producers ───────────────────────────────────────────────────────

    def _chunk_paragraph(self, text: str, parent: dict) -> list[dict]:
        sentences = self._sentences(text)
        if not sentences:
            return []

        chunks: list[dict] = []
        window: list[str]  = []

        for s in sentences:
            window.append(s)
            if self._word_count(" ".join(window)) >= self.max_words:
                overlap = self.overlap_sentences
                emit    = " ".join(window[:-overlap] if overlap and len(window) > overlap else window)
                window  = window[-overlap:] if overlap and len(window) > overlap else []
                if emit.strip():
                    chunks.append(self._child(emit, "paragraph", parent))

        if window:
            chunks.append(self._child(" ".join(window), "paragraph", parent))

        return chunks

    def _chunk_table(self, html: str, parent: dict) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            self.logger.warning("Empty table under '%s' — skipped", " > ".join(parent["heading_path"]))
            return []

        headers = []
        for row in rows:
            ths = row.find_all("th")
            if ths:
                headers = [th.get_text(strip=True) for th in ths]
                break

        section    = " > ".join(parent["heading_path"])
        header_str = " | ".join(headers)
        chunks: list[dict] = []

        for row in rows:
            values = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if not values or values == headers:
                continue
            text = f"Tableau - {section} | En-têtes: {header_str} | Ligne: {' | '.join(values)}"
            chunks.append(self._child(text, "table_row", parent))

        return chunks

    def _chunk_figure(self, content: str, parent: dict) -> list[dict]:
        m       = FIGURE_RE.search(content)
        caption = m.group(1).strip() if m else ""
        path    = m.group(2).strip() if m else ""
        if not caption and not path:
            return []
        text  = f"Figure : {caption}" if caption else "Figure sans légende"
        chunk = self._child(text, "figure", parent)
        if path:
            chunk["figure_ref"] = path
        return [chunk]

    # ── Post-processing ───────────────────────────────────────────────────────

    def _merge_small(self, chunks: list[dict]) -> list[dict]:
        out = chunks[:]
        i = 0
        while i < len(out) - 1:
            cur, nxt = out[i], out[i + 1]
            if (
                cur["element_type"] == "paragraph"
                and nxt["element_type"] == "paragraph"
                and cur["parent_id"] == nxt["parent_id"]
                and self._word_count(cur["content"]) < self.min_words
            ):
                nxt["content"]               = f"{cur['content']} {nxt['content']}".strip()
                nxt["content_for_embedding"] = f"Section : {' > '.join(nxt['heading_path'])}\n{nxt['content']}"
                nxt["token_estimate"]        = self._word_count(nxt["content"])
                out.pop(i)
            else:
                i += 1

        if len(out) >= 2:
            last, prev = out[-1], out[-2]
            if (
                last["element_type"] == "paragraph"
                and prev["element_type"] == "paragraph"
                and last["parent_id"] == prev["parent_id"]
                and self._word_count(last["content"]) < self.min_words
            ):
                prev["content"]               = f"{prev['content']} {last['content']}".strip()
                prev["content_for_embedding"] = f"Section : {' > '.join(prev['heading_path'])}\n{prev['content']}"
                prev["token_estimate"]        = self._word_count(prev["content"])
                out.pop()

        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, markdown: str) -> tuple[list[dict], list[dict]]:
        """Chunk markdown. Returns (parents, children) as plain dicts."""
        parents_raw, blocks = self._extract_blocks(markdown)

        children: list[dict] = []
        for block in blocks:
            kind, content, parent = block["kind"], block["content"], block["parent"]
            if kind == "paragraph":
                children.extend(self._chunk_paragraph(content, parent))
            elif kind == "table":
                children.extend(self._chunk_table(content, parent))
            elif kind == "figure":
                children.extend(self._chunk_figure(content, parent))

        children = self._merge_small(children)

        parents_out = [
            {
                "parent_id":    p["parent_id"],
                "heading_path": p["heading_path"],
                "doc_id":       p["doc_id"],
                "full_content": "\n".join(p["content_lines"]),
            }
            for p in parents_raw
            if p["content_lines"]
        ]

        self.logger.info("Chunking done — %d parents, %d children", len(parents_out), len(children))
        return parents_out, children

    def run_to_output(self, markdown: str, output_dir: str | None = None) -> ChunkRunOutput:
        """Chunk markdown, write JSON files to disk, return ChunkRunOutput."""
        out_dir = Path(output_dir) / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        parents, children = self.run(markdown)

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