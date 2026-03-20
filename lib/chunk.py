"""Markdown chunking pipeline for RAG indexing."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from lib.utils.logger import get_logger


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FIGURE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


@dataclass(frozen=True)
class ChunkingConfig:
    md_input_path: Path
    output_dir: Path
    doc_id: str
    max_words_per_chunk: int
    overlap_sentences: int
    min_words_per_chunk: int


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or self._load_config_from_env()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(name="Chunker", log_level=logging.INFO)

    @staticmethod
    def _load_config_from_env() -> ChunkingConfig:
        load_dotenv()

        max_words_per_chunk = int(os.getenv("MAX_WORDS_PER_CHUNK", "400"))
        min_words_per_chunk = int(os.getenv("MIN_WORDS_PER_CHUNK", "30"))

        return ChunkingConfig(
            md_input_path=Path(os.getenv("MD_INPUT_PATH", "output/full_document.md")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "output/chunks")),
            doc_id=os.getenv("DOC_ID", "document_001"),
            max_words_per_chunk=max_words_per_chunk,
            overlap_sentences=max(0, int(os.getenv("OVERLAP_SENTENCES", "1"))),
            min_words_per_chunk=max(1, min_words_per_chunk),
        )

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _new_parent(self, heading_text: str) -> dict[str, Any]:
        return {
            "parent_id": str(uuid.uuid4()),
            "heading_path": [heading_text],
            "doc_id": self.config.doc_id,
            "content_lines": [],
        }

    def _make_child_chunk(self, text: str, element_type: str, parent: dict[str, Any]) -> dict[str, Any]:
        heading_str = " > ".join(parent["heading_path"])
        return {
            "chunk_id": str(uuid.uuid4()),
            "parent_id": parent["parent_id"],
            "element_type": element_type,
            "heading_path": parent["heading_path"],
            "doc_id": self.config.doc_id,
            "content": text,
            "content_for_embedding": f"Section : {heading_str}\n{text}",
            "token_estimate": self.word_count(text),
        }

    def _build_blocks(self, markdown: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parents: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []

        current_parent = self._new_parent("[Racine]")
        parents.append(current_parent)

        paragraph_buffer: list[str] = []
        table_buffer: list[str] = []

        def flush_paragraph() -> None:
            if not paragraph_buffer:
                return
            text = " ".join(paragraph_buffer).strip()
            if text:
                blocks.append(
                    {
                        "element_type": "paragraph",
                        "content": text,
                        "parent": current_parent,
                    }
                )
            paragraph_buffer.clear()

        def flush_table() -> None:
            if not table_buffer:
                return
            html = "\n".join(table_buffer).strip()
            if html:
                blocks.append(
                    {
                        "element_type": "table",
                        "content": html,
                        "parent": current_parent,
                    }
                )
            table_buffer.clear()

        for raw_line in markdown.splitlines():
            stripped = raw_line.strip()

            heading_match = HEADING_RE.match(stripped)
            if heading_match and not table_buffer:
                flush_paragraph()
                heading_text = heading_match.group(2).strip()
                current_parent = self._new_parent(heading_text)
                parents.append(current_parent)
                continue

            current_parent["content_lines"].append(raw_line)

            if table_buffer:
                table_buffer.append(raw_line)
                if "</table>" in stripped.lower():
                    flush_table()
                continue

            if "<table" in stripped.lower():
                flush_paragraph()
                table_buffer.append(raw_line)
                if "</table>" in stripped.lower():
                    flush_table()
                continue

            if FIGURE_RE.search(stripped):
                flush_paragraph()
                blocks.append(
                    {
                        "element_type": "figure",
                        "content": stripped,
                        "parent": current_parent,
                    }
                )
                continue

            if stripped:
                paragraph_buffer.append(stripped)
            else:
                flush_paragraph()

        flush_paragraph()
        flush_table()
        self.logger.info("Block extraction complete: %s parents, %s blocks", len(parents), len(blocks))
        return parents, blocks

    def _paragraph_to_chunks(self, text: str, parent: dict[str, Any]) -> list[dict[str, Any]]:
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks: list[dict[str, Any]] = []
        window: list[str] = []

        for sentence in sentences:
            window.append(sentence)
            if self.word_count(" ".join(window)) >= self.config.max_words_per_chunk:
                if self.config.overlap_sentences and len(window) > self.config.overlap_sentences:
                    emit = " ".join(window[:-self.config.overlap_sentences])
                    window = window[-self.config.overlap_sentences :]
                else:
                    emit = " ".join(window)
                    window = []

                if emit.strip():
                    chunks.append(self._make_child_chunk(emit, "paragraph", parent))

        if window:
            chunks.append(self._make_child_chunk(" ".join(window), "paragraph", parent))

        return chunks

    def _table_to_chunks(self, html: str, parent: dict[str, Any]) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            self.logger.warning("Encountered table block without rows under heading: %s", " > ".join(parent["heading_path"]))
            return []

        section = " > ".join(parent["heading_path"])
        headers: list[str] = []

        for row in rows:
            ths = row.find_all("th")
            if ths:
                headers = [th.get_text(strip=True) for th in ths]
                break

        header_str = " | ".join(headers) if headers else ""
        chunks: list[dict[str, Any]] = []

        for row in rows:
            cells = row.find_all(["td", "th"])
            values = [cell.get_text(strip=True) for cell in cells]
            if not values or values == headers:
                continue

            row_str = " | ".join(values)
            text = f"Tableau - {section} | En-tetes: {header_str} | Ligne: {row_str}"
            chunks.append(self._make_child_chunk(text, "table_row", parent))

        self.logger.info("Table rows chunked under '%s': %s chunks", section, len(chunks))
        return chunks

    def _figure_to_chunks(self, content: str, parent: dict[str, Any]) -> list[dict[str, Any]]:
        match = FIGURE_RE.search(content)
        caption = match.group(1).strip() if match else ""
        path = match.group(2).strip() if match else ""

        if not caption and not path:
            return []

        text = f"Figure : {caption}" if caption else "Figure sans legende"
        chunk = self._make_child_chunk(text, "figure", parent)
        chunk["figure_ref"] = path
        return [chunk]

    def _merge_small_paragraph_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not chunks:
            return chunks

        merged = chunks[:]
        i = 0
        while i < len(merged) - 1:
            current = merged[i]
            nxt = merged[i + 1]

            can_merge = (
                current["element_type"] == "paragraph"
                and nxt["element_type"] == "paragraph"
                and current["parent_id"] == nxt["parent_id"]
                and self.word_count(current["content"]) < self.config.min_words_per_chunk
            )

            if can_merge:
                nxt["content"] = f"{current['content']} {nxt['content']}".strip()
                nxt["content_for_embedding"] = (
                    f"Section : {' > '.join(nxt['heading_path'])}\n{nxt['content']}"
                )
                nxt["token_estimate"] = self.word_count(nxt["content"])
                merged.pop(i)
            else:
                i += 1

        if len(merged) >= 2:
            last = merged[-1]
            prev = merged[-2]
            if (
                last["element_type"] == "paragraph"
                and prev["element_type"] == "paragraph"
                and last["parent_id"] == prev["parent_id"]
                and self.word_count(last["content"]) < self.config.min_words_per_chunk
            ):
                prev["content"] = f"{prev['content']} {last['content']}".strip()
                prev["content_for_embedding"] = (
                    f"Section : {' > '.join(prev['heading_path'])}\n{prev['content']}"
                )
                prev["token_estimate"] = self.word_count(prev["content"])
                merged.pop()

        return merged

    def _write_outputs(self, parents: list[dict[str, Any]], children: list[dict[str, Any]]) -> None:
        (self.config.output_dir / "chunks_vector.json").write_text(
            json.dumps(children, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        parent_out = [
            {
                "parent_id": parent["parent_id"],
                "heading_path": parent["heading_path"],
                "doc_id": parent["doc_id"],
                "full_content": "\n".join(parent["content_lines"]),
            }
            for parent in parents
            if parent["content_lines"]
        ]
        (self.config.output_dir / "chunks_parent.json").write_text(
            json.dumps(parent_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info("Outputs written to %s (vector: %s, parent: %s)", self.config.output_dir, len(children), len(parent_out))

    def _print_summary(self, parents: list[dict[str, Any]], children: list[dict[str, Any]]) -> None:
        type_counts: dict[str, int] = {}
        for child in children:
            kind = child["element_type"]
            type_counts[kind] = type_counts.get(kind, 0) + 1

        documented_parents = sum(1 for parent in parents if parent["content_lines"])

        print("\n" + "=" * 50)
        print("RESUME")
        print("=" * 50)
        print(f"  Blocs parents  : {documented_parents}")
        print(f"  Chunks enfants : {len(children)}")
        for kind, count in type_counts.items():
            print(f"    {kind:20s} : {count}")
        print(f"\nSortie : {self.config.output_dir}/")
        print("  chunks_vector.json  -> base vectorielle")
        print("  chunks_parent.json  -> base documentaire")

    def run(self) -> None:
        self.logger.info("Chunking run started")
        markdown = self.config.md_input_path.read_text(encoding="utf-8")
        print(f"Fichier lu : {self.config.md_input_path} ({self.word_count(markdown)} mots)")
        self.logger.info("Markdown loaded from %s", self.config.md_input_path)

        parents, blocks = self._build_blocks(markdown)

        children: list[dict[str, Any]] = []
        for block in blocks:
            parent = block["parent"]
            element_type = block["element_type"]
            content = block["content"]

            if element_type == "paragraph":
                children.extend(self._paragraph_to_chunks(content, parent))
            elif element_type == "table":
                print(f"  -> Tableau dans : {' > '.join(parent['heading_path'])}")
                children.extend(self._table_to_chunks(content, parent))
            elif element_type == "figure":
                children.extend(self._figure_to_chunks(content, parent))

        children = self._merge_small_paragraph_chunks(children)
        self.logger.info("Child chunk generation complete: %s chunks", len(children))
        self._write_outputs(parents, children)
        self._print_summary(parents, children)
        self.logger.info("Chunking run completed")


def run() -> None:
    Chunker().run()


if __name__ == "__main__":
    run()
