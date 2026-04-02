"""Chunker — splits a DoclingDocument into parent + child chunks for RAG."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any
import re
import base64
from io import BytesIO

from docling.chunking import HybridChunker , BaseChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument

from transformers import AutoTokenizer

from lib.chunking_pipeline.serializer import SerializerProvider 
from lib.models.main import ChunkRunOutput
from lib.utils.llm_client import LLMClient
from lib.utils.logger import get_logger
from lib.utils.table import extract_table_row_chunks
from lib.utils.timer import Timer

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Convert a :class:`DoclingDocument` into parent/child chunk pairs."""

    def __init__(
        self,
        doc_id: str | None = None,
        max_words: int | None = None,
        tokenizer_model: str | None = None,
        description_api_url: str | None = None,
        description_api_key: str | None = None,
        description_api_model: str | None = None,
        output_image_dir: str = "./output",
    ) -> None:
        self.doc_id = doc_id or "default-doc"
        self.max_words = max_words or _DEFAULT_MAX_TOKENS
        self.tokenizer_model = tokenizer_model or _DEFAULT_MODEL
        self.logger = get_logger(name="Chunker", log_level=logging.INFO)

        self.output_dir = Path(output_image_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.description_client: LLMClient | None = (
            LLMClient(
                api_url=description_api_url,
                api_key=description_api_key or "",
                model=description_api_model,
                timeout=90,
            )
            if description_api_url and description_api_model
            else None
        )

        # Table buffer — class-level state
        self._table_buffer: list[str] = []
        self._table_buffer_heading: str | None = None
        self._table_buffer_page_ref: int | None = None
        self._progress_log_every = 10

    @staticmethod
    def _worker_tag() -> str:
        return f"pid={os.getpid()} thread={threading.current_thread().name}"

    # ------------------------------------------------------------------
    # HybridChunker helpers
    # ------------------------------------------------------------------

    def _build_chunker(self) -> HybridChunker:
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_model),
            max_tokens=self.max_words,
        )
        return HybridChunker(
            tokenizer=tokenizer,
            repeat_table_header=False,
            serializer_provider=SerializerProvider(
                output_dir=self.output_dir,
                doc_id=self.doc_id,
            ),
        )

    @staticmethod
    def _chunk_page_no(chunk: Any) -> int | None:
        for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
            prov = getattr(item, "prov", None)
            if prov and getattr(prov[0], "page_no", None) is not None:
                return int(prov[0].page_no)
        return None

    @staticmethod
    def _chunk_heading(chunk: Any) -> str:
        headings = getattr(getattr(chunk, "meta", None), "headings", None) or []
        return str(headings[-1]).strip() if headings else "[Document root]"

    @staticmethod
    def _element_type(chunk: Any) -> str:
        for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
            if item.label == "table":
                return "table"
            if item.label == "picture":
                return "picture"
        return "text"
    # ------------------------------------------------------------------
    # Visual description
    # ------------------------------------------------------------------

    def _describe_visual(self, content: str, element_type: str) -> tuple[str, str] | str:
        """Call LLM to summarize table/picture content.
        For tables: returns (fixed_html, summary).
        For images: returns summary string.
        """
        if not self.description_client:
            return (content, "") if element_type == "table" else content
        try:
            if element_type == "table":
                result = self.description_client.fix_and_summarize_table(html_table=content)
                return result["fixed_html"], result["summary"]
            else:
                description = self.description_client.describe_image(
                    image_input=content,
                    max_words=self.max_words,
                )
                return description
        except Exception as exc:
            self.logger.warning("LLM description failed: %s", exc)
            return (content, "") if element_type == "table" else content

    # ------------------------------------------------------------------
    # Table buffer management
    # ------------------------------------------------------------------

    def _table_buffer_accumulate(self, content: str, heading: str, page_ref: int | None) -> None:
        """Accumulate a table chunk into the buffer."""
        if not self._table_buffer:
            self._table_buffer_heading = heading
            self._table_buffer_page_ref = page_ref
        self._table_buffer.append(content)

    def _table_buffer_flush(self) -> list[dict[str, Any]]:
        """Flush buffer — returns summary chunk + one chunk per row-header pair."""
        if not self._table_buffer:
            return []

        with Timer(function_name="table_buffer_flush"):
            combined_html = "".join(self._table_buffer)
            heading = self._table_buffer_heading or "[Document root]"
            page_ref = self._table_buffer_page_ref
            chunks: list[dict[str, Any]] = []

            self.logger.info(
                "Flushing table buffer (%d fragments) at heading=%s page=%s [%s]",
                len(self._table_buffer),
                heading,
                page_ref,
                self._worker_tag(),
            )

            # 1. Send to LLM to fix structure + get summary
            fixed_html, summary = self._describe_visual(combined_html, "table")

            # 2. Row chunks — parse the LLM-corrected HTML
            row_chunks = extract_table_row_chunks(
                combined_html=fixed_html,
                heading=heading,
                page_ref=page_ref,
                doc_id=self.doc_id,
                logger=self.logger,
            )
            chunks.extend(row_chunks)

            # 3. Summary chunk
            if summary:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "_heading": heading,
                    "element_type": "table_summary",
                    "page_ref": page_ref,
                    "doc_id": self.doc_id,
                    "content": summary,
                    "content_for_embedding": summary,
                    "token_estimate": len(summary.split()),
                })

            self.logger.info(
                "Table buffer flush produced %d chunks (rows=%d, summary=%s) [%s]",
                len(chunks),
                len(row_chunks),
                bool(summary),
                self._worker_tag(),
            )

            # Reset buffer
            self._table_buffer.clear()
            self._table_buffer_heading = None
            self._table_buffer_page_ref = None
            return chunks

    def _handle_picture_content(self, content: str, chunk: BaseChunker, doc: DoclingDocument) -> str:
        pictures_seen = 0
        for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
            label = str(getattr(item, "label", "")).lower()
            if "picture" not in label:
                continue
            pictures_seen += 1

            image = item.get_image(doc=doc)
            if image is None:
                self.logger.warning("Could not extract image for picture item, skipping.")
                continue

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            base64_string = base64.b64encode(buffer.getvalue()).decode()

            if not base64_string:
                continue

            with Timer(function_name="picture_description"):
                image_description = self._describe_visual(base64_string, "picture")
            clean_text = re.sub(r"Picture_.*?_Picture", "", content, flags=re.DOTALL).strip()
            content = f"{clean_text}\n\n{image_description}".strip() if clean_text else image_description

        if pictures_seen:
            self.logger.info(
                "Processed %d picture item(s) for current chunk [%s]",
                pictures_seen,
                self._worker_tag(),
            )

        return content
    
    # ------------------------------------------------------------------
    # Core chunking pipeline
    # ------------------------------------------------------------------

    def _build_children(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        with Timer(function_name="build_children"):
            chunker = self._build_chunker()
            children: list[dict[str, Any]] = []
            seen = 0
            table_seen = 0
            picture_seen = 0
            text_seen = 0

            # Reset buffer state at start of each run
            self._table_buffer.clear()
            self._table_buffer_heading = None
            self._table_buffer_page_ref = None

            self.logger.info("Starting child chunk build [%s]", self._worker_tag())

            for chunk in chunker.chunk(dl_doc=doc):
                seen += 1
                content = chunker.contextualize(chunk)
                if not content:
                    if seen % self._progress_log_every == 0:
                        self.logger.info(
                            "Progress: seen=%d children=%d (empty chunks skipped) [%s]",
                            seen,
                            len(children),
                            self._worker_tag(),
                        )
                    continue

                element_type = self._element_type(chunk)
                heading = self._chunk_heading(chunk)
                page_ref = self._chunk_page_no(chunk)

                if element_type == "table":
                    table_seen += 1
                    self._table_buffer_accumulate(chunk.text, heading, page_ref)
                    if seen % self._progress_log_every == 0:
                        self.logger.info(
                            "Progress: seen=%d children=%d tables_buffered=%d [%s]",
                            seen,
                            len(children),
                            len(self._table_buffer),
                            self._worker_tag(),
                        )
                    continue

                # Non-table chunk — flush buffer first
                children.extend(self._table_buffer_flush())

                if element_type == "picture":
                    picture_seen += 1
                    content = self._handle_picture_content(content, chunk, doc)
                else:
                    text_seen += 1

                children.append({
                    "chunk_id": str(uuid.uuid4()),
                    "_heading": heading,
                    "element_type": element_type,
                    "page_ref": page_ref,
                    "doc_id": self.doc_id,
                    "content": content,
                    "content_for_embedding": content,
                    "token_estimate": len(content.split()),
                })

                if seen % self._progress_log_every == 0:
                    self.logger.info(
                        "Progress: seen=%d children=%d text=%d picture=%d table=%d [%s]",
                        seen,
                        len(children),
                        text_seen,
                        picture_seen,
                        table_seen,
                        self._worker_tag(),
                    )

            # Flush remaining table buffer at end of document
            children.extend(self._table_buffer_flush())

            self.logger.info(
                "HybridChunker produced %d children (seen=%d, text=%d, picture=%d, table=%d) [%s]",
                len(children),
                seen,
                text_seen,
                picture_seen,
                table_seen,
                self._worker_tag(),
            )
            return children

    # ------------------------------------------------------------------
    # Parent grouping
    # ------------------------------------------------------------------

    def _build_parents(
        self, children: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with Timer(function_name="build_parents"):
            grouped: dict[str, dict[str, Any]] = {}
            order: list[str] = []

            for child in children:
                heading = child.pop("_heading", "[Document root]")

                if heading not in grouped:
                    grouped[heading] = {
                        "parent_id": str(uuid.uuid4()),
                        "doc_id": self.doc_id,
                        "heading": heading,
                        "page_no": child.get("page_ref"),
                        "full_content": [],
                    }
                    order.append(heading)

                parent = grouped[heading]
                if parent["page_no"] is None:
                    parent["page_no"] = child.get("page_ref")

                parent["full_content"].append(child["content"])
                child["parent_id"] = parent["parent_id"]

            parents = [
                {
                    **{k: v for k, v in grouped[h].items() if k != "full_content"},
                    "full_content": "\n".join(grouped[h]["full_content"]),
                }
                for h in order
                if grouped[h]["full_content"]
            ]

            self.logger.info(
                "Parent grouping complete: parents=%d children=%d [%s]",
                len(parents),
                len(children),
                self._worker_tag(),
            )
            return parents, children

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, document: DoclingDocument) -> ChunkRunOutput:
        with Timer(function_name="chunker_run_total"):
            self.logger.info("Chunker run started for doc_id=%s [%s]", self.doc_id, self._worker_tag())

            children = self._build_children(document)
            parents, children = self._build_parents(children)
            self.logger.info("Chunking complete — %d parents, %d children", len(parents), len(children))

            with Timer(function_name="write_children_json"):
                with open(self.output_dir / f"{self.doc_id}_children.json", "w", encoding="utf-8") as fp:
                    json.dump(children, fp, indent=2, ensure_ascii=False)

            with Timer(function_name="write_parents_json"):
                with open(self.output_dir / f"{self.doc_id}_parents.json", "w", encoding="utf-8") as fp:
                    json.dump(parents, fp, indent=2, ensure_ascii=False)

            self.logger.info("Chunker run finished for doc_id=%s [%s]", self.doc_id, self._worker_tag())
            return ChunkRunOutput(
                chunks_vector=children,
                chunks_parent=parents,
            )