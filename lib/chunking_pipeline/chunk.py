"""Chunker — splits a DoclingDocument into parent + child chunks for RAG."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any
import pandas as pd
import re
from io import StringIO
import base64
from io import BytesIO

from docling.chunking import HybridChunker , BaseChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument

from transformers import AutoTokenizer
from typing_extensions import override

from lib.chunking_pipeline.serializer import SerializerProvider 
from lib.models.main import ChunkRunOutput
from lib.utils.llm_client import LLMClient
from lib.utils.logger import get_logger

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

    def _describe_visual(self, content: str, element_type: str) -> str:
        """Call LLM to summarize table/picture content."""
        if not self.description_client:
            return content
        try:
            if element_type == "table":
                description = self.description_client.summarize_table(html_table=content)
            else:
                description = self.description_client.describe_image(
                    image_input=content,
                    max_words=self.max_words,
                )
                
            return description
        except Exception as exc:
            self.logger.warning("LLM description failed: %s", exc)
            return content

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

        combined_html = "".join(self._table_buffer)
        heading = self._table_buffer_heading or "[Document root]"
        page_ref = self._table_buffer_page_ref
        chunks: list[dict[str, Any]] = []

        # 1. Row chunks — one per row-header combination extracted from HTML
        row_chunks = self._extract_table_row_chunks(combined_html, heading, page_ref)
        chunks.extend(row_chunks)
        
        
        # 2. Summary chunk — LLM summary of the full table
        summary = self._describe_visual(combined_html, "table")
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

        # Reset buffer
        self._table_buffer.clear()
        self._table_buffer_heading = None
        self._table_buffer_page_ref = None

        return chunks
    
    def _extract_table_row_chunks(
        self, combined_html: str, heading: str, page_ref: int | None
    ) -> list[dict[str, Any]]:
        """Parse HTML table and return one chunk per row-header combination using Pandas."""
        try:
            # 1. Parse tables. read_html returns a list of DataFrames.
            combined_html = StringIO(combined_html)
            tables = pd.read_html(combined_html)
            if not tables:
                return []

            df = tables[0]
            # Replace NaNs with empty strings to avoid "nan" text in chunks
            df = df.fillna("")
            
            headers = df.columns.tolist()
            row_chunks: list[dict[str, Any]] = []

            # 2. Iterate through DataFrame rows
            for _, row in df.iterrows():
                
                pairs = [
                    f"{headers[i]}: {val}" 
                    for i, val in enumerate(row) 
                    if str(val).strip()
                ]

                if not pairs:
                    continue

                content = " | ".join(pairs)
                
                row_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "_heading": heading,
                    "element_type": "table_row",
                    "page_ref": page_ref,
                    "doc_id": self.doc_id,
                    "content": content,
                    "content_for_embedding": content,
                    "token_estimate": len(content.split()),
                })

            return row_chunks

        except Exception as exc:
            self.logger.warning("Row extraction failed: %s", exc)
            return []
        
    # ------------------------------------------------------------------
    # Image base64 management
    # ------------------------------------------------------------------
    def _handle_picture_content(self, content: str, chunk: BaseChunker, doc: DoclingDocument) -> str:
        for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
            label = str(getattr(item, "label", "")).lower()
            if "picture" not in label:
                continue

            image = item.get_image(doc=doc)
            if image is None:
                self.logger.warning("Could not extract image for picture item, skipping.")
                continue

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            base64_string = base64.b64encode(buffer.getvalue()).decode()

            if not base64_string:
                continue

            image_description = self._describe_visual(base64_string, "picture")
            clean_text = re.sub(r"Picture_.*?_Picture", "", content, flags=re.DOTALL).strip()
            content = f"{clean_text}\n\n{image_description}".strip() if clean_text else image_description

        return content
    
    # ------------------------------------------------------------------
    # Core chunking pipeline
    # ------------------------------------------------------------------

    def _build_children(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        chunker = self._build_chunker()
        children: list[dict[str, Any]] = []

        # Reset buffer state at start of each run
        self._table_buffer.clear()
        self._table_buffer_heading = None
        self._table_buffer_page_ref = None

        for chunk in chunker.chunk(dl_doc=doc):
            content = chunker.contextualize(chunk)
            if not content:
                continue
                             
            element_type = self._element_type(chunk)
            heading = self._chunk_heading(chunk)
            page_ref = self._chunk_page_no(chunk)

            if element_type == "table":
                self._table_buffer_accumulate(content, heading, page_ref)
                continue
            
            # Non-table chunk — flush buffer first
            children.extend(self._table_buffer_flush())
            
            if element_type == "picture":
                content = self._handle_picture_content(content, chunk, doc)
        
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

        # Flush remaining table buffer at end of document
        children.extend(self._table_buffer_flush())

        self.logger.info("HybridChunker produced %d children", len(children))
        return children

    # ------------------------------------------------------------------
    # Parent grouping
    # ------------------------------------------------------------------

    def _build_parents(
        self, children: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

        return parents, children

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, document: DoclingDocument) -> ChunkRunOutput:
        children = self._build_children(document)
        parents, children = self._build_parents(children)
        self.logger.info("Chunking complete — %d parents, %d children", len(parents), len(children))

        with open(self.output_dir / f"{self.doc_id}_children.json", "w", encoding="utf-8") as fp:
            json.dump(children, fp, indent=2, ensure_ascii=False)
        with open(self.output_dir / f"{self.doc_id}_parents.json", "w", encoding="utf-8") as fp:
            json.dump(parents, fp, indent=2, ensure_ascii=False)
        return ChunkRunOutput(
            chunks_vector=children,
            chunks_parent=parents,
        )