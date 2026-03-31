"""Chunker — splits a DoclingDocument into parent + child chunks for RAG."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.html import HTMLTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem
from transformers import AutoTokenizer
from typing_extensions import override

from lib.models.main import ChunkRunOutput
from lib.utils.llm_client import LLMClient
from lib.utils.logger import get_logger

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Custom Serializers
# ---------------------------------------------------------------------------

class PictureSerializer(MarkdownPictureSerializer):
    def __init__(self, output_dir: Path, doc_id: str) -> None:
        self.output_dir = output_dir
        self.doc_id = doc_id

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        text_parts: list[str] = []

        if item.meta is not None and item.meta.description is not None:
            text_parts.append(f"Picture description: {item.meta.description.text}")

        pil_image = item.get_image(doc)
        if pil_image:
            filename = f"{self.doc_id}_picture_{uuid.uuid4().hex[:8]}.png"
            filepath = self.output_dir / filename
            with filepath.open("wb") as fp:
                pil_image.save(fp, "PNG")
            text_parts.append(f"image_ref:{str(filepath)}")  # <-- inject path

        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)
    

class TableSerializer(HTMLTableSerializer):
    """Serializes table as HTML and saves image to disk."""

    def __init__(self, output_dir: Path, doc_id: str) -> None:
        self.output_dir = output_dir
        self.doc_id = doc_id

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        html_result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        text_parts: list[str] = [html_result.text] if html_result.text else []

        pil_image = item.get_image(doc)
        if pil_image:
            filename = f"{self.doc_id}_table_{uuid.uuid4().hex[:8]}.png"
            filepath = self.output_dir / filename
            with filepath.open("wb") as fp:
                pil_image.save(fp, "PNG")

        text_res = "\n\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


class SerializerProvider(ChunkingSerializerProvider):
    def __init__(self, output_dir: Path, doc_id: str) -> None:
        self.output_dir = output_dir
        self.doc_id = doc_id

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=PictureSerializer(
                output_dir=self.output_dir,
                doc_id=self.doc_id,
            ),
            table_serializer=TableSerializer(
                output_dir=self.output_dir,
                doc_id=self.doc_id,
            ),
        )


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
        output_image_dir: str = "./output_images",
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
            merge_peers=True,
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
            if isinstance(item, PictureItem):
                return "picture"
            if isinstance(item, TableItem):
                return "table"
        return "text"

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
                    system_prompt=(
                        "Analyze and describe this image in French. "
                        "Provide structured, professional content suitable for financial documents. "
                        "Focus on key data, visual elements, and important details."
                    ),
                )

            return description

        except Exception as exc:
            self.logger.warning("LLM description failed: %s", exc)
            return content
        
    # ------------------------------------------------------------------
    # Core chunking pipeline
    # ------------------------------------------------------------------

    def _build_children(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        chunker = self._build_chunker()
        children: list[dict[str, Any]] = []

        # Buffer for accumulating consecutive table chunks
        table_buffer: list[str] = []
        table_buffer_heading: str | None = None
        table_buffer_page_ref: int | None = None

        def _flush_table_buffer() -> None:
            """Summarize buffered table content and append as a summary chunk."""
            if not table_buffer:
                return

            combined_html = "\n".join(table_buffer)
            summary = self._describe_visual(combined_html, "table")

            children.append({
                "chunk_id": str(uuid.uuid4()),
                "_heading": table_buffer_heading or "[Document root]",
                "element_type": "table",
                "page_ref": table_buffer_page_ref,
                "doc_id": self.doc_id,
                "content": summary,
                "content_for_embedding": summary,
                "token_estimate": len(summary.split()),
            })

            table_buffer.clear()


        for chunk in chunker.chunk(dl_doc=doc):
            content = chunker.contextualize(chunk)
            if not content:
                continue

            element_type = self._element_type(chunk)
            heading = self._chunk_heading(chunk)
            page_ref = self._chunk_page_no(chunk)

            if element_type == "table":
                if not table_buffer:
                    table_buffer_heading = heading
                    table_buffer_page_ref = page_ref
                table_buffer.append(content)
                continue

            # Non-table chunk encountered — flush buffer first
            _flush_table_buffer()

            # Always scan for image_ref regardless of element_type
            image_path = None
            clean_lines = []
            for line in content.splitlines():
                if line.startswith("image_ref:"):
                    image_path = line[len("image_ref:"):].strip()
                else:
                    clean_lines.append(line)

            if image_path:
                # Replace the image_ref line with LLM description
                image_description = self._describe_visual(image_path, "picture")
                clean_text = "\n".join(clean_lines).strip()
                content = f"{clean_text}\n\n{image_description}".strip() if clean_text else image_description

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

        # Flush any remaining table buffer at end of document
        _flush_table_buffer()

        self.logger.info("HybridChunker produced %d children", len(children))
        return children
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

        return ChunkRunOutput(
            chunks_vector=children,
            chunks_parent=parents,
        )