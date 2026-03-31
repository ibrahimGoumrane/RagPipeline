"""Chunker — splits a DoclingDocument into parent + child chunks for RAG."""

from __future__ import annotations

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
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer, MarkdownTableSerializer
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem
from transformers import AutoTokenizer
from typing_extensions import override

from lib.models.main import ChunkRunOutput
from lib.utils.llm_client import LLMClient
from lib.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Custom Serializers
# ---------------------------------------------------------------------------

class ImageRefPictureSerializer(MarkdownPictureSerializer):
    """Saves picture to disk and injects the file path into the chunk content."""

    def __init__(self, output_dir: Path, doc_id: str, llm_client: LLMClient | None = None) -> None:
        self.output_dir = output_dir
        self.doc_id = doc_id
        self.llm_client = llm_client

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

        # 1. Save image to disk
        pil_image = item.get_image(doc)
        image_ref: str | None = None

        if pil_image:
            filename = f"{self.doc_id}_picture_{uuid.uuid4().hex[:8]}.png"
            filepath = self.output_dir / filename
            with filepath.open("wb") as fp:
                pil_image.save(fp, "PNG")
            image_ref = str(filepath)
            text_parts.append(f"image_ref:{image_ref}")

        # 2. Optionally describe via LLM
        if self.llm_client and image_ref:
            try:
                description = self.llm_client.describe_image(
                    image_input=image_ref,
                    system_prompt=(
                        "Analyze and describe this image in French. "
                        "Provide structured, professional content suitable for financial documents. "
                        "Focus on key data, visual elements, and important details."
                    ),
                )
                if description:
                    text_parts.append(description)
            except Exception as exc:
                pass  # image_ref is still in text_parts

        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


class ImageRefTableSerializer(MarkdownTableSerializer):
    """Saves table as image to disk, injects file path + markdown into chunk content."""

    def __init__(self, output_dir: Path, doc_id: str, llm_client: LLMClient | None = None) -> None:
        self.output_dir = output_dir
        self.doc_id = doc_id
        self.llm_client = llm_client

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        # 1. Get markdown from parent serializer (keeps table content as text too)
        md_result = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **kwargs,
        )

        text_parts: list[str] = [md_result.text] if md_result.text else []

        # 2. Save table image to disk
        pil_image = item.get_image(doc)
        image_ref: str | None = None

        if pil_image:
            filename = f"{self.doc_id}_table_{uuid.uuid4().hex[:8]}.png"
            filepath = self.output_dir / filename
            with filepath.open("wb") as fp:
                pil_image.save(fp, "PNG")
            image_ref = str(filepath)
            text_parts.append(f"image_ref:{image_ref}")

        # 3. Optionally describe via LLM
        if self.llm_client and image_ref:
            try:
                description = self.llm_client.describe_image(
                    image_input=image_ref,
                    system_prompt=(
                        "Analyze and describe this table in French. "
                        "Extract the key data, column headers, and summarize the main takeaways "
                        "suitable for financial documents."
                    ),
                )
                if description:
                    text_parts.append(description)
            except Exception as exc:
                pass

        text_res = "\n\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


class ImageRefSerializerProvider(ChunkingSerializerProvider):
    """Provides a ChunkingDocSerializer with custom picture and table serializers."""

    def __init__(self, output_dir: Path, doc_id: str, llm_client: LLMClient | None = None) -> None:
        self.output_dir = output_dir
        self.doc_id = doc_id
        self.llm_client = llm_client

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=ImageRefPictureSerializer(
                output_dir=self.output_dir,
                doc_id=self.doc_id,
                llm_client=self.llm_client,
            ),
            table_serializer=ImageRefTableSerializer(
                output_dir=self.output_dir,
                doc_id=self.doc_id,
                llm_client=self.llm_client,
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
        output_image_dir: str = "./output_images"
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
        """Instantiate a token-aware HybridChunker with custom serializers."""
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_model),
            max_tokens=self.max_words,
        )
        return HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
            repeat_table_header=True,
            serializer_provider=ImageRefSerializerProvider(  # <-- plugged in here
                output_dir=self.output_dir,
                doc_id=self.doc_id,
                llm_client=self.description_client,
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
        """Infer element type from doc_items in chunk metadata."""
        for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
            if isinstance(item, PictureItem):
                return "picture"
            if isinstance(item, TableItem):
                return "table"
        return "text"

    @staticmethod
    def _extract_image_refs(content: str) -> tuple[list[str], str]:
        """
        Pull out image_ref: lines from serialized content.
        Returns (image_refs, cleaned_content).
        """
        lines = content.splitlines()
        image_refs: list[str] = []
        clean_lines: list[str] = []

        for line in lines:
            if line.startswith("image_ref:"):
                image_refs.append(line[len("image_ref:"):].strip())
            else:
                clean_lines.append(line)

        return image_refs, "\n".join(clean_lines).strip()

    # ------------------------------------------------------------------
    # Core chunking pipeline
    # ------------------------------------------------------------------

    def _build_children(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """Run HybridChunker and return a list of child chunk dicts."""
        chunker = self._build_chunker()
        children: list[dict[str, Any]] = []

        for chunk in chunker.chunk(dl_doc=doc):
            content = chunker.contextualize(chunk)
            if not content:
                continue

            # Extract image refs injected by the custom serializers
            image_refs, clean_content = self._extract_image_refs(content)

            if not clean_content and not image_refs:
                continue

            children.append({
                "chunk_id": str(uuid.uuid4()),
                "_heading": self._chunk_heading(chunk),
                "element_type": self._element_type(chunk),
                "page_ref": self._chunk_page_no(chunk),
                "doc_id": self.doc_id,
                "content": clean_content,
                "content_for_embedding": clean_content,
                "token_estimate": len(clean_content.split()),
                "image_refs": image_refs,  # <-- naturally populated from serializer
            })

        self.logger.info("HybridChunker produced %d children", len(children))
        return children

    def _build_parents(
        self, children: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Group children by heading to produce parent records."""
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
                    "image_refs": [],  # <-- aggregate image refs at parent level too
                }
                order.append(heading)

            parent = grouped[heading]
            if parent["page_no"] is None:
                parent["page_no"] = child.get("page_ref")

            parent["full_content"].append(child["content"])
            parent["image_refs"].extend(child.get("image_refs", []))
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
        """Chunk *document* and return in-memory parent/child outputs."""
        children = self._build_children(document)
        parents, children = self._build_parents(children)
        self.logger.info("Chunking complete — %d parents, %d children", len(parents), len(children))

        import json
        # export children 
        with open(self.output_dir / f"{self.doc_id}_children.json", "w", encoding="utf-8") as fp:
            json.dump(children, fp)

        return ChunkRunOutput(
            chunks_vector=children,
            chunks_parent=parents,
        )