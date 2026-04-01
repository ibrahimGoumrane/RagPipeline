"""Serializer for chunking pipeline, with custom handling for pictures and tables."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.html import  HTMLTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer, MarkdownTableSerializer
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem
from typing_extensions import override

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
        return create_ser_result(text=doc_serializer.post_process(text=f"Picture_{uuid.uuid4()}_Picture"), span_source=item)


class TableSerializer(HTMLTableSerializer):
    """Serializes table as Markdown and saves image to disk."""

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
        md_result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        text_parts: list[str] = [md_result.text] if md_result.text else []

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
