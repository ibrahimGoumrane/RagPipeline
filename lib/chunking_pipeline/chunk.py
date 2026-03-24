"""Chunker — splits a DoclingDocument into parent + child chunks for RAG.

Strategy (two-tier):
  Parent  — one record per unique section heading; holds the full concatenated
             text of all children beneath it.  Used for context re-ranking.
  Child   — token-bounded content chunks produced by HybridChunker + contextualised
             by their heading.  These are what gets embedded and vector-searched.

Supported element types injected into children:
  • Text / SectionHeader  — via HybridChunker directly
  • Table                 — exported to HTML; optionally summarised by an LLM
  • Picture               — exported to Markdown (base64 or placeholder)
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem
from transformers import AutoTokenizer

from lib.models.main import ChunkRunOutput
from lib.utils.llm_client import LLMClient
from lib.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MAX_TOKENS = 512
_BROKEN_IMAGE_PLACEHOLDER = (
    "<!-- 🖼️❌ Image not available. "
    "Please use `PdfPipelineOptions(generate_picture_images=True)` -->"
)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Convert a :class:`DoclingDocument` into parent/child chunk pairs.

    Args:
        doc_id: Identifier attached to every chunk for downstream filtering.
        max_words: Token budget passed to :class:`HybridChunker`.
                   Defaults to ``512``.
        tokenizer_model: Hugging Face tokenizer id used by HybridChunker.
                         Defaults to ``sentence-transformers/all-MiniLM-L6-v2``.
        description_api_url: Base URL of an LLM used to summarise tables.
                             When omitted tables are stored as raw HTML.
        description_api_key: Bearer token for the description LLM.
        description_api_model: Model name for the description LLM.
    """

    def __init__(
        self,
        doc_id: str | None = None,
        max_words: int | None = None,
        tokenizer_model: str | None = None,
        description_api_url: str | None = None,
        description_api_key: str | None = None,
        description_api_model: str | None = None,
    ) -> None:
        self.doc_id = doc_id or "default-doc"
        self.max_words = max_words or _DEFAULT_MAX_TOKENS
        self.tokenizer_model = tokenizer_model or _DEFAULT_MODEL
        self.logger = get_logger(name="Chunker", log_level=logging.INFO)

        # Optional LLM client for table summarisation
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
        """Instantiate a token-aware :class:`HybridChunker`."""
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_model),
            max_tokens=self.max_words,
        )
        return HybridChunker(tokenizer=tokenizer, merge_peers=True)

    @staticmethod
    def _chunk_page_no(chunk: Any) -> int | None:
        """Return the first page number referenced by *chunk*, or ``None``."""
        for item in getattr(getattr(chunk, "meta", None), "doc_items", None) or []:
            prov = getattr(item, "prov", None)
            if prov and getattr(prov[0], "page_no", None) is not None:
                return int(prov[0].page_no)
        return None

    @staticmethod
    def _chunk_heading(chunk: Any) -> str:
        """Return the deepest heading in *chunk*'s metadata, or a sentinel."""
        headings = getattr(getattr(chunk, "meta", None), "headings", None) or []
        return str(headings[-1]).strip() if headings else "[Document root]"

    # ------------------------------------------------------------------
    # Element-level serialisers
    # ------------------------------------------------------------------

    # def _picture_text(self, item: PictureItem, doc: DoclingDocument) -> str:
    #     """Serialise a picture to Markdown, falling back to its caption text."""
    #     md = item.export_to_markdown(doc=doc)
    #     if isinstance(md, str):
    #         cleaned = md.replace(_BROKEN_IMAGE_PLACEHOLDER, "").strip()
    #         if cleaned:
    #             return cleaned

    #     # Fallback: resolve the first caption ref to its text
    #     if getattr(item, "captions", None):
    #         cap = item.captions[0].resolve(doc)
    #         cap_text = getattr(cap, "text", "") or ""
    #         if cap_text.strip():
    #             return f"Figure: {cap_text.strip()}"

    #     return "Figure: (no caption)"

    # def _table_text(self, item: TableItem, doc: DoclingDocument, page_no: int) -> str:
    #     """Export a table to HTML, optionally prefixed with an LLM summary."""
    #     try:
    #         html = item.export_to_html(doc=doc)
    #     except Exception:
    #         self.logger.warning("Table export failed on page %d", page_no)
    #         return "Unparsed Table"

    #     if not self.description_client:
    #         return html

    #     try:
    #         summary = self.description_client.summarize_table(html)
    #         if summary:
    #             return f"Table summary: {summary}\n\n{html}"
    #     except Exception as exc:
    #         self.logger.warning("Table summary failed on page %d: %s", page_no, exc)

    #     return html

    # ------------------------------------------------------------------
    # Core chunking pipeline
    # ------------------------------------------------------------------

    def _build_children(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """Run HybridChunker and return a list of child chunk dicts."""
        chunker = self._build_chunker()
        children: list[dict[str, Any]] = []

        for chunk in chunker.chunk(dl_doc=doc):
            
            # Contextualise the chunk's text content and skip if empty after stripping
            content = chunker.contextualize(chunk).strip()
            if not content:
                continue

            children.append({
                "chunk_id": str(uuid.uuid4()),
                # Temporary key used by _build_parents; removed afterwards
                "_heading": self._chunk_heading(chunk),
                "element_type": "text",
                "page_ref": self._chunk_page_no(chunk),
                "doc_id": self.doc_id,
                "content": content,
                "content_for_embedding": content,
                "token_estimate": len(content.split()),
            })

        self.logger.info("HybridChunker produced %d children", len(children))
        return children

    def _build_parents(
        self, children: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Group children by heading to produce parent records.

        Mutates each child dict in-place: replaces ``_heading`` with
        ``parent_id``.

        Returns:
            ``(parents, children)`` — both lists are ready to serialise.
        """
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
            # Use the first page encountered for this heading
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

    def run(self, document: DoclingDocument, output_dir: str | None = None) -> ChunkRunOutput:
        """Chunk *document* and write JSON outputs to *output_dir*.

        Args:
            document:   Parsed :class:`DoclingDocument` to chunk.
            output_dir: Root directory for outputs.  A ``chunks/`` sub-folder
                        is created automatically.  Defaults to ``output/``.

        Returns:
            :class:`ChunkRunOutput` with in-memory chunk lists and file paths.
        """
        out_dir = Path(output_dir or "output") / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        children = self._build_children(document)
        parents, children = self._build_parents(children)
        self.logger.info("Chunking complete — %d parents, %d children", len(parents), len(children))

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