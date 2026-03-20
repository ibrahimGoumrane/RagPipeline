"""
Docling extraction — converts a PDF to a single markdown string.

Usage:
    extractor = DoclingExtractor("path/to/file.pdf")
    result    = extractor.run()
    print(result.markdown_content)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ListItem, PictureItem, SectionHeaderItem, TableItem, TextItem

from hierarchical.postprocessor import ResultPostprocessor
from lib.models.main import ExtractRunOutput
from lib.utils.logger import get_logger


class DoclingExtractor:
    def __init__(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        use_image_processor: bool | None = None,
        use_hierarchical_headings: bool | None = None,
        model_api_url: str | None = None,
        model_api_model: str | None = None,
    ):
        self.pdf_path                  = pdf_path
        self.output_dir                = Path(output_dir)
        self.use_image_processor       = use_image_processor  
        self.use_hierarchical_headings = use_hierarchical_headings
        self.model_api_url             = model_api_url  
        self.model_api_model           = model_api_model 
        self.logger = get_logger(name="DoclingExtractor", log_level=logging.INFO)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _pipeline_options(self) -> PdfPipelineOptions:
        options = PdfPipelineOptions()
        options.do_ocr = True
        options.ocr_options.lang = ["fr", "en"]
        options.do_table_structure = True
        options.generate_picture_images = True
        options.do_picture_description = self.use_image_processor
        options.enable_remote_services = self.use_image_processor

        if self.use_image_processor:
            options.picture_description_options = PictureDescriptionApiOptions(
                url=self.model_api_url,
                params={"model": self.model_api_model, "temperature": 0.0},
                headers={"Authorization": f"Bearer {os.getenv('API_KEY', '')}"},
                prompt=(
                    "You are analyzing a figure from a French report. "
                    "If the figure contains explicit numeric values (table or grid), "
                    "convert it to HTML using <table>, <tr>, <th>, <td>. "
                    "For multiple distinct data regions output multiple HTML tables. "
                    "If the figure is a chart without precise point labels, "
                    "explain what it shows inside a <chart>...</chart> tag."
                ),
                timeout=300,
            )

        return options

    def _convert(self):
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self._pipeline_options())}
        )
        return converter.convert(self.pdf_path)

    # ── Block building ────────────────────────────────────────────────────────

    def _build_blocks(self, doc) -> List[str]:
        blocks: List[str] = []

        for item, level in doc.iterate_items():

            if isinstance(item, (TextItem, SectionHeaderItem, ListItem)):
                text = (item.text or "").strip()
                if not text:
                    continue

                if isinstance(item, SectionHeaderItem):
                    depth = getattr(item, "level", level)
                    h = max(1, min(6, int(depth) + 1)) if depth is not None else 2
                    blocks.append(f"{'#' * h} {text}")
                elif isinstance(item, ListItem):
                    blocks.append(f"- {text}")
                else:
                    blocks.append(text)

            elif isinstance(item, TableItem):
                html = ""
                try:
                    html = item.export_to_html(doc=doc)
                except Exception:
                    try:
                        html = item.export_to_dataframe(doc=doc).to_html(index=False)
                    except Exception:
                        self.logger.warning("Table could not be exported — skipped")
                if html.strip():
                    blocks.append(html)

            elif isinstance(item, PictureItem):
                caption, description = "", ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                try:
                    if item.meta and item.meta.description:
                        description = item.meta.description.text or ""
                except Exception:
                    pass

                joined = "\n\n".join(p for p in [f"*Figure caption:* {caption}" if caption else "", description] if p)
                if joined:
                    blocks.append(joined)

        return blocks

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> ExtractRunOutput:
        """Convert the PDF, write markdown to disk, return ExtractRunOutput."""
        result = self._convert()

        if self.use_hierarchical_headings:
            try:
                ResultPostprocessor(result, source=self.pdf_path).process()
            except Exception as exc:
                self.logger.warning("Hierarchical postprocessing failed: %s", exc)

        blocks   = self._build_blocks(result.document)
        markdown = "\n\n".join(blocks).strip() + "\n"

        output_file = self.output_dir / "full_document.md"
        output_file.write_text(markdown, encoding="utf-8")
        self.logger.info("Extraction complete — %d blocks written to %s", len(blocks), output_file)

        return ExtractRunOutput(
            markdown_content=markdown,
            output_file=str(output_file),
            output_dir=str(self.output_dir),
        )