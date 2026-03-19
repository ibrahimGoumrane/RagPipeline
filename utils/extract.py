"""
Docling extraction script for complex PDF documents

Setup:
    1. Replace your pyproject.toml with the one provided
    2. Run: uv sync
    3. Start vLLM in WSL first (for picture description):
           vllm serve "Qwen/Qwen2-VL-2B-Instruct-AWQ" --quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90
    4. Run: uv run python docling_extract.py
"""

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv 
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling_core.types.doc.document import TextItem, SectionHeaderItem, ListItem, TableItem, PictureItem
from hierarchical.postprocessor import ResultPostprocessor
from utils.logger import get_logger

class DoclingExtractor:
    def __init__(
        self,
        pdf_path: str | None = None,
        output_dir: str | None = None,
        use_image_processor: bool | None = None,
        model_api_url: str | None = None,
        model_api_model: str | None = None,
        use_hierarchical_headings: bool | None = None,
    ):
        load_dotenv()

        self.pdf_path = pdf_path or os.getenv("PDF_PATH", "RFA2024-pages.pdf")
        self.output_dir = Path(output_dir or os.getenv("OUTPUT_DIR", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_image_processor = use_image_processor if use_image_processor is not None else os.getenv("USE_IMAGE_PROCESSOR", "True").lower() == "true"
        self.model_api_url = model_api_url or os.getenv("MODEL_API_URL", "http://localhost:8000/v1/chat/completions")
        self.model_api_model = model_api_model or os.getenv("MODEL_API_MODEL", "Qwen/Qwen2-VL-2B-Instruct-AWQ")
        self.use_hierarchical_headings = (
            use_hierarchical_headings
            if use_hierarchical_headings is not None
            else os.getenv("USE_HIERARCHICAL_HEADINGS", "True").lower() == "true"
        )

        self.logger = get_logger(name="DoclingExtract", log_level=logging.DEBUG)
        self.logger.info("Starting extraction for PDF: %s", self.pdf_path)

    def _build_pipeline_options(self) -> PdfPipelineOptions:
        options = PdfPipelineOptions()
        options.do_ocr = True
        options.ocr_options.lang = ["fr", "en"]
        options.do_table_structure = True
        options.generate_picture_images = True

        # Picture description via vLLM API — no local model loading, no quantization issues
        options.do_picture_description = self.use_image_processor
        options.enable_remote_services = self.use_image_processor
        if self.use_image_processor:
            options.picture_description_options = PictureDescriptionApiOptions(
                url=self.model_api_url,
                params=dict(
                    model=self.model_api_model,
                    temperature=0.0,
                ),
                headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"},
                prompt=(
                    "You are analyzing a figure from a French report. "
                    "Do not limit the number of lines in your extraction when the figure contains many rows of data. "
                    "If the figure is data-heavy and contains explicit numeric values (for example a table or a grid with numbers), "
                    "convert the extracted content into complete HTML table(s) using <table>, <tr>, <th>, and <td>. "
                    "If the image contains multiple distinct data regions, output multiple HTML tables instead of merging unrelated data into one table. "
                    "Do not wrap table output inside <chart> tags. "
                    "If the figure is a chart where exact values are not explicitly available (for example a line chart without precise point labels), "
                    "explain what the chart shows and wrap the full explanation inside a <chart>...</chart> tag."
                ),
                timeout =300,  # seconds (5 minutes for image processing)
            )

        return options

    def _convert(self):
        self.logger.info("Stage 1/4 - Converting document")
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self._build_pipeline_options())}
        )
        result = converter.convert(self.pdf_path)
        self.logger.info("Conversion completed")
        return result

    def _apply_hierarchy_postprocess(self, result) -> None:
        # Enrich section hierarchy using docling-hierarchical-pdf when available.
        if not self.use_hierarchical_headings:
            return

        self.logger.info("Stage 2/4 - Applying hierarchical heading postprocessor")
        try:
            ResultPostprocessor(result, source=self.pdf_path).process()
            self.logger.info("Hierarchical heading postprocessing applied")
        except Exception as exc:
            self.logger.warning("Hierarchical postprocessing failed: %s", exc)

    def _build_markdown_blocks(self, doc) -> tuple[List[str], int, int]:
        blocks: List[str] = []
        table_count = 0
        figure_count = 0

        self.logger.info("Stage 3/4 - Building sequential markdown in source order")

        for item, level in doc.iterate_items():
            if isinstance(item, (TextItem, SectionHeaderItem, ListItem)):
                text = (item.text or "").strip()
                if not text:
                    continue

                if isinstance(item, SectionHeaderItem):
                    heading_level = 2
                    inferred_level = getattr(item, "level", level)
                    if inferred_level is not None:
                        heading_level = max(1, min(6, int(inferred_level) + 1))
                    blocks.append(f"{'#' * heading_level} {text}")
                elif isinstance(item, ListItem):
                    blocks.append(f"- {text}")
                else:
                    blocks.append(text)

            elif isinstance(item, TableItem):
                table_count += 1
                table_html = ""

                # Prefer native HTML export, then dataframe HTML fallback.
                try:
                    table_html = item.export_to_html(doc=doc)
                except Exception:
                    try:
                        df = item.export_to_dataframe(doc=doc)
                        table_html = df.to_html(index=False)
                    except Exception:
                        table_html = ""

                if table_html.strip():
                    blocks.append(table_html)
                else:
                    self.logger.warning("Table %s could not be exported to HTML; skipped", table_count)

                self.logger.info("Table %s -> inline in markdown", table_count)

            elif isinstance(item, PictureItem):
                figure_count += 1

                caption = ""
                description = ""

                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass

                try:
                    if item.meta and item.meta.description:
                        description = item.meta.description.text or ""
                except Exception:
                    pass

                figure_block = []
                if caption:
                    figure_block.append(f"*Figure caption:* {caption}")
                if description:
                    figure_block.append(description)

                if figure_block:
                    blocks.append("\n\n".join(figure_block))

                self.logger.info("Figure %s -> inline in markdown", figure_count)
                if description:
                    self.logger.debug("VLM figure description preview: %s...", description[:100])

        return blocks, table_count, figure_count

    def _write_markdown(self, blocks: List[str]) -> Path:
        sequential_md = "\n\n".join(blocks).strip() + "\n"
        output_file = self.output_dir / "full_document.md"
        output_file.write_text(sequential_md, encoding="utf-8")
        self.logger.info("Stage 4/4 - Markdown written to %s", output_file)
        return output_file

    def _log_summary(self, table_count: int, figure_count: int, output_file: Path) -> None:
        self.logger.info("=" * 50)
        self.logger.info("DOCLING EXTRACTION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info("Tables (inline): %s", table_count)
        self.logger.info("Figures (inline): %s", figure_count)
        self.logger.info("Output directory: %s", self.output_dir)
        self.logger.info("Output file: %s", output_file.name)

    def run(self) -> Path:
        result = self._convert()
        self._apply_hierarchy_postprocess(result)
        doc = result.document
        self.logger.info("Document object ready")

        blocks, table_count, figure_count = self._build_markdown_blocks(doc)
        output_file = self._write_markdown(blocks)
        self._log_summary(table_count, figure_count, output_file)
        return output_file


def main() -> None:
    extractor = DoclingExtractor()
    extractor.run()


if __name__ == "__main__":
    main()