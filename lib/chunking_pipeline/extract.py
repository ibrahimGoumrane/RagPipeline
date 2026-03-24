"""Docling extraction — converts a PDF to a single markdown string."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from lib.models.main import ExtractRunOutput
from lib.utils.logger import get_logger


class DoclingExtractor:
    def __init__(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        accelerator_device: str | None = None,
        accelerator_num_threads: int | None = None,
        use_image_processor: bool | None = None,
        model_api_url: str | None = None,
        model_api_model: str | None = None,
    ):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.start_page = start_page
        self.end_page = end_page
        self.accelerator_device = (accelerator_device or "AUTO").upper()
        self.accelerator_num_threads = accelerator_num_threads
        self.use_image_processor = use_image_processor
        self.model_api_url = model_api_url
        self.model_api_model = model_api_model
        self.logger = get_logger(name="DoclingExtractor", log_level=logging.INFO)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _accelerator_device(self) -> AcceleratorDevice:
        mapping = {
            "AUTO": AcceleratorDevice.AUTO,
            "CPU": AcceleratorDevice.CPU,
            "MPS": AcceleratorDevice.MPS,
            "CUDA": AcceleratorDevice.CUDA,
            "XPU": AcceleratorDevice.XPU,
        }
        return mapping.get(self.accelerator_device, AcceleratorDevice.AUTO)

    def _pipeline_options(self) -> PdfPipelineOptions:
        options = PdfPipelineOptions()
        options.accelerator_options = AcceleratorOptions(
            num_threads=self.accelerator_num_threads,
            device=self._accelerator_device(),
        )
        options.do_ocr = False
        options.ocr_options.lang = ["fr", "en"]
        options.do_table_structure = True
        options.generate_picture_images = False
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
        if self.start_page is not None and self.end_page is not None:
            return converter.convert(self.pdf_path, page_range=(self.start_page, self.end_page))

        return converter.convert(self.pdf_path)

    def run(self) -> ExtractRunOutput:
        result = self._convert()
        markdown = result.document.export_to_markdown().strip() + "\n"
        output_file = self.output_dir / "full_document.md"
        output_file.write_text(markdown, encoding="utf-8")

        self.logger.info("Extraction complete — markdown written to %s", output_file)

        return ExtractRunOutput(
            document=result.document,
            output_file=str(output_file),
            output_dir=str(self.output_dir),
        )
