"""Docling extraction — converts a PDF to a single markdown string."""

from __future__ import annotations

import logging

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from lib.models.main import ExtractRunOutput
from lib.utils.logger import get_logger


class DoclingExtractor:
    def __init__(
        self,
        pdf_path: str,
        start_page: int | None = None,
        end_page: int | None = None,
        accelerator_device: str | None = None,
        accelerator_num_threads: int | None = None,
    ):
        self.pdf_path = pdf_path
        self.start_page = start_page
        self.end_page = end_page
        self.accelerator_device = (accelerator_device or "AUTO").upper()
        self.accelerator_num_threads = accelerator_num_threads
        self.logger = get_logger(name="DoclingExtractor", log_level=logging.INFO)


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
        options.generate_picture_images = True

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
        return ExtractRunOutput(
            document=result.document,
        )
