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
        start_page: int | None = None,
        end_page: int | None = None,
        accelerator_device: str | None = None,
        accelerator_num_threads: int | None = None,
        use_image_processor: bool | None = None,
        model_api_url: str | None = None,
        model_api_model: str | None = None,
        model_api_key: str | None = None,
    ):
        self.pdf_path = pdf_path
        self.start_page = start_page
        self.end_page = end_page
        self.accelerator_device = (accelerator_device or "AUTO").upper()
        self.accelerator_num_threads = accelerator_num_threads
        self.use_image_processor = use_image_processor
        self.model_api_url = model_api_url
        self.model_api_model = model_api_model
        self.model_api_key = model_api_key or os.getenv("API_KEY", "")
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
        options.generate_picture_images = False
        options.do_picture_description = self.use_image_processor
        options.enable_remote_services = self.use_image_processor
        if self.use_image_processor:
            if not self.model_api_url:
                raise ValueError("MODEL_API_URL is required when image processor is enabled")
            if not self.model_api_model:
                raise ValueError("MODEL_API_MODEL is required when image processor is enabled")
            options.picture_description_options = PictureDescriptionApiOptions(
                url=self.model_api_url,
                params={"model": self.model_api_model, "temperature": 0.0},
                headers={"Authorization": f"Bearer {self.model_api_key}"},
        
                prompt=(
                    "Reponds en francais, concis et professionnel. "
                    "Donne uniquement le resultat final, sans explication. "
                    "Si valeurs numeriques explicites: produis un ou plusieurs tableaux HTML "
                    "avec seulement <table>, <tr>, <th>, <td>. "
                    "Sinon: reponds dans <chart>...</chart>. "
                    "Pas de markdown, pas de texte hors le contenu de image."
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

        return ExtractRunOutput(
            document=result.document,
        )
