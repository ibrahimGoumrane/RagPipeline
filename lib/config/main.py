"""
Environment configuration — load once, import anywhere.

    from lib.config import cfg

    _ = cfg.doc_id
    _ = cfg.model_api_url
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Output
    output_dir: Path

    # Chunking
    md_input_path: Path
    doc_id: str
    num_workers: int
    max_words_per_chunk: int
    tokenizer_model: str

    # Extractor
    pdf_path: str
    use_image_processor: bool
    model_api_url: str
    model_api_model: str
    api_key: str
    accelerator_device: str
    accelerator_num_threads: int

    # Description model
    description_api_url: str
    description_api_key: str
    description_api_model: str


cfg = Config(
    output_dir               = Path(os.getenv("OUTPUT_DIR", "output")),
    md_input_path            = Path(os.getenv("MD_INPUT_PATH", "output/full_document.md")),
    doc_id                   = os.getenv("DOC_ID", "document_001"),
    num_workers              = int(os.getenv("NUM_WORKERS", "4")),
    max_words_per_chunk      = int(os.getenv("MAX_WORDS_PER_CHUNK", "400")),
    tokenizer_model          = os.getenv("TOKENIZER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    use_image_processor      = os.getenv("USE_IMAGE_PROCESSOR", "true").lower() == "true",
    model_api_url            = os.getenv("MODEL_API_URL", ""),
    model_api_model          = os.getenv("MODEL_API_MODEL", ""),
    api_key                  = os.getenv("API_KEY", ""),
    accelerator_device       = os.getenv("ACCELERATOR_DEVICE", "AUTO"),
    accelerator_num_threads  = int(os.getenv("ACCELERATOR_NUM_THREADS", "8")),
    description_api_url      = os.getenv("DESCRIPTION_API_URL", ""),
    description_api_key      = os.getenv("DESCRIPTION_API_KEY", ""),
    description_api_model    = os.getenv("DESCRIPTION_API_MODEL", ""),
    pdf_path                 = os.getenv("PDF_PATH", ""),
)


def get_config() -> Config:
    """Return loaded configuration object."""
    return cfg