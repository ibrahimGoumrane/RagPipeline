"""
Environment configuration — load once, import anywhere.

    from lib.config import cfg

    print(cfg.doc_id)
    print(cfg.model_api_url)
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
    min_words_per_chunk: int
    max_words_per_chunk: int
    overlap_sentences: int

    # Extractor
    pdf_path: str
    use_image_processor: bool
    use_hierarchical_headings: bool
    model_api_url: str
    model_api_model: str
    api_key: str

    # Description model
    description_api_url: str
    description_api_key: str
    description_api_model: str


cfg = Config(
    output_dir               = Path(os.getenv("OUTPUT_DIR", "output")),
    md_input_path            = Path(os.getenv("MD_INPUT_PATH", "output/full_document.md")),
    doc_id                   = os.getenv("DOC_ID", "document_001"),
    min_words_per_chunk      = int(os.getenv("MIN_WORDS_PER_CHUNK", "200")),
    max_words_per_chunk      = int(os.getenv("MAX_WORDS_PER_CHUNK", "400")),
    overlap_sentences        = int(os.getenv("OVERLAP_SENTENCES", "1")),
    use_image_processor      = os.getenv("USE_IMAGE_PROCESSOR", "true").lower() == "true",
    use_hierarchical_headings= os.getenv("USE_HIERARCHICAL_HEADINGS", "true").lower() == "true",
    model_api_url            = os.getenv("MODEL_API_URL", ""),
    model_api_model          = os.getenv("MODEL_API_MODEL", ""),
    api_key                  = os.getenv("API_KEY", ""),
    description_api_url      = os.getenv("DESCRIPTION_API_URL", ""),
    description_api_key      = os.getenv("DESCRIPTION_API_KEY", ""),
    description_api_model    = os.getenv("DESCRIPTION_API_MODEL", ""),
    pdf_path                 = os.getenv("PDF_PATH", ""),
)