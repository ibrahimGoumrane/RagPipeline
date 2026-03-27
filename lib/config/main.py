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

    # Embed / Store
    embedding_api_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int

    milvus_host: str
    milvus_port: int
    milvus_children_collection: str
    milvus_parent_collection: str
    milvus_metric_type: str
    milvus_hnsw_m: int
    milvus_hnsw_ef_construction: int
    milvus_search_ef: int

    retrieve_top_k: int
    rerank_top_k: int
    similarity_floor: float
    reranker_model: str
    reranker_batch_size: int


cfg = Config(
    output_dir               = Path(os.getenv("OUTPUT_DIR", "output")),
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

    embedding_api_url        = os.getenv("EMBEDDING_API_URL", ""),
    embedding_api_key        = os.getenv("EMBEDDING_API_KEY", ""),
    embedding_model          = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
    embedding_dim            = int(os.getenv("EMBEDDING_DIM", "2560")),
    embedding_batch_size     = int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),

    milvus_host              = os.getenv("MILVUS_HOST", "localhost"),
    milvus_port              = int(os.getenv("MILVUS_PORT", "19530")),
    milvus_children_collection = os.getenv("MILVUS_CHILDREN_COLLECTION", "rag_chunks_children"),
    milvus_parent_collection = os.getenv("MILVUS_PARENT_COLLECTION", "rag_chunks_parent"),
    milvus_metric_type       = os.getenv("MILVUS_METRIC_TYPE", "COSINE"),
    milvus_hnsw_m            = int(os.getenv("MILVUS_HNSW_M", "16")),
    milvus_hnsw_ef_construction = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "200")),
    milvus_search_ef         = int(os.getenv("MILVUS_SEARCH_EF", "100")),

    retrieve_top_k           = int(os.getenv("RETRIEVE_TOP_K", "100")),
    rerank_top_k             = int(os.getenv("RERANK_TOP_K", "10")),
    similarity_floor         = float(os.getenv("SIMILARITY_FLOOR", "0.45")),
    reranker_model           = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    reranker_batch_size      = int(os.getenv("RERANKER_BATCH_SIZE", "32")),
)


def get_config() -> Config:
    """Return loaded configuration object."""
    return cfg