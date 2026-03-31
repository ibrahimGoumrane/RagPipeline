"""Worker module for processing one page-range chunk of a PDF."""

import uuid

from lib.models.main import WorkRunOutput

from .chunk import Chunker
from .embed import Embed
from .extract import DoclingExtractor
from .store import Store


class Work:
    def __init__(
        self,
        *,
        pdf_path: str,
        doc_id: str,
        max_words_per_chunk: int | None,
        tokenizer_model: str | None,
        use_image_processor: bool,
        model_api_url: str | None,
        model_api_model: str | None,
        model_api_key: str | None,
        accelerator_device: str | None,
        accelerator_num_threads: int | None,
        description_api_url: str | None,
        description_api_key: str | None,
        description_api_model: str | None,
        embedding_api_url: str | None,
        embedding_api_key: str | None,
        embedding_model: str | None,
        embedding_dim: int,
        embedding_batch_size: int,
        milvus_host: str,
        milvus_port: int,
        milvus_children_collection: str,
        milvus_parent_collection: str,
        milvus_metric_type: str,
        milvus_hnsw_m: int,
        milvus_hnsw_ef_construction: int,
        milvus_search_ef: int,
        start_page: int,
        end_page: int,
    ):
        self.worker_id = str(uuid.uuid4())
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.max_words_per_chunk = max_words_per_chunk
        self.tokenizer_model = tokenizer_model
        self.use_image_processor = use_image_processor
        self.model_api_url = model_api_url
        self.model_api_model = model_api_model
        self.model_api_key = model_api_key
        self.accelerator_device = accelerator_device
        self.accelerator_num_threads = accelerator_num_threads
        self.description_api_url = description_api_url
        self.description_api_key = description_api_key
        self.description_api_model = description_api_model
        self.embedding_api_url = embedding_api_url
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_batch_size = embedding_batch_size
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.milvus_children_collection = milvus_children_collection
        self.milvus_parent_collection = milvus_parent_collection
        self.milvus_metric_type = milvus_metric_type
        self.milvus_hnsw_m = milvus_hnsw_m
        self.milvus_hnsw_ef_construction = milvus_hnsw_ef_construction
        self.milvus_search_ef = milvus_search_ef
        self.start_page = start_page
        self.end_page = end_page

    def run(self) -> WorkRunOutput:
        extractor = DoclingExtractor(
            pdf_path=self.pdf_path,
            start_page=self.start_page,
            end_page=self.end_page,
            # use_image_processor=self.use_image_processor,
            # model_api_url=self.model_api_url,
            # model_api_model=self.model_api_model,
            # model_api_key=self.model_api_key,
            accelerator_device=self.accelerator_device,
            accelerator_num_threads=self.accelerator_num_threads,
        )
        extract_result = extractor.run()

        chunker = Chunker(
            doc_id=self.doc_id,
            max_words=self.max_words_per_chunk,
            tokenizer_model=self.tokenizer_model,
            description_api_url=self.description_api_url,
            description_api_key=self.description_api_key,
            description_api_model=self.description_api_model,
        )

        chunk_result = chunker.run(document=extract_result.document)

        # store = Store(
        #     host=self.milvus_host,
        #     port=self.milvus_port,
        #     children_collection=self.milvus_children_collection,
        #     parent_collection=self.milvus_parent_collection,
        #     vector_dim=self.embedding_dim,
        #     metric_type=self.milvus_metric_type,
        #     hnsw_m=self.milvus_hnsw_m,
        #     hnsw_ef_construction=self.milvus_hnsw_ef_construction,
        #     search_ef=self.milvus_search_ef,
        # )
        # embedder = Embed(
        #     store=store,
        #     embedding_api_url=self.embedding_api_url,
        #     embedding_api_key=self.embedding_api_key,
        #     embedding_model=self.embedding_model,
        #     vector_dim=self.embedding_dim,
        #     batch_size=self.embedding_batch_size,
        # )
        # embedder.ingest(chunk_result)

        return WorkRunOutput(
            worker_id=self.worker_id,
            status="success",
        )
