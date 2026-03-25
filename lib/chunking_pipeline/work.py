"""Worker module for processing one page-range chunk of a PDF."""

from pathlib import Path
import uuid

from lib.models.main import WorkRunOutput

from .chunk import Chunker
from .extract import DoclingExtractor


class Work:
    def __init__(
        self,
        *,
        pdf_path: str,
        output_dir: str,
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
        start_page: int,
        end_page: int,
    ):
        self.worker_id = str(uuid.uuid4())
        self.pdf_path = pdf_path
        self.output_dir = output_dir
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
        self.start_page = start_page
        self.end_page = end_page

    def run(self) -> WorkRunOutput:
        worker_output_dir = Path(self.output_dir) / "workers" / f"worker_{self.worker_id}"

        extractor = DoclingExtractor(
            pdf_path=self.pdf_path,
            output_dir=str(worker_output_dir),
            start_page=self.start_page,
            end_page=self.end_page,
            use_image_processor=self.use_image_processor,
            model_api_url=self.model_api_url,
            model_api_model=self.model_api_model,
            model_api_key=self.model_api_key,
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

        chunk_result = chunker.run(document=extract_result.document, output_dir=str(worker_output_dir))

        return WorkRunOutput(
            worker_id=self.worker_id,
            chunks_vector=chunk_result.chunks_vector,
            chunks_parent=chunk_result.chunks_parent,
            chunks_vector_path=chunk_result.chunks_vector_path,
            chunks_parent_path=chunk_result.chunks_parent_path,
            status="success",
        )
