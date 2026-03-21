"""Worker module for processing one page-range chunk of a PDF."""

from pathlib import Path
import uuid

from lib.chunk import Chunker
from lib.extract import DoclingExtractor
from lib.models.main import WorkRunOutput


class Work:
    def __init__(
        self,
        *,
        pdf_path: str,
        output_dir: str,
        doc_id: str,
        max_words_per_chunk: int | None,
        min_words_per_chunk: int | None,
        use_image_processor: bool,
        use_hierarchical_headings: bool,
        model_api_url: str | None,
        model_api_model: str | None,
        start_page: int,
        end_page: int,
    ):
        self.worker_id = str(uuid.uuid4())
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.doc_id = doc_id
        self.max_words_per_chunk = max_words_per_chunk
        self.min_words_per_chunk = min_words_per_chunk
        self.use_image_processor = use_image_processor
        self.use_hierarchical_headings = use_hierarchical_headings
        self.model_api_url = model_api_url
        self.model_api_model = model_api_model
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
            use_hierarchical_headings=self.use_hierarchical_headings,
            model_api_url=self.model_api_url,
            model_api_model=self.model_api_model,
        )
        extract_result = extractor.run()

        chunker = Chunker(
            doc_id=self.doc_id,
            max_words=self.max_words_per_chunk,
            min_words=self.min_words_per_chunk,
        )

        chunk_result = chunker.run_to_output(
            document=extract_result.document,
            output_dir=str(worker_output_dir),
        )

        return WorkRunOutput(
            worker_id=self.worker_id,
            chunks_vector=chunk_result.chunks_vector,
            chunks_parent=chunk_result.chunks_parent,
            chunks_vector_path=chunk_result.chunks_vector_path,
            chunks_parent_path=chunk_result.chunks_parent_path,
            status="success",
        )
