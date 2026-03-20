"""Worker module for processing one page-range chunk of a PDF."""

from pathlib import Path

from lib.chunk import Chunker
from lib.config.main import cfg
from lib.extract import DoclingExtractor
from lib.models.main import WorkRunOutput


class Work:
    def __init__(self, worker_id: int, start_page: int, end_page: int, overlap_pages: int = 0):
        self.worker_id = worker_id
        self.start_page = start_page
        self.end_page = end_page
        self.overlap_pages = overlap_pages

    def run(self) -> WorkRunOutput:
        worker_output_dir = Path(cfg.output_dir) / "workers" / f"worker_{self.worker_id}"

        extractor = DoclingExtractor(
            pdf_path=cfg.pdf_path,
            output_dir=str(worker_output_dir),
            start_page=self.start_page,
            end_page=self.end_page,
            use_image_processor=cfg.use_image_processor,
            use_hierarchical_headings=cfg.use_hierarchical_headings,
            model_api_url=cfg.model_api_url,
            model_api_model=cfg.model_api_model,
        )
        extract_result = extractor.run()

        chunker = Chunker(
            doc_id=cfg.doc_id,
            max_words=cfg.max_words_per_chunk,
            min_words=cfg.min_words_per_chunk,
            overlap_sentences=cfg.overlap_sentences,
        )

        exclude_pages: set[int] = set()
        if self.worker_id > 0 and self.overlap_pages > 0:
            # These are the leading overlap pages in this worker range.
            exclude_pages = set(range(self.start_page, self.start_page + self.overlap_pages))

        chunk_result = chunker.run_to_output(
            document=extract_result.document,
            output_dir=str(worker_output_dir),
            exclude_pages=exclude_pages,
        )

        return WorkRunOutput(
            worker_id=self.worker_id,
            chunks_vector=chunk_result.chunks_vector,
            chunks_parent=chunk_result.chunks_parent,
            chunks_vector_path=chunk_result.chunks_vector_path,
            chunks_parent_path=chunk_result.chunks_parent_path,
            status="success",
        )
