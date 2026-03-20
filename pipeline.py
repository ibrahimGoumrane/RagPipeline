"""Pipeline entrypoint."""

from lib import cfg , DoclingExtractor, Chunker


def main() -> None:
    # cfg is read here, once, at run time
    # In the future we can easily extend this to read from a config file or environment variables
    
    extractor = DoclingExtractor(
        pdf_path=cfg.pdf_path,
        output_dir=cfg.output_dir,
        use_image_processor=cfg.use_image_processor,
        use_hierarchical_headings=cfg.use_hierarchical_headings,
        model_api_url=cfg.model_api_url,
        model_api_model=cfg.model_api_model,
    )
    result = extractor.run()

    chunker = Chunker(
        doc_id=cfg.doc_id,
        max_words=cfg.max_words_per_chunk,
        min_words=cfg.min_words_per_chunk,
        overlap_sentences=cfg.overlap_sentences,
    )
    chunker.run_to_output(result.markdown_content)


if __name__ == "__main__":
    main()