"""Top-level launcher delegating to the class-based chunking pipeline."""

from lib.chunking_pipeline import ChunkingPipeline
from lib.config import get_config


def main() -> None:
    ChunkingPipeline(config=get_config()).run()


if __name__ == "__main__":
    main()