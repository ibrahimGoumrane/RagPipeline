"""Pipeline entrypoint that reuses the utils package extractor."""

from utils import DoclingExtractor


def main() -> None:
	extractor = DoclingExtractor()
	extractor.run()


if __name__ == "__main__":
	main()
