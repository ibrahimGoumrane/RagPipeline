from .extract import DoclingExtractor
from .logger import get_logger

"""Utils package for the document extraction and logging.

Documentation:
1. DoclingExtractor: A class that encapsulates the logic for extracting structured data from PDF documents using the Docling library. It handles configuration, pipeline setup, and logging.
2. get_logger: A function that sets up and returns a logger instance for consistent logging across the application. It configures both file and console handlers with appropriate formatting.
"""

__all__ = ["DoclingExtractor", "get_logger"]

