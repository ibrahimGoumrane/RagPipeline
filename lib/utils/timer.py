"""
This module provides a simple Timer class for measuring elapsed time in code execution. It can be used as a context manager to easily time blocks of code.

Example usage:
    with Timer("chunking_step") as timer:
        # Some code to time
        do_something()
"""

import logging
from time import perf_counter
from .logger import get_logger

class Timer:
    def __init__(self, function_name: str = "unnamed_operation", log_file_name: str = "time_elapsed.log"):
        self.function_name = function_name
        self.log_file_name = log_file_name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        self.logger = get_logger(
            name="Timer",
            log_level=logging.INFO,
            log_file_name=self.log_file_name,
        )

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.logger.info("%s took %.2f seconds", self.function_name, self.elapsed)

