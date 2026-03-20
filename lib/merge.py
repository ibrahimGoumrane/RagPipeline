"""Merge pipeline stage."""

from lib.models.main import MergeRunOutput


class Merge:
    def __init__(self):
        pass

    def run(self) -> MergeRunOutput:
        # TODO: Implement merge logic here
        
        return MergeRunOutput(
            status="success",
            merged_file_path=None
        )
