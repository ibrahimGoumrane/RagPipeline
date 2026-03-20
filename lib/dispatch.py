"""Dispatch pipeline stage."""

from lib.models.main import DispatchRunOutput


class Dispatch:
    def __init__(self):
        pass

    def run(self) -> DispatchRunOutput:
        # TODO: Implement dispatch logic here
        
        return DispatchRunOutput(
            status="success",
            message="Dispatched"
        )
