"""Testing module for retrieval examples and validation."""

from __future__ import annotations

import logging
from pathlib import Path

from lib.utils.logger import get_logger

from .embed import Embed


class Test:
    """Manages retrieval testing and validation against indexed chunks."""

    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger(name="Test", log_level=logging.INFO)

    def run_retrieval_examples(self, embedder: Embed) -> None:
        """Run fixed retrieval examples and persist raw retrieval outputs as JSON."""
        questions = [
            {
                "id": "q01",
                "question": "What is the specific social object (mission) of the UM6P Hospital entity integrated in 2024?",
            },
            {
                "id": "q02",
                "question": "Comparing 2023 and 2024, which product family saw a decrease in revenue, and by how much in millions of dirhams?",
            },
            {
                "id": "q03",
                "question": "Which geographical zone represents the largest share of OCP turnover in 2024, and what is its percentage?",
            },
            {
                "id": "q04",
                "question": "OCP is moving toward a multibusiness model. Name at least four new Strategic Business Units mentioned in the report.",
            },
            {
                "id": "q05",
                "question": "How does OCP determine when to recognize revenue for local sales versus export sales?",
            },
            {
                "id": "q06",
                "question": "As of October 2024, what is the annual capacity of the new desalination unit at Jorf Lasfar intended for Casablanca South?",
            },
            {
                "id": "q07",
                "question": "Which two subsidiaries were removed from the consolidation scope in December 2024, and why?",
            },
            {
                "id": "q08",
                "question": "According to Note 3.3, what was the total export turnover for the Nutricrops SBU in 2024?",
            },
            {
                "id": "q09",
                "question": "How does OCP Group manage transaction exchange rate risk for MAD versus Dollar and Euro exposure?",
            },
            {
                "id": "q10",
                "question": "In the OCP Green Water concession, what is the value of returnable assets and what is the concession duration?",
            },
        ]

        output_dir = Path(self.cfg.output_dir) / "retrieval" / "examples"

        self.logger.info("Running retrieval examples on %d questions", len(questions))

        for item in questions:
            embedder.retrieve_relevant_docs(
                query=item["question"],
                top_n_candidates=self.cfg.retrieve_top_k,
                top_k_final=self.cfg.rerank_top_k,
                doc_ids=[self.cfg.doc_id],
                persist=True,
                output_path=str(output_dir / f"{item['id']}.json"),
            )

        self.logger.info("Retrieval examples complete. JSON outputs saved to %s", output_dir)
