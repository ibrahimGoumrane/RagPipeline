"""Embedding and retrieval stage for the RAG chunk pipeline."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from lib.models.main import ChunkRunOutput
from lib.utils.llm_client import LLMClient
from lib.utils.logger import get_logger

from .store import Store


# Module-level cache so CrossEncoder weights are loaded from disk only once,
# regardless of how many Embed instances are created (e.g. in a worker pool).
_CROSS_ENCODER_CACHE: dict[str, CrossEncoder] = {}


class Embed:
    def __init__(
        self,
        store: Store | None = None,
        embedding_api_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        vector_dim: int | None = None,
        batch_size: int | None = None,
        retrieve_top_k: int | None = None,
        rerank_top_k: int | None = None,
        similarity_floor: float | None = None,
        reranker_model: str | None = None,
        reranker_batch_size: int | None = None,
    ) -> None:
        self.logger = get_logger(name="Embed", log_level=logging.INFO)

        self.embedding_api_url  = embedding_api_url  
        self.embedding_api_key  = embedding_api_key  
        self.embedding_model    = embedding_model   
        self.vector_dim          = vector_dim
        self.batch_size          = batch_size
        self.retrieve_top_k      = retrieve_top_k
        self.rerank_top_k        = rerank_top_k
        self.similarity_floor    = similarity_floor
        self.reranker_model      = reranker_model
        self.reranker_batch_size = reranker_batch_size

        self.client = LLMClient(
            api_url=self.embedding_api_url,
            api_key=self.embedding_api_key,
            model=self.embedding_model,
            timeout=120,
        )
        self.store = store

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """L2-normalise using numpy — significantly faster than pure Python at 2560 dims."""
        arr  = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm <= 0.0:
            return vec
        return (arr / norm).tolist()

    # ── text extraction ───────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(chunk: dict[str, Any]) -> str:
        return str(chunk.get("content_for_embedding")).strip()

    # ── file I/O ──────────────────────────────────────────────────────────────

    def _write_json_file(self, path: Path | str | None, data: dict[str, Any], default_path: Path) -> str:
        """Write data as JSON to disk and return the path."""
        target_path = Path(path) if path else default_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target_path)

    # ── embedding ─────────────────────────────────────────────────────────────
    
    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Call embedding API with max 2 attempts."""
        for attempt in range(2):
            try:
                return self.client.generate_embeddings(batch)
            except Exception as exc:
                if attempt == 1:
                    raise
                self.logger.warning("Embedding batch failed (attempt 1/2), retrying: %s", exc)
                time.sleep(1.0)

    def generate_embeddings(self, text_data: list[str]) -> list[list[float]]:
        if not text_data:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(text_data), self.batch_size):
            batch = text_data[start : start + self.batch_size]
            raw   = self._embed_batch_with_retry(batch)

            # Adding normalization here to ensure all vectors are on the same scale.
            for vec in raw:
                vectors.append(self._normalize(vec)) 
                

        if len(vectors) != len(text_data):
            raise ValueError("Embedding API returned unexpected number of vectors")

        return vectors
    # ── reranking ─────────────────────────────────────────────────────────────
    def _get_cross_encoder(self) -> CrossEncoder:
        """Return a cached CrossEncoder — model weights are loaded from disk only once."""
        if self.reranker_model not in _CROSS_ENCODER_CACHE:
            self.logger.info("Loading cross-encoder: %s", self.reranker_model)
            _CROSS_ENCODER_CACHE[self.reranker_model] = CrossEncoder(self.reranker_model)
        return _CROSS_ENCODER_CACHE[self.reranker_model]
    

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        model = self._get_cross_encoder()
        pairs = [[query, str(item.get("content") or "")] for item in candidates]
        scores = model.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []
        for item, score in zip(candidates, scores, strict=True):
            enriched = dict(item)
            enriched["cross_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda row: row["cross_score"], reverse=True)
        return reranked


    
    # ── ingest ────────────────────────────────────────────────────────────────

    def ingest(
        self,
        chunk_output: ChunkRunOutput,
        embedded_chunks_path: str | None = None,
    ) -> dict[str, Any]:
        payload  = chunk_output.model_dump()
        children = payload.get("chunks_vector") or []
        parents  = payload.get("chunks_parent") or []

        if not children:
            return {
                "status":               "success",
                "inserted_children":    0,
                "inserted_parents":     0,
                "embedded_chunks_path": None,
            }

        texts   = [self._chunk_text(chunk) for chunk in children]
        vectors = self.generate_embeddings(texts)

        embedded_children: list[dict[str, Any]] = []
        for chunk, vector in zip(children, vectors, strict=True):
            item = dict(chunk)
            item["embedding"] = vector
            item["metadata"] = {
                "doc_id":         item.get("doc_id"),
                "parent_id":      item.get("parent_id"),
                "page_ref":       item.get("page_ref"),
                "element_type":   item.get("element_type"),
                "token_estimate": item.get("token_estimate"),
            }
            embedded_children.append(item)

        store_result = self.store.save(
            ChunkRunOutput(
                chunks_vector=embedded_children,
                chunks_parent=parents,
            )
        )

        if embedded_chunks_path:
            embedded_chunks_path = self._write_json_file(
                embedded_chunks_path,
                embedded_children,
                Path(embedded_chunks_path),
            )

        return {
            "status":               "success",
            "inserted_children":    store_result["inserted_children"],
            "inserted_parents":     store_result["inserted_parents"],
            "embedded_chunks_path": embedded_chunks_path,
            "children_collection":  store_result["children_collection"],
            "parent_collection":    store_result["parent_collection"],
        }

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve_relevant_docs(
        self,
        query: str,
        *,
        top_n_candidates: int | None = None,
        top_k_final: int | None = None,
        doc_ids: list[str] | None = None,
        output_path: str | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Embed the query, run ANN retrieval, rerank, and return the top-k chunks.

        Args:
            query:            Natural language question.
            top_n_candidates: How many candidates to fetch from Milvus (default: retrieve_top_k).
            top_k_final:      How many chunks to return after reranking (default: rerank_top_k).
            doc_ids:          Optional filter — restrict search to one or more documents.
            output_path:      Custom path for the JSON output file.
            persist:          Write results to disk only when True or when output_path is given.
                              Keeps the hot retrieval path free of unnecessary I/O.
        """
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")



        query_vec  = self.generate_embeddings([query])[0]
        candidates = self.store.retrieve(
            query_embedding=query_vec,
            top_k=top_n_candidates or self.retrieve_top_k,
            similarity_floor=self.similarity_floor,
            doc_ids=doc_ids,
            retrieve_parent=False,
        )

        reranked = self._rerank(query, candidates)
        final_k  = max(1, int(top_k_final or self.rerank_top_k))
        selected = reranked[:final_k]

        parent_ids = [item.get("parent_id") for item in selected if item.get("parent_id")]
        parents    = self.store.get_parents_by_ids(parent_ids=parent_ids, doc_ids=doc_ids)
        parent_map = {row.get("parent_id"): row for row in parents}
        ##### Need abstraction after this to avoid parent map coupling in the output format
        for rank, item in enumerate(selected, start=1):
            item["rank"]   = rank
            item["parent"] = parent_map.get(item.get("parent_id"))

        payload = {
            "query":           query,
            "doc_ids":         doc_ids or [],
            "candidate_count": len(candidates),
            "selected_count":  len(selected),
            "created_at":      datetime.now(timezone.utc).isoformat(),
            "children":        selected,
            "parents":         list(parent_map.values()),
        }

        if persist or output_path:
            payload["output_path"] = self._write_json_file(
                output_path,
                payload,
                Path(output_path) if output_path else Path("retrieved_top10.json"),
            )

        return payload

