"""Milvus-backed storage abstraction for child vectors and parent records."""

from __future__ import annotations

from typing import Any

# Use the modern high-level client
from pymilvus import DataType, MilvusClient

from lib.models.main import ChunkRunOutput
from lib.utils.logger import get_logger


def _quote(s: str) -> str:
    """Safely quote a string value for a Milvus filter expression."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Store:
    def __init__(
        self,
        host: str,
        port: int,
        children_collection: str,
        parent_collection: str,
        vector_dim: int,
        metric_type: str,
        hnsw_m: int,
        hnsw_ef_construction: int,
        search_ef: int,
    ) -> None:
        self.host = host
        self.port = port
        self.uri = f"http://{self.host}:{self.port}"
        
        self.children_collection = children_collection
        self.parent_collection = parent_collection
        self.vector_dim = vector_dim
        self.metric_type = metric_type.upper()
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.search_ef = search_ef
        self.token = "root:Milvus"
        self.logger = get_logger(name="Store")

        # 1. Initialize the modern MilvusClient (Handles connections automatically)
        self.client = MilvusClient(uri=self.uri , token=self.token)
        
        self._bootstrap_collections()

    # ── collection bootstrap ──────────────────────────────────────────────────

    def _bootstrap_collections(self) -> None:
        """Create collections using explicit schema + index params flow."""
        
        # --- Children Collection ---
        if not self.client.has_collection(self.children_collection):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="element_type", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="page_ref", datatype=DataType.INT64)
            schema.add_field(field_name="token_estimate", datatype=DataType.INT64)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_name="idx_children_embedding",
                index_type="HNSW",
                metric_type=self.metric_type,
                params={"M": self.hnsw_m, "efConstruction": self.hnsw_ef_construction},
            )

            self.client.create_collection(
                collection_name=self.children_collection,
                schema=schema,
                index_params=index_params,
            )

            self.logger.info("Created Milvus children collection: %s", self.children_collection)
        else:
            self.client.load_collection(self.children_collection)

        # --- Parent Collection ---
        if not self.client.has_collection(self.parent_collection):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
            schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="heading", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="page_no", datatype=DataType.INT64)
            schema.add_field(field_name="full_content", datatype=DataType.VARCHAR, max_length=65535)
            # MilvusClient create_collection expects at least one vector field.
            # Parent rows are queried by ids only, so we store a placeholder vector.
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_name="idx_parent_embedding",
                index_type="HNSW",
                metric_type=self.metric_type,
                params={"M": self.hnsw_m, "efConstruction": self.hnsw_ef_construction},
            )

            self.client.create_collection(
                collection_name=self.parent_collection,
                schema=schema,
                index_params=index_params,
            )
            self.logger.info("Created Milvus parent collection: %s", self.parent_collection)
        else:
            self.client.load_collection(self.parent_collection)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_string(value: Any, max_length: int = 65535) -> str:
        text = "" if value is None else str(value)
        return text[:max_length] if len(text) > max_length else text

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _delete_by_doc_ids(self, collection_name: str, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        quoted = ", ".join(_quote(doc_id) for doc_id in doc_ids)
        self.client.delete(collection_name=collection_name, filter=f"doc_id in [{quoted}]")

    def clear_doc(self, doc_id: str) -> None:
        cleaned = (doc_id or "").strip()
        if not cleaned:
            return
        self._delete_by_doc_ids(self.children_collection, [cleaned])
        self._delete_by_doc_ids(self.parent_collection, [cleaned])

    # ── write ─────────────────────────────────────────────────────────────────

    def insert_parents(self, parents: list[dict[str, Any]]) -> int:
        if not parents:
            return 0

        rows: list[dict[str, Any]] = []
        for row in parents:
            parent_id = Store._safe_string(row.get("parent_id"), 128)
            doc_id    = Store._safe_string(row.get("doc_id"), 256)
            if not parent_id or not doc_id:
                continue
            rows.append(
                {
                    "parent_id":    parent_id,
                    "doc_id":       doc_id,
                    "heading":      Store._safe_string(row.get("heading"), 2048),
                    "page_no":      Store._safe_int(row.get("page_no")),
                    "full_content": Store._safe_string(row.get("full_content"), 65535),
                    "embedding":    [0.0] * self.vector_dim,
                }
            )

        if not rows:
            return 0

        self.client.insert(collection_name=self.parent_collection, data=rows)
        return len(rows)

    def insert_children(
        self,
        chunks_with_embeddings: list[dict[str, Any]],
    ) -> int:
        if not chunks_with_embeddings:
            return 0

        rows: list[dict[str, Any]] = []
        for chunk in chunks_with_embeddings:
            embedding = chunk.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != self.vector_dim:
                continue

            chunk_id = Store._safe_string(chunk.get("chunk_id"), 128)
            doc_id   = Store._safe_string(chunk.get("doc_id"), 256)
            if not chunk_id or not doc_id:
                continue

            rows.append(
                {
                    "chunk_id":       chunk_id,
                    "parent_id":      Store._safe_string(chunk.get("parent_id"), 128),
                    "doc_id":         doc_id,
                    "element_type":   Store._safe_string(chunk.get("element_type"), 64),
                    "content":        Store._safe_string(chunk.get("content"), 65535),
                    "page_ref":       Store._safe_int(chunk.get("page_ref")),
                    "token_estimate": Store._safe_int(chunk.get("token_estimate")),
                    "embedding":      [float(v) for v in embedding],
                }
            )

        if not rows:
            return 0

        self.client.insert(collection_name=self.children_collection, data=rows)
        return len(rows)

    def save(self, data: ChunkRunOutput) -> dict[str, Any]:
        payload  = data.model_dump()
        parents  = payload.get("chunks_parent") or []
        children = payload.get("chunks_vector") or []

        inserted_parents  = self.insert_parents(parents)
        inserted_children = self.insert_children(children)

        return {
            "status":              "success",
            "inserted_parents":    inserted_parents,
            "inserted_children":   inserted_children,
            "children_collection": self.children_collection,
            "parent_collection":   self.parent_collection,
        }

    # ── read ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 100,
        similarity_floor: float = 0.45,
        *,
        doc_ids: list[str] | None = None,
        retrieve_parent: bool = True,
    ) -> list[dict[str, Any]]:

        filter_doc_ids = [d for d in (doc_ids or []) if d]

        expr: str | None = None
        if filter_doc_ids:
            quoted_ids = ", ".join(_quote(pid) for pid in filter_doc_ids)
            expr = f"doc_id in [{quoted_ids}]"

        hits = self.client.search(
            collection_name=self.children_collection,
            anns_field="embedding",
            data=[query_embedding],
            limit=max(1, int(top_k)),
            output_fields=[
                "chunk_id", "parent_id", "doc_id", "content",
                "page_ref", "token_estimate", "element_type",
            ],
            search_params={
                "metric_type": self.metric_type,
                "params": {"ef": max(1, int(self.search_ef))},
            },
            filter=expr,
        )

        results: list[dict[str, Any]] = []
        for hit in hits[0] if hits else []:
            entity   = hit.get("entity", {})
            distance = float(hit.get("distance", 0.0))

            if distance < similarity_floor:
                continue

            results.append(
                {
                    "chunk_id":       entity.get("chunk_id"),
                    "parent_id":      entity.get("parent_id"),
                    "doc_id":         entity.get("doc_id"),
                    "content":        entity.get("content", ""),
                    "page_ref":       entity.get("page_ref"),
                    "token_estimate": entity.get("token_estimate"),
                    "element_type":   entity.get("element_type"),
                    "bi_score":       distance,
                    "raw_score":      distance,
                }
            )

        if not retrieve_parent:
            return results

        parent_ids = [item.get("parent_id") for item in results if item.get("parent_id")]
        if not parent_ids:
            return []

        parent_rows = self.get_parents_by_ids(parent_ids=parent_ids, doc_ids=filter_doc_ids or None)
        score_by_parent: dict[str, float] = {}
        for item in results:
            parent_id = item.get("parent_id")
            if not parent_id:
                continue
            score = float(item.get("bi_score", 0.0))
            best = score_by_parent.get(parent_id)
            if best is None or score > best:
                score_by_parent[parent_id] = score

        parent_results: list[dict[str, Any]] = []
        for parent in parent_rows:
            parent_id = parent.get("parent_id")
            enriched = dict(parent)
            enriched["bi_score"] = score_by_parent.get(parent_id, 0.0)
            parent_results.append(enriched)

        parent_results.sort(key=lambda row: row.get("bi_score", 0.0), reverse=True)
        return parent_results

    def get_parents_by_ids(
        self,
        parent_ids: list[str],
        *,
        doc_ids: list[str] = [],
    ) -> list[dict[str, Any]]:

        cleaned = [pid for pid in dict.fromkeys(parent_ids) if pid]
        if not cleaned:
            return []

        quoted_ids = ", ".join(_quote(pid) for pid in cleaned)
        expr = f"parent_id in [{quoted_ids}]"
        if doc_ids:
            quoted_doc_ids = ", ".join(_quote(pid) for pid in doc_ids if pid)
            expr += f" and doc_id in [{quoted_doc_ids}]"

        rows = self.client.query(
            collection_name=self.parent_collection,
            filter=expr,
            output_fields=["parent_id", "doc_id", "heading", "page_no", "full_content"],
        )
        return list(rows)