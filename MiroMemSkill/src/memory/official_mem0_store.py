# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Official Mem0 + Qdrant storage adapter for MiroMemSkill.

Mem0 owns vector CRUD, extraction/consolidation, and SQLite history. The
application keeps its domain-specific temporal rules and statistical ledgers.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from src.memory.vector_store import MemoryConfigError, MemoryRecord
from src.utils.env_loader import load_project_env

# Disable anonymous telemetry unless the operator explicitly opted in.
os.environ.setdefault("MEM0_TELEMETRY", "false")

from mem0 import Memory  # noqa: E402  (environment must be set before import)


DEFAULT_COLLECTION = "miromemskill"
DEFAULT_EMBEDDING_DIMS = 2048
DEFAULT_MAX_RECORDS = 10_000

OFFICIAL_MEM0_INSTRUCTIONS = """\
Store only concise, reusable and evidence-backed lessons that can improve a
future task. Do not store the full answer, task boilerplate, stock-price
outcome, or an unconditional bullish/bearish direction prior. Prefer a
conditional heuristic with its scope and failure condition. Extract at most
one memory from one evaluated task. Preserve uncertainty and do not invent
facts absent from the supplied trace.
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _normalize_iso_date(value: Any) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) < 8:
        return ""
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}T00:00:00Z"


def normalize_mem0_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Normalize payload fields so Qdrant can apply native date filters."""
    result = dict(metadata or {})
    available_after = _normalize_iso_date(result.get("available_after"))
    if available_after:
        result["available_after"] = available_after
    return result


class OfficialMem0Store:
    """Compatibility adapter backed by official ``mem0ai.Memory``."""

    is_official_mem0 = True

    def __init__(
        self,
        store_dir: str | Path,
        namespace: str = "default",
        embedding_model: str = "embedding-3",
        embedding_dims: int = DEFAULT_EMBEDDING_DIMS,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        collection_name: str | None = None,
        history_db_path: str | Path | None = None,
    ):
        load_project_env()
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.embedding_dims = int(embedding_dims)
        self.collection_name = (
            collection_name or os.getenv("MEM0_QDRANT_COLLECTION") or DEFAULT_COLLECTION
        )
        self.qdrant_host = qdrant_host or os.getenv("MEM0_QDRANT_HOST", "127.0.0.1")
        self.qdrant_port = int(qdrant_port or _env_int("MEM0_QDRANT_PORT", 6333))
        self.history_db_path = Path(
            history_db_path
            or os.getenv("MEM0_HISTORY_DB_PATH", "")
            or self.store_dir / "mem0_history.db"
        )
        self._ledger_lock_path = self.store_dir / f"{namespace}.ledger.lock"
        self._max_records = _env_int("MEM0_MAX_RECORDS", DEFAULT_MAX_RECORDS)

        embedding_key = os.getenv("GLM_API_KEY", "")
        if not embedding_key:
            raise MemoryConfigError(
                "GLM_API_KEY is required for official Mem0 embeddings."
            )
        embedding_base_url = os.getenv(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        )

        llm_key = (
            os.getenv("REFLECTION_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or embedding_key
        )
        llm_base_url = (
            os.getenv("REFLECTION_LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or embedding_base_url
        )
        llm_model = (
            os.getenv("REFLECTION_LLM_MODEL_NAME")
            or os.getenv("DEEPSEEK_MODEL_NAME")
            or os.getenv("GLM_MODEL_NAME")
            or "deepseek-chat"
        )

        self.memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": self.collection_name,
                        "embedding_model_dims": self.embedding_dims,
                        "host": self.qdrant_host,
                        "port": self.qdrant_port,
                        "path": None,
                        "on_disk": True,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": self.embedding_model,
                        "api_key": embedding_key,
                        "openai_base_url": embedding_base_url,
                        "embedding_dims": self.embedding_dims,
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm_model,
                        "api_key": llm_key,
                        "openai_base_url": llm_base_url,
                        "temperature": 0.2,
                        "max_tokens": 4096,
                    },
                },
                "history_db_path": str(self.history_db_path),
                "custom_instructions": OFFICIAL_MEM0_INSTRUCTIONS,
            }
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Cross-process lock only for JSONL statistical sidecar ledgers."""
        with self._ledger_lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _scope_filters(
        self, filters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        scoped = dict(filters or {})
        scoped["user_id"] = self.namespace
        return scoped

    @staticmethod
    def _to_record(item: dict[str, Any]) -> MemoryRecord:
        metadata = dict(item.get("metadata") or {})
        return MemoryRecord(
            id=str(item["id"]),
            content=str(item.get("memory", "") or ""),
            metadata=metadata,
            embedding=None,
            created_at=str(item.get("created_at") or _utc_now()),
            updated_at=str(
                item.get("updated_at") or item.get("created_at") or _utc_now()
            ),
        )

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        embedding: Optional[list[float]] = None,
        source_task: str = "",
    ) -> MemoryRecord:
        """Store deterministic content without Mem0 LLM rewriting."""
        del embedding  # Official Mem0 owns embedding generation.
        payload = normalize_mem0_metadata(metadata)
        if source_task:
            payload.setdefault("source_task", source_task)
        result = self.memory.add(
            content,
            user_id=self.namespace,
            metadata=payload,
            infer=False,
        )
        items = result.get("results", [])
        if not items:
            raise RuntimeError("official Mem0 add returned no record")
        record = self.get(str(items[0]["id"]))
        if record is None:
            raise RuntimeError("official Mem0 added a record that cannot be read back")
        return record

    def add_inferred(
        self,
        messages: str | list[dict[str, str]],
        metadata: Optional[dict[str, Any]] = None,
        prompt: str = OFFICIAL_MEM0_INSTRUCTIONS,
    ) -> list[dict[str, Any]]:
        result = self.memory.add(
            messages,
            user_id=self.namespace,
            metadata=normalize_mem0_metadata(metadata),
            infer=True,
            prompt=prompt,
        )
        return list(result.get("results", []))

    def update(
        self,
        record_id: str,
        new_content: str,
        metadata_patch: Optional[dict[str, Any]] = None,
        source_task: str = "",
    ) -> Optional[MemoryRecord]:
        current = self.get(record_id)
        if current is None:
            return None
        metadata = dict(current.metadata)
        metadata.update(metadata_patch or {})
        if source_task:
            metadata["source_task"] = source_task
        self.memory.update(
            record_id,
            data=new_content,
            metadata=normalize_mem0_metadata(metadata),
        )
        return self.get(record_id)

    def delete(self, record_id: str, source_task: str = "", reason: str = "") -> bool:
        del source_task, reason  # Mem0 records the DELETE in its SQLite history.
        if self.get(record_id) is None:
            return False
        self.memory.delete(record_id)
        return True

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        try:
            item = self.memory.get(record_id)
        except ValueError:
            return None
        if item and item.get("user_id") != self.namespace:
            return None
        return self._to_record(item) if item else None

    def all_records(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryRecord]:
        result = self.memory.get_all(
            filters=self._scope_filters(filters),
            top_k=self._max_records,
            show_expired=True,
        )
        return [self._to_record(item) for item in result.get("results", [])]

    def search(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        top_k: int = 5,
        predicate: Optional[Callable[[MemoryRecord], bool]] = None,
        min_score: float = 0.0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[MemoryRecord, float]]:
        if top_k <= 0 or (not query and query_embedding is None):
            return []
        if query_embedding is not None and not query:
            raise ValueError("official Mem0 search requires query text")

        # Native Qdrant filters are exact. The predicate path only remains for
        # legacy callers and deliberately over-fetches before post-filtering.
        fetch_k = max(top_k * 20, 200) if predicate else top_k
        result = self.memory.search(
            str(query),
            top_k=fetch_k,
            filters=self._scope_filters(filters),
            threshold=min_score,
            show_expired=True,
        )
        picked: list[tuple[MemoryRecord, float]] = []
        for item in result.get("results", []):
            record = self._to_record(item)
            if predicate is not None and not predicate(record):
                continue
            picked.append((record, float(item.get("score", 0.0))))
            if len(picked) >= top_k:
                break
        return picked

    def embed(self, text: str) -> list[float]:
        return list(self.memory.embedding_model.embed(text, "search"))

    def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        return len(self.all_records(filters=filters))

    def history(self, record_id: str) -> list[dict[str, Any]]:
        return list(self.memory.history(record_id))

    def reset_namespace(self) -> None:
        self.memory.delete_all(user_id=self.namespace)
