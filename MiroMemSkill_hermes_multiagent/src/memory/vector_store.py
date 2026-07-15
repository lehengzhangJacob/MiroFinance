# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding-backed memory store with file locking and an operation audit log.

Mem0-style storage layer: every record carries an embedding (GLM embedding-3,
mandatory —构造时无 key 直接失败，绝不静默降级为关键词检索) plus structured
metadata used for retrieval filters (source market month, functional stance).
All mutations (ADD/UPDATE/DELETE) are serialized through an fcntl file lock —
benchmark reflection, MCP tool processes and the injection path may touch the
same JSONL concurrently — and appended to a history JSONL for auditability.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.env_loader import load_project_env


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


class MemoryConfigError(RuntimeError):
    """Raised when the store cannot operate as configured (e.g. no embedding key)."""


@dataclass
class MemoryRecord:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", _utc_now()),
            updated_at=data.get("updated_at", data.get("created_at", _utc_now())),
        )


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    """JSONL-persisted vector store: one memories file + one history file per namespace."""

    def __init__(
        self,
        store_dir: str | Path,
        namespace: str = "default",
        embedding_model: str = "embedding-3",
    ):
        load_project_env()
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.memories_path = self.store_dir / f"{namespace}_memories.jsonl"
        self.history_path = self.store_dir / f"{namespace}_history.jsonl"
        self._lock_path = self.store_dir / f"{namespace}.lock"

        self._api_key = os.getenv("GLM_API_KEY", "")
        self._base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        if not self._api_key:
            raise MemoryConfigError(
                "GLM_API_KEY is required for memory embeddings (agent/llm_key). "
                "Refusing to run with keyword-only retrieval."
            )

    # ------------------------------------------------------------------ io

    @contextmanager
    def _locked(self):
        """Exclusive cross-process lock around any read-modify-write cycle."""
        with open(self._lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def _load(self) -> list[MemoryRecord]:
        if not self.memories_path.exists():
            return []
        records = []
        with open(self.memories_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(MemoryRecord.from_dict(json.loads(line)))
        return records

    def _write_all(self, records: list[MemoryRecord]) -> None:
        tmp = self.memories_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, self.memories_path)

    def _log_history(self, op: str, record_id: str, detail: dict[str, Any]) -> None:
        event = {"ts": _utc_now(), "op": op, "id": record_id, **detail}
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- embedding

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        url = f"{self._base_url.rstrip('/')}/embeddings"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.embedding_model, "input": text[:3000]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    # ------------------------------------------------------------ mutations

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        embedding: Optional[list[float]] = None,
        source_task: str = "",
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content.strip(),
            metadata=metadata or {},
            embedding=embedding if embedding is not None else self.embed(content),
        )
        with self._locked():
            with open(self.memories_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            self._log_history(
                "ADD", record.id, {"content": record.content, "source_task": source_task}
            )
        return record

    def update(
        self,
        record_id: str,
        new_content: str,
        metadata_patch: Optional[dict[str, Any]] = None,
        source_task: str = "",
    ) -> Optional[MemoryRecord]:
        new_embedding = self.embed(new_content)
        with self._locked():
            records = self._load()
            target = next((r for r in records if r.id == record_id), None)
            if target is None:
                return None
            old_content = target.content
            target.content = new_content.strip()
            target.embedding = new_embedding
            target.updated_at = _utc_now()
            if metadata_patch:
                target.metadata.update(metadata_patch)
            self._write_all(records)
            self._log_history(
                "UPDATE",
                record_id,
                {"old_content": old_content, "content": target.content, "source_task": source_task},
            )
            return target

    def delete(self, record_id: str, source_task: str = "", reason: str = "") -> bool:
        with self._locked():
            records = self._load()
            target = next((r for r in records if r.id == record_id), None)
            if target is None:
                return False
            records = [r for r in records if r.id != record_id]
            self._write_all(records)
            self._log_history(
                "DELETE",
                record_id,
                {"content": target.content, "reason": reason, "source_task": source_task},
            )
            return True

    # -------------------------------------------------------------- queries

    def all_records(self) -> list[MemoryRecord]:
        with self._locked():
            return self._load()

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return next((r for r in self.all_records() if r.id == record_id), None)

    def search(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        top_k: int = 5,
        predicate: Optional[Callable[[MemoryRecord], bool]] = None,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine search over records passing the metadata predicate."""
        if query_embedding is None:
            if not query:
                return []
            query_embedding = self.embed(query)
        candidates = [
            r
            for r in self.all_records()
            if r.embedding and (predicate is None or predicate(r))
        ]
        scored = [(r, cosine(query_embedding, r.embedding)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(r, s) for r, s in scored[:top_k] if s >= min_score]

    def count(self) -> int:
        return len(self.all_records())
