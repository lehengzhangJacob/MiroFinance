# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent memory store with GLM embedding retrieval and BM25 fallback."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from src.utils.env_loader import load_project_env


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


@dataclass
class MemoryEntry:
    id: str
    kind: str  # episodic | semantic
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "tags": self.tags,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            kind=data.get("kind", "episodic"),
            content=data["content"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", _utc_now()),
        )


class MemoryStore:
    """JSONL-backed memory with embedding retrieval and BM25 fallback."""

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
        self.episodic_path = self.store_dir / f"{namespace}_episodic.jsonl"
        self.semantic_path = self.store_dir / f"{namespace}_semantic.jsonl"
        self._entries: list[MemoryEntry] = []
        self._load_all()

        self._api_key = os.getenv("GLM_API_KEY", "")
        self._base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        embedding_switch = os.getenv("MEMSKILL_EMBEDDING_ENABLED", "true").lower() != "false"
        self._embedding_enabled = bool(self._api_key) and embedding_switch

    def _load_all(self) -> None:
        self._entries = []
        for path in (self.episodic_path, self.semantic_path):
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._entries.append(MemoryEntry.from_dict(json.loads(line)))

    def _path_for_kind(self, kind: str) -> Path:
        return self.episodic_path if kind == "episodic" else self.semantic_path

    def _append_entry(self, entry: MemoryEntry) -> None:
        path = self._path_for_kind(entry.kind)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._entries.append(entry)

    def add(
        self,
        content: str,
        kind: str = "episodic",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        compute_embedding: bool = True,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            kind=kind,
            content=content.strip(),
            tags=tags or [],
            metadata=metadata or {},
        )
        if compute_embedding and self._embedding_enabled:
            try:
                entry.embedding = self._get_embedding(entry.content)
            except Exception:
                pass
        self._append_entry(entry)
        return entry

    def _get_embedding(self, text: str) -> list[float]:
        url = f"{self._base_url.rstrip('/')}/embeddings"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.embedding_model, "input": text[:8000]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _bm25_score(self, query: str, doc: str, avg_dl: float, df: Counter, N: int) -> float:
        k1, b = 1.5, 0.75
        q_terms = _tokenize(query)
        d_terms = _tokenize(doc)
        if not q_terms or not d_terms:
            return 0.0
        dl = len(d_terms)
        tf = Counter(d_terms)
        score = 0.0
        for term in set(q_terms):
            if term not in tf:
                continue
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
            freq = tf[term]
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        return score

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        kinds: Optional[list[str]] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        candidates = [
            e for e in self._entries if kinds is None or e.kind in kinds
        ]
        if not candidates:
            return []

        docs = [_tokenize(e.content) for e in candidates]
        N = len(candidates)
        df: Counter = Counter()
        for terms in docs:
            df.update(set(terms))
        avg_dl = sum(len(d) for d in docs) / max(N, 1)

        scored = [
            (entry, self._bm25_score(query, entry.content, avg_dl, df, N))
            for entry in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(e, s) for e, s in scored[:top_k] if s > 0]

    def search(
        self,
        query: str,
        top_k: int = 5,
        kinds: Optional[list[str]] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        candidates = [
            e for e in self._entries if kinds is None or e.kind in kinds
        ]
        if not candidates:
            return []

        if self._embedding_enabled:
            try:
                q_emb = self._get_embedding(query)
                scored: list[tuple[MemoryEntry, float]] = []
                for entry in candidates:
                    if entry.embedding:
                        scored.append((entry, self._cosine(q_emb, entry.embedding)))
                    else:
                        scored.append((entry, 0.0))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = [(e, s) for e, s in scored[:top_k] if s > 0.05]
                if top:
                    return top
            except Exception:
                pass

        return self._bm25_search(query, top_k, kinds)

    def format_search_results(self, results: list[tuple[MemoryEntry, float]]) -> str:
        if not results:
            return "No relevant memories found."
        lines = []
        for i, (entry, score) in enumerate(results, 1):
            tag_str = ", ".join(entry.tags) if entry.tags else "none"
            lines.append(
                f"{i}. [{entry.kind}|score={score:.3f}|tags={tag_str}] {entry.content}"
            )
        return "\n".join(lines)

    def count(self) -> dict[str, int]:
        return {
            "episodic": sum(1 for e in self._entries if e.kind == "episodic"),
            "semantic": sum(1 for e in self._entries if e.kind == "semantic"),
            "total": len(self._entries),
        }
