# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Construct the configured memory storage backend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.memory.official_mem0_store import OfficialMem0Store
from src.memory.vector_store import VectorStore


def create_memory_store(
    *,
    store_dir: str | Path,
    namespace: str,
    backend: str = "mem0_qdrant",
    embedding_model: str = "embedding-3",
    embedding_dims: int = 2048,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    collection_name: str | None = None,
    history_db_path: str | Path | None = None,
) -> Any:
    namespace = str(namespace or "").strip()
    if (
        not namespace
        or len(namespace) > 128
        or re.fullmatch(r"[A-Za-z0-9_.-]+", namespace) is None
    ):
        raise ValueError(
            "memory namespace must be 1-128 characters using only "
            "letters, digits, underscore, dot, or hyphen"
        )
    normalized = str(backend or "mem0_qdrant").strip().lower()
    if normalized in {"mem0_qdrant", "mem0", "qdrant"}:
        return OfficialMem0Store(
            store_dir=store_dir,
            namespace=namespace,
            embedding_model=embedding_model,
            embedding_dims=embedding_dims,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
            history_db_path=history_db_path,
        )
    if normalized in {"jsonl", "legacy_jsonl"}:
        return VectorStore(
            store_dir=store_dir,
            namespace=namespace,
            embedding_model=embedding_model,
        )
    raise ValueError(f"Unsupported memory backend: {backend!r}")
