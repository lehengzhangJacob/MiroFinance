#!/usr/bin/env python3
"""Run the pinned shared open-market evaluator with explicit data inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


DEFAULT_SNAPSHOT = Path(
    os.environ.get(
        "ASHARE_COMPARE_SNAPSHOT",
        "/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_20260714",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(snapshot: Path) -> dict:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise RuntimeError(f"unsupported comparison manifest: {manifest_path}")
    return manifest


def artifact_path(snapshot: Path, manifest: dict, name: str) -> Path:
    item = manifest["artifacts"][name]
    path = snapshot / item["path"]
    actual = sha256(path)
    if actual != item["sha256"]:
        raise RuntimeError(
            f"{name} artifact hash mismatch: expected={item['sha256']} "
            f"actual={actual} path={path}"
        )
    return path


def load_reference_evaluator(
    evaluator_path: Path,
    expected_sha256: str,
) -> ModuleType:
    actual = sha256(evaluator_path)
    if actual != expected_sha256:
        raise RuntimeError(
            "reference evaluator hash mismatch: "
            f"expected={expected_sha256} actual={actual}"
        )
    spec = importlib.util.spec_from_file_location(
        "_miroflow_pinned_open_evaluator",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    args, evaluator_args = parser.parse_known_args()

    snapshot = args.snapshot.expanduser().resolve()
    manifest = load_manifest(snapshot)
    evaluator_path = artifact_path(snapshot, manifest, "evaluator")
    default_tasks = artifact_path(snapshot, manifest, "tasks")
    default_db = artifact_path(snapshot, manifest, "database")
    tasks = (args.tasks or default_tasks).expanduser().resolve()
    database = (args.db or default_db).expanduser().resolve()
    if sha256(tasks) != manifest["artifacts"]["tasks"]["sha256"]:
        raise RuntimeError(f"tasks do not match frozen manifest: {tasks}")
    if sha256(database) != manifest["artifacts"]["database"]["sha256"]:
        raise RuntimeError(f"database does not match frozen manifest: {database}")

    reference = load_reference_evaluator(
        evaluator_path,
        manifest["artifacts"]["evaluator"]["sha256"],
    )
    reference.TASKS = tasks
    reference.DB_PATH = database
    original_argv = sys.argv
    try:
        sys.argv = [str(evaluator_path), *evaluator_args]
        print(
            "pinned evaluator inputs: "
            f"tasks={manifest['artifacts']['tasks']['sha256']} "
            f"db={manifest['artifacts']['database']['sha256']} "
            f"evaluator={manifest['artifacts']['evaluator']['sha256']}"
        )
        reference.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
