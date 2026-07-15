#!/usr/bin/env python3
"""Archive, inspect, and reset MiroMemSkill runtime memory safely.

Static skill libraries under ``memory_bank/skills*`` are intentionally outside
the cleanup scope. Only top-level runtime files are archived and removed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import tarfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = REPO_ROOT / "memory_bank"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / ".memory_archives"

RUNTIME_PATTERNS = (
    "*_memories.jsonl",
    "*_history.jsonl",
    "*_samples.jsonl",
    "*_outcomes.jsonl",
    "*_episodic.jsonl",
    "*_semantic.jsonl",
    "*.lock",
    "*.bak*",
    "mem0_history.db",
    "mem0_history.db-*",
)


def _runtime_files(memory_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in RUNTIME_PATTERNS:
        files.update(path for path in memory_dir.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int | None:
    if path.suffix not in {".jsonl", ".lock"}:
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _manifest(memory_dir: Path, files: list[Path]) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "memory_dir": str(memory_dir.resolve()),
        "preserved_directories": [
            str(path.relative_to(memory_dir))
            for path in (memory_dir / "skills", memory_dir / "skills_ashare")
            if path.is_dir()
        ],
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "lines": _line_count(path),
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }


def _acquire_runtime_locks(files: list[Path], stack: ExitStack) -> None:
    """Fail rather than reset while another process holds a namespace lock."""
    for path in files:
        if path.suffix != ".lock":
            continue
        handle: BinaryIO = stack.enter_context(path.open("a+b"))
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"runtime lock is busy: {path}") from exc


def _write_archive(
    archive_path: Path,
    manifest: dict,
    files: list[Path],
) -> None:
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    with tarfile.open(tmp_path, "w:gz") as tar:
        manifest_info = tarfile.TarInfo("manifest.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(datetime.now().timestamp())
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for path in files:
            tar.add(path, arcname=f"memory_bank/{path.name}", recursive=False)
    os.replace(tmp_path, archive_path)


def _verify_archive(archive_path: Path, manifest: dict) -> None:
    expected = {item["path"]: item for item in manifest["files"]}
    with tarfile.open(archive_path, "r:gz") as tar:
        members = {
            member.name.removeprefix("memory_bank/"): member
            for member in tar.getmembers()
            if member.isfile() and member.name.startswith("memory_bank/")
        }
        if set(members) != set(expected):
            raise RuntimeError("archive member set does not match manifest")
        for name, item in expected.items():
            extracted = tar.extractfile(members[name])
            if extracted is None:
                raise RuntimeError(f"cannot read archived member: {name}")
            digest = hashlib.sha256(extracted.read()).hexdigest()
            if digest != item["sha256"]:
                raise RuntimeError(f"checksum mismatch for archived member: {name}")


def archive_reset(memory_dir: Path, archive_root: Path, dry_run: bool) -> int:
    files = _runtime_files(memory_dir)
    if not files:
        print(f"No runtime memory files found in {memory_dir}")
        return 0

    manifest = _manifest(memory_dir, files)
    print(f"Runtime files: {len(files)}")
    print(f"Runtime bytes: {sum(item['bytes'] for item in manifest['files'])}")
    for item in manifest["files"]:
        print(f"  {item['path']} ({item['bytes']} bytes)")
    if dry_run:
        print("Dry run: no archive created and no files removed.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = archive_root / stamp
    target_dir.mkdir(parents=True, exist_ok=False)
    archive_path = target_dir / "memory_bank.tar.gz"
    manifest_path = target_dir / "manifest.json"

    with ExitStack() as stack:
        _acquire_runtime_locks(files, stack)
        _write_archive(archive_path, manifest, files)
        _verify_archive(archive_path, manifest)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in files:
            path.unlink()

    remaining = _runtime_files(memory_dir)
    if remaining:
        raise RuntimeError(f"runtime files remain after reset: {remaining}")
    print(f"Archive verified: {archive_path}")
    print(f"Manifest: {manifest_path}")
    print("Runtime memory reset complete; static skills were preserved.")
    return 0


def status(memory_dir: Path) -> int:
    files = _runtime_files(memory_dir)
    print(f"Memory directory: {memory_dir}")
    print(f"Runtime files: {len(files)}")
    for path in files:
        print(f"  {path.name} ({path.stat().st_size} bytes)")
    for name in ("skills", "skills_ashare"):
        path = memory_dir / name
        count = (
            sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0
        )
        print(f"Preserved {name}: {count} files")
    return 0


def qdrant_reset(
    *,
    host: str,
    port: int,
    collection: str,
    history_db: Path,
    confirmed: bool,
) -> int:
    """Delete the Mem0 Qdrant collection(s) and local SQLite history."""
    if not confirmed:
        raise RuntimeError("qdrant-reset is destructive; pass --yes to confirm")

    from qdrant_client import QdrantClient

    client = QdrantClient(host=host, port=port)
    collection_names = {item.name for item in client.get_collections().collections}
    targets = sorted(
        name
        for name in collection_names
        if name == collection or name.startswith(f"{collection}_")
    )
    for name in targets:
        client.delete_collection(name)
        print(f"Deleted Qdrant collection: {name}")
    if not targets:
        print(f"No Qdrant collection matched: {collection}")

    for path in (
        history_db,
        Path(f"{history_db}-wal"),
        Path(f"{history_db}-shm"),
    ):
        if path.exists():
            path.unlink()
            print(f"Deleted Mem0 history file: {path}")
    print("Official Mem0 backend reset complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("status", "archive-reset", "qdrant-reset"),
        help="Inspect, archive/reset sidecars, or reset the official backend.",
    )
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--qdrant-host", default=os.getenv("MEM0_QDRANT_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.getenv("MEM0_QDRANT_PORT", "6333")),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MEM0_QDRANT_COLLECTION", "miromemskill"),
    )
    parser.add_argument("--history-db", type=Path)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    memory_dir = args.memory_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    if not memory_dir.is_dir():
        parser.error(f"memory directory does not exist: {memory_dir}")

    if args.command == "status":
        return status(memory_dir)
    if args.command == "qdrant-reset":
        return qdrant_reset(
            host=args.qdrant_host,
            port=args.qdrant_port,
            collection=args.collection,
            history_db=(
                args.history_db.resolve()
                if args.history_db
                else (memory_dir / "mem0_history.db").resolve()
            ),
            confirmed=args.yes,
        )
    return archive_reset(memory_dir, archive_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
