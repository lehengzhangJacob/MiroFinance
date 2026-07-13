# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Pinned parity wrapper for the shared open-universe A-share MCP server.

The comparison intentionally runs the exact same market-tool implementation
for MiroFlow and MiroMemSkill.  This lightweight wrapper verifies the source
hash before loading it, so a later edit cannot silently invalidate a paired
run.  ``ASHARE_OPEN_SERVER`` can override the default sibling-repository path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType


EXPECTED_SERVER_SHA256 = (
    "6aa0f8ce4a43b42a25267d74a8beaf9fb650ae219e5d8155dede18c313a2c404"
)
DEFAULT_SERVER_PATH = (
    Path(__file__).resolve().parents[4]
    / "MiroMemSkill"
    / "src"
    / "tool"
    / "mcp_servers"
    / "ashare_open_mcp_server.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shared_server() -> ModuleType:
    source = Path(
        os.environ.get("ASHARE_OPEN_SERVER", str(DEFAULT_SERVER_PATH))
    ).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"shared ashare-open server not found: {source}")
    actual = _sha256(source)
    expected = os.environ.get(
        "ASHARE_OPEN_SERVER_SHA256", EXPECTED_SERVER_SHA256
    ).strip()
    if actual != expected:
        raise RuntimeError(
            "shared ashare-open server hash mismatch: "
            f"expected={expected} actual={actual} path={source}"
        )
    spec = importlib.util.spec_from_file_location(
        "_miroflow_shared_ashare_open_server",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared ashare-open server: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SHARED = load_shared_server()
mcp = _SHARED.mcp


if __name__ == "__main__":
    mcp.run()
