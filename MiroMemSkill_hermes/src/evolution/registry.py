# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed skill registry with CAS promotion and holdout leases.

Layout under ``<repo>/.evolution``::

    registry.json                  # single source of truth (atomic writes)
    candidates/<short_id>/<name>.md  # immutable materialized skill snapshots
    backups/<ts>_<short_id>.md     # pre-promotion copies of the active file

The registry never mutates candidate files after creation, and the active
production skill file is only rewritten through :meth:`SkillRegistry.promote`,
which is guarded by a compare-and-swap on the active file's current digest.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import stat
from pathlib import Path

from src.evolution.types import SkillArtifact, sha256_text


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class RegistryError(RuntimeError):
    pass


class HoldoutLeaseError(RegistryError):
    """A candidate tried to run the sealed holdout more than once."""


class PromotionError(RegistryError):
    pass


class SkillRegistry:
    def __init__(self, repo_root: str | Path, skill_rel_path: str):
        self.repo_root = Path(repo_root).resolve()
        self.root = self.repo_root / ".evolution"
        self.candidates_dir = self.root / "candidates"
        self.backups_dir = self.root / "backups"
        self.registry_path = self.root / "registry.json"
        self._lock_path = self.root / ".registry.lock"
        self.skill_rel_path = skill_rel_path
        self.skill_path = self.repo_root / skill_rel_path

    # ------------------------------------------------------------------ io

    def _load(self) -> dict:
        if not self.registry_path.exists():
            raise RegistryError(
                f"registry not initialized: {self.registry_path} missing; run init first"
            )
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self.registry_path)

    def _locked(self):
        """Context manager: exclusive advisory lock for mutations."""
        registry = self

        class _Lock:
            def __enter__(self):
                registry.root.mkdir(parents=True, exist_ok=True)
                self.fh = open(registry._lock_path, "w")
                fcntl.flock(self.fh, fcntl.LOCK_EX)
                return self

            def __exit__(self, *exc):
                fcntl.flock(self.fh, fcntl.LOCK_UN)
                self.fh.close()
                return False

        return _Lock()

    # ---------------------------------------------------------- lifecycle

    def init_baseline(self, force: bool = False) -> SkillArtifact:
        """Register the current production skill file as the baseline."""
        with self._locked():
            if self.registry_path.exists() and not force:
                raise RegistryError(
                    f"{self.registry_path} already exists (use force to re-init)"
                )
            text = self.skill_path.read_text(encoding="utf-8")
            artifact = SkillArtifact.from_text(text)
            self._materialize(artifact)
            data = {
                "skill_name": artifact.name,
                "skill_rel_path": self.skill_rel_path,
                "baseline_digest": artifact.digest,
                "active_digest": artifact.digest,
                "created_at": _utcnow(),
                "candidates": {
                    artifact.digest: self._record(artifact, "baseline", "", "baseline")
                },
                "holdout_leases": {},
                "promotions": [],
            }
            self._save(data)
            return artifact

    @staticmethod
    def _record(
        artifact: SkillArtifact, generator: str, rationale: str, status: str
    ) -> dict:
        return {
            "digest": artifact.digest,
            "short_id": artifact.short_id,
            "name": artifact.name,
            "parent_digest": artifact.parent_digest,
            "generator": generator,
            "rationale": rationale,
            "status": status,
            "created_at": _utcnow(),
            "reports": {},
        }

    def _materialize(self, artifact: SkillArtifact) -> Path:
        """Write the immutable candidate snapshot dir (single .md, read-only)."""
        dest_dir = self.candidates_dir / artifact.short_id
        dest = dest_dir / f"{artifact.name}.md"
        if dest.exists():
            existing = dest.read_text(encoding="utf-8")
            if sha256_text(existing) != artifact.digest:
                raise RegistryError(
                    f"materialized file {dest} exists with different content"
                )
            return dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(artifact.text, encoding="utf-8")
        dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return dest_dir

    def register_candidate(
        self,
        text: str,
        parent_digest: str,
        generator: str,
        rationale: str = "",
    ) -> SkillArtifact:
        with self._locked():
            data = self._load()
            if parent_digest not in data["candidates"]:
                raise RegistryError(f"unknown parent digest {parent_digest[:12]}")
            artifact = SkillArtifact.from_text(text, parent_digest=parent_digest)
            if artifact.digest in data["candidates"]:
                return artifact  # idempotent re-registration
            self._materialize(artifact)
            data["candidates"][artifact.digest] = self._record(
                artifact, generator, rationale, "proposed"
            )
            self._save(data)
            return artifact

    # ------------------------------------------------------------ queries

    def resolve(self, digest_or_short: str) -> dict:
        data = self._load()
        if digest_or_short in data["candidates"]:
            return data["candidates"][digest_or_short]
        matches = [
            rec
            for d, rec in data["candidates"].items()
            if d.startswith(digest_or_short) or rec["short_id"] == digest_or_short
        ]
        if len(matches) != 1:
            raise RegistryError(
                f"cannot resolve candidate {digest_or_short!r} "
                f"({len(matches)} matches)"
            )
        return matches[0]

    def artifact(self, digest_or_short: str) -> SkillArtifact:
        rec = self.resolve(digest_or_short)
        path = self.candidates_dir / rec["short_id"] / f"{rec['name']}.md"
        text = path.read_text(encoding="utf-8")
        artifact = SkillArtifact.from_text(text, parent_digest=rec["parent_digest"])
        if artifact.digest != rec["digest"]:
            raise RegistryError(
                f"materialized candidate {rec['short_id']} digest mismatch "
                "(tampered snapshot)"
            )
        return artifact

    def skill_dir(self, digest_or_short: str) -> Path:
        rec = self.resolve(digest_or_short)
        return self.candidates_dir / rec["short_id"]

    def active_digest(self) -> str:
        return self._load()["active_digest"]

    def baseline_digest(self) -> str:
        return self._load()["baseline_digest"]

    def status(self) -> dict:
        return self._load()

    # ---------------------------------------------------------- mutations

    def update_status(self, digest_or_short: str, status: str) -> None:
        with self._locked():
            data = self._load()
            rec = self.resolve(digest_or_short)
            data["candidates"][rec["digest"]]["status"] = status
            self._save(data)

    def attach_report(
        self, digest_or_short: str, report_kind: str, report: dict
    ) -> None:
        with self._locked():
            data = self._load()
            rec = self.resolve(digest_or_short)
            data["candidates"][rec["digest"]]["reports"][report_kind] = {
                "at": _utcnow(),
                **report,
            }
            self._save(data)

    def acquire_holdout_lease(self, digest_or_short: str, run_id: str) -> None:
        """One sealed-holdout run per candidate, enforced in code."""
        with self._locked():
            data = self._load()
            rec = self.resolve(digest_or_short)
            lease = data["holdout_leases"].get(rec["digest"])
            if lease is not None:
                raise HoldoutLeaseError(
                    f"candidate {rec['short_id']} already used its holdout run "
                    f"(run_id={lease['run_id']} at {lease['acquired_at']})"
                )
            data["holdout_leases"][rec["digest"]] = {
                "run_id": run_id,
                "acquired_at": _utcnow(),
            }
            self._save(data)

    def promote(self, digest_or_short: str, run_id: str = "") -> dict:
        """CAS-guarded promotion of a candidate into the production file."""
        with self._locked():
            data = self._load()
            rec = self.resolve(digest_or_short)
            candidate = self.artifact(rec["digest"])
            current_text = self.skill_path.read_text(encoding="utf-8")
            current_digest = SkillArtifact.from_text(current_text).digest
            if current_digest != data["active_digest"]:
                raise PromotionError(
                    "active skill file changed outside the registry "
                    f"(file={current_digest[:12]}, registry={data['active_digest'][:12]}); "
                    "refusing to promote"
                )
            self.backups_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = self.backups_dir / f"{ts}_{current_digest[:12]}.md"
            backup.write_text(current_text, encoding="utf-8")
            tmp = self.skill_path.with_suffix(".md.tmp")
            tmp.write_text(candidate.text, encoding="utf-8")
            os.replace(tmp, self.skill_path)
            previous = data["active_digest"]
            data["active_digest"] = candidate.digest
            data["candidates"][candidate.digest]["status"] = "promoted"
            data["promotions"].append(
                {
                    "digest": candidate.digest,
                    "previous_digest": previous,
                    "run_id": run_id,
                    "backup": str(backup.relative_to(self.repo_root)),
                    "at": _utcnow(),
                }
            )
            self._save(data)
            return data["promotions"][-1]

    def rollback(self, target_digest_or_short: str) -> None:
        """Restore a previously registered artifact as the active skill."""
        with self._locked():
            data = self._load()
            rec = self.resolve(target_digest_or_short)
            target = self.artifact(rec["digest"])
            tmp = self.skill_path.with_suffix(".md.tmp")
            tmp.write_text(target.text, encoding="utf-8")
            os.replace(tmp, self.skill_path)
            previous = data["active_digest"]
            data["active_digest"] = target.digest
            data["promotions"].append(
                {
                    "digest": target.digest,
                    "previous_digest": previous,
                    "rollback": True,
                    "at": _utcnow(),
                }
            )
            self._save(data)
