#!/usr/bin/env python3
"""Offline heuristic rewrite of the intentionally bad commit_message_writer skill.

Used when LLM/GEPA is unavailable (e.g. API balance). Applies the same rubric
the golden dataset expects — Conventional Commits, 72-char subject, dash bullets.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "hermes-agent" / "skills" / "dev" / "commit_message_writer" / "SKILL.md"

GOOD_BODY = """## Task

Given a git diff summary, write a **Conventional Commit** message.

## Required format

1. **Header**: `type(scope): subject`
   - **type** (required): one of `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `style`, `build`, `perf`
   - **scope** (optional): affected module, e.g. `auth`, `api`, `ci`
   - **subject**: imperative mood, **≤ 72 characters**, **no trailing period**, normal sentence case (not ALL CAPS)
2. **Blank line** after the header.
3. **Body** (optional): `-` dash bullets, one point per line.

## Procedure

1. Read the diff and pick the correct **type** (feat=new feature, fix=bugfix, docs=documentation, etc.).
2. Write a concise subject describing *what* changed, not how you feel about it.
3. Add 0–3 body bullets for non-obvious context only.

## Hard constraints

- Do **not** use ALL CAPS subjects.
- Do **not** use emoji in commit messages.
- Do **not** use numbered lists (`1.`) in the body — use `-` bullets only.
- Do **not** put file paths or essays in the subject line.

## Example

```
feat(auth): add JWT refresh token endpoint

- add POST /auth/refresh in routes.py
- extend test_tokens.py coverage
```
"""


def main():
    raw = SKILL_PATH.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise SystemExit("SKILL.md missing frontmatter")

    parts = raw.split("---", 2)
    frontmatter = parts[1].strip()
    # Update description in frontmatter
    lines = []
    for line in frontmatter.splitlines():
        if line.strip().startswith("description:"):
            lines.append('description: Write Conventional Commit messages from diff summaries (GEPA/heuristic-fixed)')
        else:
            lines.append(line)
    frontmatter = "\n".join(lines)

    fixed = f"---\n{frontmatter}\n---\n\n{GOOD_BODY}\n"
    out_dir = ROOT / "output" / "commit_message_writer" / "heuristic_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_skill.md").write_text(raw, encoding="utf-8")
    (out_dir / "evolved_skill.md").write_text(fixed, encoding="utf-8")
    SKILL_PATH.write_text(fixed, encoding="utf-8")
    print(f"Fixed skill deployed → {SKILL_PATH}")
    print(f"Diff saved under {out_dir}/")


if __name__ == "__main__":
    main()
