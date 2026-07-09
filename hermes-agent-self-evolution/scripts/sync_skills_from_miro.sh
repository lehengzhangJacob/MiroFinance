#!/usr/bin/env bash
# Copy MiroMemSkill seed skills into the local hermes-agent stub (all paths stay
# inside this repo — nothing is written to the parent agent/ tree).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/../MiroMemSkill/memory_bank/skills_ashare}"
DST="$ROOT/hermes-agent/skills/ashare"

mkdir -p "$DST"
for f in "$SRC"/*.md; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .md)"
    mkdir -p "$DST/$name"
    cp "$f" "$DST/$name/SKILL.md"
    echo "  $name -> hermes-agent/skills/ashare/$name/SKILL.md"
done
# Packaged skills (directory with SKILL.md + scripts/config): copy whole tree.
for d in "$SRC"/*/; do
    [ -f "$d/SKILL.md" ] || continue
    name="$(basename "$d")"
    mkdir -p "$DST/$name"
    rsync -a --delete --exclude='__pycache__' --exclude='.pytest_cache' \
        "$d" "$DST/$name/"
    echo "  $name/ -> hermes-agent/skills/ashare/$name/ (packaged)"
done
echo "done ($(ls "$DST" | wc -l) skills)"
