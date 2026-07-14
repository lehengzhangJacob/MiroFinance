#!/usr/bin/env bash
# Download large snapshot/arm DBs from GitHub Release (avoids Git LFS quota).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="${MIROFINANCE_REPO:-lehengzhangJacob/MiroFinance}"
TAG="${ASHARE_OPEN_RELEASE_TAG:-ashare-open-20260714}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

download() {
  local name="$1" dest="$2" sha256="$3"
  local url="${BASE}/${name}"
  local tmp="${dest}.part"
  echo "==> ${name} -> ${dest}"
  curl -fL --retry 3 --continue-at - -o "${tmp}" "${url}"
  echo "${sha256}  ${tmp}" | sha256sum -c -
  mv -f "${tmp}" "${dest}"
}

mkdir -p "${ROOT}/arms/20260714_memfix02_full"

download "ashare_pools_snapshot.db" \
  "${ROOT}/ashare_pools_snapshot.db" \
  "0149c0f50b987ff0e981269a5f3e367ef0d99b76c4d730a12f4ffa0169a4c37b"

download "miromemskill_memfix02_full.db" \
  "${ROOT}/arms/20260714_memfix02_full/miromemskill.db" \
  "7f55f7bbbc6d3a36ec2ca84853fd50a2b653904dd1ae6bba61f3bcc11635c62f"

echo "==> assets ready under ${ROOT}"
