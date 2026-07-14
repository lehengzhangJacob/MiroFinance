#!/usr/bin/env bash
# Upload large A-share snapshot DBs to GitHub Release (background-friendly).
set -euo pipefail

AGENT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REPO="${MIROFINANCE_REPO:-lehengzhangJacob/MiroFinance}"
TAG="${ASHARE_OPEN_RELEASE_TAG:-ashare-open-20260714}"
STAGING="${AGENT_ROOT}/shared/ashare_open_stocks_glm52_20260714/.release_staging"
LOG_DIR="${AGENT_ROOT}/MiroFlow/logs/nohup"
LOG="${LOG_DIR}/release_upload_${TAG}.log"

mkdir -p "${STAGING}" "${LOG_DIR}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

log "start upload tag=${TAG} repo=${REPO}"

if [[ ! -f "${STAGING}/ashare_pools_snapshot.db" ]]; then
  cp -f "${AGENT_ROOT}/shared/ashare_open_stocks_glm52_20260714/ashare_pools_snapshot.db" \
    "${STAGING}/ashare_pools_snapshot.db"
fi
if [[ ! -f "${STAGING}/miromemskill_memfix02_full.db" ]]; then
  cp -f "${AGENT_ROOT}/shared/ashare_open_stocks_glm52_20260714/arms/20260714_memfix02_full/miromemskill.db" \
    "${STAGING}/miromemskill_memfix02_full.db"
fi

log "upload ashare_pools_snapshot.db (~611MB)"
gh release upload "${TAG}" \
  "${STAGING}/ashare_pools_snapshot.db" \
  --repo "${REPO}" \
  --clobber 2>&1 | tee -a "${LOG}"

log "upload miromemskill_memfix02_full.db (~611MB)"
gh release upload "${TAG}" \
  "${STAGING}/miromemskill_memfix02_full.db" \
  --repo "${REPO}" \
  --clobber 2>&1 | tee -a "${LOG}"

log "publish release (draft -> latest)"
gh release edit "${TAG}" \
  --repo "${REPO}" \
  --draft=false 2>&1 | tee -a "${LOG}"

log "done: https://github.com/${REPO}/releases/tag/${TAG}"
