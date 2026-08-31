#!/usr/bin/env bash
#
# bootstrap.sh — fetch the pinned third-party eval harness (exllamav3) into
# third_party/. Idempotent: safe to run repeatedly.
#
# This script NEVER downloads model weights. Weight fetching is an node-side,
# out-of-band step; this repo only ever references checkpoints by env var.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

EXL3_REPO="https://github.com/turboderp-org/exllamav3.git"
EXL3_REV="0c49587a7c235e6303a6bbedc8b665272ad3a2ea"   # must match manifests/runtime-pins.json
EXL3_DIR="third_party/exllamav3"

# ---- disk free check (informational, plus a weight-download guard) ----------------
disk_free_gb() {
  # GNU df; this script targets Linux nodes (Linux nodes), not macOS.
  df -BG --output=avail "${1:-.}" | tail -n 1 | tr -dc '0-9'
}
free_gb="$(disk_free_gb .)"
echo "Disk free: ${free_gb} GB (repo root)"

# The 20GB refusal below applies ONLY to model-weight downloads. bootstrap.sh never
# downloads weights, so it never trips this guard; it lives here so any future
# weight-download path has the refusal logic in one place.
download_weights_guard() {
  if (( free_gb < 20 )); then
    echo "ERROR: refusing weight download with <20GB free (have ${free_gb} GB)." >&2
    exit 1
  fi
}
# Intentionally NOT called by bootstrap.sh: no weights are downloaded here.

# ---- clone / verify exllamav3 at the pinned rev -----------------------------------
if [[ -d "$EXL3_DIR/.git" ]]; then
  current="$(git -C "$EXL3_DIR" rev-parse HEAD)"
  if [[ "$current" == "$EXL3_REV" ]]; then
    echo "exllamav3 already at pinned rev ${EXL3_REV}"
  else
    echo "exllamav3 present at ${current}; switching to pinned rev ${EXL3_REV}"
    git -C "$EXL3_DIR" fetch origin "$EXL3_REV" 2>/dev/null || git -C "$EXL3_DIR" fetch origin
    git -C "$EXL3_DIR" checkout "$EXL3_REV"
  fi
else
  mkdir -p third_party
  echo "Cloning exllamav3 (${EXL3_REPO}) ..."
  git clone "$EXL3_REPO" "$EXL3_DIR"
  git -C "$EXL3_DIR" checkout "$EXL3_REV"
fi

echo
echo "Bootstrap complete. exllamav3 is at $(git -C "$EXL3_DIR" rev-parse HEAD)."
echo
echo "Next steps:"
echo "  1. python3 -m pytest tests/ -q                 # offline integrity checks"
echo "  2. Copy this repo to the node (code+configs only; NEVER weights via this shared tree)."
echo "  3. On the DGX Spark node:"
echo "       DRY_RUN=1 ./scripts/run_fidelity_smoke.sh   # preview the qbench command"
echo "       BASE_MODEL_DIR=/path/to/zai-org/GLM-5.3-Flash \\"
echo "       QUANT_MODEL_DIR=/path/to/LibertAIDAI/GLM-5.3-Flash-NVFP4 \\"
echo "         ./scripts/run_fidelity_smoke.sh"
