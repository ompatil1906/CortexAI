#!/usr/bin/env bash
# Deploy the SupportLM demo Space.
#
# Exists because the Space could not be created at first: the Hub rejects
# ZeroGPU for accounts under 30 days old (HTTP 402, "You must be subscribed to
# PRO to host Spaces with ZeroGPU"), and cpu-basic is rejected the same way.
# Once the account ages past 30 days, or on a PRO plan, this just works.
#
#   ./scripts/deploy_space.sh
#
# Requires: hf auth login, and either a >30-day-old account or PRO.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_DIR="${REPO_ROOT}/space_supportlm"
SPACE_ID="${SPACE_ID:-Ompatil19/supportlm-demo}"
HF="${HF:-${REPO_ROOT}/venv/bin/hf}"

cd "${REPO_ROOT}"

if [[ ! -x "${HF}" ]]; then
  HF="$(command -v hf || true)"
fi
if [[ -z "${HF}" ]]; then
  echo "error: hf CLI not found. Install with: curl -LsSf https://hf.co/cli/install.sh | bash" >&2
  exit 1
fi

echo "==> auth"
"${HF}" auth whoami >/dev/null 2>&1 || {
  echo "error: not logged in. Run: ${HF} auth login" >&2
  exit 1
}

echo "==> creating ${SPACE_ID} (zero-a10g)"
# --exist-ok so re-runs update the existing Space instead of failing.
"${HF}" repos create "${SPACE_ID}" \
  --type space --space-sdk gradio --flavor zero-a10g --public --exist-ok

echo "==> uploading app files"
"${HF}" upload "${SPACE_ID}" "${SPACE_DIR}" \
  --repo-type space \
  --exclude "__pycache__/**" --exclude "*.pyc" --exclude ".DS_Store"

cat <<EOF

==> pushed. Now watch it build and come up:

  ${HF} spaces info ${SPACE_ID} --expand runtime
  ${HF} spaces logs ${SPACE_ID} --follow

Space URL: https://huggingface.co/spaces/${SPACE_ID}

First boot downloads Qwen2.5-1.5B in bf16 (~3 GB), so allow several minutes
before the endpoint answers. Confirm it actually works rather than trusting
RUNNING: call it through gradio_client and read the log for silent CPU
fallbacks, which a healthy-looking Space can still be doing.
EOF