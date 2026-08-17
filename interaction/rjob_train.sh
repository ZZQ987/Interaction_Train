#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LAUNCHER_SCRIPT="${SCRIPT_DIR}/actually-run-shell.sh"
CONFIG_FILE="${SCRIPT_DIR}/configs/minicpmo_32gpu_event_clean.sh"

if [[ ! -x "${LAUNCHER_SCRIPT}" ]]; then
  echo "Missing or non-executable launcher script: ${LAUNCHER_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing experiment config: ${CONFIG_FILE}" >&2
  exit 2
fi

exec "${LAUNCHER_SCRIPT}" --config "${CONFIG_FILE}" "$@"
