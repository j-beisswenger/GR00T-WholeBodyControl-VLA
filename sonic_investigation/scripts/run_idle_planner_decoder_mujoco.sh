#!/usr/bin/env bash
# Launch run_idle_planner_decoder_mujoco.py through `mjpython` (required for the
# MuJoCo passive viewer on macOS) with DYLD_FALLBACK_LIBRARY_PATH pointing at the
# venv's base-prefix lib/ — otherwise mjpython fails to dlopen libpython3.13.dylib
# when the interpreter was installed via uv.
set -euo pipefail

# Resolve repo root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PY="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "error: ${VENV_PY} not found — activate the venv or update VENV_PY in this script." >&2
    exit 1
fi

# Ask the venv where its libpython lives. base_prefix points at the uv-managed
# interpreter; its lib/ holds libpython3.13.dylib.
LIBPY_DIR="$("${VENV_PY}" -c 'import sys, os; print(os.path.join(sys.base_prefix, "lib"))')"

if [[ ! -f "${LIBPY_DIR}/libpython3.13.dylib" ]]; then
    echo "warning: libpython3.13.dylib not found in ${LIBPY_DIR}; mjpython may still fail." >&2
fi

# mjpython lives in the venv's bin (installed by `pip install mujoco`).
MJPYTHON="${REPO_ROOT}/.venv/bin/mjpython"
if [[ ! -x "${MJPYTHON}" ]]; then
    MJPYTHON="$(command -v mjpython || true)"
fi
if [[ -z "${MJPYTHON}" ]]; then
    echo "error: mjpython not found." >&2
    exit 1
fi

exec env \
    DYLD_FALLBACK_LIBRARY_PATH="${LIBPY_DIR}:${DYLD_FALLBACK_LIBRARY_PATH:-}" \
    "${MJPYTHON}" "${SCRIPT_DIR}/run_idle_planner_decoder_mujoco.py" "$@"
