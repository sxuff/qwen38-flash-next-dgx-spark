#!/usr/bin/env bash
set -euo pipefail
# Prompt kernels default on. Forward --prefill-kernels (compatibility),
# --legacy-prefill-kernels (fallback), and --dry-run unchanged to Python.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/install_service.py" "$@"
