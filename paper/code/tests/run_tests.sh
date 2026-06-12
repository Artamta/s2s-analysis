#!/usr/bin/env bash
# Run the full test suite. Usage:  bash paper/code/tests/run_tests.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."          # -> repository root
exec python -m pytest paper/code/tests -v "$@"
