#!/usr/bin/env bash
set -euo pipefail
uv run server --port "${PORT:-8000}"
