#!/usr/bin/env bash
set -euo pipefail
uv run pytest tests/test_models.py tests/test_reset_step_state.py
