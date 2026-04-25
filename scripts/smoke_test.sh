#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/track_pytest.py tests/test_models.py tests/test_reset_step_state.py
