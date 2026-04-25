#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/generate_scenario_cache.py --train-per-bucket 3 --validation-per-bucket 3 --heldout-per-bucket 3
