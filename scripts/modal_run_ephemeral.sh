#!/usr/bin/env bash
set -euo pipefail
modal run scripts/modal_ephemeral_train.py --mode "${MODE:-smoke}" --episodes "${EPISODES:-4}" --seed-start "${SEED_START:-0}"
