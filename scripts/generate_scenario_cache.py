"""Prepare the validated CyberSecurity_OWASP scenario cache.

This command is intentionally offline/cache-prep work. Runtime ``reset()`` can
load these bundles in required mode without compiling a fresh scenario during a
Modal smoke or training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from CyberSecurity_OWASP.config import load_scenario_authoring_config
from CyberSecurity_OWASP.server.scenario_cache import prepare_scenario_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated scenario cache bundles.")
    parser.add_argument("--config", default="", help="Path to scenario authoring JSON config.")
    parser.add_argument("--cache-dir", default="", help="Output scenario cache directory.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--difficulty-buckets", type=int, default=0)
    parser.add_argument("--train-per-bucket", type=int, default=0)
    parser.add_argument("--validation-per-bucket", type=int, default=0)
    parser.add_argument("--heldout-per-bucket", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing bundles.")
    args = parser.parse_args()

    if args.difficulty_buckets:
        os.environ["CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS"] = str(args.difficulty_buckets)
    if args.train_per_bucket:
        os.environ["CYBERSECURITY_OWASP_TRAIN_SCENARIOS_PER_BUCKET"] = str(args.train_per_bucket)
    if args.validation_per_bucket:
        os.environ["CYBERSECURITY_OWASP_VALIDATION_SCENARIOS_PER_BUCKET"] = str(args.validation_per_bucket)
    if args.heldout_per_bucket:
        os.environ["CYBERSECURITY_OWASP_HELDOUT_SCENARIOS_PER_BUCKET"] = str(args.heldout_per_bucket)
    if args.config:
        os.environ["CYBERSECURITY_OWASP_SCENARIO_CONFIG"] = args.config
    if args.cache_dir:
        os.environ["CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR"] = args.cache_dir

    settings = load_scenario_authoring_config()
    cache_dir = Path(args.cache_dir or settings.runtime.cache_dir)
    result = prepare_scenario_cache(
        cache_dir=cache_dir,
        settings=settings,
        seed_start=args.seed_start,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
