"""Run pytest and record the result as a Trackio run."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from training.trackio_utils import build_run_name, get_git_sha, log_trackio_metrics, trackio_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest with Trackio tracking.")
    parser.add_argument("pytest_args", nargs="*", help="Arguments passed through to pytest.")
    parser.add_argument("--run-name", default="", help="Trackio run name override.")
    parser.add_argument("--difficulty", type=int, default=0)
    args, passthrough = parser.parse_known_args()

    run_name = args.run_name or build_run_name(
        "pytest",
        "smoke",
        args.difficulty,
        git_sha=get_git_sha(),
    )
    pytest_args = [*args.pytest_args, *passthrough] or ["tests"]
    command = [sys.executable, "-m", "pytest", *pytest_args]
    started = time.perf_counter()

    with trackio_run(
        run_name=run_name,
        run_type="pytest",
        config={
            "command": " ".join(command),
            "pytest_args": pytest_args,
        },
        group="smoke",
    ):
        completed = subprocess.run(command)
        duration = time.perf_counter() - started
        log_trackio_metrics(
            {
                "smoke/pytest_exit_code": completed.returncode,
                "smoke/pytest_passed": completed.returncode == 0,
                "smoke/duration_seconds": duration,
            },
            step=0,
        )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
