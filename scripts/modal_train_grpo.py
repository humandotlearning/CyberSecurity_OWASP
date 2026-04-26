"""Persistent Modal GRPO launcher for CyberSecurity_OWASP.

This packages the local repository into a Modal GPU image, runs a small
tool-use GRPO job against the in-process CyberSecurity_OWASP environment, logs
metrics/traces to Trackio, and saves LoRA checkpoints in a persistent Modal
volume.

Example:

    uv run --extra modal modal run scripts/modal_train_grpo.py \
        --max-steps 10 \
        --dataset-size 16 \
        --num-generations 6 \
        --difficulty 0
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import modal


APP_NAME = "CyberSecurity_OWASP-grpo"
VOLUME_NAME = "CyberSecurity_OWASP-grpo-runs"
CACHE_VOLUME_NAME = "CyberSecurity_OWASP-model-cache"
SCENARIO_CACHE_VOLUME_NAME = "CyberSecurity_OWASP-scenario-cache"
SECRET_NAME = "CyberSecurity_OWASP-secrets"
RUNS_DIR = pathlib.Path("/runs")
CACHE_DIR = pathlib.Path("/cache")
SCENARIO_CACHE_DIR = pathlib.Path("/scenario-cache")
HF_HOME_DIR = CACHE_DIR / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
TORCH_HOME_DIR = CACHE_DIR / "torch"
XDG_CACHE_DIR = CACHE_DIR / "xdg"
UNSLOTH_CACHE_DIR = CACHE_DIR / "unsloth"
TRITON_CACHE_DIR = CACHE_DIR / "triton"
REMOTE_PROJECT = "/root/CyberSecurity_OWASP"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC_REPO_URL = "https://github.com/humandotlearning/CyberSecurity_OWASP.git"
PUBLIC_REPO_BRANCH = "master"
DEFAULT_GEMMA_MODEL = "unsloth/gemma-4-E2B-it"
GRPO_TRAINING_TIMEOUT_SECONDS = 24 * 60 * 60
_IMAGE_NOTICE_PRINTED = False


def _ensure_gemma4_model(model_name: str) -> str:
    if model_name != DEFAULT_GEMMA_MODEL:
        raise ValueError(
            "CyberSecurity_OWASP GRPO training is pinned to "
            f"{DEFAULT_GEMMA_MODEL}, matching the Unsloth Gemma 4 E2B RL notebook. "
            f"Received {model_name!r}."
        )
    return model_name


def _model_repo_slug(model_name: str) -> str:
    return (
        model_name.replace("/", "-")
        .replace("_", "-")
        .replace(".", "-")
        .lower()
    )


def _hf_model_cache_path(model_name: str) -> pathlib.Path:
    return HF_HUB_CACHE_DIR / f"models--{model_name.replace('/', '--')}"


def _resolve_grpo_batch_config(
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    world_size: int = 1,
) -> tuple[int, int]:
    if num_generations < 1:
        raise ValueError("--num-generations must be at least 1.")
    if per_device_train_batch_size < 1:
        raise ValueError("--per-device-train-batch-size must be at least 1.")
    if world_size < 1:
        raise ValueError("world_size must be at least 1.")

    resolved_gradient_accumulation_steps = (
        gradient_accumulation_steps
        if gradient_accumulation_steps > 0
        else max(2, num_generations)
    )
    if resolved_gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1.")

    effective_batch_size = (
        per_device_train_batch_size
        * resolved_gradient_accumulation_steps
        * world_size
    )
    if effective_batch_size % num_generations:
        raise ValueError(
            "Invalid GRPO batch shape: "
            "per_device_train_batch_size * gradient_accumulation_steps * world_size "
            f"must be divisible by num_generations. Got "
            f"{per_device_train_batch_size} * "
            f"{resolved_gradient_accumulation_steps} * {world_size} = "
            f"{effective_batch_size}, which is not divisible by {num_generations}."
        )
    return resolved_gradient_accumulation_steps, effective_batch_size


def _validate_vllm_config(*, use_vllm: bool, vllm_gpu_memory_utilization: float) -> None:
    if not use_vllm:
        return
    if not 0.0 < vllm_gpu_memory_utilization <= 0.95:
        raise ValueError(
            "--vllm-gpu-memory-utilization must be in the interval (0.0, 0.95] "
            "when --use-vllm is enabled."
        )


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        for part in stripped.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            candidates.append(part)

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded

    start = stripped.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        loaded = json.loads(stripped[start : index + 1])
                    except Exception:
                        break
                    if isinstance(loaded, dict):
                        return loaded
        start = stripped.find("{", start + 1)
    return None


def _configure_modal_cache_env() -> dict[str, str]:
    values = {
        "HF_HOME": str(HF_HOME_DIR),
        "HF_HUB_CACHE": str(HF_HUB_CACHE_DIR),
        "TRANSFORMERS_CACHE": str(HF_HUB_CACHE_DIR),
        "TORCH_HOME": str(TORCH_HOME_DIR),
        "XDG_CACHE_HOME": str(XDG_CACHE_DIR),
        "UNSLOTH_CACHE_DIR": str(UNSLOTH_CACHE_DIR),
        "UNSLOTH_COMPILE_CACHE": str(UNSLOTH_CACHE_DIR / "compile"),
        "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    }
    for key, value in values.items():
        os.environ[key] = value
    for path in {
        CACHE_DIR,
        HF_HOME_DIR,
        HF_HUB_CACHE_DIR,
        TORCH_HOME_DIR,
        XDG_CACHE_DIR,
        UNSLOTH_CACHE_DIR,
        UNSLOTH_CACHE_DIR / "compile",
        TRITON_CACHE_DIR,
    }:
        path.mkdir(parents=True, exist_ok=True)
    return values


def _configure_scenario_cache_env(*, required: bool = True) -> dict[str, str]:
    values = {
        "CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR": str(SCENARIO_CACHE_DIR),
        "CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE": "require" if required else "fallback",
    }
    for key, value in values.items():
        os.environ[key] = value
    SCENARIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return values


def _print_image_startup_notice() -> None:
    global _IMAGE_NOTICE_PRINTED
    if _IMAGE_NOTICE_PRINTED:
        return
    _IMAGE_NOTICE_PRINTED = True
    print(
        "Modal startup phase 1/5: building or validating the GPU training image. "
        "If this takes minutes, it is Modal image packaging/dependency cache work, "
        "not model-weight download."
    )
    print(
        "Later remote phases will print: cache hit/miss, snapshot_download progress, "
        "Unsloth weight loading, GRPO heartbeat, Trackio upload, and volume commits."
    )


def _load_local_env_file() -> None:
    env_path = PROJECT_ROOT / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"TRACKIO_PROJECT"}:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _modal_secrets() -> list[modal.Secret]:
    if _is_config_mode():
        return []
    return [modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])]


def _is_config_mode() -> bool:
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--mode" and index + 1 < len(args):
            return args[index + 1] == "config"
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1] == "config"
    return False


def _is_prepare_cache_mode() -> bool:
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--mode" and index + 1 < len(args):
            return args[index + 1] == "prepare-cache"
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1] == "prepare-cache"
    return False


_load_local_env_file()


def _cli_arg_value(name: str, default: str = "") -> str:
    args = sys.argv[1:]
    flag = f"--{name}"
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def _source_mode() -> str:
    return _cli_arg_value("source-mode", os.environ.get("MODAL_SOURCE_MODE", "local"))


def _training_image() -> modal.Image:
    if _is_prepare_cache_mode():
        return _scenario_cache_image()
    if not _is_prepare_cache_mode():
        _print_image_startup_notice()
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "build-essential", "curl")
        .uv_pip_install(
            "torch==2.10.0",
            "triton>=3.4.0",
            "torchvision==0.25.0",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "huggingface_hub",
            "peft",
            "pillow",
            "tokenizers",
            "nvidia-ml-py",
            "trackio>=0.25.0",
            "transformers>=5.5.0",
            "trl>=0.28.0",
            "openenv-core[core]>=0.2.3",
        )
        .uv_pip_install(
            "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
            "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        )
        .uv_pip_install("timm", extra_options="--no-deps")
        .uv_pip_install("pydantic==2.10.6")
        .uv_pip_install("mergekit", "immutables==0.21", extra_options="--no-deps")
        .uv_pip_install("llm-blender", "weave")
        .uv_pip_install("trl>=0.28.0", "transformers>=5.5.0", "jmespath")
    )

    if _source_mode() == "public":
        repo_url = _cli_arg_value("repo-url", PUBLIC_REPO_URL)
        repo_branch = _cli_arg_value("repo-branch", PUBLIC_REPO_BRANCH)
        image = image.run_commands(
            f"git clone --depth 1 --branch {repo_branch} {repo_url} {REMOTE_PROJECT}",
            f"python -m pip install --no-deps -e {REMOTE_PROJECT}",
        )
    else:
        image = image.add_local_dir(
            PROJECT_ROOT,
            remote_path=REMOTE_PROJECT,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                ".env",
                ".env.*",
                "__pycache__",
                ".pytest_cache",
                "outputs",
                "*.pyc",
            ],
        )
        image = image.run_commands(
            f"python -m pip install --no-deps -e {REMOTE_PROJECT}",
        )

    return image.run_commands(
        "python -c \"import os, torch; import transformers.utils.hub as hub; "
        "hub.TRANSFORMERS_CACHE = getattr(hub, 'TRANSFORMERS_CACHE', "
        "os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')); "
        "from trl import GRPOConfig, GRPOTrainer; "
        "from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import "
        "CybersecurityOwaspEnvironment; print('trainer import ok', torch.__version__)\"",
    ).workdir(REMOTE_PROJECT)


def _scenario_cache_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .uv_pip_install("openenv-core[core]>=0.2.3", "trackio>=0.25.0")
    )

    if _source_mode() == "public":
        repo_url = _cli_arg_value("repo-url", PUBLIC_REPO_URL)
        repo_branch = _cli_arg_value("repo-branch", PUBLIC_REPO_BRANCH)
        image = image.run_commands(
            f"git clone --depth 1 --branch {repo_branch} {repo_url} {REMOTE_PROJECT}",
            f"python -m pip install --no-deps -e {REMOTE_PROJECT}",
        )
    else:
        image = image.add_local_dir(
            PROJECT_ROOT,
            remote_path=REMOTE_PROJECT,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                ".env",
                ".env.*",
                "__pycache__",
                ".pytest_cache",
                "outputs",
                "*.pyc",
            ],
        )
        image = image.run_commands(
            f"python -m pip install --no-deps -e {REMOTE_PROJECT}",
        )
    return image.workdir(REMOTE_PROJECT)


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
scenario_cache_volume = modal.Volume.from_name(SCENARIO_CACHE_VOLUME_NAME, create_if_missing=True)
secrets = _modal_secrets()
scenario_cache_image = _scenario_cache_image()
training_image = _training_image()


@app.function(
    image=scenario_cache_image,
    timeout=2 * 60 * 60,
    volumes={SCENARIO_CACHE_DIR: scenario_cache_volume},
)
def prepare_modal_scenario_cache(
    seed_start: int = 0,
    difficulty_buckets: int = 0,
    train_per_bucket: int = 0,
    validation_per_bucket: int = 0,
    heldout_per_bucket: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    if difficulty_buckets:
        os.environ["CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS"] = str(difficulty_buckets)
    if train_per_bucket:
        os.environ["CYBERSECURITY_OWASP_TRAIN_SCENARIOS_PER_BUCKET"] = str(train_per_bucket)
    if validation_per_bucket:
        os.environ["CYBERSECURITY_OWASP_VALIDATION_SCENARIOS_PER_BUCKET"] = str(validation_per_bucket)
    if heldout_per_bucket:
        os.environ["CYBERSECURITY_OWASP_HELDOUT_SCENARIOS_PER_BUCKET"] = str(heldout_per_bucket)
    _configure_scenario_cache_env(required=False)
    from CyberSecurity_OWASP.config import load_scenario_authoring_config
    from CyberSecurity_OWASP.server.scenario_cache import prepare_scenario_cache

    settings = load_scenario_authoring_config()
    result = prepare_scenario_cache(
        cache_dir=SCENARIO_CACHE_DIR,
        settings=settings,
        seed_start=seed_start,
        force=force,
    )
    scenario_cache_volume.commit()
    result["scenario_cache_volume"] = SCENARIO_CACHE_VOLUME_NAME
    return result


@app.function(
    image=scenario_cache_image,
    timeout=60 * 10,
    volumes={SCENARIO_CACHE_DIR: scenario_cache_volume},
)
def verify_modal_scenario_cache_for_training(
    split: str = "train",
    difficulty: int = 0,
    dataset_size: int = 2,
    seed_start: int = 0,
) -> dict[str, Any]:
    _configure_scenario_cache_env(required=True)
    scenario_cache_volume.reload()

    from CyberSecurity_OWASP.config import load_scenario_authoring_config
    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )
    from CyberSecurity_OWASP.server.curriculum import CurriculumController
    from CyberSecurity_OWASP.server.scenario_cache import ScenarioCache

    settings = load_scenario_authoring_config()
    scenario_profile = CurriculumController(settings=settings).select_profile(
        seed=seed_start,
        split=split,
        requested_difficulty=difficulty,
    )
    resolved_difficulty = int(scenario_profile["difficulty"])
    cache = ScenarioCache(SCENARIO_CACHE_DIR, settings=settings)
    coverage = cache.assert_coverage(split=split, difficulty=resolved_difficulty)
    entries = cache.validated_entries(split=split, difficulty=resolved_difficulty)
    if not entries:
        entries = cache.validated_entries(split=split)
    if not entries:
        raise RuntimeError(f"No validated scenario cache entries found for split={split!r}.")
    sample_entry = entries[0]

    env = CybersecurityOwaspEnvironment()
    try:
        obs = env.reset(
            seed=int(sample_entry["seed"]),
            split=str(sample_entry["split"]),
            difficulty=int(sample_entry["difficulty"]),
        )
        if not env.state.cache_hit:
            raise RuntimeError("Scenario cache preflight reset did not hit cache.")
        if env.state.metrics.get("scenario_compile_latency_ms", 0.0):
            raise RuntimeError("Scenario cache preflight unexpectedly compiled a scenario.")
        sample = {
            "phase": obs.phase,
            "task_id": env.state.task_id,
            "cache_hit": env.state.cache_hit,
            "scenario_hash": env.state.scenario_hash,
            "reset_latency_ms": env.state.reset_latency_ms,
            "bundle_load_latency_ms": env.state.metrics.get(
                "scenario_bundle_load_latency_ms",
                0.0,
            ),
        }
    finally:
        env.close()

    return {
        "scenario_cache_volume": SCENARIO_CACHE_VOLUME_NAME,
        "scenario_cache_dir": str(SCENARIO_CACHE_DIR),
        "scenario_cache_mode": "require",
        "split": split,
        "difficulty": "adaptive",
        "initial_difficulty": resolved_difficulty,
        "dataset_size": dataset_size,
        "available_scenarios": len(cache.validated_entries(split=split)),
        "coverage": coverage,
        "sample_reset": sample,
    }


@app.function(
    image=training_image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={RUNS_DIR: volume, CACHE_DIR: cache_volume, SCENARIO_CACHE_DIR: scenario_cache_volume},
    secrets=secrets,
)
def check_training_imports() -> dict[str, str]:
    cache_env = _configure_modal_cache_env()
    scenario_cache_env = _configure_scenario_cache_env(required=False)

    import torch
    import trackio
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastVisionModel

    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )

    env = CybersecurityOwaspEnvironment()
    obs = env.reset(seed=0, split="validation", difficulty=0)
    return {
        "torch": torch.__version__,
        "trackio": getattr(trackio, "__version__", "unknown"),
        "dataset": Dataset.__name__,
        "grpo_config": GRPOConfig.__name__,
        "grpo_trainer": GRPOTrainer.__name__,
        "unsloth_vision_model": FastVisionModel.__name__,
        "env": CybersecurityOwaspEnvironment.__name__,
        "reset_phase": obs.phase,
        "hf_home": cache_env["HF_HOME"],
        "hf_hub_cache": cache_env["HF_HUB_CACHE"],
        "scenario_cache_dir": scenario_cache_env["CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR"],
    }


@app.function(
    image=training_image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={RUNS_DIR: volume, CACHE_DIR: cache_volume, SCENARIO_CACHE_DIR: scenario_cache_volume},
    secrets=secrets,
)
def run_cybersecurity_owasp_baseline(
    max_steps: int = 50,
    dataset_size: int = 1,
    difficulty: int = 0,
    split: str = "train",
    model_name: str = DEFAULT_GEMMA_MODEL,
    max_seq_length: int = 4096,
    max_completion_length: int = 768,
    trackio_space_id: str = "Humanlearning/CyberSecurity_OWASP-trackio",
    trackio_project: str = "CyberSecurity_OWASP-grpo",
    num_generations: int = 1,
    trace_log_every: int = 1,
    seed_start: int = 0,
    git_sha: str = "nogit",
    run_name: str = "baseline",
    source_mode: str = "local",
    repo_url: str = PUBLIC_REPO_URL,
    repo_branch: str = PUBLIC_REPO_BRANCH,
) -> dict[str, str | int | float]:
    import statistics
    import time

    import torch
    from huggingface_hub import snapshot_download, whoami
    from unsloth import FastVisionModel
    import transformers.utils.hub as transformers_hub

    from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
    from CyberSecurity_OWASP.config import load_scenario_authoring_config
    from CyberSecurity_OWASP.reward_config import load_reward_settings
    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )
    from CyberSecurity_OWASP.server.curriculum import CurriculumController
    from CyberSecurity_OWASP.server.scenario_cache import ScenarioCache
    from training.trackio_utils import (
        aggregate_episode_metrics,
        episode_record_from_state,
        log_reward_config,
        log_trace_table,
        log_trackio_metrics,
        reward_config_trackio_config,
        trackio_run,
    )

    model_name = _ensure_gemma4_model(model_name)
    if int(num_generations) != 1:
        raise ValueError("Baseline mode runs the untrained model with --num-generations 1.")

    cache_env = _configure_modal_cache_env()
    scenario_cache_env = _configure_scenario_cache_env(required=True)
    transformers_hub.TRANSFORMERS_CACHE = cache_env["HF_HUB_CACHE"]
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(f"HF_TOKEN is missing from the Modal secret {SECRET_NAME}.")
    try:
        whoami(token=hf_token)
    except Exception as exc:
        raise RuntimeError("HF_TOKEN could not be validated before baseline run.") from exc

    os.environ["TRACKIO_SPACE_ID"] = trackio_space_id
    os.environ["TRACKIO_PROJECT"] = trackio_project
    reward_settings = load_reward_settings()
    reward_tracking_config = reward_config_trackio_config(reward_settings)
    run_name = run_name or "baseline"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_volume.reload()
        print(f"Reloaded Modal model cache volume: {CACHE_VOLUME_NAME}")
    except Exception as exc:
        print(f"Model cache volume reload skipped: {exc!r}")
    try:
        scenario_cache_volume.reload()
        print(f"Reloaded Modal scenario cache volume: {SCENARIO_CACHE_VOLUME_NAME}")
    except Exception as exc:
        print(f"Scenario cache volume reload skipped: {exc!r}")

    settings = load_scenario_authoring_config()
    scenario_profile = CurriculumController(settings=settings).select_profile(
        seed=seed_start,
        split=split,
        requested_difficulty=difficulty,
    )
    resolved_difficulty = int(scenario_profile["difficulty"])
    scenario_cache = ScenarioCache(SCENARIO_CACHE_DIR, settings=settings)
    coverage = scenario_cache.assert_coverage(
        split=split,
        difficulty=resolved_difficulty,
    )
    entries = scenario_cache.validated_entries(
        split=split,
        difficulty=resolved_difficulty,
    ) or scenario_cache.validated_entries(split=split)
    if not entries:
        raise RuntimeError(f"No validated scenario cache entries found for split={split!r}.")

    print(f"Baseline run name: {run_name}")
    print(f"Source mode: {source_mode}")
    if source_mode == "public":
        print(f"Installed CyberSecurity_OWASP from public repo: {repo_url}@{repo_branch}")
    else:
        print("Packaged local CyberSecurity_OWASP repo.")
    print(f"Trackio Space: {trackio_space_id}")
    print(f"Trackio Project: {trackio_project}")
    print(f"Reward config: {reward_tracking_config['reward_config_id']}")
    print(f"Reward config hash: {reward_tracking_config['reward_config_hash']}")
    print(f"Scenario cache dir: {scenario_cache_env['CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR']}")
    print(f"Scenario cache coverage: {coverage}")
    print(
        "Baseline generation config: "
        f"episodes={dataset_size}, max_episode_steps={max_steps}, "
        f"num_generations={num_generations}, max_completion_length={max_completion_length}, "
        f"trace_log_every={trace_log_every}"
    )

    expected_model_cache = _hf_model_cache_path(model_name)
    print(f"Expected HF model cache path: {expected_model_cache}")
    print(f"Model cache hit before load: {expected_model_cache.exists()}")
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(HF_HUB_CACHE_DIR),
            token=hf_token,
        )
        print(f"Model snapshot ready: {snapshot_path}")
        cache_volume.commit()
    except Exception as exc:
        print(f"Explicit model snapshot prefetch failed; loading directly. Error: {exc!r}")

    model_api = FastVisionModel
    model, tokenizer = model_api.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        cache_dir=str(HF_HUB_CACHE_DIR),
        token=hf_token,
    )
    if hasattr(model_api, "for_inference"):
        model_api.for_inference(model)
    model.eval()
    cache_volume.commit()
    device = next(model.parameters()).device
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def render_prompt(observation, actions: list[dict[str, Any]]) -> str:
        recent_actions = actions[-8:]
        return (
            "You are the untrained baseline model for a defensive local AppSec "
            "repair environment. Use only the listed local tools. Return exactly "
            "one JSON object and no markdown.\n\n"
            f"{observation.scenario_prompt}\n\n"
            f"Current phase: {observation.phase}\n"
            f"Available actions: {observation.available_actions}\n"
            f"Last tool result: {observation.last_tool_result}\n"
            f"Recent actions: {json.dumps(recent_actions, sort_keys=True)}\n\n"
            'Required format: {"tool_name":"inspect_policy_graph","arguments":{}}'
        )

    def generate_action_text(prompt: str) -> tuple[str, list[int], list[int]]:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = prompt
        for candidate in (tokenizer, text_tokenizer):
            if hasattr(candidate, "apply_chat_template"):
                try:
                    prompt_text = candidate.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    break
                except Exception:
                    prompt_text = prompt
        encode = tokenizer
        try:
            inputs = encode(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
        except Exception:
            inputs = text_tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        input_ids = inputs.get("input_ids")
        input_len = int(input_ids.shape[-1]) if input_ids is not None else 0
        pad_token_id = getattr(text_tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(text_tokenizer, "eos_token_id", None)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        output_ids = generated[0]
        completion_ids = output_ids[input_len:]
        decode = getattr(text_tokenizer, "decode", None) or getattr(tokenizer, "decode")
        text = decode(completion_ids, skip_special_tokens=True)
        prompt_ids = (
            [int(item) for item in input_ids[0].detach().cpu().tolist()]
            if input_ids is not None
            else []
        )
        return text, prompt_ids, [int(item) for item in completion_ids.detach().cpu().tolist()]

    def action_from_completion(raw_text: str) -> tuple[CyberSecurityOWASPAction, str | None]:
        loaded = _extract_first_json_object(raw_text)
        if loaded is None:
            return CyberSecurityOWASPAction(tool_name="noop", arguments={}), "invalid_json"
        arguments = loaded.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        payload = {
            "tool_name": loaded.get("tool_name", "noop"),
            "arguments": arguments,
        }
        try:
            return CyberSecurityOWASPAction(**payload), None
        except Exception as exc:
            return (
                CyberSecurityOWASPAction(tool_name="noop", arguments={}),
                f"invalid_action_schema: {exc}",
            )

    episode_records: list[dict[str, Any]] = []
    raw_traces: list[dict[str, Any]] = []
    invalid_model_outputs = 0
    generation_started = time.monotonic()
    config = {
        "base_model": model_name,
        "algo": "baseline",
        "difficulty": difficulty,
        "split": split,
        "max_episode_steps": max_steps,
        "dataset_size": dataset_size,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "git_sha": git_sha,
        **reward_tracking_config,
    }

    with trackio_run(
        run_name=run_name,
        run_type="baseline",
        config=config,
        project=trackio_project,
        space_id=trackio_space_id,
        group="baseline",
        auto_log_gpu=True,
    ):
        log_reward_config(reward_settings, step=0)
        for episode_index in range(max(1, int(dataset_size))):
            entry = entries[(seed_start + episode_index) % len(entries)]
            env = CybersecurityOwaspEnvironment()
            try:
                observation = env.reset(
                    seed=int(entry["seed"]),
                    split=str(entry["split"]),
                    difficulty=int(entry["difficulty"]),
                )
                env.state.max_steps = int(max_steps)
                actions: list[dict[str, Any]] = []
                model_steps: list[dict[str, Any]] = []
                prompt_token_count = 0
                completion_token_count = 0

                for step_index in range(int(max_steps)):
                    if observation.done:
                        break
                    prompt = render_prompt(observation, actions)
                    raw_text, prompt_ids, completion_ids = generate_action_text(prompt)
                    prompt_token_count += len(prompt_ids)
                    completion_token_count += len(completion_ids)
                    action, invalid_reason = action_from_completion(raw_text)
                    if invalid_reason:
                        invalid_model_outputs += 1
                    observation = env.step(action)
                    action_dump = action.model_dump()
                    actions.append(action_dump)
                    model_steps.append(
                        {
                            "step": step_index + 1,
                            "raw_completion": raw_text,
                            "action": action_dump,
                            "invalid_model_output": invalid_reason,
                            "observation_message": observation.message,
                            "reward": observation.reward,
                            "done": observation.done,
                        }
                    )

                env.state.completion_tokens = completion_token_count
                env.state.metrics["prompt_tokens"] = prompt_token_count
                env.state.metrics["completion_tokens"] = completion_token_count
                final_observation = observation.model_dump()
                record = episode_record_from_state(
                    env.state,
                    run_context={
                        "base_model": model_name,
                        "algo": "baseline",
                        "reward_version": "reward_v2",
                        "env_version": "0.1.0",
                        **reward_tracking_config,
                    },
                    final_observation=final_observation,
                )
                record.update(
                    {
                        "reward_total": float(env.state.accumulated_reward),
                        "success": bool(env.state.success),
                        "episode_length": int(env.state.step_count),
                        "invalid_model_output_count": sum(
                            1 for item in model_steps if item["invalid_model_output"]
                        ),
                        "prompt_tokens": prompt_token_count,
                        "completion_tokens": completion_token_count,
                    }
                )
                episode_records.append(record)
                raw_traces.append(
                    {
                        "episode_index": episode_index,
                        "task_id": env.state.task_id,
                        "seed": env.state.seed,
                        "split": env.state.split,
                        "difficulty": env.state.difficulty,
                        "domain": env.state.domain,
                        "bug_family": env.state.bug_family,
                        "steps": model_steps,
                    }
                )
            finally:
                env.close()

            metrics = aggregate_episode_metrics(episode_records)
            metrics.update(
                {
                    "baseline/episode_count": float(len(episode_records)),
                    "baseline/reward_total_mean": statistics.mean(
                        float(item.get("reward_total", 0.0)) for item in episode_records
                    ),
                    "baseline/success_rate": statistics.mean(
                        1.0 if item.get("success") else 0.0 for item in episode_records
                    ),
                    "baseline/invalid_model_output_rate": invalid_model_outputs
                    / max(1.0, sum(float(item.get("episode_length", 0)) for item in episode_records)),
                    "baseline/num_generations": float(num_generations),
                    "baseline/max_episode_steps": float(max_steps),
                    "baseline/max_completion_length": float(max_completion_length),
                }
            )
            log_trackio_metrics(metrics, step=episode_index + 1)
            if trace_log_every > 0 and (
                episode_index == 0 or (episode_index + 1) % trace_log_every == 0
            ):
                log_trace_table(
                    [episode_records[-1]],
                    table_name="baseline_traces",
                    step=episode_index + 1,
                )

    elapsed_s = time.monotonic() - generation_started
    summary = {
        "run_name": run_name,
        "trackio_space_id": trackio_space_id,
        "trackio_project": trackio_project,
        "model_name": model_name,
        "dataset_size": len(episode_records),
        "max_episode_steps": int(max_steps),
        "difficulty": int(difficulty),
        "split": split,
        "num_generations": int(num_generations),
        "max_completion_length": int(max_completion_length),
        "mean_reward": (
            statistics.mean(float(item.get("reward_total", 0.0)) for item in episode_records)
            if episode_records
            else 0.0
        ),
        "success_rate": (
            statistics.mean(1.0 if item.get("success") else 0.0 for item in episode_records)
            if episode_records
            else 0.0
        ),
        "invalid_model_output_count": int(invalid_model_outputs),
        "elapsed_s": elapsed_s,
        **reward_tracking_config,
    }
    artifact_path = output_dir / "baseline_rollouts.json"
    artifact_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "episodes": episode_records,
                "raw_traces": raw_traces,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    volume.commit()
    cache_volume.commit()
    scenario_cache_volume.commit()
    print(f"Baseline artifact saved to {artifact_path}")
    return {**summary, "artifact_path": str(artifact_path)}


@app.function(
    image=training_image,
    gpu="L4",
    timeout=GRPO_TRAINING_TIMEOUT_SECONDS,
    volumes={RUNS_DIR: volume, CACHE_DIR: cache_volume, SCENARIO_CACHE_DIR: scenario_cache_volume},
    secrets=secrets,
)
def train_cybersecurity_owasp_grpo(
    env_repo_id: str = "",
    output_repo_id: str = "",
    max_steps: int = 10,
    dataset_size: int = 16,
    difficulty: int = 0,
    split: str = "train",
    model_name: str = DEFAULT_GEMMA_MODEL,
    max_seq_length: int = 4096,
    max_completion_length: int = 768,
    lora_rank: int = 32,
    trackio_space_id: str = "Humanlearning/CyberSecurity_OWASP-trackio",
    trackio_project: str = "CyberSecurity_OWASP-grpo",
    num_generations: int = 6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 0,
    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.2,
    trace_log_every: int = 5,
    seed_start: int = 0,
    git_sha: str = "nogit",
    run_name: str = "",
    source_mode: str = "local",
    repo_url: str = PUBLIC_REPO_URL,
    repo_branch: str = PUBLIC_REPO_BRANCH,
    push_to_hub: bool = False,
) -> dict[str, str | int | float]:
    import inspect
    import statistics
    import threading
    import time

    model_name = _ensure_gemma4_model(model_name)
    cache_env = _configure_modal_cache_env()
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    (
        resolved_gradient_accumulation_steps,
        effective_train_batch_size,
    ) = _resolve_grpo_batch_config(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        world_size=world_size,
    )
    _validate_vllm_config(
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    trace_log_every = max(0, int(trace_log_every))

    import torch
    from unsloth import FastVisionModel
    import transformers.utils.hub as transformers_hub
    from datasets import Dataset
    from huggingface_hub import snapshot_download, whoami
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer, clone_chat_template
    from trl.chat_template_utils import add_response_schema

    import trackio

    from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
    from CyberSecurity_OWASP.config import load_scenario_authoring_config
    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )
    from CyberSecurity_OWASP.reward_config import (
        compute_token_penalty,
        load_reward_settings,
    )
    from CyberSecurity_OWASP.server.curriculum import CurriculumController
    from CyberSecurity_OWASP.server.scenario_cache import ScenarioCache
    from training.trackio_utils import (
        aggregate_episode_metrics,
        episode_record_from_state,
        episode_trace_fingerprint,
        log_reward_config,
        log_gpu_metrics,
        log_trace_table,
        log_trackio_metrics,
        reward_config_trackio_config,
        train_metric_aliases,
    )
    from training.grpo_curriculum import (
        ScenarioGroupRegistry,
        build_scenario_group_rows,
    )

    transformers_hub.TRANSFORMERS_CACHE = cache_env["HF_HUB_CACHE"]

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            f"HF_TOKEN is missing from the Modal secret {SECRET_NAME}."
        )

    user = whoami(token=hf_token)["name"]
    env_repo_id = env_repo_id or f"{user}/CyberSecurity_OWASP"
    output_repo_id = output_repo_id or (
        f"{user}/CyberSecurity_OWASP-{_model_repo_slug(model_name)}-grpo-lora"
    )
    if not trackio_space_id:
        trackio_space_id = "Humanlearning/CyberSecurity_OWASP-trackio"
        if hf_token:
            try:
                from huggingface_hub import whoami

                user = whoami(token=hf_token)["name"]
                if user == "humandotlearning":
                    trackio_space_id = f"{user}/CyberSecurity_OWASP-trackio"
            except Exception:
                pass

    os.environ["TRACKIO_SPACE_ID"] = trackio_space_id
    os.environ["TRACKIO_PROJECT"] = trackio_project
    os.environ.setdefault("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    reward_settings = load_reward_settings()
    reward_tracking_config = reward_config_trackio_config(reward_settings)

    model_slug = model_name.replace("/", "-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = run_name or (
        f"CyberSecurity_OWASP-{model_slug}-grpo-level{difficulty}-{stamp}-{git_sha[:8]}"
    )
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_volume.reload()
        print(f"Reloaded Modal model cache volume: {CACHE_VOLUME_NAME}")
    except Exception as exc:
        print(f"Model cache volume reload skipped: {exc!r}")
    try:
        scenario_cache_volume.reload()
        print(f"Reloaded Modal scenario cache volume: {SCENARIO_CACHE_VOLUME_NAME}")
    except Exception as exc:
        print(f"Scenario cache volume reload skipped: {exc!r}")
    cache_env = _configure_modal_cache_env()
    scenario_cache_env = _configure_scenario_cache_env(required=True)
    scenario_settings = load_scenario_authoring_config()
    scenario_profile = CurriculumController(settings=scenario_settings).select_profile(
        seed=seed_start,
        split=split,
        requested_difficulty=difficulty,
    )
    scenario_cache = ScenarioCache(SCENARIO_CACHE_DIR, settings=scenario_settings)
    scenario_cache_coverage = scenario_cache.assert_coverage(
        split=split,
        difficulty=int(scenario_profile["difficulty"]),
    )
    scenario_entries = scenario_cache.validated_entries(split=split)
    scenario_registry = ScenarioGroupRegistry(
        scenario_entries,
        split=split,
        initial_difficulty=int(scenario_profile["difficulty"]),
        rng_seed=seed_start,
        max_level=scenario_settings.curriculum.difficulty_bucket_count - 1,
    )

    training_prompt = (
        "You are a defensive AppSec repair agent in the local CyberSecurity_OWASP "
        "OpenEnv environment. Use only the provided local tools. Do not target real "
        "systems. Work step by step: inspect policy and generated code, reproduce the "
        "authorization issue locally, submit a policy-tied diagnosis, patch the generated "
        "app, run visible tests, then submit the fix. Do not write explanations unless "
        "a tool argument needs evidence text."
    )

    dataset = Dataset.from_list(
        build_scenario_group_rows(
            dataset_size=dataset_size,
            training_prompt=training_prompt,
            seed_start=seed_start,
            split=split,
            difficulty=difficulty,
            difficulty_policy="adaptive",
        )
    )

    def _state_snapshot(env: CybersecurityOwaspEnvironment) -> dict[str, Any]:
        state = env.state
        return {
            "episode_id": state.episode_id,
            "task_id": state.task_id,
            "seed": state.seed,
            "split": state.split,
            "difficulty": state.difficulty,
            "difficulty_tier": state.difficulty_tier,
            "domain": state.domain,
            "bug_family": state.bug_family,
            "template_id": state.template_id,
            "cache_hit": state.cache_hit,
            "scenario_hash": state.scenario_hash,
            "phase": state.phase,
            "step_count": state.step_count,
            "done": state.done,
            "success": state.success,
            "failure_reason": state.failure_reason,
            "anti_cheat_flags": list(state.anti_cheat_flags),
        }

    class CyberSecurityOWASPToolEnv:
        def __init__(self):
            self._env = CybersecurityOwaspEnvironment()
            self.reward = 0.0
            self.reward_breakdown: dict[str, float] = {}
            self.done = False
            self.success = False
            self.invalid_actions = 0
            self.scenario_group_id = -1
            self.scenario_assignment: dict[str, Any] = {}
            self.trace_messages: list[dict[str, str]] = []
            self.trace_metadata: dict[str, Any] = {}

        def reset(self, **kwargs) -> str:
            group_id = int(kwargs.get("scenario_group_id", kwargs.get("seed", seed_start)))
            assignment = scenario_registry.assignment_for(
                scenario_group_id=group_id,
                requested_seed=int(kwargs.get("seed", seed_start)),
                requested_difficulty=int(kwargs.get("difficulty", difficulty)),
                split=str(kwargs.get("split", split)),
                difficulty_policy=str(kwargs.get("difficulty_policy", "adaptive")),
            )
            seed = int(assignment["seed"])
            current_difficulty = int(assignment["difficulty"])
            current_split = str(assignment["split"])
            obs = self._env.reset(
                seed=seed,
                split=current_split,
                difficulty=current_difficulty,
            )
            self.scenario_group_id = group_id
            self.scenario_assignment = assignment
            self.reward = 0.0
            self.reward_breakdown = {}
            self.done = bool(obs.done)
            self.success = False
            self.invalid_actions = 0
            self.trace_messages = [
                {
                    "role": "user",
                    "content": (
                        f"{training_prompt}\n\n"
                        f"{obs.scenario_prompt}\n\n"
                        f"Initial message: {obs.message}"
                    ),
                }
            ]
            self.trace_metadata = _state_snapshot(self._env)
            self.trace_metadata.update(
                {
                    "scenario_group_id": self.scenario_group_id,
                    "scenario_assignment": dict(self.scenario_assignment),
                    "scenario_prompt_length": len(obs.scenario_prompt),
                    "reward_config_id": reward_tracking_config["reward_config_id"],
                    "reward_config_hash": reward_tracking_config["reward_config_hash"],
                    "reward_stage": reward_tracking_config["reward_stage"],
                    "reward_mode": reward_tracking_config["reward_mode"],
                }
            )
            return obs.scenario_prompt

        def _step(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
            if self.done:
                raise ValueError("Episode is already over.")
            action = CyberSecurityOWASPAction(
                tool_name=tool_name,
                arguments=arguments or {},
            )
            obs = self._env.step(action)
            if not obs.last_action_valid:
                self.invalid_actions += 1
            self.reward = float(self._env.state.accumulated_reward)
            self.reward_breakdown = dict(obs.reward_breakdown or {})
            self.done = bool(obs.done)
            self.success = bool(self._env.state.success)
            self.trace_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"{tool_name}({arguments or {}})",
                    },
                    {"role": "tool", "content": obs.message},
                ]
            )
            self.trace_metadata.update(_state_snapshot(self._env))
            self.trace_metadata.update(
                {
                    "last_action_valid": obs.last_action_valid,
                    "last_action_error": obs.last_action_error,
                    "reward": self.reward,
                    "reward_breakdown": self.reward_breakdown,
                    "invalid_actions": self.invalid_actions,
                    "scenario_cache_hit": self._env.state.cache_hit,
                    "scenario_hash": self._env.state.scenario_hash,
                    "scenario_group_id": self.scenario_group_id,
                    "scenario_assignment": dict(self.scenario_assignment),
                }
            )
            return obs.message

        def inspect_policy_graph(self) -> str:
            """Return public policy hints for the generated local scenario."""
            return self._step("inspect_policy_graph")

        def list_routes(self) -> str:
            """List generated local app route summaries."""
            return self._step("list_routes")

        def read_openapi(self) -> str:
            """Read generated OpenAPI metadata for the local app."""
            return self._step("read_openapi")

        def read_file(self, path: str) -> str:
            """
            Read an editable generated workspace file by relative path.

            Args:
                path: Relative path inside the generated editable workspace.

            Returns:
                The file contents or a safe tool error observation.
            """
            return self._step("read_file", {"path": path})

        def search_code(self, query: str) -> str:
            """
            Search editable generated workspace files for a string.

            Args:
                query: Search text to find in editable generated app files.

            Returns:
                Matching file lines or a no-match message.
            """
            return self._step("search_code", {"query": query})

        def send_local_request(
            self,
            path: str,
            method: str = "GET",
            user_id: str | None = None,
        ) -> str:
            """
            Send a request to the generated local app only.

            Args:
                path: Local route path such as /health or /invoices/<id>.
                method: HTTP method to use for the local request.
                user_id: Optional generated user identifier for authentication.

            Returns:
                JSON response from the simulated local app request.
            """
            return self._step(
                "send_local_request",
                {"path": path, "method": method, "user_id": user_id},
            )

        def compare_identities(
            self,
            path: str,
            first_user_id: str,
            second_user_id: str,
            method: str = "GET",
        ) -> str:
            """
            Compare one local request as two generated users.

            Args:
                path: Local route path to request as both generated users.
                first_user_id: First generated user identifier.
                second_user_id: Second generated user identifier.
                method: HTTP method to use for both local requests.

            Returns:
                JSON summary of both simulated local responses.
            """
            return self._step(
                "compare_identities",
                {
                    "path": path,
                    "method": method,
                    "first_user_id": first_user_id,
                    "second_user_id": second_user_id,
                },
            )

        def submit_diagnosis(
            self,
            bug_class: str,
            route: str,
            violated_policy_rule: str,
            evidence_trace_ids: list[str],
            fix_plan: str,
        ) -> str:
            """
            Submit structured diagnosis for the suspected authorization bug.

            Args:
                bug_class: Short class such as idor_ownership_bug.
                route: Method and route pattern believed to be vulnerable.
                violated_policy_rule: Policy rule that the behavior violates.
                evidence_trace_ids: Request trace IDs from local evidence tools.
                fix_plan: Concise secure repair plan.

            Returns:
                Diagnosis acceptance result and next phase information.
            """
            return self._step(
                "submit_diagnosis",
                {
                    "bug_class": bug_class,
                    "route": route,
                    "violated_policy_rule": violated_policy_rule,
                    "evidence_trace_ids": evidence_trace_ids,
                    "fix_plan": fix_plan,
                },
            )

        def patch_file(
            self,
            path: str,
            content: str | None = None,
            diff: str | None = None,
        ) -> str:
            """
            Patch an editable generated app file with full content or a unified diff.

            Args:
                path: Relative path of the editable generated app file to patch.
                content: Complete replacement file content, when using full-file patching.
                diff: Unified diff to apply, when using diff patching.

            Returns:
                Patch application result.
            """
            args: dict[str, Any] = {"path": path}
            if content is not None:
                args["content"] = content
            if diff is not None:
                args["diff"] = diff
            return self._step("patch_file", args)

        def run_visible_tests(self) -> str:
            """Run visible tests only; hidden tests are never exposed."""
            return self._step("run_visible_tests")

        def submit_fix(self) -> str:
            """Submit the final patch to the hidden deterministic verifier."""
            return self._step("submit_fix")

        def noop(self) -> str:
            """Take no action."""
            return self._step("noop")

        def _score(self, completion_tokens: int = 0) -> float:
            token_penalty = compute_token_penalty(completion_tokens)
            self._env.state.completion_tokens = int(completion_tokens)
            self._env.state.metrics["completion_tokens"] = int(completion_tokens)
            self._env.state.metrics["token_penalty"] = token_penalty
            return float(self._env.state.accumulated_reward + token_penalty)

        def __del__(self):
            try:
                self._env.close()
            except Exception:
                pass

    trace_step = {"value": 0}
    logged_trace_fingerprints: set[str] = set()

    def _completion_to_text(completion) -> str:
        if completion is None:
            return ""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            parts = []
            for item in completion:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(completion)

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def cybersecurity_owasp_reward(environments, **kwargs) -> list[float]:
        completions = kwargs.get("completions") or kwargs.get("completion") or []
        completion_texts = [_completion_to_text(item) for item in completions]
        completion_tokens = [len(text.split()) for text in completion_texts]
        rewards = [
            float(env._score(completion_tokens[index] if index < len(completion_tokens) else 0))
            for index, env in enumerate(environments)
        ]
        trace_step["value"] += 1

        episode_records = []
        for index, (env, reward) in enumerate(zip(environments, rewards)):
            record = episode_record_from_state(
                env._env.state,
                run_context={
                    "base_model": model_name,
                    "algo": "grpo",
                    "reward_version": "reward_v2",
                    "env_version": "0.1.0",
                    **reward_tracking_config,
                },
            )
            record.update(
                {
                    "reward_total": reward,
                    "reward_token_penalty": float(env._env.state.metrics.get("token_penalty", 0.0)),
                    "completion_tokens": completion_tokens[index] if index < len(completion_tokens) else 0,
                    "success": bool(getattr(env, "success", False)),
                }
            )
            episode_records.append(record)

        group_successes: dict[int, list[float]] = {}
        for env in environments:
            group_id = int(getattr(env, "scenario_group_id", -1))
            if group_id < 0:
                continue
            group_successes.setdefault(group_id, []).append(1.0 if getattr(env, "success", False) else 0.0)
        for group_id, successes in group_successes.items():
            scenario_registry.record_group_outcome(group_id, _mean(successes))

        batch_fingerprints = [
            episode_trace_fingerprint(record)
            for record in episode_records
        ]
        sampled_traces = []
        seen_this_batch: set[str] = set()
        duplicate_trace_suppressed_count = 0
        for index, (env, record, reward, fingerprint) in enumerate(
            zip(environments, episode_records, rewards, batch_fingerprints)
        ):
            if fingerprint in seen_this_batch or fingerprint in logged_trace_fingerprints:
                duplicate_trace_suppressed_count += 1
                continue
            seen_this_batch.add(fingerprint)
            if len(sampled_traces) < 4:
                sampled_traces.append((index, env, record, reward, fingerprint))

        should_log_trace_artifacts = trace_log_every > 0 and (
            trace_step["value"] == 1
            or trace_step["value"] % trace_log_every == 0
        )
        canonical_metrics = aggregate_episode_metrics(episode_records)
        metrics = {
            **canonical_metrics,
            **train_metric_aliases(canonical_metrics),
            **scenario_registry.metrics(
                episode_records,
                unique_trace_count=len(set(batch_fingerprints)),
                duplicate_trace_suppressed_count=duplicate_trace_suppressed_count,
            ),
        }
        metrics["train/per_device_train_batch_size"] = float(per_device_train_batch_size)
        metrics["train/gradient_accumulation_steps"] = float(
            resolved_gradient_accumulation_steps
        )
        metrics["train/effective_train_batch_size"] = float(effective_train_batch_size)
        metrics["train/num_generations"] = float(num_generations)
        metrics["train/use_vllm"] = float(bool(use_vllm))
        metrics["train/vllm_gpu_memory_utilization"] = (
            float(vllm_gpu_memory_utilization) if use_vllm else 0.0
        )
        metrics["train/trace_log_every"] = float(trace_log_every)
        metrics["train/trace_artifacts_logged"] = float(should_log_trace_artifacts)
        if rewards:
            metrics["train/reward_mean"] = _mean(rewards)
            metrics["train/reward_std"] = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

        try:
            log_trackio_metrics(metrics, step=trace_step["value"])
        except Exception as exc:
            print(f"Trackio metric logging skipped: {exc!r}")

        if should_log_trace_artifacts and sampled_traces:
            try:
                log_trace_table(
                    [record for _, _, record, _, _ in sampled_traces],
                    table_name="sample_traces",
                    step=trace_step["value"],
                )
            except Exception as exc:
                print(f"Trackio sample trace table logging skipped: {exc!r}")

            for index, env, _record, reward, fingerprint in sampled_traces:
                logged_trace_fingerprints.add(fingerprint)
                messages = list(getattr(env, "trace_messages", []))
                if index < len(completions):
                    completion_text = _completion_to_text(completions[index])
                    if completion_text:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"Raw generated completion:\n{completion_text}",
                            }
                        )
                metadata = dict(getattr(env, "trace_metadata", {}))
                metadata.update(
                    {
                        "sample_index": index,
                        "reward": reward,
                        "trace_step": trace_step["value"],
                        "trace_fingerprint": fingerprint,
                        "num_generations": num_generations,
                        "run_name": run_name,
                        "reward_config_id": reward_tracking_config["reward_config_id"],
                        "reward_config_hash": reward_tracking_config["reward_config_hash"],
                        "reward_stage": reward_tracking_config["reward_stage"],
                        "reward_mode": reward_tracking_config["reward_mode"],
                    }
                )
                try:
                    trackio.log(
                        {
                            f"cybersecurity_owasp_trace/sample_{index}": trackio.Trace(
                                messages=messages,
                                metadata=metadata,
                            )
                        },
                        step=trace_step["value"],
                    )
                except Exception as exc:
                    print(f"Trackio trace logging skipped: {exc!r}")
        elif sampled_traces:
            print(
                "Trackio trace artifacts throttled at reward callback "
                f"{trace_step['value']}; set --trace-log-every 1 for every callback "
                "or 0 to disable trace artifacts."
            )

        if rewards:
            print(
                "Reward batch: "
                f"mean={statistics.mean(rewards):.3f}, "
                f"min={min(rewards):.3f}, max={max(rewards):.3f}"
            )
        return rewards

    class TrackioSystemMetricsCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            try:
                reward_summary = log_reward_config(reward_settings, step=int(state.global_step or 0))
                metrics = log_gpu_metrics(step=int(state.global_step or 0))
                log_trackio_metrics(
                    {
                        "system/model_cache_hit": float(cache_hit),
                        "system/scenario_cache_required": 1.0,
                        "system/scenario_cache_entries": float(
                            scenario_cache_coverage.get("entries", 0)
                        ),
                        "system/hub_push_enabled": float(push_to_hub),
                    },
                    step=int(state.global_step or 0),
                )
                print(
                    "Trackio reward config logged: "
                    f"{reward_summary['reward_config_id']} "
                    f"({reward_summary['reward_config_hash']})"
                )
            except Exception as exc:
                print(f"Trackio initialization metrics skipped: {exc!r}")
                return control
            if metrics:
                system_summary = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(metrics.items())
                    if key.startswith("system/")
                )
                print(f"Trackio GPU metrics initialized: {system_summary}")
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                metrics = log_gpu_metrics(step=int(state.global_step or 0))
            except Exception as exc:
                print(f"Trackio GPU metrics skipped: {exc!r}")
                return control
            if metrics:
                summary = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items())[:4])
                print(f"Trackio GPU metrics logged at step {state.global_step}: {summary}")
            return control

        def on_train_end(self, args, state, control, **kwargs):
            try:
                log_gpu_metrics(step=int(state.global_step or 0))
            except Exception as exc:
                print(f"Trackio final GPU metrics skipped: {exc!r}")
            return control

    print(f"CUDA available: {torch.cuda.is_available()}")
    if source_mode == "public":
        print(f"Installed CyberSecurity_OWASP from public repo: {repo_url}@{repo_branch}")
    else:
        print(f"Packaged local CyberSecurity_OWASP repo; default env repo id: {env_repo_id}")
    print(f"Trackio Space: {trackio_space_id}")
    print(f"Trackio Project: {trackio_project}")
    print(f"Output repo: {output_repo_id}")
    print(f"Run name: {run_name}")
    print(f"Reward config: {reward_tracking_config['reward_config_id']}")
    print(f"Reward config hash: {reward_tracking_config['reward_config_hash']}")
    print(f"Model cache volume: {CACHE_VOLUME_NAME}")
    print(f"Scenario cache volume: {SCENARIO_CACHE_VOLUME_NAME}")
    print(f"Scenario cache dir: {scenario_cache_env['CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR']}")
    print("Scenario cache mode: require")
    print(f"Scenario cache coverage: {scenario_cache_coverage}")
    print(f"HF_HOME: {cache_env['HF_HOME']}")
    print(f"HF_HUB_CACHE: {cache_env['HF_HUB_CACHE']}")
    print(f"Torch cache: {cache_env['TORCH_HOME']}")
    print(f"Unsloth cache: {cache_env['UNSLOTH_CACHE_DIR']}")
    print(f"Triton cache: {cache_env['TRITON_CACHE_DIR']}")
    print(f"Hub push enabled: {push_to_hub}")
    print(
        "GRPO throughput config: "
        f"per_device_train_batch_size={per_device_train_batch_size}, "
        f"gradient_accumulation_steps={resolved_gradient_accumulation_steps}, "
        f"num_generations={num_generations}, "
        f"world_size={world_size}, "
        f"effective_train_batch_size={effective_train_batch_size}"
    )
    print(
        "Generation acceleration config: "
        f"use_vllm={use_vllm}, "
        f"vllm_gpu_memory_utilization={vllm_gpu_memory_utilization}, "
        f"trace_log_every={trace_log_every}"
    )

    expected_model_cache = _hf_model_cache_path(model_name)
    cache_hit = expected_model_cache.exists()
    print(f"Expected HF model cache path: {expected_model_cache}")
    print(f"Model cache hit before load: {cache_hit}")
    if cache_hit:
        print("Using cached model snapshot from the persistent Modal volume when valid.")
    else:
        print(
            "Model cache miss. Downloading model weights once into the persistent "
            "Modal cache volume; Hugging Face progress output should follow."
        )
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(HF_HUB_CACHE_DIR),
            token=hf_token,
        )
        print(f"Model snapshot ready: {snapshot_path}")
        cache_volume.commit()
        print(f"Committed Modal model cache volume after snapshot download: {CACHE_VOLUME_NAME}")
    except Exception as exc:
        print(
            "Explicit model snapshot prefetch failed; Unsloth will attempt the "
            f"model load directly. Error: {exc!r}"
        )

    print(f"Loading model with Unsloth from_pretrained: {model_name}")
    model_api = FastVisionModel
    model_load_values = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "load_in_4bit": False,
        "fast_inference": use_vllm,
        "gpu_memory_utilization": vllm_gpu_memory_utilization if use_vllm else None,
        "cache_dir": str(HF_HUB_CACHE_DIR),
        "token": hf_token,
    }
    from_pretrained_parameters = inspect.signature(model_api.from_pretrained).parameters
    from_pretrained_accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in from_pretrained_parameters.values()
    )
    skipped_model_load_keys = sorted(
        key
        for key, value in model_load_values.items()
        if value is not None
        and key not in from_pretrained_parameters
        and not from_pretrained_accepts_kwargs
    )
    if skipped_model_load_keys:
        print(f"Skipping unsupported from_pretrained keys: {skipped_model_load_keys}")
    model, tokenizer = model_api.from_pretrained(
        **{
            key: value
            for key, value in model_load_values.items()
            if value is not None
            and (key in from_pretrained_parameters or from_pretrained_accepts_kwargs)
        }
    )
    print("Model load complete.")
    cache_volume.commit()
    print(f"Committed Modal model cache volume after model load: {CACHE_VOLUME_NAME}")
    try:
        tokenizer = add_response_schema(tokenizer)
    except Exception as exc:
        print(
            "Tokenizer response schema add skipped for Gemma 4 processor, "
            "matching the Unsloth Gemma 4 GRPO notebook pattern: "
            f"{exc!r}"
        )

    model = model_api.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if hasattr(model_api, "for_training"):
        model_api.for_training(model)
    print("LoRA adapter attached and model switched to training mode.")

    grpo_config_values = {
        "temperature": 1.0,
        "learning_rate": 5e-6,
        "weight_decay": 0.001,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": resolved_gradient_accumulation_steps,
        "num_generations": num_generations,
        "max_prompt_length": max_seq_length,
        "max_completion_length": max_completion_length,
        "max_steps": max_steps,
        "save_steps": max(10, max_steps),
        "report_to": "trackio",
        "project": trackio_project,
        "trackio_space_id": trackio_space_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "push_to_hub": push_to_hub,
        "hub_model_id": output_repo_id,
        "hub_private_repo": True,
        "hub_strategy": "every_save",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "use_vllm": use_vllm,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "delta": 1.5,
        "loss_type": "bnpo",
        "mask_truncated_completions": False,
    }
    grpo_config_parameters = set(inspect.signature(GRPOConfig).parameters)
    skipped_config_keys = sorted(set(grpo_config_values) - grpo_config_parameters)
    if skipped_config_keys:
        print(f"Skipping unsupported GRPOConfig keys: {skipped_config_keys}")
    training_args = GRPOConfig(
        **{
            key: value
            for key, value in grpo_config_values.items()
            if key in grpo_config_parameters
        }
    )

    trainer_values = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": cybersecurity_owasp_reward,
        "args": training_args,
        "train_dataset": dataset,
        "environment_factory": CyberSecurityOWASPToolEnv,
        "callbacks": [TrackioSystemMetricsCallback()],
    }
    trainer_parameters = set(inspect.signature(GRPOTrainer).parameters)
    skipped_trainer_keys = sorted(set(trainer_values) - trainer_parameters)
    if skipped_trainer_keys:
        print(f"Skipping unsupported GRPOTrainer keys: {skipped_trainer_keys}")
    trainer = GRPOTrainer(
        **{
            key: value
            for key, value in trainer_values.items()
            if key in trainer_parameters
        }
    )
    print("Starting GRPO trainer.train().")
    heartbeat_stop = threading.Event()

    def _training_heartbeat() -> None:
        start_time = time.monotonic()
        while not heartbeat_stop.wait(30):
            elapsed = int(time.monotonic() - start_time)
            print(
                "Training heartbeat: still inside trainer.train() "
                f"after {elapsed}s. For this smoke, the slow part is usually "
                f"Gemma generation/backprop on L4: {num_generations} completions "
                f"up to {max_completion_length} tokens, plus Trackio upload."
            )

    heartbeat_thread = threading.Thread(
        target=_training_heartbeat,
        name="grpo-training-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        trainer.train()
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
    print("GRPO trainer.train() complete.")
    if push_to_hub:
        print(f"Pushing LoRA adapter to Hugging Face Hub: {output_repo_id}")
        trainer.push_to_hub()
        print("Hub push complete.")
    else:
        print("Skipping Hub push for this run. Pass --push-to-hub to upload adapters.")
    volume.commit()
    cache_volume.commit()
    scenario_cache_volume.commit()
    print(f"Committed run volume: {VOLUME_NAME}")
    print(f"Committed model cache volume: {CACHE_VOLUME_NAME}")
    print(f"Committed scenario cache volume: {SCENARIO_CACHE_VOLUME_NAME}")
    try:
        trackio.finish()
    except RuntimeError as exc:
        print(f"Trackio finish skipped because the trainer already finalized it: {exc}")

    return {
        "run_name": run_name,
        "env_repo_id": env_repo_id,
        "output_repo_id": output_repo_id,
        "trackio_space_id": trackio_space_id,
        "trackio_project": trackio_project,
        "max_steps": max_steps,
        "dataset_size": dataset_size,
        "difficulty": difficulty,
        "split": split,
        "model_name": model_name,
        "max_completion_length": max_completion_length,
        "num_generations": num_generations,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": resolved_gradient_accumulation_steps,
        "effective_train_batch_size": effective_train_batch_size,
        "use_vllm": int(bool(use_vllm)),
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "trace_log_every": trace_log_every,
        "source_mode": source_mode,
        "repo_url": repo_url,
        "repo_branch": repo_branch,
        "push_to_hub": push_to_hub,
        "scenario_cache_volume": SCENARIO_CACHE_VOLUME_NAME,
        "scenario_cache_mode": "require",
        **reward_tracking_config,
    }


@app.local_entrypoint()
def main(
    mode: str = "train",
    env_repo_id: str = "",
    output_repo_id: str = "",
    max_steps: int = 10,
    dataset_size: int = 16,
    difficulty: int = 0,
    split: str = "train",
    model_name: str = DEFAULT_GEMMA_MODEL,
    max_seq_length: int = 4096,
    max_completion_length: int = 768,
    lora_rank: int = 32,
    trackio_space_id: str = "Humanlearning/CyberSecurity_OWASP-trackio",
    trackio_project: str = "CyberSecurity_OWASP-grpo",
    num_generations: int = 6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 0,
    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.2,
    trace_log_every: int = 5,
    seed_start: int = 0,
    git_sha: str = "nogit",
    run_name: str = "",
    source_mode: str = "local",
    repo_url: str = PUBLIC_REPO_URL,
    repo_branch: str = PUBLIC_REPO_BRANCH,
    detach: bool = False,
    push_to_hub: bool = False,
    cache_seed_start: int = 0,
    cache_difficulty_buckets: int = 0,
    cache_train_per_bucket: int = 0,
    cache_validation_per_bucket: int = 0,
    cache_heldout_per_bucket: int = 0,
    cache_force: bool = False,
) -> None:
    model_name = _ensure_gemma4_model(model_name)
    if mode == "prepare-cache":
        result = prepare_modal_scenario_cache.remote(
            seed_start=cache_seed_start,
            difficulty_buckets=cache_difficulty_buckets,
            train_per_bucket=cache_train_per_bucket,
            validation_per_bucket=cache_validation_per_bucket,
            heldout_per_bucket=cache_heldout_per_bucket,
            force=cache_force,
        )
        print(f"Prepared scenario cache: {result}")
        return
    if mode == "config":
        result = check_training_imports.remote()
        print(result)
        return
    if mode == "baseline":
        if int(num_generations) != 1:
            raise ValueError("baseline mode expects --num-generations 1.")
        trace_log_every = max(0, int(trace_log_every))
        run_name = run_name or "baseline"
        preflight = verify_modal_scenario_cache_for_training.remote(
            split=split,
            difficulty=difficulty,
            dataset_size=dataset_size,
            seed_start=seed_start,
        )
        print(f"CPU scenario cache preflight passed: {preflight}")
        kwargs = dict(
            max_steps=max_steps,
            dataset_size=dataset_size,
            difficulty=difficulty,
            split=split,
            model_name=model_name,
            max_seq_length=max_seq_length,
            max_completion_length=max_completion_length,
            trackio_space_id=trackio_space_id,
            trackio_project=trackio_project,
            num_generations=num_generations,
            trace_log_every=trace_log_every,
            seed_start=seed_start,
            git_sha=git_sha,
            run_name=run_name,
            source_mode=source_mode,
            repo_url=repo_url,
            repo_branch=repo_branch,
        )
        if detach:
            call = run_cybersecurity_owasp_baseline.spawn(**kwargs)
            print(f"Spawned Modal baseline call: {call.object_id}")
        else:
            result = run_cybersecurity_owasp_baseline.remote(**kwargs)
            print(f"Baseline result: {result}")
        return
    if mode != "train":
        raise ValueError("mode must be 'prepare-cache', 'train', 'baseline', or 'config'")

    (
        resolved_gradient_accumulation_steps,
        effective_train_batch_size,
    ) = _resolve_grpo_batch_config(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        world_size=1,
    )
    _validate_vllm_config(
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    trace_log_every = max(0, int(trace_log_every))

    trackio_space_id = trackio_space_id or os.environ.get(
        "TRACKIO_SPACE_ID",
        "Humanlearning/CyberSecurity_OWASP-trackio",
    )
    trackio_project = trackio_project or os.environ.get(
        "TRACKIO_PROJECT", "CyberSecurity_OWASP-grpo"
    )
    resolved_trackio_space_id = trackio_space_id
    resolved_output_repo_id = output_repo_id
    if not resolved_trackio_space_id or not resolved_output_repo_id:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import whoami

                user = whoami(token=hf_token)["name"]
                if not resolved_trackio_space_id:
                    resolved_trackio_space_id = (
                        f"{user}/CyberSecurity_OWASP-trackio"
                        if user == "humandotlearning"
                        else "Humanlearning/CyberSecurity_OWASP-trackio"
                    )
                resolved_output_repo_id = (
                    resolved_output_repo_id
                    or f"{user}/CyberSecurity_OWASP-{_model_repo_slug(model_name)}-grpo-lora"
                )
            except Exception as exc:
                print(f"Could not resolve Hugging Face defaults locally: {exc!r}")

    if git_sha == "nogit":
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_sha = "nogit"

    model_slug = model_name.replace("/", "-")
    local_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = run_name or (
        f"CyberSecurity_OWASP-{model_slug}-grpo-level{difficulty}-"
        f"{local_stamp}-{git_sha[:8]}"
    )

    print(f"Run name: {run_name}")
    print(f"Source mode: {source_mode}")
    if source_mode == "public":
        print(f"Public repo: {repo_url}@{repo_branch}")
    if resolved_trackio_space_id:
        print(f"Trackio Space: https://huggingface.co/spaces/{resolved_trackio_space_id}")
    else:
        print("Trackio Space: derived remotely from HF_TOKEN as <hf-user>/CyberSecurity_OWASP-trackio")
    if resolved_output_repo_id:
        print(f"Output model repo: https://huggingface.co/{resolved_output_repo_id}")
    else:
        print(
            "Output model repo: derived remotely from HF_TOKEN as "
            f"<hf-user>/CyberSecurity_OWASP-{_model_repo_slug(model_name)}-grpo-lora"
        )
    print(f"Hub push enabled: {push_to_hub}")
    print(f"Model cache volume: {CACHE_VOLUME_NAME}")
    print(f"Scenario cache volume: {SCENARIO_CACHE_VOLUME_NAME}")
    print(
        "GRPO throughput config: "
        f"per_device_train_batch_size={per_device_train_batch_size}, "
        f"gradient_accumulation_steps={resolved_gradient_accumulation_steps}, "
        f"num_generations={num_generations}, "
        f"effective_train_batch_size={effective_train_batch_size}"
    )
    print(
        "Generation acceleration config: "
        f"use_vllm={use_vllm}, "
        f"vllm_gpu_memory_utilization={vllm_gpu_memory_utilization}, "
        f"trace_log_every={trace_log_every}"
    )
    print("Launch phases:")
    print(
        "1. Modal image build/validation: happens before remote Python logs; "
        "slow when local source or dependency layers changed."
    )
    print("2. CPU-only scenario cache preflight in CyberSecurity_OWASP-scenario-cache.")
    print("3. GPU container start on one L4 only after cache preflight passes.")
    print("4. Model cache check in CyberSecurity_OWASP-model-cache.")
    print("5. Cached snapshot load into GPU RAM with Unsloth progress.")
    print("6. GRPO steps, Trackio sync, and volume commit.")
    print(
        "If there is a long pause after trainer.train() starts, watch for "
        "Training heartbeat lines every 30 seconds."
    )

    kwargs = dict(
        env_repo_id=env_repo_id,
        output_repo_id=output_repo_id,
        max_steps=max_steps,
        dataset_size=dataset_size,
        difficulty=difficulty,
        split=split,
        model_name=model_name,
        max_seq_length=max_seq_length,
        max_completion_length=max_completion_length,
        lora_rank=lora_rank,
        trackio_space_id=trackio_space_id,
        trackio_project=trackio_project,
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=resolved_gradient_accumulation_steps,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        trace_log_every=trace_log_every,
        seed_start=seed_start,
        git_sha=git_sha,
        run_name=run_name,
        source_mode=source_mode,
        repo_url=repo_url,
        repo_branch=repo_branch,
        push_to_hub=push_to_hub,
    )
    preflight = verify_modal_scenario_cache_for_training.remote(
        split=split,
        difficulty=difficulty,
        dataset_size=dataset_size,
        seed_start=seed_start,
    )
    print(f"CPU scenario cache preflight passed: {preflight}")
    if detach:
        call = train_cybersecurity_owasp_grpo.spawn(**kwargs)
        print(f"Spawned Modal training call: {call.object_id}")
    else:
        result = train_cybersecurity_owasp_grpo.remote(**kwargs)
        print(f"Training result: {result}")
