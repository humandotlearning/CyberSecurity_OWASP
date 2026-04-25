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
    from CyberSecurity_OWASP.reward_config import compute_token_penalty
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
    available_scenarios = int(
        coverage.get("counts", {})
        .get(split, {})
        .get(str(resolved_difficulty), 0)
    )
    if available_scenarios < dataset_size:
        raise RuntimeError(
            "Scenario cache does not cover this Modal dataset. Run "
            "--mode prepare-cache with a larger per-bucket count before training. "
            f"available={available_scenarios}, requested_dataset_size={dataset_size}, "
            f"split={split}, difficulty={resolved_difficulty}"
        )

    env = CybersecurityOwaspEnvironment()
    try:
        obs = env.reset(seed=seed_start, split=split, difficulty=difficulty)
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
        "difficulty": resolved_difficulty,
        "dataset_size": dataset_size,
        "available_scenarios": available_scenarios,
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
    from CyberSecurity_OWASP.reward_config import compute_token_penalty
    from CyberSecurity_OWASP.server.curriculum import CurriculumController
    from CyberSecurity_OWASP.server.scenario_cache import ScenarioCache
    from training.trackio_utils import (
        aggregate_episode_metrics,
        episode_record_from_state,
        episode_trace_fingerprint,
        log_gpu_metrics,
        log_trace_table,
        log_trackio_metrics,
        train_metric_aliases,
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
    available_scenarios = int(
        scenario_cache_coverage.get("counts", {})
        .get(split, {})
        .get(str(int(scenario_profile["difficulty"])), 0)
    )
    if available_scenarios < dataset_size:
        raise RuntimeError(
            "Scenario cache does not cover this Modal dataset. Run "
            "--mode prepare-cache with a larger per-bucket count before training. "
            f"available={available_scenarios}, requested_dataset_size={dataset_size}, "
            f"split={split}, difficulty={scenario_profile['difficulty']}"
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
        [
            {
                "prompt": [{"role": "user", "content": training_prompt}],
                "seed": seed_start + index,
                "difficulty": difficulty,
                "split": split,
            }
            for index in range(dataset_size)
        ]
    )

    def _state_snapshot(env: CybersecurityOwaspEnvironment) -> dict[str, Any]:
        state = env.state
        return {
            "episode_id": state.episode_id,
            "task_id": state.task_id,
            "seed": state.seed,
            "split": state.split,
            "difficulty": state.difficulty,
            "domain": state.domain,
            "bug_family": state.bug_family,
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
            self.trace_messages: list[dict[str, str]] = []
            self.trace_metadata: dict[str, Any] = {}

        def reset(self, **kwargs) -> str:
            seed = int(kwargs.get("seed", seed_start))
            current_difficulty = int(kwargs.get("difficulty", difficulty))
            current_split = str(kwargs.get("split", split))
            obs = self._env.reset(
                seed=seed,
                split=current_split,
                difficulty=current_difficulty,
            )
            self.reward = 0.0
            self.reward_breakdown = {}
            self.done = bool(obs.done)
            self.success = False
            self.invalid_actions = 0
            self.trace_messages = [
                {
                    "role": "user",
                    "content": (
                        f"{training_prompt}\n\nInitial observation:\n"
                        f"Phase: {obs.phase}\n"
                        f"Task: {obs.task_brief}\n"
                        f"Available actions: {obs.available_actions}\n"
                        f"Workspace summary: {obs.workspace_summary}\n"
                        f"Policy hint: {obs.visible_policy_hint}\n"
                        f"Message: {obs.message}"
                    ),
                }
            ]
            self.trace_metadata = _state_snapshot(self._env)
            return obs.message

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

        canonical_metrics = aggregate_episode_metrics(episode_records)
        metrics = {
            **canonical_metrics,
            **train_metric_aliases(canonical_metrics),
        }
        if rewards:
            metrics["train/reward_mean"] = _mean(rewards)
            metrics["train/reward_std"] = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

        try:
            log_trackio_metrics(metrics, step=trace_step["value"])
        except Exception as exc:
            print(f"Trackio metric logging skipped: {exc!r}")

        sampled_traces = []
        seen_this_batch: set[str] = set()
        for index, (env, record, reward) in enumerate(zip(environments, episode_records, rewards)):
            fingerprint = episode_trace_fingerprint(record)
            if fingerprint in seen_this_batch or fingerprint in logged_trace_fingerprints:
                continue
            seen_this_batch.add(fingerprint)
            logged_trace_fingerprints.add(fingerprint)
            sampled_traces.append((index, env, record, reward, fingerprint))
            if len(sampled_traces) >= 4:
                break

        try:
            log_trace_table(
                [record for _, _, record, _, _ in sampled_traces],
                table_name="sample_traces",
                step=trace_step["value"],
            )
        except Exception as exc:
            print(f"Trackio sample trace table logging skipped: {exc!r}")

        for index, env, _record, reward, fingerprint in sampled_traces:
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
                    "run_name": run_name,
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
            except Exception as exc:
                print(f"Trackio GPU metrics initialization skipped: {exc!r}")
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
    model, tokenizer = model_api.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        cache_dir=str(HF_HUB_CACHE_DIR),
        token=hf_token,
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
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": max(2, num_generations),
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
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "delta": 1.5,
        "loss_type": "bnpo",
        "mask_truncated_completions": True,
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
        "source_mode": source_mode,
        "repo_url": repo_url,
        "repo_branch": repo_branch,
        "push_to_hub": push_to_hub,
        "scenario_cache_volume": SCENARIO_CACHE_VOLUME_NAME,
        "scenario_cache_mode": "require",
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
    seed_start: int = 0,
    git_sha: str = "nogit",
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
    if mode != "train":
        raise ValueError("mode must be 'prepare-cache', 'train', or 'config'")

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
    run_name = (
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
