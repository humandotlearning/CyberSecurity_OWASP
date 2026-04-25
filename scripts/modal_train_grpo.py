"""Persistent Modal GRPO launcher for CyberSecurity_OWASP.

This packages the local repository into a Modal GPU image, runs a small
tool-use GRPO job against the in-process CyberSecurity_OWASP environment, logs
metrics/traces to Trackio, and saves LoRA checkpoints in a persistent Modal
volume.

Example:

    uv run --extra modal modal run scripts/modal_train_grpo.py \
        --max-steps 10 \
        --dataset-size 16 \
        --num-generations 2 \
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
SECRET_NAME = "CyberSecurity_OWASP-secrets"
RUNS_DIR = pathlib.Path("/runs")
REMOTE_PROJECT = "/root/CyberSecurity_OWASP"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC_REPO_URL = "https://github.com/humandotlearning/CyberSecurity_OWASP.git"
PUBLIC_REPO_BRANCH = "master"


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
            f"python -m pip install -e {REMOTE_PROJECT}",
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
            f"python -m pip install -e {REMOTE_PROJECT}",
        )

    return image.run_commands(
        "python -c \"import os, torch; import transformers.utils.hub as hub; "
        "hub.TRANSFORMERS_CACHE = getattr(hub, 'TRANSFORMERS_CACHE', "
        "os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')); "
        "from trl import GRPOConfig, GRPOTrainer; "
        "from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import "
        "CybersecurityOwaspEnvironment; print('trainer import ok', torch.__version__)\"",
    ).workdir(REMOTE_PROJECT)


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = _modal_secrets()


@app.function(
    image=_training_image(),
    gpu=["L4", "A10G"],
    timeout=4 * 60 * 60,
    volumes={RUNS_DIR: volume},
    secrets=secrets,
)
def check_training_imports() -> dict[str, str]:
    import torch
    import trackio
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

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
        "unsloth_model": FastLanguageModel.__name__,
        "env": CybersecurityOwaspEnvironment.__name__,
        "reset_phase": obs.phase,
    }


@app.function(
    image=_training_image(),
    gpu=["L4", "A10G"],
    timeout=4 * 60 * 60,
    volumes={RUNS_DIR: volume},
    secrets=secrets,
)
def train_cybersecurity_owasp_grpo(
    env_repo_id: str = "",
    output_repo_id: str = "",
    max_steps: int = 10,
    dataset_size: int = 16,
    difficulty: int = 0,
    split: str = "train",
    model_name: str = "Qwen/Qwen3-1.7B",
    max_seq_length: int = 4096,
    max_completion_length: int = 768,
    lora_rank: int = 32,
    trackio_space_id: str = "",
    trackio_project: str = "CyberSecurity_OWASP-grpo",
    num_generations: int = 2,
    seed_start: int = 0,
    git_sha: str = "nogit",
    run_name: str = "",
    source_mode: str = "local",
    repo_url: str = PUBLIC_REPO_URL,
    repo_branch: str = PUBLIC_REPO_BRANCH,
) -> dict[str, str | int | float]:
    import inspect
    import statistics

    import torch
    from unsloth import FastLanguageModel
    import transformers.utils.hub as transformers_hub
    from datasets import Dataset
    from huggingface_hub import whoami
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer, clone_chat_template
    from trl.chat_template_utils import add_response_schema

    import trackio

    from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )

    if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
        transformers_hub.TRANSFORMERS_CACHE = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            f"HF_TOKEN is missing from the Modal secret {SECRET_NAME}."
        )

    user = whoami(token=hf_token)["name"]
    env_repo_id = env_repo_id or f"{user}/CyberSecurity_OWASP"
    output_repo_id = output_repo_id or f"{user}/CyberSecurity_OWASP-qwen3-1.7b-grpo-lora"
    trackio_space_id = trackio_space_id or f"{user}/CyberSecurity_OWASP-trackio"

    os.environ["TRACKIO_SPACE_ID"] = trackio_space_id
    os.environ["TRACKIO_PROJECT"] = trackio_project

    model_slug = model_name.replace("/", "-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = run_name or (
        f"CyberSecurity_OWASP-{model_slug}-grpo-level{difficulty}-{stamp}-{git_sha[:8]}"
    )
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_prompt = (
        "You are a defensive AppSec repair agent in the local CyberSecurity_OWASP "
        "OpenEnv environment. Use only the provided local tools. Do not target real "
        "systems. Work step by step: inspect policy and generated code, reproduce the "
        "authorization issue locally, submit a policy-tied finding, patch the generated "
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
            self.reward = float(obs.reward_breakdown.get("total", obs.reward or 0.0))
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

        def submit_finding(
            self,
            summary: str,
            evidence: str,
            policy_rule: str,
        ) -> str:
            """
            Submit structured evidence for the suspected authorization bug.

            Args:
                summary: Concise description of the suspected access-control bug.
                evidence: Local reproduction evidence from policy, code, or requests.
                policy_rule: Policy rule that the observed behavior violates.

            Returns:
                Finding acceptance result and next phase information.
            """
            return self._step(
                "submit_finding",
                {
                    "summary": summary,
                    "evidence": evidence,
                    "policy_rule": policy_rule,
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

        def _score(self) -> float:
            return float(self.reward)

        def __del__(self):
            try:
                self._env.close()
            except Exception:
                pass

    trace_step = {"value": 0}

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
        rewards = [float(env._score()) for env in environments]
        completions = kwargs.get("completions") or kwargs.get("completion") or []
        trace_step["value"] += 1

        breakdowns = [getattr(env, "reward_breakdown", {}) or {} for env in environments]
        metrics = {
            "train/reward_total_mean": _mean(rewards),
            "train/reward_discovery_mean": _mean(
                [float(item.get("discovery", 0.0)) for item in breakdowns]
            ),
            "train/reward_security_mean": _mean(
                [float(item.get("security", 0.0)) for item in breakdowns]
            ),
            "train/reward_regression_mean": _mean(
                [float(item.get("regression", 0.0)) for item in breakdowns]
            ),
            "train/reward_public_routes_mean": _mean(
                [float(item.get("public_routes", 0.0)) for item in breakdowns]
            ),
            "train/reward_patch_quality_mean": _mean(
                [float(item.get("patch_quality", 0.0)) for item in breakdowns]
            ),
            "train/reward_visible_tests_mean": _mean(
                [float(item.get("visible_tests", 0.0)) for item in breakdowns]
            ),
            "train/reward_anti_cheat_mean": _mean(
                [float(item.get("anti_cheat", 0.0)) for item in breakdowns]
            ),
            "train/success_rate": _mean(
                [1.0 if bool(getattr(env, "success", False)) else 0.0 for env in environments]
            ),
            "train/invalid_action_rate": _mean(
                [float(getattr(env, "invalid_actions", 0)) for env in environments]
            ),
            "train/episode_length_mean": _mean(
                [
                    float(getattr(env, "trace_metadata", {}).get("step_count", 0))
                    for env in environments
                ]
            ),
        }

        try:
            trackio.log(metrics, step=trace_step["value"])
        except Exception as exc:
            print(f"Trackio metric logging skipped: {exc!r}")

        for index, env in enumerate(environments):
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
                    "reward": rewards[index],
                    "trace_step": trace_step["value"],
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
        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                metrics = trackio.log_gpu()
            except Exception as exc:
                print(f"Trackio GPU metrics skipped: {exc!r}")
                return control
            if metrics:
                summary = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items())[:4])
                print(f"Trackio GPU metrics logged at step {state.global_step}: {summary}")
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        token=hf_token,
    )
    try:
        tokenizer = add_response_schema(tokenizer)
    except Exception as exc:
        print(f"Tokenizer response schema add failed before cloning: {exc!r}")
        model, tokenizer, added_tokens = clone_chat_template(
            model,
            tokenizer,
            "Qwen/Qwen3-0.6B",
        )
        print(f"Cloned Qwen3 chat template; added {len(added_tokens)} tokens.")
        tokenizer = add_response_schema(tokenizer)

    model = FastLanguageModel.get_peft_model(
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
    FastLanguageModel.for_training(model)

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
        "trackio_space_id": trackio_space_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "push_to_hub": True,
        "hub_model_id": output_repo_id,
        "hub_private_repo": True,
        "hub_strategy": "every_save",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
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
    trainer.train()
    trainer.push_to_hub()
    volume.commit()

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
    model_name: str = "Qwen/Qwen3-1.7B",
    max_seq_length: int = 4096,
    max_completion_length: int = 768,
    lora_rank: int = 32,
    trackio_space_id: str = "",
    trackio_project: str = "CyberSecurity_OWASP-grpo",
    num_generations: int = 2,
    seed_start: int = 0,
    git_sha: str = "nogit",
    source_mode: str = "local",
    repo_url: str = PUBLIC_REPO_URL,
    repo_branch: str = PUBLIC_REPO_BRANCH,
    detach: bool = False,
) -> None:
    if mode == "config":
        result = check_training_imports.remote()
        print(result)
        return
    if mode != "train":
        raise ValueError("mode must be 'train' or 'config'")

    trackio_space_id = trackio_space_id or os.environ.get("TRACKIO_SPACE_ID", "")
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
                resolved_trackio_space_id = (
                    resolved_trackio_space_id or f"{user}/CyberSecurity_OWASP-trackio"
                )
                resolved_output_repo_id = (
                    resolved_output_repo_id
                    or f"{user}/CyberSecurity_OWASP-qwen3-1.7b-grpo-lora"
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
            "<hf-user>/CyberSecurity_OWASP-qwen3-1.7b-grpo-lora"
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
    )
    if detach:
        call = train_cybersecurity_owasp_grpo.spawn(**kwargs)
        print(f"Spawned Modal training call: {call.object_id}")
    else:
        result = train_cybersecurity_owasp_grpo.remote(**kwargs)
        print(f"Training result: {result}")
