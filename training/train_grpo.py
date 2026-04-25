"""Modal-only GRPO config helper for CyberSecurity_OWASP.

This module intentionally does not run local training.
Use `scripts/modal_train_grpo.py` (persistent) or
`scripts/modal_ephemeral_train.py` (smoke) for execution.
"""

from __future__ import annotations

import os

from training.trackio_utils import build_run_name, get_git_sha


DEFAULT_GEMMA_MODEL = os.getenv("MODEL_NAME", "unsloth/gemma-4-E2B-it")


def ensure_gemma4_model(model_name: str) -> str:
    if model_name != "unsloth/gemma-4-E2B-it":
        raise ValueError(
            "CyberSecurity_OWASP GRPO is pinned to unsloth/gemma-4-E2B-it, "
            "matching the Unsloth Gemma 4 E2B RL notebook."
        )
    return model_name


def build_grpo_config():
    """Build the TRL GRPOConfig used by the Modal training pipeline."""

    from trl import GRPOConfig

    model_name = ensure_gemma4_model(os.getenv("MODEL_NAME", DEFAULT_GEMMA_MODEL))
    difficulty = int(os.getenv("DIFFICULTY", "0"))
    output_dir = os.getenv(
        "OUTPUT_DIR",
        f"CyberSecurity_OWASP-{model_name.replace('/', '-')}-grpo",
    )
    trackio_space_id = os.getenv("TRACKIO_SPACE_ID", "Humanlearning/CyberSecurity_OWASP-trackio")
    os.environ.setdefault("TRACKIO_PROJECT", "CyberSecurity_OWASP-grpo")
    run_name = os.getenv(
        "RUN_NAME",
        build_run_name(model_name, "grpo", difficulty, git_sha=get_git_sha()),
    )
    return GRPOConfig(
        output_dir=output_dir,
        report_to="trackio",
        trackio_space_id=trackio_space_id,
        run_name=run_name,
        logging_steps=1,
        save_steps=25,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_generations=6,
        max_prompt_length=4096,
        max_completion_length=768,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "CyberSecurity_OWASP GRPO config helper."
            " Actual GRPO training is executed on Modal only."
        )
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="Optional curriculum difficulty included in the generated run name.",
    )
    parser.add_argument("--model-name", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional GRPO output_dir override.",
    )
    args = parser.parse_args()

    os.environ["MODEL_NAME"] = ensure_gemma4_model(args.model_name)
    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir

    config = build_grpo_config()
    print("GRPO config (Modal execution):")
    print(config)
    print(
        "Run on Modal, for example:\n"
        "uv run --extra modal modal run scripts/modal_train_grpo.py "
        f"--model-name {args.model_name} --difficulty {args.difficulty}"
    )


if __name__ == "__main__":
    main()
