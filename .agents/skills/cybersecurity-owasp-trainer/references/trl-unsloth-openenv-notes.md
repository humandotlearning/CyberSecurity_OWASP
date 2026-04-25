# TRL, Unsloth, And OpenEnv Notes

Sources checked for this skill:

- TRL GRPO Trainer: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL OpenEnv integration: https://huggingface.co/docs/trl/en/openenv
- Unsloth RL Guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Unsloth Advanced RL Documentation: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
- Unsloth vLLM deployment/saving guide: https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide

Recheck these pages before major dependency upgrades because TRL, OpenEnv integration, vLLM, and Unsloth RL APIs move quickly.

## TRL GRPO

- GRPO is an online RL method that samples multiple completions per prompt, scores them with reward functions, and optimizes relative advantage within the group.
- `GRPOTrainer` accepts one or more reward functions. Custom reward functions receive prompts, completions, completion IDs, trainer state, and dataset columns through keyword arguments.
- Multiple reward functions are summed unless reward weights are configured. Use separate component functions for logging and diagnosis.
- TRL logs component reward means/stds, total reward, completion length, KL when enabled, entropy, clipping metrics, and token/step timing.
- vLLM is the main acceleration path for generation. Colocate mode shares the trainer process/GPU; server mode is better when inference has separate GPUs.

## TRL With OpenEnv

- Use environment training when state carries across turns and observations depend on prior actions.
- Current TRL docs prefer `environment_factory` for automatic multi-turn tool loops. It exposes public methods as tools and uses an `environments` argument in reward functions.
- `rollout_func` is still appropriate when the repo needs a custom generation, parsing, artifact, or client loop. CyberSecurity_OWASP currently has this shape in `training/rollout.py`.
- If migrating from `rollout_func` to `environment_factory`, preserve typed action validation, phase gating, reward breakdowns, anti-cheat flags, and rollout artifact output.
- For concurrent training, match OpenEnv server session capacity to the generation batch. Create clients lazily in `reset` and close old sessions before reopening.

## Unsloth RL Guidance

- Use Unsloth for memory-efficient LoRA/QLoRA GRPO when local hardware is constrained.
- Start from a capable instruct model or lightly format-tuned model. If success probability is effectively zero, RL will not bootstrap.
- Keep reward functions/verifiers simple and trustworthy first; add shaping only after sparse reward blocks learning.
- Unsloth recipes commonly use Qwen, Gemma, Llama, Phi, Mistral, and gpt-oss variants. For this repo, prefer the configured `Qwen/Qwen3-1.7B` or another small instruct/coder checkpoint for smoke runs.
- For Unsloth-specific GRPO recipes, use more than two generations per prompt when hardware allows. Keep the repo's small `num_generations=2` only as a low-cost smoke/debug default unless tests prove it is sufficient.
- Pin torch, CUDA, vLLM, TRL, and Unsloth versions for any serious run, then run a short smoke test before scaling.

## Saving And Serving

- Save LoRA adapters directly when adapters are enough for evaluation or continued training.
- Use Unsloth-supported merged save methods for deployment formats, such as merged 16-bit for vLLM serving.
- Avoid manually upcasting a 4-bit model and merging LoRA weights outside the supported save path.
- After saving, immediately run post-training inference against a small held-out set to prove the artifact loads and still follows the JSON action protocol.
