---
name: cybersecurity-owasp-trainer
description: Train, debug, evaluate, and document CyberSecurity_OWASP model runs with OpenEnv, TRL/GRPO, optional Unsloth/QLoRA, Trackio, rollout artifacts, baseline-vs-trained evaluation, and reward-hacking safeguards. Use when working on training scripts, launch commands, rollouts, Trackio metrics, model saving, or hackathon demo evidence for this repo.
---

# CyberSecurity_OWASP Trainer

## Overview

Use this skill to run or modify the CyberSecurity_OWASP training and evaluation loop without weakening the verifier, reward integrity, or hackathon evidence trail. Training is expected to run on Modal only.

Important: do **not** run GRPO/PPO training loops locally in this repo. Use Modal launchers (`scripts/modal_ephemeral_train.py` for smoke and `scripts/modal_train_grpo.py` for GRPO).

## References

- Load `references/hackathon-training-notes.md` when checking hackathon expectations, demo evidence, reward-hacking safeguards, or scaling order.
- Load `references/trl-unsloth-openenv-notes.md` before changing TRL, OpenEnv training integration, Unsloth/QLoRA settings, vLLM settings, or model saving.
- Use the repo's existing `openenv-cli` skill for OpenEnv CLI command details.
- Use the repo's existing `hugging-face-trackio` skill for Trackio API, dashboard, alert, or metric retrieval details.

## Preflight Gate

Do not start real training until all checks below are true:

- `reset`, `step`, `state`, typed actions, observations, and terminal states work deterministically.
- Verifier and reward tests cover exploit blocking, regression preservation, public routes, visible tests, app boot, anti-cheat, invalid actions, and no repeated reward inflation.
- Hidden tests and protected files cannot be read or patched through environment actions.
- `send_local_request` is restricted to the generated local app.
- A local server or Docker server can run, and at least one manual episode completes.
- Scripted random, bad, and oracle policies run without crashing; oracle gets high reward on easy seeds.
- At least 10 validation rollouts complete and sampled rollout artifacts look behaviorally plausible.
- The validated scenario cache exists, is mounted, and meets the configured split/difficulty minimums.
- Modal smoke and GRPO runs use `CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require`; runtime `reset()` must not compile scenarios or call an LLM during training/eval.
- Trackio run config is set and can log a smoke metric locally or to the canonical Space.

If any gate fails, fix the environment, verifier, reward engine, or rollout parser before touching trainer scale.

## Repo Training Path

Prefer the existing repo modules:

- `training/rollout.py`: full OpenEnv episode loop, action JSON parsing, reward trace, rollout artifact fields.
- `training/reward_funcs.py`: component reward functions exposed to TRL/GRPO.
- `training/train_grpo.py`: `GRPOConfig`/model defaults and launch intent (does not run local training).
- `training/eval_before_after.py`: baseline-vs-trained and held-out summary metrics.
- `training/trackio_utils.py`: run naming, canonical metric names, Trackio init/log/finalize helpers.

Default environment values:

```powershell
$env:MODEL_NAME = "unsloth/gemma-4-E2B-it"
$env:TRACKIO_SPACE_ID = "Humanlearning/CyberSecurity_OWASP-trackio"
$env:TRACKIO_PROJECT = "CyberSecurity_OWASP"
$env:DIFFICULTY = "0"
$env:CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR = "scenario_cache"
$env:CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE = "fallback"
```

Use level-0 debug runs before scaling, and verify them through Modal smoke/ephemeral runs.

Modal uses two persistent cache volumes:

- `CyberSecurity_OWASP-model-cache`: Hugging Face, torch, Unsloth, Triton, and model artifacts.
- `CyberSecurity_OWASP-scenario-cache`: validated executable scenario bundles for `reset()`.

Scenario/curriculum authoring is config-driven through `configs/scenario_authoring.small.json`. The default offline author model is `deepseek-ai/DeepSeek-V4-Pro`; this is not the RL training policy model. The RL training model is pinned to `unsloth/gemma-4-E2B-it`, matching the Unsloth Gemma 4 E2B RL notebook.

## Training Workflow

1. Validate the environment first: run the targeted tests that cover models, reset/step/state, rewards, anti-cheat, seed reproducibility, invalid actions, rollouts, config, and scenario cache.
2. Prepare the scenario cache once per generator/verifier version: `scripts/modal_train_grpo.py --mode prepare-cache` or `scripts/modal_ephemeral_train.py --mode prepare-cache`.
3. Run the CPU-only Modal scenario-cache preflight before any GPU training. If cache hit rate or coverage is below config, stop and refill the cache instead of allocating a GPU.
4. Run a frozen-model or dummy-policy rollout on Modal and inspect the action trace, observations, terminal reason, cache metadata, and reward breakdown.
5. Confirm Trackio receives component metrics and the run name follows `CyberSecurity_OWASP-<model>-<algo>-level<difficulty>-<YYYYMMDD-HHMM>-<git_sha>`.
6. Start a very small GRPO run only after the above passes. Start via `scripts/modal_train_grpo.py --mode train`.
7. Evaluate baseline, trained, and held-out splits with `training/eval_before_after.py` and save summaries under `outputs/evals/`.
8. Save sampled rollouts under `outputs/rollouts/` for baseline, mid-training, trained, and held-out evidence.

## Reward And Monitoring

Track at least these behavior columns:

- Reward components: total, discovery, security, regression, public routes, patch quality, visible tests, safety, anti-cheat.
- Rates: success, exploit-block, regression preservation, public-route preservation, anti-cheat pass, invalid action, timeout, safety violation, reward-hacking suspected.
- Efficiency: episode length mean/p95, rollouts per second, tokens per second, loss, learning rate, KL, grad norm.
- Environment timing: reset, step, verifier, reward, scenario cache hit/miss, scenario bundle load, scenario compile fallback, error rate, difficulty, seed.

Stop or roll back if reward rises while sampled traces show deny-all patches, hardcoded users/resources/tenants, fixture/test tampering, repeated invalid actions, public routes being locked, or visible-test-only optimization.

Stop or downgrade to local-dev only if Modal training/eval shows runtime scenario compilation, cache misses in required mode, or cache hit rate below the configured target.

## Parallel Modal Runs

Parallel GRPO runs are allowed, but they must not share mutable experiment
identity or mutate shared caches while another job is training.

Before launching another run:

1. Check active Modal apps:

```bash
uv run --extra modal modal app list
```

2. Inspect any active `CyberSecurity_OWASP` app before starting another job:

```bash
uv run --extra modal modal app logs <app-id>
```

3. Use both detach layers for long jobs:

```bash
uv run --extra modal modal run --detach scripts/modal_train_grpo.py \
  --max-steps 300 \
  --dataset-size 64 \
  --num-generations 8 \
  --max-completion-length 768 \
  --difficulty 0 \
  --trace-log-every 10 \
  --seed-start 10000 \
  --detach
```

The Modal CLI `--detach` keeps the remote function alive after the local
entrypoint disconnects. The launcher `--detach` prevents the parent Modal
function from waiting on the spawned GPU call. Use both; using only the script
flag can let Modal stop the run when the local client exits.

For concurrent experiments:

- Assign every run a distinct `--seed-start` range, normally at least 10,000
  seeds apart.
- Keep `CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require`.
- Do not run `prepare-cache --cache-force` while any training job is active.
- Leave `--push-to-hub` off unless every job has a unique `--output-repo-id`.
- Keep Trackio run names unique. The launcher timestamp normally handles this;
  set `RUN_NAME` only when it is globally unique.
- Use the same Trackio Space/project for comparison, but never reuse a run
  name.
- Treat `CyberSecurity_OWASP-model-cache` and
  `CyberSecurity_OWASP-scenario-cache` as shared read-mostly volumes during
  training. Run checkpoints and artifacts must live under the run-specific
  output directory.
- For clean comparisons, keep model, difficulty, dataset size, generation
  length, reward config, and cache version fixed; vary only `seed-start` or the
  one hyperparameter being tested.

On Windows, if Modal startup fails with a Unicode `charmap` encoding error,
rerun the command with UTF-8 enabled:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; uv run --extra modal modal run --detach scripts/modal_train_grpo.py --max-steps 300 --dataset-size 64 --num-generations 4 --max-completion-length 768 --difficulty 0 --trace-log-every 10 --seed-start 60000 --detach
```

## TRL, OpenEnv, And Unsloth Guidance

- Use TRL GRPO for verifier-driven rewards. Keep multiple independent reward functions for logging and diagnosis.
- Keep the existing custom rollout path unless deliberately migrating to TRL's `environment_factory`. If migrating, preserve typed actions, observations, reward component logging, anti-cheat flags, and rollout artifacts.
- Use Modal as the default training path; local-only vLLM/GRPO execution is intentionally avoided in this repository.
- For OpenEnv server training concurrency, ensure the server supports enough concurrent sessions for the generation batch.
- Keep scenario generation out of the rollout hot path. `reset()` should clone cached bundles; any LLM scenario authoring belongs to offline cache prep.
- GPU training launchers must call the CPU-only scenario-cache preflight before spawning the L4 function, so missing cache coverage fails before GPU allocation.
- Use Unsloth with LoRA or QLoRA for memory efficiency when the training machine supports it. Start from an instruct-capable checkpoint and verify the model has non-zero success probability before RL.
- Do not swap the RL model away from `unsloth/gemma-4-E2B-it` for smoke runs. Cost-control should use `--max-steps`, `--dataset-size`, `--max-completion-length`, and cache preflight, not a different model.
- Pin and smoke-test TRL, Unsloth, vLLM, CUDA, and torch versions before longer runs.
- Save LoRA adapters or use Unsloth-supported merged save paths. Do not naively upcast a 4-bit model and merge adapters manually.

## Demo Evidence

- Report baseline vs trained success and reward improvements.
- Include held-out split results, exploit-block rate, regression-preservation rate, public-route preservation rate, and anti-cheat pass rate.
- Show representative rollout traces before and after training.
- Explain how hidden verifier checks, anti-cheat checks, randomized scenarios, and held-out combinations reduce reward hacking and overfitting.
