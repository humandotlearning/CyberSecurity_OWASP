---
name: cybersecurity-owasp-trainer
description: Train, debug, evaluate, and document CyberSecurity_OWASP model runs with OpenEnv, TRL/GRPO, optional Unsloth/QLoRA, Trackio, rollout artifacts, baseline-vs-trained evaluation, and reward-hacking safeguards. Use when working on training scripts, launch commands, rollouts, Trackio metrics, model saving, or hackathon demo evidence for this repo.
---

# CyberSecurity_OWASP Trainer

## Overview

Use this skill to run or modify the CyberSecurity_OWASP training and evaluation loop without weakening the verifier, reward integrity, or hackathon evidence trail. Treat the environment and reward engine as the product; training only starts after those are stable.

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
- Trackio run config is set and can log a smoke metric locally or to the canonical Space.

If any gate fails, fix the environment, verifier, reward engine, or rollout parser before touching trainer scale.

## Repo Training Path

Prefer the existing repo modules:

- `training/rollout.py`: full OpenEnv episode loop, action JSON parsing, reward trace, rollout artifact fields.
- `training/reward_funcs.py`: component reward functions exposed to TRL/GRPO.
- `training/train_grpo.py`: `GRPOConfig`, model defaults, Trackio reporting, vLLM settings.
- `training/eval_before_after.py`: baseline-vs-trained and held-out summary metrics.
- `training/trackio_utils.py`: run naming, canonical metric names, Trackio init/log/finalize helpers.

Default environment values:

```powershell
$env:MODEL_NAME = "Qwen/Qwen3-1.7B"
$env:TRACKIO_SPACE_ID = "Humanlearning/CyberSecurity_OWASP-trackio"
$env:TRACKIO_PROJECT = "CyberSecurity_OWASP"
$env:DIFFICULTY = "0"
```

Use level-0 debug runs before scaling. Do not increase batch size, prompt count, scenario diversity, or difficulty until sampled artifacts show real discover-then-patch behavior rather than formatting compliance only.

## Training Workflow

1. Validate the environment first: run the targeted tests that cover models, reset/step/state, rewards, anti-cheat, seed reproducibility, invalid actions, and rollouts.
2. Run a tiny smoke path that constructs `GRPOConfig` without starting expensive training.
3. Run a frozen-model or dummy-policy rollout and inspect the action trace, observations, terminal reason, and reward breakdown.
4. Confirm Trackio receives component metrics and the run name follows `CyberSecurity_OWASP-<model>-<algo>-level<difficulty>-<YYYYMMDD-HHMM>-<git_sha>`.
5. Start a very small GRPO run only after the above passes. Watch completions and rollout artifacts during the run, not just aggregate reward.
6. Evaluate baseline, trained, and held-out splits with `training/eval_before_after.py` and save summaries under `outputs/evals/`.
7. Save sampled rollouts under `outputs/rollouts/` for baseline, mid-training, trained, and held-out evidence.

## Reward And Monitoring

Track at least these behavior columns:

- Reward components: total, discovery, security, regression, public routes, patch quality, visible tests, safety, anti-cheat.
- Rates: success, exploit-block, regression preservation, public-route preservation, anti-cheat pass, invalid action, timeout, safety violation, reward-hacking suspected.
- Efficiency: episode length mean/p95, rollouts per second, tokens per second, loss, learning rate, KL, grad norm.
- Environment timing: reset, step, verifier, reward, scenario compile, error rate, difficulty, seed.

Stop or roll back if reward rises while sampled traces show deny-all patches, hardcoded users/resources/tenants, fixture/test tampering, repeated invalid actions, public routes being locked, or visible-test-only optimization.

## TRL, OpenEnv, And Unsloth Guidance

- Use TRL GRPO for verifier-driven rewards. Keep multiple independent reward functions for logging and diagnosis.
- Keep the existing custom rollout path unless deliberately migrating to TRL's `environment_factory`. If migrating, preserve typed actions, observations, reward component logging, anti-cheat flags, and rollout artifacts.
- Use vLLM colocate for small local runs when memory allows; use server mode only when a separate inference GPU/server is available.
- For OpenEnv server training concurrency, ensure the server supports enough concurrent sessions for the generation batch.
- Use Unsloth with LoRA or QLoRA for memory efficiency when the training machine supports it. Start from an instruct-capable checkpoint and verify the model has non-zero success probability before RL.
- Pin and smoke-test TRL, Unsloth, vLLM, CUDA, and torch versions before longer runs.
- Save LoRA adapters or use Unsloth-supported merged save paths. Do not naively upcast a 4-bit model and merge adapters manually.

## Demo Evidence

- Report baseline vs trained success and reward improvements.
- Include held-out split results, exploit-block rate, regression-preservation rate, public-route preservation rate, and anti-cheat pass rate.
- Show representative rollout traces before and after training.
- Explain how hidden verifier checks, anti-cheat checks, randomized scenarios, and held-out combinations reduce reward hacking and overfitting.
