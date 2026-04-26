---
title: CyberSecurity_OWASP Environment Server
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - cybersecurity
  - owasp
---

# CyberSecurity_OWASP

[Hugging Face Space](https://huggingface.co/spaces/Humanlearning/CyberSecurity_OWASP) | [Mini-blog](blog/blog.md)

`CyberSecurity_OWASP` is an OpenEnv-compliant reinforcement-learning environment for a single LLM agent that performs a defensive authorization-repair workflow:

```text
inspect generated app + policy -> discover authorization bug -> submit diagnosis -> patch code -> preserve intended behavior
```

The current implementation includes a functional closed-loop MVP scenario: an invoices FastAPI-style app with one injected OWASP A01 BOLA/IDOR defect, config-driven curriculum settings, cache-backed scenario reset, an ephemeral app sandbox, multi-layer deterministic verifier checks, anti-cheat safeguards, JSONL episode artifacts, and decomposed reward.

## Diagrams

![CyberSecurity_OWASP architecture](assets/architecture_diagram.svg)

![CyberSecurity_OWASP RL training flow](assets/env_rl_training_flow_diagram.svg)

Editable Mermaid sources are available in `assets/architecture_diagram.mmd` and `assets/env_rl_training_flow_diagram.mmd`.

## Quick Start

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run python scripts/generate_scenario_cache.py --train-per-bucket 3 --validation-per-bucket 3 --heldout-per-bucket 3
uv run server --port 8000
```

Then connect with the OpenEnv client:

```python
from CyberSecurity_OWASP import CyberSecurityOWASPAction, CyberSecurityOWASPEnv

with CyberSecurityOWASPEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=7)
    print(result.observation.task_brief)
    result = env.step(CyberSecurityOWASPAction(tool_name="list_routes"))
    print(result.observation.last_tool_result)
```

## Action Space

The agent emits one JSON action at a time:

```json
{"tool_name":"read_file","arguments":{"path":"app/routes/invoices.py"}}
```

Supported tools:

- `inspect_policy_graph`
- `list_routes`
- `read_openapi`
- `read_file`
- `search_code`
- `send_local_request`
- `compare_identities`
- `submit_diagnosis`
- `patch_file`
- `run_visible_tests`
- `submit_fix`
- `noop`

Tools are phase-gated:

- `discover`: inspect policy/routes/files, run safe local requests, compare identities, submit diagnosis.
- `patch`: read/search, patch editable app files, run visible tests, submit final fix.
- `done`: stable terminal observation only.

## Reward

Terminal reward uses stable components:

```python
{
    "discovery": 0.0,
    "security": 0.0,
    "regression": 0.0,
    "public_routes": 0.0,
    "patch_quality": 0.0,
    "visible_tests": 0.0,
    "safety": 0.0,
    "anti_cheat": 0.0,
    "terminal_total": 0.0,
    "progressive": 0.0,
    "step_penalty": 0.0,
    "speed_bonus": 0.0,
    "token_penalty": 0.0,
    "behavior_penalty": 0.0,
    "train_total": 0.0,
    "total": 0.0,
}
```

The verifier rewards blocking the hidden exploit while preserving legitimate owner/admin behavior and intentionally public routes. Terminal scoring requires visible checks, hidden authorization checks, a policy-oracle matrix, regression checks, public-route preservation, and patch-quality checks. It penalizes deny-all fixes, hardcoded IDs, repeated/invalid action patterns, hidden file probes, external URL attempts, and test/fixture tampering.

Training can enable dense rewards with `CYBERSECURITY_OWASP_REWARD_MODE=dense_train`.
Dense mode adds configurable progressive rewards, small efficiency penalties, and capped behavior penalties from `training/configs/grpo_small.yaml`; evaluation defaults to sparse terminal scoring.

## Scenario Cache And Generation

Scenario generation is an offline/cache-prep concern. `reset(seed)` asks the `CurriculumController` for a difficulty tier and target weakness, then loads a validated executable bundle from the scenario cache when `CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require`. Local development defaults to `fallback`, which compiles deterministically on a cache miss.

The scenario/curriculum author is config-driven through `configs/scenario_authoring.small.json`. The default offline author model is `deepseek-ai/DeepSeek-V4-Pro` with Hugging Face provider settings, thinking mode enabled, `temperature=1.0`, and `top_p=1.0`. This model config is for scenario authoring, not the RL policy model.

The cache bundle contract is:

- `scenario.json`
- `app_source/`
- `policy_graph.json`
- `visible_tests.py`
- `hidden_tests.py`
- `oracle_tests.py`
- `expected_exploit_trace.json`
- `reward_config.json`
- `metadata.json`

Cache keys include difficulty, authorization bug type, app family, framework, policy shape, tenant model, exploit depth, patch scope, regression risk, generator version, verifier version, and scenario hash.

The MVP compiler currently generates:

- invoices domain policy graph;
- bounded adversarial target metadata such as same-role cross-object access, cross-tenant access, public-route overlocking traps, alternate route/service reachability, or visible-test-only edge cases;
- randomized users, tenants, invoices, and IDs;
- generated app files under `app/`;
- visible tests under `tests/test_visible.py`;
- hidden facts, oracle tuples, scenario family metadata, and verifier targets kept out of observations.

Additional domains and bug families are scaffolded for extension.

## Runtime Components

The OpenEnv runtime is split into small server modules:

- `server/curriculum.py` tracks mastery, weak spots, reward trend, and difficulty tier.
- `server/scenario_cache.py` writes and loads validated executable scenario bundles.
- `server/adversarial_designer.py` chooses safe synthetic scenario targets from tracked weaknesses.
- `server/scenario_factory.py` compiles the generated app during cache prep or local fallback.
- `server/app_sandbox.py` handles editable workspace reads, patches, local requests, and OpenAPI summaries.
- `server/action_tools.py` dispatches typed tools through the sandbox.
- `server/authz_oracle.py` builds the hidden allowed/denied user-resource-action matrix.
- `server/verifier.py` aggregates visible tests, hidden tests, oracle matrix, regression/public-route checks, and patch quality.
- `server/episode_logger.py` appends JSONL rollouts under `outputs/rollouts/`.

The agent sees partial observations only: product rules, fixture aliases, route summaries, visible test results, and action errors. Hidden tests, oracle tuples, injected bug labels, and held-out scenario-family labels stay internal.

## Testing

```bash
uv run --extra dev pytest
```

The suite covers model serialization, reset/step/state behavior, seed reproducibility, invalid actions, reward outcomes, anti-cheat checks, scripted rollout policies, curriculum selection, adversarial targeting, held-out scenario families, oracle checks, verifier aggregation, and episode artifact logging.

## Training Scaffold

Training files are under `training/`:

- `rollout.py`
- `reward_funcs.py`
- `train_grpo.py`
- `eval_before_after.py`
- `trackio_utils.py`
- `configs/grpo_small.yaml`

The training scaffold is intentionally minimal until the environment/verifier behavior is stable. Trackio metric names and GRPO defaults follow the project brief.

`training/train_grpo.py` in this repo is a config helper only; it does not execute training locally.
Use the Modal launchers in `scripts/modal_train_grpo.py` (persistent) and
`scripts/modal_ephemeral_train.py` (smoke) for real GRPO runs.

Modal smoke and GRPO runs use `CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require` and mount the persistent `CyberSecurity_OWASP-scenario-cache` volume. Prepare that cache before smoke/training:

```bash
uv run --extra modal modal run scripts/modal_train_grpo.py --mode prepare-cache
uv run --extra modal modal run scripts/modal_ephemeral_train.py --mode prepare-cache
```

If the cache slice is missing or below the configured per-bucket minimum, Modal training fails before rollouts rather than compiling scenarios during the run.
The persistent GRPO launcher runs a CPU-only scenario-cache preflight before it starts the L4 GPU function, so missing cache coverage fails before GPU allocation.

## Trackio Run Tracking

Trackio is the default tracker for official runs. Set `TRACKIO_SPACE_ID` to log to a hosted Hugging Face Trackio Space; otherwise Trackio records locally.

```bash
export TRACKIO_SPACE_ID=<hf-user>/CyberSecurity_OWASP-trackio
export TRACKIO_PROJECT=CyberSecurity_OWASP-grpo
```

Use the tracked smoke wrapper instead of invoking pytest directly when producing run artifacts:

```bash
bash scripts/smoke_test.sh
uv run python scripts/track_pytest.py tests
```

Evaluation summaries saved through `training.eval_before_after.save_eval_summary(...)`, Modal smoke runs, and GRPO training configs all initialize Trackio runs with CyberSecurity_OWASP run names.

Training, baseline, and smoke runs also log the effective reward config at step
0. In Trackio, open **Media & Tables** and select the `reward_config` table to
see the actual values for each reward key, including stage-specific values,
caps, thresholds, terminate flags, and descriptions. Scalar metrics under
`reward_config/<key>/<field>` expose the same numeric values for plotting and
filtering, for example `reward_config/policy_inspected/value` and
`reward_config/shaping_weight/resolved`.

Each run config includes `reward_config_id`, `reward_config_hash`,
`reward_config_source`, `reward_mode`, and `reward_stage`. For manual ablations,
compare runs with the same scenario/model settings and different
`reward_config_hash` values to see which reward weights produced each training
curve.

## Modal Ephemeral Runs

Modal Labs support is kept in a separate launcher script so the local OpenEnv server and core training scaffold stay unchanged.

Install the optional local Modal client:

```bash
uv sync --extra modal
```

Run a temporary Modal app for a cheap environment/training smoke check:

```bash
uv run --extra modal modal run scripts/modal_ephemeral_train.py --mode prepare-cache
uv run --extra modal modal run scripts/modal_ephemeral_train.py --mode smoke --episodes 4
```

The app is ephemeral: Modal starts it for the command and stops it when the command exits. The remote result is written locally under `outputs/rollouts/` and the summary metrics are logged to Trackio.

You can also validate the GRPO config construction remotely:

```bash
uv run --extra modal modal run scripts/modal_ephemeral_train.py --mode grpo-config
```

The shell wrapper is equivalent:

```bash
MODE=smoke EPISODES=4 uv run --extra modal bash scripts/modal_run_ephemeral.sh
```

## Synthetic SFT Before GRPO

Use supervised fine-tuning to warm-start `unsloth/gemma-4-E2B-it` before GRPO.
The SFT generator executes every teacher action in the real environment and
keeps only trajectories that pass the deterministic reward verifier.

Generate a 300-train-episode curriculum SFT dataset across levels `0,1,2,3`:

```bash
uv run python scripts/generate_sft_dataset.py \
  --teacher-model deepseek-ai/DeepSeek-V4-Pro \
  --target-model unsloth/gemma-4-E2B-it \
  --difficulty-levels 0,1,2,3 \
  --difficulty-buckets 4 \
  --episodes 75 \
  --validation-episodes 20 \
  --workers 8 \
  --out-dir outputs/sft
```

`--episodes` is per difficulty level when `--difficulty-levels` is set, so
`--episodes 75` across four levels gives 300 total train episodes. Expect
roughly 2,400-4,500 chat-format JSONL rows because each successful trajectory
contributes one row per action step. The script writes JSONL rows under
`outputs/sft/`, trajectory artifacts under `outputs/sft/trajectories/`, a
dataset card at `outputs/sft/README.md`, and `outputs/sft/manifest.json` with
reward summaries and curriculum coverage.

Verify reward metadata before any training run:

```bash
uv run python scripts/generate_sft_dataset.py \
  --verify-only \
  --difficulty-levels 0,1,2,3 \
  --out-dir outputs/sft
```

Push the verified dataset to Hugging Face Hub:

```bash
uv run python scripts/generate_sft_dataset.py \
  --push-only \
  --difficulty-levels 0,1,2,3 \
  --out-dir outputs/sft \
  --dataset-repo-id Humanlearning/CyberSecurity_OWASP-sft-dataset
```

The canonical dataset repo name is
`Humanlearning/CyberSecurity_OWASP-sft-dataset`. The upload is refused if
reward verification fails or `HF_TOKEN` is missing.

You can also generate and push in one command by adding `--push-to-hub` to the
generation command.

For local CI or smoke checks, add `--dry-run-oracle`; official SFT data should
use the teacher path and still pass the verifier gate above.

Launch SFT on Modal after reward verification passes:

```bash
uv run --extra modal modal run --detach scripts/modal_train_sft.py \
  --local-train-path outputs/sft/train.jsonl \
  --local-validation-path outputs/sft/validation.jsonl \
  --local-manifest-path outputs/sft/manifest.json \
  --required-difficulties 0,1,2,3 \
  --trackio-space-id Humanlearning/CyberSecurity_OWASP-trackio \
  --trackio-project CyberSecurity_OWASP-sft \
  --output-repo-id Humanlearning/CyberSecurity_OWASP-unsloth-gemma-4-e2b-it-sft-lora \
  --push-to-hub \
  --detach
```

`scripts/modal_train_sft.py` re-checks the JSONL reward metadata locally before
upload and again inside Modal before loading the model. It refuses to start SFT
unless all required curriculum difficulties are represented and the verifier
reward metadata passes. The default SFT config trains the full dataset
(`--max-steps -1`) with bf16/tf32, LoRA rank 32, and Modal GPU fallback
`H200 -> H100 -> A100-80GB -> L40S`. TRL does not support packing or
assistant-only loss for the Gemma 4 vision-language loader, so both remain
disabled for this model. The script pre-tokenizes the small JSONL dataset
serially before constructing `SFTTrainer`, which avoids TRL multiprocessing
around the Gemma/Unsloth config object. It also uses the base Transformers loss
path to avoid a TRL entropy-metric incompatibility with Gemma 4 lazy logits. A
warm run for the 300-400 episode dataset should usually finish in about 20-60
minutes; first image or model-cache builds can push that closer to 45-90
minutes.

Continue GRPO from the SFT LoRA:

The GRPO launcher downloads the Hub adapter, attaches a matching trainable
Unsloth LoRA to Gemma 4, and then loads the adapter safetensors. This keeps the
SFT handoff compatible with Gemma 4's Unsloth linear wrappers.

```bash
uv run --extra modal modal run --detach scripts/modal_train_grpo.py \
  --initial-adapter-repo-id Humanlearning/CyberSecurity_OWASP-unsloth-gemma-4-e2b-it-sft-lora \
  --max-steps 300 \
  --dataset-size 64 \
  --num-generations 8 \
  --difficulty 0 \
  --trace-log-every 10 \
  --detach
```

## Modal GRPO Training

The persistent GPU training launcher packages this local repo into Modal, trains
a small LoRA GRPO run, logs metrics and traces to Trackio, stores checkpoints in
the `CyberSecurity_OWASP-grpo-runs` Modal volume, and pushes the output adapter
to Hugging Face Hub.

Create a Modal secret named `CyberSecurity_OWASP-secrets` with `HF_TOKEN`, then
run the import/config check:

```bash
uv run --extra modal modal run scripts/modal_train_grpo.py --mode config
```

Run the default smoke GRPO job:

```bash
uv run --extra modal modal run scripts/modal_train_grpo.py --mode prepare-cache
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --max-steps 10 \
  --dataset-size 16 \
  --num-generations 6 \
  --difficulty 0
```

For GPU-utilization tuning on the same single L4, start with a larger but still
bounded no-code trial:

```bash
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --max-steps 30 \
  --dataset-size 64 \
  --num-generations 8 \
  --max-completion-length 256 \
  --difficulty 0
```

The launcher exposes GRPO throughput knobs for follow-up trials:

```bash
# larger generation group, no vLLM
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --max-steps 30 --dataset-size 64 --num-generations 8 \
  --max-completion-length 256 --trace-log-every 5

# vLLM colocate on the same L4
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --max-steps 30 --dataset-size 64 --num-generations 8 \
  --max-completion-length 256 --use-vllm \
  --vllm-gpu-memory-utilization 0.35 --trace-log-every 5

# larger microbatch if the vLLM trial does not OOM
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --max-steps 30 --dataset-size 64 --num-generations 8 \
  --per-device-train-batch-size 2 --gradient-accumulation-steps 4 \
  --max-completion-length 256 --use-vllm \
  --vllm-gpu-memory-utilization 0.45 --trace-log-every 5
```

`per_device_train_batch_size * gradient_accumulation_steps * world_size` must
be divisible by `num_generations`; the launcher validates this before the GPU
container starts. Scalar Trackio metrics still log every reward callback, while
sample trace tables and Trace objects are throttled by `--trace-log-every`
(`1` restores every-callback logging, `0` disables trace artifacts).

### Parallel Modal GRPO Runs

Parallel Modal GRPO runs are safe when each run has its own seed range, run
name, and output target, while the shared cache volumes remain read-only.
Before launching another job, check what is already active:

```bash
uv run --extra modal modal app list
uv run --extra modal modal app logs <app-id>
```

Launch long-running parallel jobs with both Modal CLI detach and the launcher
detach flag. The CLI-level `--detach` keeps the remote function alive after the
local entrypoint exits; the launcher `--detach` prevents the parent Modal
function from waiting on the GPU call.

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

For multiple concurrent experiments:

- Use a unique `--seed-start` range for every run, normally spaced by at least
  10,000 seeds.
- Keep `CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require`; do not compile
  scenarios during training.
- Do not run `prepare-cache --cache-force` while training jobs are active.
- Keep `--push-to-hub` disabled unless each run has a unique
  `--output-repo-id`.
- Let the launcher generate unique timestamped Trackio run names, or set an
  explicit `RUN_NAME` only when it is globally unique.
- Use the same Trackio Space/project for comparable metrics, but never reuse a
  run name.
- Treat `CyberSecurity_OWASP-model-cache` and
  `CyberSecurity_OWASP-scenario-cache` as shared read-mostly infrastructure
  during training. Run outputs and checkpoints should stay under each run's
  unique output directory.

If a Windows shell fails with a Unicode `charmap` encoding error during Modal
startup, rerun with UTF-8 enabled for that command:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; uv run --extra modal modal run --detach scripts/modal_train_grpo.py --max-steps 300 --dataset-size 64 --num-generations 4 --max-completion-length 768 --difficulty 0 --trace-log-every 10 --seed-start 60000 --detach
```

If running from a public repository and you do not want Modal to package the
local workspace, use public source mode:

```bash
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --source-mode public \
  --repo-url https://github.com/humandotlearning/CyberSecurity_OWASP.git \
  --repo-branch master \
  --max-steps 10 \
  --dataset-size 16 \
  --num-generations 6 \
  --difficulty 0
```

Defaults are derived from `HF_TOKEN`:

- Trackio Space: `<hf-user>/CyberSecurity_OWASP-trackio`
- Trackio project: `CyberSecurity_OWASP-grpo`
- Training model: `unsloth/gemma-4-E2B-it`
- Output repo: `<hf-user>/CyberSecurity_OWASP-unsloth-gemma-4-e2b-it-grpo-lora`

Override these with `--trackio-space-id`, `--trackio-project`, and
`--output-repo-id` when needed. The persistent GRPO launcher intentionally rejects non-Gemma model overrides so smoke runs match the Unsloth Gemma 4 E2B RL notebook.

## Docker / Spaces

```bash
docker build -t CyberSecurity_OWASP:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 CyberSecurity_OWASP:latest
openenv push --repo-id <username>/CyberSecurity_OWASP
```
