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

`CyberSecurity_OWASP` is an OpenEnv-compliant reinforcement-learning environment for a single LLM agent that performs a defensive authorization-repair workflow:

```text
inspect generated app + policy -> discover authorization bug -> submit finding -> patch code -> preserve intended behavior
```

The current implementation includes a functional MVP scenario: an invoices FastAPI-style app with one injected OWASP A01 BOLA/IDOR defect, visible tests, hidden deterministic verifier checks, anti-cheat safeguards, and decomposed reward.

## Diagrams

![CyberSecurity_OWASP architecture](assets/architecture_diagram.svg)

![CyberSecurity_OWASP RL training flow](assets/env_rl_training_flow_diagram.svg)

Editable Mermaid sources are available in `assets/architecture_diagram.mmd` and `assets/env_rl_training_flow_diagram.mmd`.

## Quick Start

```bash
uv sync --extra dev
uv run --extra dev pytest
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
- `submit_finding`
- `patch_file`
- `run_visible_tests`
- `submit_fix`
- `noop`

Tools are phase-gated:

- `discover`: inspect policy/routes/files, run safe local requests, compare identities, submit finding.
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
    "total": 0.0,
}
```

The verifier rewards blocking the hidden exploit while preserving legitimate owner/admin behavior and intentionally public routes. It penalizes deny-all fixes, hardcoded IDs, hidden file probes, external URL attempts, and test/fixture tampering.

## Scenario Generation

`reset(seed)` compiles a fresh isolated workspace under a temp directory. The MVP compiler generates:

- invoices domain policy graph;
- randomized users, tenants, invoices, and IDs;
- generated app files under `app/`;
- visible tests under `tests/test_visible.py`;
- hidden facts kept only in state for deterministic verification.

Additional domains and bug families are scaffolded for extension.

## Testing

```bash
uv run --extra dev pytest
```

The suite covers model serialization, reset/step/state behavior, seed reproducibility, invalid actions, reward outcomes, anti-cheat checks, and scripted rollout policies.

## Training Scaffold

Training files are under `training/`:

- `rollout.py`
- `reward_funcs.py`
- `train_grpo.py`
- `eval_before_after.py`
- `trackio_utils.py`
- `configs/grpo_small.yaml`

The training scaffold is intentionally minimal until the environment/verifier behavior is stable. Trackio metric names and GRPO defaults follow the project brief.

## Modal Ephemeral Runs

Modal Labs support is kept in a separate launcher script so the local OpenEnv server and core training scaffold stay unchanged.

Install the optional local Modal client:

```bash
uv sync --extra modal
```

Run a temporary Modal app for a cheap environment/training smoke check:

```bash
uv run --extra modal modal run scripts/modal_ephemeral_train.py --mode smoke --episodes 4
```

The app is ephemeral: Modal starts it for the command and stops it when the command exits. The remote result is written locally under `outputs/rollouts/`.

You can also validate the GRPO config construction remotely:

```bash
uv run --extra modal modal run scripts/modal_ephemeral_train.py --mode grpo-config
```

The shell wrapper is equivalent:

```bash
MODE=smoke EPISODES=4 uv run --extra modal bash scripts/modal_run_ephemeral.sh
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
uv run --extra modal modal run scripts/modal_train_grpo.py \
  --max-steps 10 \
  --dataset-size 16 \
  --num-generations 2 \
  --difficulty 0
```

Defaults are derived from `HF_TOKEN`:

- Trackio Space: `<hf-user>/CyberSecurity_OWASP-trackio`
- Trackio project: `CyberSecurity_OWASP-grpo`
- Output repo: `<hf-user>/CyberSecurity_OWASP-qwen3-1.7b-grpo-lora`

Override these with `--trackio-space-id`, `--trackio-project`, and
`--output-repo-id` when needed.

## Docker / Spaces

```bash
docker build -t CyberSecurity_OWASP:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 CyberSecurity_OWASP:latest
openenv push --repo-id <username>/CyberSecurity_OWASP
```
