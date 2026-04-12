---
title: Cyber Analyst Environment Server
emoji: 🎯
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Cyber Analyst Environment

Cyber Analyst is an OpenEnv implementation of the "SecOps Evidence Gym". It benchmarks a bounded, safe security-triage workflow: investigate synthetic artifacts, cite evidence IDs, validate candidate findings with deterministic verifiers, and submit a remediation report.

The environment contains no live targets, no real secrets, no exploit workflow, no shell, and no outbound investigation tools. All evidence is static synthetic lab data.

## Motivation

Frontier models are becoming much stronger at security-relevant reasoning. Anthropic's April 7, 2026 report, [Assessing Claude Mythos Preview's cybersecurity capabilities](https://red.anthropic.com/2026/mythos-preview/), describes a model that can identify and exploit subtle vulnerabilities across real software targets, and argues that the same capability jump should be directed toward defense.

That creates a practical gap: many modern applications are built quickly, including "vibe coded" apps whose security review may not keep pace with generation speed. This environment is a small, safe training and evaluation surface for the defensive side of that gap. The goal is to help train and benchmark smaller, more accessible models to behave like careful application-security analysts: gather evidence, avoid unsupported claims, validate findings, and recommend concrete fixes.

## Environment Description

Each episode simulates a synthetic microservice organization with three services:

- `gateway`
- `profile-service`
- `admin-service`

The agent starts from an alert and can inspect only closed-world artifact collections:

- `repo_snapshot`: static code/config snippets
- `telemetry`: sanitized log events
- `headers`: static response-header snapshots
- `dependencies`: static dependency manifest excerpts

The episode budget is 12 steps. Seeds deterministically vary benign details such as service aliases and evidence ordering while keeping the same task ground truth reproducible.

## Tasks

The manifest ships three graded tasks:

| Task id | Difficulty | Task description | Expected difficulty |
| --- | --- | --- | --- |
| `secret_exposure_easy` | easy | Find a synthetic API-key-like secret in a repo snapshot and propose removal plus rotation. | Easiest path: one focused `search_repo` call can surface the relevant evidence, then the agent must create, validate, and report the finding. |
| `missing_security_headers_medium` | medium | Detect missing HSTS/CSP headers in a synthetic gateway header snapshot. | Requires choosing the purpose-built `check_security_headers` tool and mapping missing headers to remediation instead of over-searching unrelated artifacts. |
| `authz_boundary_hard` | hard | Detect an admin route role-policy mismatch without exploitation. | Requires correlating route/role policy evidence with a supporting log event and recommending least-privilege policy remediation plus regression testing. |

## Action Space

Each `step` accepts exactly one bounded simulator tool call:

```python
CyberAnalystAction(
    tool_name="search_repo",
    args={"query": "api key"},
)
```

Approved tools:

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_assets` | `{}` | List synthetic services, routes, and artifact collections. |
| `get_log_events` | `{"service_id": "str", "query": "str"}` | Return sanitized telemetry evidence IDs for a service/query. |
| `check_security_headers` | `{"service_id": "str"}` | Inspect a service header snapshot and return pass/fail evidence. |
| `search_repo` | `{"query": "str"}` | Search synthetic repo/config snippets for evidence IDs. |
| `scan_dependencies` | `{}` | Inspect a synthetic dependency manifest excerpt. |
| `create_finding` | `{"finding_type": "str", "evidence_ids": ["str"], "severity_guess": "str", "remediation": "str"}` | Store a candidate finding for verifier review. |
| `validate_finding` | `{"finding_id": "str"}` | Run the deterministic verifier for a candidate finding. |
| `submit_report` | `{"report_json": {"findings": [...]}}` | Submit the final structured report and end the episode. |

Unsupported tools return an observation error instead of running arbitrary commands. Repeating the exact same action is penalized, and six repeated identical actions hard-stop the episode.

## Observation Space

Each observation is a `CyberAnalystObservation` with:

| Field | Definition |
| --- | --- |
| `task_id` | Current benchmark task ID. |
| `alert` | Initial alert or task prompt. |
| `phase` | Current episode phase, usually `investigate` or `done`. |
| `tool_catalog` | Approved tool list and argument schemas. |
| `tool_result` | Result returned by the latest tool call. |
| `evidence_ids` | Evidence IDs discovered so far. |
| `candidate_findings` | Candidate findings created by the agent. |
| `verified_findings` | Verifier-confirmed findings. |
| `step_budget_remaining` | Steps remaining before timeout. |
| `score_breakdown` | Deterministic final scoring explanation after report submission. |
| `error` | Non-fatal environment error, if any. |
| `done` | Whether the episode has ended. |
| `reward` | Step reward clamped to the validator-compatible range. |

`submit_report` also returns `trajectory_jsonl`, a JSONL export of the episode events up to report submission. This is intended for offline inspection and future training data extraction.

## Scoring

Final reports are scored deterministically:

- base score: `0.05`
- verified correct finding with matching report impact: `+0.60`
- valid evidence ID in the report: `+0.15`
- actionable remediation keywords: `+0.15`
- hallucinated or unverified finding claims: `-0.40` each
- submitting without verifier validation: `-0.20`

Rewards and final scores are clamped to `0.01..0.99` for validator compatibility.

## Baseline Scores

The current deterministic oracle rollout follows the intended evidence -> finding -> validation -> report path for each task. These scores were measured locally against the environment with `seed=7`.

| Task id | Baseline type | Steps | Final score | Step rewards |
| --- | --- | ---: | ---: | --- |
| `secret_exposure_easy` | deterministic oracle | 4 | `0.95` | `0.05, 0.06, 0.11, 0.98` |
| `missing_security_headers_medium` | deterministic oracle | 4 | `0.95` | `0.05, 0.06, 0.11, 0.98` |
| `authz_boundary_hard` | deterministic oracle | 6 | `0.95` | `0.03, 0.05, 0.05, 0.06, 0.11, 0.98` |

A hallucinated one-step report scores `0.01`; repeated identical actions hard-stop at a low score.

## Setup

From this directory, install dependencies:

```bash
uv sync
```

Run the local server:

```bash
uv run server
```

Health check:

```bash
curl http://localhost:8000/health
```

Then connect with the client:

```python
from Cyber_analyst import CyberAnalystAction, CyberAnalystEnv

with CyberAnalystEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="secret_exposure_easy", seed=7)
    result = env.step(CyberAnalystAction(tool_name="search_repo", args={"query": "api key"}))
    print(result.observation.tool_result)
```

## Baseline Inference

`inference.py` runs a model-backed baseline over the configured task set and prints strict parser-friendly logs:

```text
[START] task=<task_id> env=Cyber_analyst model=<model_name>
[STEP] step=<n> action=<compact_json_action> reward=<0.00> done=<true|false> error=<msg|null>
[END] task=<task_id> success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

The script uses the OpenAI SDK with Hugging Face Inference Providers by default:

```powershell
$env:ENV_URL = "http://localhost:8000"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "google/gemma-4-31B-it:fastest"
$env:HF_TOKEN = "<your-hugging-face-token>"
python inference.py
```

Use `$env:TASK_NAME = "<task_id>"` to run one task instead of all three.

## Validation

Useful local checks:

```bash
python -m py_compile server/Cyber_analyst_environment.py inference.py
python -m pytest tests
.\.venv\Scripts\openenv.exe validate . --json
```

## Docker

Build the environment image from this directory:

```bash
docker build -t cyber-analyst-env:latest -f server/Dockerfile .
```

Run:

```bash
docker run -p 8000:8000 cyber-analyst-env:latest
```

Health check:

```bash
curl http://localhost:8000/health
```

## Deployment

Deploy to Hugging Face Spaces with OpenEnv:

```bash
openenv push --repo-id <your-hf-username>/Cyber_analyst
```

The deployed Space exposes `/health`, `/docs`, `/ws`, and the optional `/web` interface when web UI support is enabled by the OpenEnv runtime.

## Adding Scenarios

Add new safe scenarios in `server/tasks.py` by extending `SCENARIOS` with:

- a stable `task_id`
- synthetic `assets`, `repo`, `logs`, `headers`, and `dependencies` entries
- `ground_truth_id`, `finding_type`, `required_evidence`, `impact_keywords`, and `remediation_keywords`

Then add a grader adapter in `server/graders.py` and a matching `tasks` entry in `openenv.yaml`. Keep all artifacts synthetic, keep correctness deterministic, and avoid adding real targets or arbitrary execution tools.
