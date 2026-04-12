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

Cyber Analyst is an OpenEnv implementation of the "SecOps Evidence Gym" design from `docs/deep-research-report.md`. It benchmarks a bounded, safe security triage workflow: investigate synthetic artifacts, cite evidence IDs, validate candidate findings with deterministic verifiers, and submit a remediation report.

The environment contains no live targets, no real secrets, no exploit workflow, no shell, and no outbound investigation tools. All evidence is static synthetic lab data.

## Tasks

The manifest ships three graded tasks:

| Task id | Difficulty | Goal |
| --- | --- | --- |
| `secret_exposure_easy` | easy | Find a synthetic API-key-like secret in a repo snapshot and propose removal plus rotation. |
| `missing_security_headers_medium` | medium | Detect missing HSTS/CSP headers in a synthetic gateway header snapshot. |
| `authz_boundary_hard` | hard | Detect an admin route role-policy mismatch without exploitation. |

## Action Contract

Use one bounded tool call per `step`:

```python
CyberAnalystAction(
    tool_name="search_repo",
    args={"query": "api key"},
)
```

Approved tools:

- `list_assets()`
- `get_log_events(service_id, query)`
- `check_security_headers(service_id)`
- `search_repo(query)`
- `scan_dependencies()`
- `create_finding(finding_type, evidence_ids, severity_guess, remediation)`
- `validate_finding(finding_id)`
- `submit_report(report_json)`

## Observation Contract

Each observation includes:

- `alert`: task prompt
- `tool_catalog`: approved tool list
- `tool_result`: latest tool result
- `evidence_ids`: discovered evidence IDs
- `candidate_findings`: created findings
- `verified_findings`: verifier-confirmed findings
- `score_breakdown`: deterministic scoring explanation
- `step_budget_remaining`, `error`, `done`, and `reward`

Rewards and final scores are clamped to `0.01..0.99` for validator compatibility.

`submit_report` also returns `trajectory_jsonl`, a JSONL export of the episode
events up to report submission. This is intended for offline inspection and
future training data extraction.

## Local Run

From this directory:

```bash
uv run server
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

`inference.py` runs a deterministic scripted baseline and prints strict parser-friendly logs:

```text
[START] task=<task_id> env=Cyber_analyst model=<model_name>
[STEP] step=<n> action=<tool_name> reward=<0.00> done=<true|false> error=<msg|null>
[END] task=<task_id> success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

LLM calls are not enabled by default. The script already includes OpenAI SDK configuration compatible with Hugging Face Inference Providers so model-backed report drafting can be added later:

```bash
set ENV_URL=http://localhost:8000
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=openai/gpt-oss-120b:novita
set HF_TOKEN=<your-hugging-face-token>
python inference.py
```

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
