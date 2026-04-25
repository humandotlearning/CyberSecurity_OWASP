#!/usr/bin/env bash
set -euo pipefail
openenv push --repo-id "${HF_REPO_ID:?set HF_REPO_ID, e.g. username/CyberSecurity_OWASP}"
