#!/usr/bin/env bash
set -euo pipefail
docker run --rm -p "${PORT:-8000}:8000" CyberSecurity_OWASP:latest
