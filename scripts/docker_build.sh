#!/usr/bin/env bash
set -euo pipefail
docker build -t CyberSecurity_OWASP:latest -f server/Dockerfile .
