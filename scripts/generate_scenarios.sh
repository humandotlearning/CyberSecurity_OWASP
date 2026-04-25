#!/usr/bin/env bash
set -euo pipefail
uv run python -c "from CyberSecurity_OWASP.scenario_compiler import compile_scenario; [compile_scenario(i) for i in range(3)]; print('generated 3 smoke scenarios')"
