ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env

COPY pyproject.toml uv.lock ./
COPY README.md openenv.yaml ./
COPY __init__.py client.py models.py ./
COPY bug_mutator.py evals.py fixture_generator.py policy_graph.py rewards.py safety.py scenario_compiler.py template_renderer.py validators.py ./
COPY server ./server
COPY training ./training
COPY scripts ./scripts
COPY tests ./tests

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM ${BASE_IMAGE}

WORKDIR /app/env
COPY --from=builder /app/env /app/env
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "CyberSecurity_OWASP.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
