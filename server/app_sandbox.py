"""Ephemeral generated app sandbox operations."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

try:
    from ..models import CyberSecurityOWASPState
    from ..safety import is_local_route
    from ..validators import is_path_allowed, simulate_request
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPState
    from safety import is_local_route
    from validators import is_path_allowed, simulate_request


class AppSandbox:
    """Encapsulates all generated workspace reads, patches, and local requests."""

    def __init__(self, state: CyberSecurityOWASPState):
        self.state = state

    @property
    def workspace(self) -> Path:
        return Path(str(self.state.hidden_facts["workspace"]))

    def read_file(self, path: str) -> str:
        return self._resolve_path(path).read_text(encoding="utf-8")

    def search_code(self, query: str) -> str:
        if not query:
            raise ValueError("query is required")
        results: list[str] = []
        for rel in self.state.hidden_facts.get("editable_files", []):
            path = self.workspace / rel
            text = path.read_text(encoding="utf-8")
            for idx, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    results.append(f"{rel}:{idx}: {line}")
        return "\n".join(results) or "No matches."

    def patch_file(self, path: str, *, content: str | None = None, diff: str | None = None) -> dict[str, str]:
        target = self._resolve_path(path, write=True)
        before = target.read_text(encoding="utf-8")
        if content is not None:
            target.write_text(content, encoding="utf-8")
        else:
            self._apply_unified_diff(target, diff or "")
        after = target.read_text(encoding="utf-8")
        patch_diff = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=path,
                tofile=path,
            )
        )
        self.state.patch_diff = patch_diff
        self.state.patch_attempt_count += 1
        files_touched = self.state.metrics.setdefault("files_touched", [])
        if path not in files_touched:
            files_touched.append(path)
        return {"path": path, "diff": patch_diff}

    def read_openapi(self) -> str:
        routes = self.state.visible_facts.get("workspace_summary", {}).get("routes", [])
        paths: dict[str, Any] = {}
        for route in routes:
            paths.setdefault(route["path"], {})[route["method"].lower()] = {
                "x-public": bool(route.get("public", False))
            }
        return json.dumps(
            {
                "openapi": "3.1.0",
                "info": {"title": "Generated invoices app", "version": "0.1.0"},
                "paths": paths,
            },
            indent=2,
            sort_keys=True,
        )

    def send_local_request(self, method: str, path: str, user_id: str | None = None) -> dict[str, Any]:
        if not is_local_route(path):
            raise ValueError("send_local_request only accepts local route paths")
        response = simulate_request(self.state, method, path, user_id)
        trace_id = self._record_request_trace(
            method=method,
            path=path,
            user_id=user_id,
            status=int(response.get("status", 0) or 0),
        )
        return {"trace_id": trace_id, **response}

    def compare_identities(
        self,
        method: str,
        path: str,
        first_user_id: str,
        second_user_id: str,
    ) -> dict[str, Any]:
        if not is_local_route(path):
            raise ValueError("compare_identities only accepts local route paths")
        first = simulate_request(self.state, method, path, first_user_id)
        second = simulate_request(self.state, method, path, second_user_id)
        trace_id = self._record_request_trace(
            method=method,
            path=path,
            user_id=first_user_id,
            status=int(first.get("status", 0) or 0),
            comparison_user_id=second_user_id,
            comparison_status=int(second.get("status", 0) or 0),
        )
        return {
            "trace_id": trace_id,
            "first": first,
            "second": second,
        }

    def _record_request_trace(
        self,
        *,
        method: str,
        path: str,
        user_id: str | None,
        status: int,
        comparison_user_id: str | None = None,
        comparison_status: int | None = None,
    ) -> str:
        trace_id = f"req_{len(self.state.request_trace) + 1:03d}"
        hidden = self.state.hidden_facts
        unauthorized_success = (
            str(hidden.get("other_invoice_id", "")) in path
            and user_id == hidden.get("owner_user_id")
            and status == 200
        )
        if comparison_user_id is not None and comparison_status is not None:
            unauthorized_success = unauthorized_success or (
                str(hidden.get("other_invoice_id", "")) in path
                and comparison_user_id == hidden.get("owner_user_id")
                and comparison_status == 200
            )
        self.state.request_trace.append(
            {
                "trace_id": trace_id,
                "method": method.upper(),
                "path": path,
                "user_id": user_id,
                "status": status,
                "comparison_user_id": comparison_user_id,
                "comparison_status": comparison_status,
                "unauthorized_success": unauthorized_success,
            }
        )
        return trace_id

    def _resolve_path(self, path: str, *, write: bool = False) -> Path:
        allowed, normalized_or_error = is_path_allowed(self.state, path, write=write)
        if not allowed:
            raise ValueError(normalized_or_error)
        return self.workspace / normalized_or_error

    def _apply_unified_diff(self, path: Path, diff: str) -> None:
        if not diff.strip():
            raise ValueError("diff or content is required")
        original = path.read_text(encoding="utf-8").splitlines(True)
        output: list[str] = []
        old_index = 0
        lines = diff.splitlines(True)
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.startswith("@@"):
                i += 1
                continue
            old_start = int(line.split()[1].split(",")[0][1:])
            output.extend(original[old_index : old_start - 1])
            old_index = old_start - 1
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_line = lines[i]
                if hunk_line.startswith(" "):
                    output.append(original[old_index])
                    old_index += 1
                elif hunk_line.startswith("-"):
                    old_index += 1
                elif hunk_line.startswith("+"):
                    output.append(hunk_line[1:])
                elif hunk_line.startswith("\\"):
                    pass
                i += 1
        output.extend(original[old_index:])
        path.write_text("".join(output), encoding="utf-8")
