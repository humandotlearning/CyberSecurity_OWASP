"""Typed action tool dispatcher for the generated app sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass

try:
    from ..models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from .app_sandbox import AppSandbox
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from server.app_sandbox import AppSandbox


@dataclass(frozen=True)
class ToolResult:
    message: str
    visible_test_result: str | None = None


class ActionTools:
    """Executes phase-gated, safe tools against one episode state."""

    def __init__(
        self,
        state: CyberSecurityOWASPState,
        visible_policy_hint: dict,
        workspace_summary: dict,
    ):
        self.state = state
        self.visible_policy_hint = visible_policy_hint
        self.workspace_summary = workspace_summary
        self.sandbox = AppSandbox(state)

    def execute(self, action: CyberSecurityOWASPAction) -> ToolResult:
        args = action.arguments or {}
        if action.tool_name == "noop":
            return ToolResult("No operation.")
        if action.tool_name == "inspect_policy_graph":
            return ToolResult(json.dumps(self.visible_policy_hint, indent=2, sort_keys=True))
        if action.tool_name == "list_routes":
            return ToolResult(json.dumps(self.workspace_summary["routes"], indent=2))
        if action.tool_name == "read_openapi":
            return ToolResult(self.sandbox.read_openapi())
        if action.tool_name == "read_file":
            return ToolResult(self.sandbox.read_file(str(args.get("path", ""))))
        if action.tool_name == "search_code":
            return ToolResult(self.sandbox.search_code(str(args.get("query", ""))))
        if action.tool_name == "send_local_request":
            response = self.sandbox.send_local_request(
                str(args.get("method", "GET")),
                str(args.get("path", "")),
                args.get("user_id"),
            )
            return ToolResult(json.dumps(response, indent=2, sort_keys=True))
        if action.tool_name == "compare_identities":
            response = self.sandbox.compare_identities(
                str(args.get("method", "GET")),
                str(args.get("path", "")),
                str(args.get("first_user_id", "")),
                str(args.get("second_user_id", "")),
            )
            return ToolResult(json.dumps(response, indent=2, sort_keys=True))
        if action.tool_name == "patch_file":
            result = self.sandbox.patch_file(
                str(args.get("path", "")),
                content=str(args["content"]) if "content" in args else None,
                diff=str(args.get("diff", "")) if "content" not in args else None,
            )
            changed = "no diff" if not result["diff"].strip() else "diff recorded"
            return ToolResult(f"Patched {result['path']} ({changed}).")
        raise ValueError(f"Unhandled tool {action.tool_name}")
