from pathlib import Path

from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import CybersecurityOwaspEnvironment


def make_env(seed: int = 7) -> CybersecurityOwaspEnvironment:
    env = CybersecurityOwaspEnvironment()
    env.reset(seed=seed, episode_id=f"test-{seed}")
    return env


def submit_valid_finding(env: CybersecurityOwaspEnvironment):
    hidden = env.state.hidden_facts
    trace = env.step(
        CyberSecurityOWASPAction(
            tool_name="send_local_request",
            arguments={
                "method": "GET",
                "path": f"/invoices/{hidden['other_invoice_id']}",
                "user_id": hidden["owner_user_id"],
            },
        )
    )
    trace_id = "req_001"
    if '"trace_id": "req_' in trace.last_tool_result:
        trace_id = trace.last_tool_result.split('"trace_id": "', 1)[1].split('"', 1)[0]
    return env.step(
        CyberSecurityOWASPAction(
            tool_name="submit_diagnosis",
            arguments={
                "bug_class": "idor_ownership_bug",
                "route": "GET /invoices/{invoice_id}",
                "violated_policy_rule": "owner or billing_admin in same tenant only",
                "evidence_trace_ids": [trace_id],
                "fix_plan": "add tenant and owner/admin checks before returning the invoice",
            },
        )
    )


def secure_invoice_source(env: CybersecurityOwaspEnvironment) -> str:
    workspace = Path(env.state.hidden_facts["workspace"])
    source = (workspace / "app/routes/invoices.py").read_text(encoding="utf-8")
    return source.replace(
        "    # BUG: this only checks that the caller is authenticated. It forgets the\n"
        "    # owner/admin and tenant policy checks required by the policy graph.\n"
        "    return {\"status\": 200, \"body\": invoice}\n",
        "    if invoice[\"tenant_id\"] != actor[\"tenant_id\"]:\n"
        "        return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
        "    if invoice[\"owner_user_id\"] != actor[\"user_id\"] and not is_billing_admin(actor):\n"
        "        return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
        "    return {\"status\": 200, \"body\": invoice}\n",
    )


def apply_secure_patch(env: CybersecurityOwaspEnvironment):
    return env.step(
        CyberSecurityOWASPAction(
            tool_name="patch_file",
            arguments={"path": "app/routes/invoices.py", "content": secure_invoice_source(env)},
        )
    )
