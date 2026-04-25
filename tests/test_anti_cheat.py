from CyberSecurity_OWASP.models import CyberSecurityOWASPAction

from .helpers import make_env, submit_valid_finding


def test_patch_hidden_identifier_flagged():
    env = make_env(50)
    submit_valid_finding(env)
    hidden_id = env.state.hidden_facts["other_invoice_id"]
    env.step(
        CyberSecurityOWASPAction(
            tool_name="patch_file",
            arguments={"path": "app/routes/invoices.py", "content": f"# {hidden_id}\n"},
        )
    )
    assert "hardcoded_hidden_identifier" in env.state.anti_cheat_flags
