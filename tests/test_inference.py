import pytest

from Cyber_analyst.inference import (
    ModelActionError,
    action_to_log,
    error_action,
    parse_model_action,
)


def test_parse_model_action_accepts_compact_json():
    action = parse_model_action('{"tool_name":"search_repo","args":{"query":"api key"}}')

    assert action.tool_name == "search_repo"
    assert action.args == {"query": "api key"}


def test_parse_model_action_accepts_fenced_json():
    action = parse_model_action(
        """```json
{"tool_name":"list_assets","args":{}}
```"""
    )

    assert action.tool_name == "list_assets"
    assert action.args == {}


def test_parse_model_action_rejects_malformed_json():
    with pytest.raises(ModelActionError, match="model_parse_error"):
        parse_model_action("search the repo for api keys")


def test_action_to_log_is_single_line_json():
    action = parse_model_action('{"tool_name":"search_repo","args":{"query":"api\\nkey"}}')

    logged = action_to_log(action)

    assert "\n" not in logged
    assert logged == '{"args":{"query":"api\\nkey"},"tool_name":"search_repo"}'


def test_error_action_uses_strict_diagnostic_tool_name():
    action = error_action(ModelActionError("model_parse_error: empty response"))

    assert action.tool_name == "model_parse_error"
    assert action.args == {"message": "model_parse_error: empty response"}
