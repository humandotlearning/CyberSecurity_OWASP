from fastapi.testclient import TestClient

from CyberSecurity_OWASP.server.app import app


def test_space_root_redirects_to_openenv_web_ui():
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/web/"


def test_openenv_web_ui_and_api_routes_are_available():
    client = TestClient(app)

    web_response = client.get("/web/")
    health_response = client.get("/health")
    state_response = client.get("/web/state")

    assert web_response.status_code == 200
    assert "text/html" in web_response.headers["content-type"]
    assert "Reset" in web_response.text
    assert "Step" in web_response.text
    assert "Get state" in web_response.text

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}

    assert state_response.status_code == 200
    state = state_response.json()
    assert "episode_id" in state
    assert "step_count" in state


def test_web_reset_returns_cybersecurity_observation():
    client = TestClient(app)

    response = client.post("/web/reset")

    assert response.status_code == 200
    payload = response.json()
    observation = payload["observation"]
    assert observation["phase"] == "discover"
    assert "authorization" in observation["task_brief"]
    assert "inspect_policy_graph" in observation["available_actions"]
