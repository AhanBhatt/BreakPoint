from fastapi.testclient import TestClient

from breakpoint_eval.api import app


def test_health_and_compile_endpoint() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    response = client.post(
        "/compile",
        json={"categories": ["tool_misuse"], "items_per_category": 2, "variants_per_item": 1, "seed": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["accepted_items"] == 2
    assert payload["preview"][0]["category"] == "tool_misuse"
