from fastapi.testclient import TestClient

from breakpoint_eval.api import app


def test_health_and_compile_endpoint(tmp_path) -> None:
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

    actual_response = client.post(
        "/actual/compile",
        json={
            "out_dir": str(tmp_path / "actual-api"),
            "max_records": 1,
            "variants_per_item": 1,
            "external_judges": False,
        },
    )
    assert actual_response.status_code == 200
    actual_payload = actual_response.json()
    assert actual_payload["trace_count"] == 1
    assert actual_payload["total_cases"] == 2
