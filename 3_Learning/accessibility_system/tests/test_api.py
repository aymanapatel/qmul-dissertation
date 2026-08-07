from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from accessibility_system.api import create_app
from accessibility_system.repair.contracts import RepairProposal


def make_client(tmp_path: Path) -> TestClient:
    inputs = [{
        "query_id": "q-1", "condition": "graph_constrained_rag", "prompt": "Repair it",
        "citations": [{"record_id": "r-1"}], "safe_action": "propose_only_do_not_apply",
        "original_finding": {
            "finding_id": "f-1", "site_id": "example.com", "rule_id": "image-alt",
            "evidence": {"target": ["img"]},
        },
    }]
    input_path = tmp_path / "inputs.json"
    input_path.write_text(json.dumps(inputs), encoding="utf-8")
    axe = tmp_path / "axe.js"
    axe.write_text("", encoding="utf-8")
    return TestClient(create_app(
        generator_inputs=input_path, corpus_dir=tmp_path / "corpus",
        axe_js=axe, runs_dir=tmp_path / "runs",
    ))


def test_health_and_rag_endpoints(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    listing = client.get("/v1/rag/inputs", params={"rule_id": "image-alt"}).json()
    assert listing["count"] == 1
    detail = client.get("/v1/rag/inputs/q-1").json()
    assert detail["conditions"][0]["citations"][0]["record_id"] == "r-1"
    assert client.get("/v1/rag/inputs/missing").status_code == 404


def test_scan_rejects_private_network_targets(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/v1/scans", json={"urls": ["http://127.0.0.1/admin"]})
    assert response.status_code == 422
    assert "blocked" in response.json()["detail"]


def test_job_and_artifact_endpoints_are_traversal_safe(tmp_path):
    client = make_client(tmp_path)
    store = client.app.state.store
    run = store.root / "scan-existing"
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({
        "job_id": "scan-existing", "kind": "scan", "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00", "run_dir": str(run),
        "result_path": str(run / "result.json"),
    }), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert client.get("/v1/jobs/scan-existing/result").json()["status"] == "completed"
    assert client.get("/v1/jobs/scan-existing/artifacts/result.json").status_code == 200
    assert client.get("/v1/jobs/scan-existing/artifacts/../../inputs.json").status_code == 404


def test_live_suggestion_endpoint_scans_input_and_returns_structured_llm_output(tmp_path, monkeypatch):
    captured = {}

    def fake_scanner(urls, axe_js, timeout, screenshot_dir):
        captured["urls"] = urls
        captured["screenshot_dir"] = screenshot_dir
        return {"sites": [{
            "url": urls[0], "final_url": urls[0], "status": "completed",
            "violation_count": 1, "affected_node_count": 1,
            "violations_by_impact": {"serious": 1}, "screenshot_artifact": None,
            "violations": [{
                "id": "image-alt", "impact": "serious", "help": "Images must have alternate text",
                "help_url": "https://example.test/rule", "nodes": [{
                    "target": ["#hero"], "html": '<img id="hero">',
                    "failure_summary": "Element does not have an alt attribute",
                }],
            }],
        }]}

    class FakeGenerator:
        def generate(self, generator_input):
            finding = generator_input["original_finding"]
            return SimpleNamespace(
                model="test-structured-model", response_id="resp_test", usage={"total_tokens": 42},
                proposal=RepairProposal(
                    schema_version=1, proposal_id="proposal-1", query_id=generator_input["query_id"],
                    finding_id=finding["finding_id"], decision="requires_human_review", operations=[],
                    rationale="Alternative text depends on the image purpose.",
                    expected_resolution="Provide an equivalent text alternative.", cited_record_ids=[],
                    uncertainty="The image purpose is not known.", requires_human_review=True,
                    human_review_reasons=["Confirm the image purpose and wording."],
                    validation_steps=["Inspect the computed accessible name."], confidence=0.71,
                ),
            )

    monkeypatch.setattr("accessibility_system.api._safe_public_url", lambda value: None)
    inputs = tmp_path / "inputs.json"; inputs.write_text("[]")
    axe = tmp_path / "axe.js"; axe.write_text("")
    client = TestClient(create_app(
        generator_inputs=inputs, corpus_dir=tmp_path / "corpus", axe_js=axe,
        runs_dir=tmp_path / "runs", suggestion_scanner=fake_scanner,
        suggestion_generator_factory=FakeGenerator,
    ))
    accepted = client.post("/v1/suggestion-audits", json={"url": "https://input.example/page", "max_suggestions": 1})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert job["status"] == "completed"
    result = client.get(f"/v1/jobs/{job_id}/result").json()
    assert captured["urls"] == ["https://input.example/page"]
    assert result["safety"].startswith("Suggestions only")
    assert result["suggestions"][0]["model"] == "test-structured-model"
    assert result["suggestions"][0]["decision"] == "requires_human_review"
    assert result["suggestions"][0]["operations"] == []
