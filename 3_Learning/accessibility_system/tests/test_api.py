from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from accessibility_system.api import create_app


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
