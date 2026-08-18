from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from accessibility_system.api import _specialist_suggestion_inputs, create_app
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


def test_app_owns_live_model_bundle_configuration(tmp_path):
    phase5 = tmp_path / "trained" / "phase_5_multiview_final_v2"
    policy = tmp_path / "trained" / "phase_6_7_final" / "phase_6_fusion_policy.json"
    app = create_app(
        generator_inputs=tmp_path / "inputs.json", corpus_dir=tmp_path / "corpus",
        axe_js=tmp_path / "axe.js", runs_dir=tmp_path / "runs",
        phase5_dir=phase5, fusion_policy_path=policy,
    )
    assert app.state.phase5_dir == phase5.resolve()
    assert app.state.fusion_policy_path == policy.resolve()


def test_scan_rejects_private_network_targets(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/v1/scans", json={"urls": ["http://127.0.0.1/admin"]})
    assert response.status_code == 422
    assert "blocked" in response.json()["detail"]


def test_architecture_specific_findings_remain_separate():
    site = {"final_url": "https://example.com", "violations": []}
    repair_context = {
        "bounded_candidates": [{
            "operation": {
                "operation": "set_attribute", "selector": "#hero",
                "attribute_name": "alt", "css_property": None, "new_value": "Hero",
            },
            "derived_from": "visible_figcaption", "verification_level": "visible_context",
            "requires_human_review": True,
        }],
    }
    finding = {
        "rule_id": "image-alt", "criterion_id": "1.1.1", "graph_view": "a11y-tree",
        "detector_id": "a11y-tree:graphsage:image-alt", "probability": 0.9,
        "threshold": 0.7, "routing_status": "fail", "routing_confidence": 0.8,
        "architecture": "graphsage", "evidence": {
            "selector": "#hero", "html": "<img>", "repair_context": repair_context,
        },
    }
    second = {
        **finding, "architecture": "gat", "detector_id": "a11y-tree:gat:image-alt",
        "probability": 0.85,
    }
    values = _specialist_suggestion_inputs(site, {"findings": [finding, second]}, 5)
    assert len(values) == 2
    assert values[0]["original_finding"]["finding_id"] != values[1]["original_finding"]["finding_id"]
    assert [item["model_evidence"]["architecture"] for item in values] == ["graphsage", "gat"]
    evidence = values[0]["original_finding"]["evidence"]
    assert evidence["repair_context"] == repair_context
    assert "contextual semantic candidate" in values[0]["prompt"]


def test_measured_contrast_elements_become_one_visual_llm_request():
    selector_a = "main > p"
    selector_b = "main > button"

    def visual_failure(selector, tag, text, ratio):
        operation = {
            "operation": "set_style_property", "selector": selector,
            "attribute_name": None, "css_property": "color", "new_value": "#000000",
        }
        return {
            "source": "same-session-rendered-visual-capture",
            "selector": selector, "tag": tag, "text": text,
            "bounds": {"x": 10, "y": 20, "width": 100, "height": 30},
            "visual": {
                "foreground_rgb": [200, 200, 200], "background_rgb": [255, 255, 255],
                "contrast_ratio": ratio, "required_contrast_ratio": 4.5,
            },
            "contrast_failure": True,
            "contrast_failure_source": "axe-core+same-session-rendered-geometry",
            "repair_context": {
                "target": {"tag": tag, "selector": selector},
                "bounded_candidates": [{
                    "operation": operation, "derived_from": "computed_contrast",
                    "verification_level": "computed", "requires_human_review": False,
                }],
            },
        }

    failures = [
        visual_failure(selector_a, "p", "Low paragraph", 1.67),
        visual_failure(selector_b, "button", "Low button", 1.55),
    ]
    site = {
        "final_url": "https://example.com",
        "violations": [{
            "id": "color-contrast", "impact": "serious", "help": "Contrast",
            "help_url": "https://example.test/contrast",
        }],
    }
    values = _specialist_suggestion_inputs(site, {
        "findings": [],
        "visual_evidence": {
            "source": "same-session-rendered-visual-capture",
            "elements": failures, "contrast_failures": failures,
        },
    }, 5)
    assert len(values) == 1
    finding = values[0]["original_finding"]
    assert finding["rule_id"] == "color-contrast"
    assert finding["evidence"]["target"] == [selector_a, selector_b]
    assert finding["evidence"]["visual_elements"] == failures
    assert len(finding["evidence"]["repair_context"]["bounded_candidates"]) == 2
    assert values[0]["model_evidence"]["evidence_kind"] == "measured_visual"
    assert values[0]["model_evidence"]["probability"] is None


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
                    inspected_visual_elements=[],
                    human_review_reasons=["Confirm the image purpose and wording."],
                    validation_steps=["Inspect the computed accessible name."], confidence=0.71,
                ),
            )

    def fake_specialist_runner(site_dir, output_dir, progress=None):
        if progress:
            progress("build_a11y_tree", "running", "Build accessibility-tree graph", {})
            progress("build_a11y_tree", "completed", "Build accessibility-tree graph", {"node_count": 8})
        return {
            "architectures": ["mlp", "graphsage", "gat"],
            "training_artifacts": "frozen/phase5",
            "fusion_policy": "frozen/policy.json",
            "model_runs": [{
                "view": "a11y-tree", "architecture": "graphsage",
                "axe_used_for_prediction": False, "node_count": 8, "edge_count": 7,
                "rules": [], "findings": [],
            }],
            "findings": [{
                "rule_id": "image-alt", "criterion_id": "1.1.1",
                "graph_view": "a11y-tree", "architecture": "graphsage",
                "detector_id": "a11y-tree:graphsage:image-alt",
                "probability": 0.91, "threshold": 0.7,
                "routing_status": "fail", "routing_confidence": 0.82,
                "human_review_required": False, "wcag_ids": ["1.1.1"],
                "evidence": {"selector": "#hero", "html": '<img id="hero">', "text": ""},
            }],
        }

    monkeypatch.setattr("accessibility_system.api._safe_public_url", lambda value: None)
    inputs = tmp_path / "inputs.json"; inputs.write_text("[]")
    axe = tmp_path / "axe.js"; axe.write_text("")
    client = TestClient(create_app(
        generator_inputs=inputs, corpus_dir=tmp_path / "corpus", axe_js=axe,
        runs_dir=tmp_path / "runs", suggestion_scanner=fake_scanner,
        suggestion_generator_factory=FakeGenerator,
        suggestion_specialist_runner=fake_specialist_runner,
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
    assert result["suggestions"][0]["model_evidence"]["graph_view"] == "a11y-tree"
    assert result["specialist"]["architectures"] == ["mlp", "graphsage", "gat"]
    assert result["specialist"]["model_runs"][0]["axe_used_for_prediction"] is False
    assert result["suggestions"][0]["api_trace"]["request"]["system_prompt"]
    assert result["suggestions"][0]["api_trace"]["request"]["user_prompt"]["immutable_identity"]["finding_id"]
    assert result["application_api"]["submit"]["request_body"]["url"] == "https://input.example/page"
    assert {event["event_id"] for event in job["progress"]["events"]} >= {
        "capture_page", "build_a11y_tree", "call_llm_01", "finalise_result",
    }
    assert "run_dir" not in job and "result_path" not in job
    assert job["links"]["result"] == f"/v1/jobs/{job_id}/result"
