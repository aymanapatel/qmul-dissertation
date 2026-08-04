from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from accessibility_system.phase9 import _configure_logging, run as run_phase9
from accessibility_system.repair.contracts import GenerationResult, RepairOperation, RepairProposal
from accessibility_system.repair.generator import OpenAIRepairGenerator, _validate_key
from accessibility_system.repair.patches import PatchApplicationError, apply_typed_patch
from accessibility_system.repair.policy import proposal_policy_errors
from accessibility_system.repair.validators import _visual_change_independently_verified, validate_repair


AXE_JS = Path(__file__).resolve().parents[3] / "2_Data" / "browser-use" / "axe-core.min.js"


def operation(name="remove_meta_viewport_restriction", selector="meta[name=viewport]", **overrides):
    values = dict(operation=name, selector=selector, attribute_name=None, css_property=None, new_value=None)
    values.update(overrides)
    return RepairOperation(**values)


def proposal(**overrides):
    values = dict(
        schema_version=1,
        proposal_id="proposal-1",
        query_id="query-1",
        finding_id="finding-1",
        decision="propose",
        operations=[operation()],
        rationale="Remove the zoom restriction.",
        expected_resolution="Page zoom is no longer restricted.",
        cited_record_ids=["record-1"],
        uncertainty="",
        requires_human_review=False,
        human_review_reasons=[],
        validation_steps=["Rerun the originating detector and axe."],
        confidence=0.98,
    )
    values.update(overrides)
    return RepairProposal(**values)


def finding(rule="meta-viewport", status="verified_fail", selector="meta[name=viewport]", **overrides):
    values = dict(
        finding_id="finding-1", site_id="fixture", criterion_id="1.4.4",
        rule_id=rule, status=status, evidence={"target": [selector]},
    )
    values.update(overrides)
    return values


def generator_input(**overrides):
    values = dict(
        query_id="query-1", condition="graph_constrained_rag", prompt="Propose a repair.",
        citations=[{"record_id": "record-1"}], safe_action="propose_only_do_not_apply",
        original_finding=finding(),
    )
    values.update(overrides)
    return values


def valid_html() -> str:
    return """<!doctype html><html lang="en"><head><title>Fixture</title>
    <meta name="viewport" content="width=device-width, maximum-scale=1, user-scalable=no">
    </head><body><main><button id="go">Go</button></main></body></html>"""


def test_schema_is_strict_and_decisions_are_consistent():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RepairProposal.model_validate({**proposal().model_dump(), "raw_patch": "<script>bad()</script>"})
    with pytest.raises(ValidationError, match="must not contain operations"):
        proposal(decision="leave_unchanged")
    with pytest.raises(ValidationError, match="String should match pattern"):
        proposal(proposal_id="../../unsafe")
    schema = RepairProposal.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["RepairOperation"]["additionalProperties"] is False


def test_openai_adapter_uses_responses_parse_and_pydantic_format():
    expected = proposal()

    class Responses:
        def __init__(self):
            self.kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                id="response-1", model="gpt-5.6-sol", output_parsed=expected,
                usage=SimpleNamespace(model_dump=lambda mode: {"input_tokens": 10, "output_tokens": 20}),
            )

    responses = Responses()
    client = SimpleNamespace(responses=responses)
    private_input = generator_input()
    private_input["original_finding"].update({"oracle_operations": [{"secret": True}], "semantic_verified": True})
    result = OpenAIRepairGenerator(client=client, api_mode="responses").generate(private_input)
    assert result.proposal == expected
    assert responses.kwargs["text_format"] is RepairProposal
    assert responses.kwargs["model"] == "gpt-5.6-sol"
    assert responses.kwargs["store"] is False
    assert responses.kwargs["input"][0]["role"] == "system"
    payload = json.loads(responses.kwargs["input"][1]["content"])
    assert payload["immutable_identity"] == {"query_id": "query-1", "finding_id": "finding-1"}
    assert payload["allowed_cited_record_ids"] == ["record-1"]
    assert payload["allowed_selectors"] == ["meta[name=viewport]"]
    assert "oracle_operations" not in payload["original_finding"]
    assert "semantic_verified" not in payload["original_finding"]


def test_openai_adapter_passes_only_schema_feedback_on_retry():
    expected = proposal()

    class Responses:
        def parse(self, **kwargs):
            payload = json.loads(kwargs["input"][1]["content"])
            assert "set_attribute requires attribute_name" in payload["schema_correction"]
            assert "oracle_operations" not in payload["original_finding"]
            return SimpleNamespace(id="response-2", model="model", output_parsed=expected, usage={})

    value = generator_input(_schema_retry_feedback="ValidationError: set_attribute requires attribute_name")
    assert OpenAIRepairGenerator(client=SimpleNamespace(responses=Responses()), api_mode="responses").generate(value).proposal == expected


def test_openai_compatible_chat_adapter_uses_json_schema_parse():
    expected = proposal()

    class Completions:
        def __init__(self):
            self.kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(parsed=expected, refusal=None)
            return SimpleNamespace(
                id="chat-1", model="compatible-model",
                choices=[SimpleNamespace(message=message)], usage={"total_tokens": 3},
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = OpenAIRepairGenerator(
        client=client, model="compatible-model", api_mode="chat_completions",
    ).generate(generator_input())
    assert result.proposal == expected
    assert completions.kwargs["response_format"] is RepairProposal
    assert completions.kwargs["messages"][0]["role"] == "system"


def test_live_client_receives_explicit_string_and_placeholder_is_rejected(monkeypatch):
    captured = {}
    monkeypatch.setattr("accessibility_system.repair.generator.load_all", lambda: {
        "api_key": "sk-env-key", "base_url": "https://env.example/v1/",
        "model": "env-model", "api_mode": "chat_completions",
    })
    monkeypatch.setattr(
        "accessibility_system.repair.generator.OpenAI",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(responses=None),
    )
    generator = OpenAIRepairGenerator(api_key="sk-local-explicit-string")
    assert captured["api_key"] == "sk-local-explicit-string"
    assert captured["base_url"] == "https://env.example/v1"
    assert generator.model == "env-model"
    assert generator.api_mode == "chat_completions"
    with pytest.raises(RuntimeError, match="placeholder"):
        _validate_key("REPLACE_WITH_OPENAI_API_KEY")


def test_live_client_uses_env_when_no_explicit_api_key(monkeypatch):
    captured = {}
    monkeypatch.setattr("accessibility_system.repair.generator.load_all", lambda: {
        "api_key": "sk-env-key", "base_url": "https://env.example/v1",
        "model": "env-model", "api_mode": "chat_completions",
    })
    monkeypatch.setattr(
        "accessibility_system.repair.generator.OpenAI",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(responses=None),
    )
    OpenAIRepairGenerator()
    assert captured["api_key"] == "sk-env-key"
    assert captured["base_url"] == "https://env.example/v1"


def test_patch_is_atomic_typed_and_rejects_unsafe_attribute():
    source = valid_html()
    patched, evidence = apply_typed_patch(source, proposal())
    assert "user-scalable=no" not in patched
    assert "maximum-scale=1" not in patched
    assert "user-scalable=no" in source
    assert evidence["operation_count"] == 1
    unsafe = proposal(operations=[operation(
        "set_attribute", "button#go", attribute_name="onclick", new_value="bad()",
    )])
    with pytest.raises(PatchApplicationError, match="not allow-listed"):
        apply_typed_patch(source, unsafe)


def test_grounding_policy_blocks_hallucinated_identity_citation_and_selector():
    bad = proposal(
        query_id="other-query", finding_id="other-finding", cited_record_ids=["invented"],
        operations=[operation(selector="body")],
    )
    errors = proposal_policy_errors(bad, generator_input())
    assert len(errors) == 4


def test_grounding_policy_blocks_instructional_placeholder_as_repair_value():
    bad = proposal(operations=[operation(
        "set_attribute", "meta[name=viewport]", attribute_name="title",
        new_value="REQUIRES_HUMAN_REVIEW: provide appropriate text",
    )])
    errors = proposal_policy_errors(bad, generator_input())
    assert errors == [
        "operation new_value contains instructional or placeholder text: meta[name=viewport]",
    ]


@pytest.mark.skipif(not AXE_JS.is_file(), reason="local axe-core bundle is unavailable")
def test_isolated_browser_validation_accepts_verified_nonsemantic_repair(tmp_path):
    source = tmp_path / "source.html"
    source.write_text(valid_html(), encoding="utf-8")
    result = validate_repair(
        source_path=source, proposal=proposal(), original_finding=finding(),
        output_dir=tmp_path / "attempts", axe_js=AXE_JS,
    )
    assert result.outcome == "accepted"
    assert result.target_resolved is True
    assert result.new_regressions == []
    assert result.source_unchanged is True
    assert source.read_text(encoding="utf-8") == valid_html()
    assert Path(result.artifact_paths["validation"]).is_file()


def test_semantic_repair_is_not_accepted_from_syntax_alone(tmp_path):
    source = tmp_path / "source.html"
    source.write_text('<html lang="en"><head><title>X</title></head><body><main><img id="hero" src="x.png"></main></body></html>', encoding="utf-8")
    repair = proposal(operations=[operation(
        "set_attribute", "img#hero", attribute_name="alt", new_value="A person",
    )])
    original = finding(rule="image-alt", selector="img#hero", semantic_verified=False)
    result = validate_repair(
        source_path=source, proposal=repair, original_finding=original,
        output_dir=tmp_path / "attempts", axe_js=AXE_JS, run_browser=False,
    )
    assert result.target_resolved is True
    assert result.outcome == "requires_human_review"
    assert "semantic_or_contextual_correctness_not_independently_verified" in result.human_review_reasons
    assert "browser_validation_incomplete" in result.human_review_reasons


def test_exact_oracle_visible_label_can_explain_expected_pixel_change():
    visible_label = proposal(operations=[operation(
        "insert_label_before", "#email", new_value="Email address",
    )])
    verified = finding(rule="label", selector="#email", semantic_verified=True)
    assert _visual_change_independently_verified(visible_label, verified, oracle_match=True) is True
    assert _visual_change_independently_verified(visible_label, verified, oracle_match=False) is False


def test_unapplied_contextual_proposal_is_routed_to_review_not_rejected(tmp_path):
    source = tmp_path / "source.html"
    source.write_text(
        '<html lang="en"><head><title>X</title></head><body><main><img id="hero" src="x.png"></main></body></html>',
        encoding="utf-8",
    )
    review = proposal(
        decision="requires_human_review", operations=[], requires_human_review=True,
        human_review_reasons=["The image purpose is unknown."],
    )
    result = validate_repair(
        source_path=source,
        proposal=review,
        original_finding=finding(rule="image-alt", selector="img#hero"),
        output_dir=tmp_path / "attempts",
        axe_js=AXE_JS,
        run_browser=False,
    )
    assert result.target_resolved is False
    assert result.outcome == "requires_human_review"
    assert result.rejection_reasons == []
    assert "generator_requested_human_review_without_applying_a_patch" in result.human_review_reasons


def test_new_accessibility_regression_rejects_otherwise_resolved_repair(tmp_path):
    source = tmp_path / "source.html"
    source.write_text(valid_html(), encoding="utf-8")
    repair = proposal(operations=[
        operation(),
        operation("remove_attribute", "html", attribute_name="lang"),
    ])
    result = validate_repair(
        source_path=source, proposal=repair, original_finding=finding(),
        output_dir=tmp_path / "attempts", axe_js=AXE_JS, run_browser=False,
    )
    assert result.target_resolved is True
    assert result.outcome == "rejected"
    assert any(item.startswith("html-has-lang@") for item in result.new_regressions)


def test_phase9_cli_orchestration_writes_report_and_manifest_with_mocked_llm(tmp_path):
    corpus = tmp_path / "corpus" / "fixture"
    corpus.mkdir(parents=True)
    (corpus / "0.html").write_text(valid_html(), encoding="utf-8")
    inputs_path = tmp_path / "generator_inputs.json"
    inputs_path.write_text(json.dumps([generator_input()]), encoding="utf-8")

    class MockGenerator:
        def generate(self, item):
            return GenerationResult(
                proposal=proposal(), response_id="mock-response", model="mock-openai-model",
                usage={"input_tokens": 1, "output_tokens": 1}, refusal=None,
            )

    args = argparse.Namespace(
        generator_inputs=inputs_path, corpus_dir=tmp_path / "corpus", axe_js=AXE_JS,
        output_dir=tmp_path / "phase9", condition="graph_constrained_rag",
        model="gpt-5.6-sol", max_proposals=1, skip_browser=True,
    )
    report = run_phase9(args, generator=MockGenerator())
    assert report["attempt_count"] == 1
    assert report["outcome_counts"] == {"requires_human_review": 1}
    assert (args.output_dir / "phase_9_report.json").is_file()
    assert (args.output_dir / "run_manifest.json").is_file()


def test_phase9_respects_requested_api_mode(tmp_path):
    corpus = tmp_path / "corpus" / "fixture"
    corpus.mkdir(parents=True)
    (corpus / "0.html").write_text(valid_html(), encoding="utf-8")
    inputs_path = tmp_path / "generator_inputs.json"
    inputs_path.write_text(json.dumps([generator_input()]), encoding="utf-8")

    class MockGenerator:
        model = "mock-responses-model"

        def generate(self, item):
            return GenerationResult(
                proposal=proposal(), response_id="response-1", model=self.model,
                usage={}, refusal=None,
            )

    args = argparse.Namespace(
        generator_inputs=inputs_path, corpus_dir=tmp_path / "corpus", axe_js=AXE_JS,
        output_dir=tmp_path / "phase9", condition="graph_constrained_rag",
        model=None, api_mode="responses", base_url=None, max_proposals=1,
        skip_browser=True,
    )
    report = run_phase9(args, generator=MockGenerator())
    assert report["api_mode"] == "responses"
    assert report["model"] == "mock-responses-model"


def test_phase9_logs_redacted_root_cause_and_skips_after_auth_failure(tmp_path):
    corpus = tmp_path / "corpus" / "fixture"
    corpus.mkdir(parents=True)
    (corpus / "0.html").write_text(valid_html(), encoding="utf-8")
    inputs_path = tmp_path / "generator_inputs.json"
    inputs_path.write_text(json.dumps([generator_input(), generator_input()]), encoding="utf-8")

    AuthenticationError = type("AuthenticationError", (Exception,), {})

    class FailingGenerator:
        calls = 0

        def generate(self, item):
            self.calls += 1
            raise AuthenticationError("Incorrect API key provided: sk-secret-that-must-not-appear")

    args = argparse.Namespace(
        generator_inputs=inputs_path, corpus_dir=tmp_path / "corpus", axe_js=AXE_JS,
        output_dir=tmp_path / "phase9", condition="graph_constrained_rag",
        model="gpt-5.6-sol", api_mode="responses", base_url=None,
        max_proposals=2, skip_browser=True,
    )
    log_path = _configure_logging(args.output_dir, "INFO")
    failing = FailingGenerator()
    report = run_phase9(args, generator=failing)
    assert failing.calls == 1
    assert report["outcome_counts"] == {
        "generation_authentication_failed": 1, "skipped_after_fatal_error": 1,
    }
    combined = log_path.read_text(encoding="utf-8") + json.dumps(report)
    assert "authentication_failed" in combined
    assert "sk-secret" not in combined
