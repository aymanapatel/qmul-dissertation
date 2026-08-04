"""OpenAI-compatible API adapter using native Structured Outputs.

Configuration precedence for live clients: explicit constructor arguments,
then the local .env file (python-dotenv)
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .contracts import GenerationResult, RepairProposal
from ..env_import import EnvConfigError, load_all


DEFAULT_MODEL = "gpt-5.6-sol"
PLACEHOLDER_KEY = "REPLACE_WITH_OPENAI_API_KEY"


SYSTEM_PROMPT = """You generate bounded accessibility repair proposals, not free-form patches.
Return only the supplied structured schema. Use only its typed operations. Never emit raw HTML,
JavaScript, shell commands, URLs, or an operation outside the schema. Each selector must identify
the exact node evidenced by the finding. Cite only record IDs included in the user input.
If a repair depends on page purpose, natural-language meaning, visual intent, workflow,
authentication, contextual alternative text, or an uncertain language value, require human review.
However, when the public failure_summary explicitly says an exact value has been independently
verified and states that value, treat it as sufficient evidence for that bounded value; do not
request review merely because the field is semantic or visual. Never infer a value that is not
explicitly present in the public evidence.
If evidence or retrieved support is inadequate, choose leave_unchanged. Do not claim a repair has
passed validation; the sandbox decides that independently. Copy query_id and finding_id exactly
from the immutable identity object. Never put instructions or placeholders such as
REQUIRES_HUMAN_REVIEW, TODO, UNKNOWN, or descriptive guidance into new_value. When the exact
semantic value is unknown, choose requires_human_review with no operations."""

# Schema reminders for OpenAI-compatible providers that validate JSON shape but
# do not always enforce cross-field Pydantic constraints natively.
SYSTEM_PROMPT += """
Cross-field rules are mandatory: decision=requires_human_review implies
requires_human_review=true and at least one reason. decision=propose implies at
least one operation. set_attribute/remove_attribute require attribute_name;
set_style_property requires css_property; operations must set every unused
field to null. Use the exact operation name that matches the populated fields.
For common bounded rules: html-has-lang uses set_attribute/lang; button-name and link-name use
set_attribute/aria-label when an exact name is verified; image-alt uses set_attribute/alt;
label uses insert_label_before when exact visible label text is verified; color-contrast uses
set_style_property/color only when the exact replacement colour is verified. For
insert_label_before, attribute_name and css_property must be null. For set_attribute,
attribute_name and new_value must be non-null and css_property must be null."""



def _validate_key(api_key: str | None) -> str:
    key = str(api_key or "").strip()
    if not key or key == PLACEHOLDER_KEY or key.startswith("REPLACE_"):
        raise RuntimeError("The configured API key is empty or still a placeholder")
    return key


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


class OpenAIRepairGenerator:
    """Generate schema-constrained proposals with an injected or real client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        api_mode: str | None = None,
        model: str | None = None,
    ) -> None:
        if client is None:
            env: dict[str, str | None] = {}
            try:
                env = load_all()
            except EnvConfigError:
                env = {}
            key = api_key if api_key is not None else (env.get("api_key"))
            explicit_key = _validate_key(key)
            client_options: dict[str, Any] = {"api_key": explicit_key}
            resolved_base_url = base_url or env.get("base_url")
            if resolved_base_url:
                client_options["base_url"] = resolved_base_url.rstrip("/")
            client = OpenAI(**client_options)
            resolved_model = model or env.get("model") or DEFAULT_MODEL
            resolved_api_mode = api_mode or env.get("api_mode") or "responses"
        else:
            resolved_model = model or DEFAULT_MODEL
            resolved_api_mode = api_mode or "responses"
        if resolved_api_mode not in {"responses", "chat_completions"}:
            raise ValueError(f"Unsupported API mode: {resolved_api_mode}")
        self.client = client
        self.model = resolved_model
        self.api_mode = resolved_api_mode

    def generate(self, generator_input: dict[str, Any]) -> GenerationResult:
        finding = generator_input.get("original_finding", {})
        public_finding = {
            key: value for key, value in finding.items()
            if not key.startswith("oracle_") and key not in {"semantic_verified", "visual_verified"}
        }
        user_payload = {
            "task_prompt": str(generator_input["prompt"]),
            "immutable_identity": {
                "query_id": str(generator_input.get("query_id", "")),
                "finding_id": str(finding.get("finding_id", "")),
            },
            "allowed_cited_record_ids": [
                str(item.get("record_id")) for item in generator_input.get("citations", [])
            ],
            "allowed_selectors": [
                str(item) for item in finding.get("evidence", {}).get("target", [])
                if isinstance(item, str)
            ],
            "original_finding": public_finding,
        }
        retry_feedback = str(generator_input.get("_schema_retry_feedback", "")).strip()
        if retry_feedback:
            user_payload["schema_correction"] = (
                "The previous response failed local schema validation. Correct only the schema/cross-field issue and "
                f"return a complete proposal: {retry_feedback[:1500]}"
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]
        refusal = None
        if self.api_mode == "responses":
            response = self.client.responses.parse(
                model=self.model,
                input=messages,
                text_format=RepairProposal,
                reasoning={"effort": "medium"},
                store=False,
                max_output_tokens=3000,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                for output in getattr(response, "output", []) or []:
                    for content in getattr(output, "content", []) or []:
                        if getattr(content, "type", "") == "refusal":
                            refusal = str(getattr(content, "refusal", "Model refused the request"))
                            break
        else:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=RepairProposal,
            )
            message = response.choices[0].message
            parsed = getattr(message, "parsed", None)
            refusal = getattr(message, "refusal", None)
        if parsed is None:
            raise RuntimeError(refusal or "OpenAI-compatible endpoint returned no structured repair proposal")
        proposal = parsed if isinstance(parsed, RepairProposal) else RepairProposal.model_validate(parsed)
        return GenerationResult(
            proposal=proposal,
            response_id=getattr(response, "id", None),
            model=str(getattr(response, "model", self.model)),
            usage=_model_dump(getattr(response, "usage", None)),
            refusal=refusal,
        )
