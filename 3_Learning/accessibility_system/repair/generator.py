"""OpenAI-compatible API adapter using native Structured Outputs.

Configuration precedence for live clients: explicit constructor arguments,
then the local .env file (python-dotenv)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import OpenAI

from .contracts import GenerationResult, RepairProposal
from ..env_import import EnvConfigError, load_all


DEFAULT_MODEL = "gpt-5.6-sol"
PLACEHOLDER_KEY = "REPLACE_WITH_OPENAI_API_KEY"
SENSITIVE_HTTP_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"})
SENSITIVE_PAYLOAD_KEYS = frozenset({"api_key", "apikey", "authorization", "token", "secret", "password"})


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


def build_prompt_messages(generator_input: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the exact public prompts sent to the structured-output provider."""

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
    return messages, user_payload


def _safe_endpoint(base_url: str | None, api_mode: str) -> str:
    """Return a display-safe endpoint without credentials, query, or fragment."""

    raw = str(base_url or "https://api.openai.com/v1").rstrip("/")
    parsed = urlsplit(raw)
    hostname = parsed.hostname or "api.openai.com"
    port = f":{parsed.port}" if parsed.port else ""
    safe_base = urlunsplit((parsed.scheme or "https", f"{hostname}{port}", parsed.path.rstrip("/"), "", ""))
    resource = "responses" if api_mode == "responses" else "chat/completions"
    return f"{safe_base}/{resource}"



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


def _redact_payload(value: Any) -> Any:
    """Recursively redact common secret-bearing fields before diagnostic logging."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower().replace("-", "_") in SENSITIVE_PAYLOAD_KEYS else _redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


def _http_body_log(body: bytes, *, include_body: bool, limit: int = 4000) -> str:
    """Return a bounded body diagnostic, never exposing values under secret keys."""
    digest = hashlib.sha256(body).hexdigest()
    prefix = f"bytes={len(body)} sha256={digest}"
    if not include_body or not body:
        return prefix
    try:
        parsed = _redact_payload(json.loads(body.decode("utf-8")))
        rendered = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        rendered = body.decode("utf-8", errors="replace")
    rendered = " ".join(rendered.split())
    return f"{prefix} body={rendered[:limit]}{'...' if len(rendered) > limit else ''}"


def _safe_http_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: "[REDACTED]" if key.lower() in SENSITIVE_HTTP_HEADERS else value
        for key, value in headers.items()
        if key.lower() in {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key", "content-type", "accept", "user-agent", "x-request-id", "openai-request-id", "retry-after"}
    }


def _http_event_hooks(logger: logging.Logger, *, include_bodies: bool) -> dict[str, list]:
    def request_hook(request: httpx.Request) -> None:
        safe_url = str(request.url.copy_with(query=None))
        logger.info(
            "http_request method=%s url=%s headers=%s %s",
            request.method, safe_url, _safe_http_headers(request.headers),
            _http_body_log(request.content, include_body=include_bodies),
        )

    def response_hook(response: httpx.Response) -> None:
        # Chat-completions is non-streaming. Reading here caches the body for
        # the OpenAI SDK while making its status/body available for diagnosis.
        response.read()
        safe_url = str(response.request.url.copy_with(query=None))
        logger.info(
            "http_response status=%d url=%s headers=%s %s",
            response.status_code, safe_url, _safe_http_headers(response.headers),
            _http_body_log(response.content, include_body=include_bodies),
        )

    return {"request": [request_hook], "response": [response_hook]}


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
        request_timeout_seconds: float | None = None,
        http_logger: logging.Logger | None = None,
        log_http_bodies: bool = False,
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
            if request_timeout_seconds is not None:
                client_options["timeout"] = float(request_timeout_seconds)
            if http_logger is not None:
                http_client = httpx.Client(
                    timeout=float(request_timeout_seconds) if request_timeout_seconds is not None else None,
                    event_hooks=_http_event_hooks(http_logger, include_bodies=log_http_bodies),
                )
                client_options["http_client"] = http_client
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
        known_base_url = resolved_base_url if "resolved_base_url" in locals() else getattr(client, "base_url", None)
        self.endpoint = _safe_endpoint(str(known_base_url) if known_base_url else None, resolved_api_mode)

    def generate(self, generator_input: dict[str, Any]) -> GenerationResult:
        messages, user_payload = build_prompt_messages(generator_input)
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
            request_trace={
                "method": "POST",
                "endpoint": self.endpoint,
                "api_mode": self.api_mode,
                "model": self.model,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_payload,
                "response_format": "RepairProposal (strict structured output)",
            },
        )
