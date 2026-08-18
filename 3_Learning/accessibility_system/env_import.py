"""Load OpenAI-compatible configuration from a local .env file via python-dotenv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent

ENV_CANDIDATES = (
    PACKAGE_DIR / ".env",
    PACKAGE_DIR.parent / ".env",
    Path.cwd() / ".env",
)


class EnvConfigError(RuntimeError):
    """Raised when the .env file is missing or a required variable is absent."""


def _find_env_file() -> Path | None:
    for candidate in ENV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def load_env() -> Path:
    """Locate and load the nearest .env file, returning its path.

    Values from .env override any ambient environment variables. Raises
    EnvConfigError if no .env file can be found.
    """
    env_file = _find_env_file()
    if env_file is None:
        raise EnvConfigError(
            "No .env file found. Copy 3_Learning/accessibility_system/.env.example "
            "to .env and set OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL."
        )
    load_dotenv(dotenv_path=env_file, override=True)
    return env_file


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("REPLACE_") or "REPLACE_WITH_" in value:
        raise EnvConfigError(
            f"The .env variable {name} is missing or still a placeholder; "
            "replace it in the git-ignored .env file before live generation."
        )
    return value


def api_key() -> str:
    return _required("OPENAI_API_KEY")


def base_url() -> str | None:
    value = os.getenv("OPENAI_BASE_URL", "").strip()
    return value.rstrip("/") if value else None


def model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.6-sol").strip() or "gpt-5.6-sol"


def api_mode() -> str:
    return os.getenv("OPENAI_API_MODE", "chat_completions").strip() or "chat_completions"


def _float_setting(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be a number, received {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise EnvConfigError(f"{name} must be between {minimum} and {maximum}, received {value}")
    return value


def _int_setting(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be an integer, received {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise EnvConfigError(f"{name} must be between {minimum} and {maximum}, received {value}")
    return value


def _choice_setting(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower() or default
    if value not in allowed:
        raise EnvConfigError(f"{name} must be one of {sorted(allowed)}, received {value!r}")
    return value


def temperature() -> float:
    return _float_setting("OPENAI_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0)


def top_p() -> float:
    return _float_setting("OPENAI_TOP_P", 1.0, minimum=0.0, maximum=1.0)


def seed() -> int:
    return _int_setting("OPENAI_SEED", 42, minimum=0, maximum=2_147_483_647)


def max_output_tokens() -> int:
    return _int_setting("OPENAI_MAX_OUTPUT_TOKENS", 3000, minimum=1, maximum=100_000)


def reasoning_effort() -> str:
    return _choice_setting(
        "OPENAI_REASONING_EFFORT", "medium",
        {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
    )


def verbosity() -> str:
    return _choice_setting("OPENAI_VERBOSITY", "low", {"low", "medium", "high"})


def load_all() -> dict[str, Any]:
    """Load .env once and return every OpenAI-compatible setting as a dict."""
    load_env()
    return {
        "api_key": api_key(),
        "base_url": base_url(),
        "model": model(),
        "api_mode": api_mode(),
        "temperature": temperature(),
        "top_p": top_p(),
        "seed": seed(),
        "max_output_tokens": max_output_tokens(),
        "reasoning_effort": reasoning_effort(),
        "verbosity": verbosity(),
    }


__all__ = [
    "EnvConfigError",
    "ENV_CANDIDATES",
    "load_env",
    "load_all",
    "api_key",
    "base_url",
    "model",
    "api_mode",
    "temperature",
    "top_p",
    "seed",
    "max_output_tokens",
    "reasoning_effort",
    "verbosity",
]
