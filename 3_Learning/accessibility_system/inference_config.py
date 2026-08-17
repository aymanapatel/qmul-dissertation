"""Model-bundle configuration owned by the accessibility-system API.

The trained checkpoints remain learning_v2 outputs, but the API—not the
learning_v2 inference helper—selects which completed bundle it serves.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .env_import import ENV_CANDIDATES


def _load_local_environment() -> None:
    """Load the same optional .env locations as the generation configuration."""

    for candidate in ENV_CANDIDATES:
        if candidate.is_file():
            load_dotenv(dotenv_path=candidate, override=True)
            return


_load_local_environment()


LEARNING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = LEARNING_ROOT / "learning_v2/artifacts_evidence-v3.0"


def _path_setting(name: str, default: Path) -> Path:
    """Return an optional absolute or working-directory-relative path override."""

    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


# Keep these paths here so application deployments do not silently inherit
# learning_v2.live_inference's separate defaults.
DEFAULT_PHASE5 = _path_setting(
    "ACCESSIBILITY_PHASE5_DIR",
    DEFAULT_TRAINING_ROOT / "phase_5_multiview_final_v2",
)
DEFAULT_FUSION_POLICY = _path_setting(
    "ACCESSIBILITY_FUSION_POLICY",
    DEFAULT_TRAINING_ROOT / "phase_6_7_final/phase_6_fusion_policy.json",
)


__all__ = ["DEFAULT_PHASE5", "DEFAULT_FUSION_POLICY", "DEFAULT_TRAINING_ROOT"]
