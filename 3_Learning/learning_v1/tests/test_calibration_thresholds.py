import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

from calibrate_thresholds import per_rule_thresholds  # noqa: E402
from wcag_rules import RULE_INDEX  # noqa: E402


def test_per_rule_thresholds_respect_precision_floor():
    rule_idx = RULE_INDEX["color-contrast"]
    probs = torch.zeros(6, 46)
    labels = torch.zeros(6, 46)
    probs[:, rule_idx] = torch.tensor([0.95, 0.90, 0.70, 0.40, 0.30, 0.20])
    labels[:, rule_idx] = torch.tensor([1, 1, 0, 1, 0, 0])
    records = [
        {
            "rule_probs": probs,
            "rule_labels": labels,
        }
    ]

    thresholds = per_rule_thresholds(records, [0.2, 0.4, 0.7, 0.9], precision_floor=0.55)
    selected = thresholds["color-contrast"]

    assert selected["precision_floor_met"] is True
    assert selected["precision"] >= 0.55
    assert selected["rule_threshold"] == 0.4
    assert selected["recall"] == 1.0
