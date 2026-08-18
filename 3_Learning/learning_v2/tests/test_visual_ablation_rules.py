import json
from argparse import Namespace

from learning_v2.visual_ablation import _completed_result, parse_args


def test_visual_ablation_defaults_to_phase5_visual_rule(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "visual_ablation",
            "--cache-dir", "cache",
            "--split", "split.json",
            "--output-dir", "output",
        ],
    )

    args = parse_args()

    assert args.rule_ids == ["color-contrast"]


def test_resume_loads_only_complete_compatible_run(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{}', encoding="utf-8")
    args = Namespace(
        output_dir=tmp_path / "output", split=split, epochs=20, patience=5,
        batch_size=1, hidden=64, layers=2, heads=4, dropout=0.2, lr=5e-4,
        negative_ratio=8.0, minimum_negatives=256, precision_floor=0.8,
        device="mps",
    )
    run_dir = args.output_dir / "seed_41/rendered-visual/mlp"
    run_dir.mkdir(parents=True)
    assert _completed_result(args, 41, "full", "mlp", ["color-contrast"]) is None

    for filename in ("best_model.pt", "history.json"):
        (run_dir / filename).write_text('{}', encoding="utf-8")
    (run_dir / "calibration.json").write_text(json.dumps({
        "node": {"precision_floor_met": False},
        "page": {"precision_floor_met": True},
        "rules": {"color-contrast": {"precision_floor_met": False}},
    }), encoding="utf-8")
    metrics = {"rule_f1": 0.5}
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "graph_source": "rendered-visual",
        "architecture": "mlp",
        "feature_variant": "full",
        "split_mode": "governed",
        "split_provenance": {
            "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        },
        "training_config": {
            "epochs": 20, "patience": 5, "batch_size": 1, "hidden": 64,
            "layers": 2, "heads": 4, "dropout": 0.2, "lr": 5e-4,
            "negative_ratio": 8.0, "minimum_negatives": 256,
            "precision_floor": 0.8, "device": "mps",
        },
        "rules": [{"rule_id": "color-contrast"}],
        "best_epoch": 9,
        "stopped_early": True,
    }), encoding="utf-8")

    result = _completed_result(args, 41, "full", "mlp", ["color-contrast"])
    assert result["best_epoch"] == 9
    assert result["test"] == metrics
    assert result["calibration_floor_policy"] == "validation_f1_fallback"
    assert result["unmet_precision_floor"] == ["node", "color-contrast"]
