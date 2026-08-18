"""Registry integrity and Plan_v3 exit-gate tests."""

import json
from pathlib import Path

import pytest

from learning_v2.catalog import CriterionRegistry, FamilyRegistry

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.fixture(scope="module")
def criteria() -> CriterionRegistry:
    return CriterionRegistry.load(CONFIGS / "wcag_criteria.json")


@pytest.fixture(scope="module")
def families() -> FamilyRegistry:
    return FamilyRegistry.load(CONFIGS / "wcag_label_families.json")


class TestCriterionRegistry:
    def test_total_criteria_count(self, criteria):
        assert len(criteria) == 87

    def test_active_non_media_count(self, criteria):
        active_non_media = [c for c in criteria.criteria.values() if c.status == "active" and c.scope != "excluded"]
        assert len(active_non_media) == 77

    def test_legacy_count(self, criteria):
        legacy = [c for c in criteria.criteria.values() if c.status == "legacy"]
        assert len(legacy) == 1
        assert legacy[0].criterion_id == "4.1.1"

    def test_excluded_count(self, criteria):
        excluded = [c for c in criteria.criteria.values() if c.scope == "excluded"]
        assert len(excluded) == 9
        assert all(c.criterion_id.startswith("1.2.") for c in excluded)

    def test_core_count(self, criteria):
        core = [c for c in criteria.criteria.values() if c.scope == "core"]
        assert len(core) == 18

    def test_control_count(self, criteria):
        control = [c for c in criteria.criteria.values() if c.scope == "control"]
        assert len(control) == 4
        control_ids = {c.criterion_id for c in control}
        assert control_ids == {"1.3.5", "2.4.2", "3.1.1", "3.1.2"}

    def test_registry_validation(self, criteria):
        errors = criteria.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_all_criteria_have_level(self, criteria):
        for cid, record in criteria.criteria.items():
            assert record.level in ("A", "AA", "AAA", "unknown"), f"{cid}: invalid level {record.level}"

    def test_all_criteria_have_detector(self, criteria):
        for cid, record in criteria.criteria.items():
            assert record.primary_detector, f"{cid}: missing primary_detector"

    def test_all_criteria_have_automation(self, criteria):
        for cid, record in criteria.criteria.items():
            assert record.automation in ("automated", "assisted", "manual"), f"{cid}: invalid automation {record.automation}"

    def test_axe_rule_mapping(self, criteria):
        # 1.1.1 should have multiple axe rules
        record = criteria["1.1.1"]
        assert len(record.axe_rule_ids) > 0
        assert "image-alt" in record.axe_rule_ids

    def test_412_criterion(self, criteria):
        record = criteria["4.1.2"]
        assert record.name == "Name, Role, Value"
        assert record.scope == "core"
        assert len(record.axe_rule_ids) > 0

    def test_411_is_legacy(self, criteria):
        record = criteria["4.1.1"]
        assert record.status == "legacy"

    def test_round_trip(self, criteria, tmp_path):
        out = tmp_path / "round_trip.json"
        criteria.save(out)
        loaded = CriterionRegistry.load(out)
        assert len(loaded) == len(criteria)
        for cid in criteria.criteria:
            assert loaded[cid].to_dict() == criteria[cid].to_dict()


class TestFamilyRegistry:
    def test_family_count(self, families):
        assert len(families) == 10

    def test_family_validation(self, families):
        errors = families.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_expected_families(self, families):
        expected = {
            "missing-accessible-name",
            "broken-form-labelling",
            "low-contrast",
            "keyboard-focus-issue",
            "poor-semantic-structure",
            "unclear-link-button-purpose",
            "dynamic-content-not-announced",
            "touch-pointer-target-issue",
            "media-alternative-issue",
            "authentication-error-prevention-issue",
        }
        assert set(families.families.keys()) == expected

    def test_all_families_have_criteria(self, families):
        for fid, record in families.families.items():
            assert len(record.wcag_criteria) > 0, f"{fid}: no criteria"

    def test_round_trip(self, families, tmp_path):
        out = tmp_path / "round_trip.json"
        families.save(out)
        loaded = FamilyRegistry.load(out)
        assert len(loaded) == len(families)
        for fid in families.families:
            assert loaded[fid].to_dict() == families[fid].to_dict()


class TestCrossRegistry:
    def test_core_criteria_have_family_links(self, criteria, families):
        core = [c for c in criteria.criteria.values() if c.scope == "core"]
        for c in core:
            if c.issue_family_ids:
                for fid in c.issue_family_ids:
                    assert fid in families, f"Criterion {c.criterion_id} references unknown family {fid}"

    def test_family_criteria_exist(self, criteria, families):
        for fid, family in families.families.items():
            for cid in family.wcag_criteria:
                assert cid in criteria, f"Family {fid} references unknown criterion {cid}"

    def test_core_criteria_cover_four_families(self, criteria, families):
        core_families = set()
        for c in criteria.criteria.values():
            if c.scope == "core":
                core_families.update(c.issue_family_ids)
        assert len(core_families) >= 4, f"Core scope should cover at least 4 families, got {len(core_families)}"
