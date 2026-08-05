from __future__ import annotations

from learning_v2.readiness_audit import _pytest_summary


def test_pytest_summary_reads_testsuites(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    assert _pytest_summary(report) == {"tests": 3, "failures": 0, "errors": 0, "skipped": 1}


def test_pytest_summary_fails_closed_when_missing(tmp_path):
    assert _pytest_summary(tmp_path / "missing.xml")["tests"] == 0
