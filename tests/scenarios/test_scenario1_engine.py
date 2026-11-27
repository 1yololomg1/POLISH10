"""Scenario 1 at engine level. Supplement Part F.

A density curve is absent over an interval. With an injected test tool group
and a test definition of shared, the gap is tool-absence evidence: recorded,
non-fillable, and present in the findings. The same gap with no tool group
is AMBIGUOUS. OD-13 is unset on the shipped path.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

from polish.model.states import GapClass, GapFillOutcome, StageStatus
from polish.stages.s00_ingest import ingest
from polish.stages.s01_depth import classify
from polish.stages.s02_identify import identify
from polish.stages.s03_units import interpret
from polish.stages.s04_validity import SharedMatch, assess, is_structurally_non_fillable

_WHEN = datetime.datetime(2026, 10, 14, tzinfo=datetime.timezone.utc)
_GROUPS = {
    "BULK_DENSITY": "TEST_DENSITY",
    "DENSITY_CORRECTION": "TEST_DENSITY",
    "PHOTOELECTRIC_FACTOR": "TEST_DENSITY",
}


def _body() -> str:
    null = "-999.25"
    rhob = ["2.3", "2.3", null, null, null, "2.4", "2.4"]
    drho = ["0.01", "0.01", null, null, null, "0.02", "0.02"]
    pef = ["3", "3", null, null, null, "4", "4"]
    rows = []
    for index in range(len(rhob)):
        rows.append(f"{float(index)} {rhob[index]} {drho[index]} {pef[index]}")
    return (
        "~VERSION INFORMATION\n"
        "VERS. 2.0 : CWLS\n"
        "WRAP. NO : WRAP\n"
        "~WELL INFORMATION\n"
        "STRT.M 0.0 : START\n"
        "STOP.M 6.0 : STOP\n"
        "STEP.M 1.0 : STEP\n"
        "NULL. -999.25 : NULL\n"
        "~CURVE INFORMATION\n"
        "DEPT.M : DEPTH\n"
        "RHOB.G/CC : RHOB\n"
        "DRHO.G/CC : DRHO\n"
        "PEF.B/E : PEF\n"
        "~A\n"
        + "\n".join(rows)
        + "\n"
    )


def _exact_shared(gap, members):
    span = set(range(gap.sample_start, gap.sample_end))
    sharing = []
    for member in members:
        if not span.issubset(set(member.absent_indexes)):
            return None
        sharing.append(member.mnemonic)
    return SharedMatch(tuple(sharing), "exact sample-index overlap", "test definition of shared")


def _run(directory: Path):
    path = directory / "well.las"
    path.write_text(_body(), encoding="utf-8", newline="\n")
    ingested = ingest(path, read_at=_WHEN)
    assert ingested.dataset is not None
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)
    depth = classify(ingested)
    identified = identify(ingested, depth, local_tables_directory=directory / "no-local-tables")
    units = interpret(ingested, identified)
    assert depth.status is StageStatus.COMPLETE
    assert identified.status is StageStatus.COMPLETE
    assert units.status is StageStatus.COMPLETE
    return ingested, depth, identified, units, before


def _rhob(result):
    matches = [item for item in result.curves if item.mnemonic == "RHOB"]
    assert len(matches) == 1
    return matches[0]


def test_scenario1_tool_absence_stays_empty_and_is_recorded(tmp_path: Path):
    ingested, depth, identified, units, before = _run(tmp_path)
    result = assess(ingested, depth, identified, units, tool_groups=_GROUPS, shared=_exact_shared)
    curve = _rhob(result)
    assert len(curve.gaps) == 1
    gap = curve.gaps[0]
    assert gap.sample_start == 2
    assert gap.sample_end == 5
    assert gap.classification.gap_class is GapClass.TOOL_ABSENCE_EVIDENCED
    assert gap.fill_outcome is GapFillOutcome.NOT_REQUESTED
    assert is_structurally_non_fillable(gap.classification.gap_class) is True
    assert gap.classification.evidence.strip() != ""
    assert gap.classification.confidence == "test definition of shared"
    assert "not proof" in gap.classification.evidence
    notices = [
        finding
        for finding in result.findings
        if finding.evidence.as_dict().get("mnemonic") == "RHOB"
        and finding.evidence.as_dict().get("gap_class") == "TOOL_ABSENCE_EVIDENCED"
    ]
    assert len(notices) == 1
    evidence = notices[0].evidence.as_dict()
    assert evidence["not_fillable"] is True
    assert evidence["confidence"] == "test definition of shared"
    assert "Not fillable" in notices[0].message
    assert tuple(item.values.tobytes() for item in ingested.dataset.curves) == before
    stored = ingested.dataset.curves[1].values
    assert all(math.isnan(float(stored[index])) for index in range(2, 5))


def test_scenario1_without_a_tool_group_is_ambiguous_and_raises_no_finding(tmp_path: Path):
    ingested, depth, identified, units, _before = _run(tmp_path)

    def _refuse(_gap, _members):
        raise AssertionError("shared was called with no tool group")

    result = assess(ingested, depth, identified, units, shared=_refuse)
    gap = _rhob(result).gaps[0]
    assert gap.classification.gap_class is GapClass.AMBIGUOUS
    assert "no co-tool evidence available" in gap.classification.evidence
    assert gap.fill_outcome is GapFillOutcome.NOT_REQUESTED
    assert is_structurally_non_fillable(gap.classification.gap_class) is False
    assert all(finding.evidence.as_dict().get("gap_class") != "TOOL_ABSENCE_EVIDENCED" for finding in result.findings)
    assert all(finding.evidence.as_dict().get("not_fillable") is not True for finding in result.findings)

