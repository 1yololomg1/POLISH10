"""Stage 4 validity. D11, D87, D44, D49, D50, D66, D26, C2, C3, UXC6, UXC9.

KEOUGH is not in this repo. Tests that need it are skipped and do not look
outside the repository.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy
import pytest

from polish.model.dataset import Curve, Dataset
from polish.model.states import (
    BlockType,
    GapClass,
    GapFillOutcome,
    OutputInclusion,
    SampleStatus,
    StageStatus,
    ValidationOutcome,
)
from polish.rules.loader import _validity_rule, load_rule, validity_rule, validity_rules
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, Runner, StageOutcome
from polish.stages.s00_ingest import IngestResult, ingest
from polish.stages.s01_depth import classify
from polish.stages.s01_depth import registered_stage as depth_stage
from polish.stages.s02_identify import identify
from polish.stages.s02_identify import registered_stage as identify_stage
from polish.stages.s03_units import interpret
from polish.stages.s03_units import registered_stage as units_stage
from polish.stages.s04_validity import (
    CONTRACT,
    GapRecord,
    SharedMatch,
    assess,
    is_structurally_non_fillable,
    legacy_depth_gap_label,
    registered_stage,
)

_WHEN = datetime.datetime(2026, 10, 14, tzinfo=datetime.timezone.utc)
_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"
_DENSITY_GROUPS = {
    "BULK_DENSITY": "TEST_DENSITY",
    "DENSITY_CORRECTION": "TEST_DENSITY",
    "PHOTOELECTRIC_FACTOR": "TEST_DENSITY",
}


def _write(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "well.las"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _las(specs: list[tuple[str, str]], columns: list[list[str]], null: str = "-999.25") -> str:
    count = len(columns[0])
    if any(len(column) != count for column in columns):
        raise AssertionError("columns differ in length")
    lines = ["DEPT.M : DEPTH"]
    for mnemonic, unit in specs:
        lines.append(f"{mnemonic}.{unit} : {mnemonic}")
    rows = []
    for index in range(count):
        rows.append(" ".join([str(float(index))] + [column[index] for column in columns]))
    stop = float(count - 1)
    return (
        "~VERSION INFORMATION\n"
        "VERS. 2.0 : CWLS\n"
        "WRAP. NO : WRAP\n"
        "~WELL INFORMATION\n"
        "STRT.M 0.0 : START\n"
        f"STOP.M {stop} : STOP\n"
        "STEP.M 1.0 : STEP\n"
        f"NULL. {null} : NULL\n"
        "~CURVE INFORMATION\n"
        + "\n".join(lines)
        + "\n~A\n"
        + "\n".join(rows)
        + "\n"
    )


def _pipeline(directory: Path, body: str):
    path = _write(directory, body)
    ingested = ingest(path, read_at=_WHEN)
    assert ingested.dataset is not None
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)
    depth = classify(ingested)
    assert depth.status is StageStatus.COMPLETE
    identified = identify(ingested, depth, local_tables_directory=directory / "no-local-tables")
    assert identified.status is StageStatus.COMPLETE
    units = interpret(ingested, identified)
    assert units.status is StageStatus.COMPLETE
    result = assess(ingested, depth, identified, units)
    assert result.status is StageStatus.COMPLETE
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
    return ingested, depth, identified, units, result


def _one(result, mnemonic: str):
    matches = [item for item in result.curves if item.mnemonic == mnemonic]
    assert len(matches) == 1, mnemonic
    return matches[0]


def _findings(result, block: BlockType, mnemonic: str):
    return [
        finding
        for finding in result.findings
        if finding.block_type is block and finding.evidence.as_dict().get("mnemonic") == mnemonic
    ]


def _exact_shared(gap, members):
    """Test definition of shared: every curve in the group is absent on the same indexes."""

    span = set(range(gap.sample_start, gap.sample_end))
    sharing = []
    for member in members:
        if not span.issubset(set(member.absent_indexes)):
            return None
        sharing.append(member.mnemonic)
    return SharedMatch(tuple(sharing), "exact sample-index overlap", "test definition of shared")


def _replace_sample(ingested: IngestResult, mnemonic: str, index: int, value: float) -> IngestResult:
    dataset = ingested.dataset
    assert dataset is not None
    curves = []
    for curve in dataset.curves:
        if curve.original_mnemonic != mnemonic:
            curves.append(curve)
            continue
        values = numpy.array(curve.values, dtype=numpy.float64, copy=True)
        values[index] = value
        curves.append(
            Curve(curve.ref, curve.original_mnemonic, curve.original_unit, curve.original_description, values)
        )
    replaced = Dataset(dataset.id, None, dataset.source_file, dataset.grid, tuple(curves), ())
    return IngestResult(
        ingested.source_file,
        replaced,
        ingested.findings,
        ingested.audit,
        ingested.null_sample_indexes,
        ingested.status,
    )


def test_contract_is_read_only_and_depends_on_earlier_stages():
    assert CONTRACT.stage_number == 4
    assert CONTRACT.name == "Validity"
    assert CONTRACT.alters_data is False
    assert CONTRACT.depends_on_stages == (0, 1, 2, 3)


def test_library_ships_the_stated_rules_and_rejects_a_missing_key():
    rules = {rule.id: rule for rule in validity_rules()}
    assert rules["SONIC_COMPRESSIONAL"].rule_type == "HARD_IMPOSSIBILITY"
    assert rules["SONIC_COMPRESSIONAL"].violation_effect == "INVALID"
    assert rules["GAMMA_RAY_TOTAL"].rule_type == "PLAUSIBILITY"
    assert rules["GAMMA_RAY_TOTAL"].violation_effect == "WARNING"
    assert rules["BULK_DENSITY"].violation_effect == "WARNING"
    assert rules["NEUTRON_POROSITY"].violation_effect == "WARNING"
    assert rules["PHOTOELECTRIC_FACTOR"].rule_type == "PLAUSIBILITY"
    assert rules["PHOTOELECTRIC_FACTOR"].violation_effect == "WARNING"
    for family in (
        "RESISTIVITY_DEEP",
        "RESISTIVITY_MEDIUM",
        "RESISTIVITY_SHALLOW",
        "RESISTIVITY_MICRO",
        "RESISTIVITY_LATEROLOG",
    ):
        assert rules[family].rule_type == "PLAUSIBILITY"
        assert rules[family].canonical_unit == "OHMM"
        assert rules[family].bound_values.minimum == -0.02
        assert rules[family].bound_values.maximum == 20000
    sp = rules["SPONTANEOUS_POTENTIAL"]
    assert sp.log_level is True
    assert sp.bound_values.outside_fraction_more_than is not None
    assert sp.bound_values.outside_fraction_more_than.numerator == 1
    assert sp.bound_values.outside_fraction_more_than.denominator == 5
    assert "Claude's reading of 'its samples', 14 Oct 2026, not the expert's words" in sp.source
    assert "family mapping proposed 14 Oct 2026, not confirmed by the domain expert" in sp.source
    finite = rules["non_finite_value"]
    assert finite.family is None
    assert finite.canonical_unit is None
    assert finite.rule_type == "DOMAIN_CONSTRAINT"
    assert finite.violation_effect == "INVALID"
    assert finite.bound_values.condition == "non_finite"
    assert finite.source == "definitional (plan fallback, owner 14 Oct 2026)"
    assert load_rule("depth.grid_rounding")["id"] == "depth.grid_rounding"
    with pytest.raises(KeyError):
        load_rule("not-a-rule")
    with pytest.raises(KeyError):
        validity_rule("not-a-rule")
    with pytest.raises(ValueError, match="missing"):
        _validity_rule({"id": "X"})


@pytest.mark.skip(
    reason=(
        "KEOUGH is not in the repo (checked 14 Oct 2026). Density samples above the "
        "tool's first measured depth, and the OD-11 interval, are not asserted. "
        "The file is not read from outside the repo."
    )
)
def test_keough_outside_interval_when_the_file_is_present():
    raise AssertionError("KEOUGH is not available.")


def test_curve_with_no_family_rule_is_not_passed(tmp_path: Path):
    _ingested, _depth, _identified, _units, result = _pipeline(
        tmp_path, _las([("CALI", "IN")], [["6.0", "8.0"]])
    )
    cali = _one(result, "CALI")
    assert cali.outcome is ValidationOutcome.NOT_ELIGIBLE
    assert cali.outcome is not ValidationOutcome.PASSED
    assert cali.rule_set_version is None
    assert cali.family_rules_applied == ()
    assert cali.ineligibility_reason == "no family rule"
    assert cali.non_finite_count == 0
    assert all(sample.status is SampleStatus.MEASURED for sample in cali.samples)
    notices = _findings(result, BlockType.INFORMATIONAL, "CALI")
    assert len(notices) == 1
    assert notices[0].evidence.as_dict()["reason"] == "no family rule"
    assert "eligibility_evidence" in notices[0].evidence.as_dict()


def test_non_finite_rule_alone_does_not_pass_and_flags_infinity(tmp_path: Path):
    ingested, depth, identified, units, result = _pipeline(
        tmp_path, _las([("CALI", "IN")], [["6.0", "8.0"]])
    )
    cali = _one(result, "CALI")
    assert cali.outcome is not ValidationOutcome.PASSED
    replaced = _replace_sample(ingested, "CALI", 0, math.inf)
    inf_result = assess(replaced, depth, identified, units)
    flagged = _one(inf_result, "CALI")
    assert flagged.outcome is ValidationOutcome.HARD_VIOLATION_EXCLUDED
    assert flagged.outcome is not ValidationOutcome.PASSED
    assert flagged.rule_set_version is None
    assert flagged.non_finite_count == 1
    assert flagged.samples[0].status is SampleStatus.INVALID
    assert flagged.samples[0].invalid_rules == ("non_finite_value",)
    assert flagged.samples[0].output_inclusion is OutputInclusion.PENDING_DECISION
    assert flagged.samples[1].status is SampleStatus.MEASURED
    hard = _findings(inf_result, BlockType.AMBIGUOUS_INTERPRETATION, "CALI")
    assert len(hard) == 1
    assert hard[0].evidence.as_dict()["rule_id"] == "non_finite_value"
    assert "non-finite value" in hard[0].message
    assert math.isinf(float(replaced.dataset.curves[1].values[0]))


def test_gapi_rule_is_not_applied_to_cps(tmp_path: Path):
    _ingested, _depth, _identified, _units, result = _pipeline(
        tmp_path, _las([("GR", "CPS")], [["5000", "10"]])
    )
    gr = _one(result, "GR")
    assert "GAMMA_RAY_TOTAL" not in gr.family_rules_applied
    assert gr.outcome is ValidationOutcome.NOT_ELIGIBLE
    assert gr.ineligibility_reason == "no exact conversion"
    assert all(sample.status is SampleStatus.MEASURED for sample in gr.samples)
    assert _findings(result, BlockType.RULE_DEPENDENT_WARNING, "GR") == []


def test_sp_flags_only_above_one_fifth_and_never_invalidates_a_sample(tmp_path: Path):
    exact = _pipeline(
        tmp_path / "exact",
        _las(
            [("SP", "MV")],
            [["100", "0", "0", "0", "0", "-999.25", "-999.25", "-999.25", "-999.25", "-999.25"]],
        ),
    )[4]
    sp = _one(exact, "SP")
    assert sp.outcome is ValidationOutcome.PASSED
    assert sp.rule_set_version == "1"
    assert _findings(exact, BlockType.RULE_DEPENDENT_WARNING, "SP") == []
    assert all(sample.status is not SampleStatus.INVALID for sample in sp.samples)
    assert sum(sample.status is SampleStatus.ABSENT for sample in sp.samples) == 5

    over = _pipeline(
        tmp_path / "over",
        _las(
            [("SP", "MV")],
            [["100", "100", "0", "0", "0", "-999.25", "-999.25", "-999.25", "-999.25", "-999.25"]],
        ),
    )[4]
    flagged = _one(over, "SP")
    assert flagged.outcome is ValidationOutcome.PLAUSIBILITY_WARNING
    assert all(sample.status is not SampleStatus.INVALID for sample in flagged.samples)
    warning = _findings(over, BlockType.INFORMATIONAL, "SP")
    assert len(warning) == 1
    assert _findings(over, BlockType.RULE_DEPENDENT_WARNING, "SP") == []
    assert "flagged for review" in warning[0].message
    assert all(sample.output_inclusion is None for sample in flagged.samples)
    evidence = warning[0].evidence.as_dict()
    assert evidence["log_level"] is True
    assert evidence["outside_count"] == 2
    assert evidence["non_absent_count"] == 5
    assert evidence["outside_fraction_more_than"] == "1/5"
    assert "depths" not in evidence


def test_dt_below_40_is_invalid_visible_and_unclipped(tmp_path: Path):
    ingested, _depth, _identified, _units, result = _pipeline(
        tmp_path, _las([("DT", "US/FT")], [["30", "40", "300", "301", "50"]])
    )
    dt = _one(result, "DT")
    assert [sample.status for sample in dt.samples] == [
        SampleStatus.INVALID,
        SampleStatus.MEASURED,
        SampleStatus.MEASURED,
        SampleStatus.INVALID,
        SampleStatus.MEASURED,
    ]
    assert dt.samples[0].output_inclusion is OutputInclusion.PENDING_DECISION
    assert dt.samples[3].output_inclusion is OutputInclusion.PENDING_DECISION
    assert dt.outcome is ValidationOutcome.HARD_VIOLATION_EXCLUDED
    stored = ingested.dataset.curves[1].values
    assert list(stored) == [30.0, 40.0, 300.0, 301.0, 50.0]
    assert stored.flags.writeable is False
    findings = _findings(result, BlockType.AMBIGUOUS_INTERPRETATION, "DT")
    assert len(findings) == 1
    evidence = findings[0].evidence.as_dict()
    assert evidence["rule_id"] == "SONIC_COMPRESSIONAL"
    assert evidence["output_inclusion"] == "PENDING_DECISION"
    assert json.loads(evidence["scope_options"]) == ["THIS_ITEM", "LOCAL_RULE"]
    assert json.loads(evidence["options"]) == ["KEEP", "EXCLUDE"]
    assert json.loads(evidence["depths"]) == [0.0, 3.0]
    assert evidence["sample_count"] == 2
    assert "Exclude these samples" in findings[0].message
    assert all(
        sample.output_inclusion is not OutputInclusion.EXCLUDED
        and sample.output_inclusion is not OutputInclusion.KEPT
        for sample in dt.samples
    )


def test_undeclared_sentinel_stays_measured(tmp_path: Path):
    ingested, _depth, _identified, _units, result = _pipeline(
        tmp_path,
        _las([("GR", "GAPI")], [["50", "-999.25"]], null="-1.0"),
    )
    gr = _one(result, "GR")
    assert gr.samples[1].status is SampleStatus.MEASURED
    assert gr.samples[1].status is not SampleStatus.ABSENT
    assert gr.outcome is ValidationOutcome.PLAUSIBILITY_WARNING
    assert float(ingested.dataset.curves[1].values[1]) == -999.25


def test_exact_conversion_is_required(tmp_path: Path):
    density, _depth, _identified, _units, result = _pipeline(
        tmp_path / "density",
        _las([("RHOB", "KG/M3")], [["2650", "1000"]]),
    )
    rhob = _one(result, "RHOB")
    assert rhob.family_rules_applied == ("BULK_DENSITY",)
    assert rhob.outcome is ValidationOutcome.PLAUSIBILITY_WARNING
    assert rhob.samples[0].status is SampleStatus.MEASURED
    assert rhob.samples[1].status is SampleStatus.MEASURED
    warning = _findings(result, BlockType.AMBIGUOUS_INTERPRETATION, "RHOB")
    assert json.loads(warning[0].evidence.as_dict()["sample_indexes"]) == [1]
    assert json.loads(warning[0].evidence.as_dict()["options"]) == ["KEEP", "EXCLUDE"]
    assert rhob.samples[1].status is SampleStatus.MEASURED
    assert rhob.samples[1].output_inclusion is OutputInclusion.PENDING_DECISION
    assert list(density.dataset.curves[1].values) == [2650.0, 1000.0]

    percent = _pipeline(tmp_path / "percent", _las([("NPHI", "PERCENT")], [["45", "45"]]))[4]
    nphi = _one(percent, "NPHI")
    assert nphi.outcome is ValidationOutcome.PASSED
    assert nphi.rule_set_version == "1"

    pu = _pipeline(tmp_path / "pu", _las([("NPHI", "PU")], [["0.2", "0.9"]]))[4]
    raw = _one(pu, "NPHI")
    assert raw.outcome is ValidationOutcome.NOT_ELIGIBLE
    assert raw.ineligibility_reason == "no exact conversion"
    assert raw.family_rules_applied == ()


def test_density_gap_shared_by_three_curves_lists_all_three(tmp_path: Path):
    _ingested, depth, identified, units, plain = _pipeline(tmp_path, _density_body())
    rhob = _one(plain, "RHOB")
    assert len(rhob.gaps) == 1
    assert rhob.gaps[0].classification.gap_class is GapClass.AMBIGUOUS
    assert "no co-tool evidence available" in rhob.gaps[0].classification.evidence
    assert rhob.gaps[0].fill_outcome is GapFillOutcome.NOT_REQUESTED
    assert is_structurally_non_fillable(rhob.gaps[0].classification.gap_class) is False
    assert _findings(plain, BlockType.INFORMATIONAL, "RHOB") == []

    def _refuse(_gap, _members):
        raise AssertionError("shared was called without a tool group")

    quiet = assess(_ingested, depth, identified, units, shared=_refuse)
    assert _one(quiet, "RHOB").gaps[0].classification.gap_class is GapClass.AMBIGUOUS

    shared = assess(_ingested, depth, identified, units, tool_groups=_DENSITY_GROUPS, shared=_exact_shared)
    for mnemonic in ("RHOB", "DRHO", "PEF"):
        curve = _one(shared, mnemonic)
        assert len(curve.gaps) == 1
        gap = curve.gaps[0]
        assert gap.classification.gap_class is GapClass.TOOL_ABSENCE_EVIDENCED
        assert gap.fill_outcome is GapFillOutcome.NOT_REQUESTED
        assert is_structurally_non_fillable(gap.classification.gap_class) is True
        assert "RHOB" in gap.classification.evidence
        assert "DRHO" in gap.classification.evidence
        assert "PEF" in gap.classification.evidence
        assert "not proof" in gap.classification.evidence
        assert gap.classification.confidence == "test definition of shared"
        notices = _findings(shared, BlockType.INFORMATIONAL, mnemonic)
        tool_notices = [item for item in notices if item.evidence.as_dict().get("gap_class") == "TOOL_ABSENCE_EVIDENCED"]
        assert len(tool_notices) == 1
        assert tool_notices[0].evidence.as_dict()["not_fillable"] is True
        assert set(json.loads(tool_notices[0].evidence.as_dict()["sharing_curves"])) == {"RHOB", "DRHO", "PEF"}
    assert "fillable" not in GapRecord.__dataclass_fields__


def test_outside_interval_and_summaries(tmp_path: Path):
    _ingested, _depth, _identified, _units, result = _pipeline(
        tmp_path,
        _las([("RHOB", "G/CC")], [["-999.25", "-999.25", "2.4", "2.4", "-999.25"]]),
    )
    rhob = _one(result, "RHOB")
    assert rhob.measured_interval is not None
    assert rhob.measured_interval.sample_start == 2
    assert rhob.measured_interval.sample_last == 3
    classes = [gap.classification.gap_class for gap in rhob.gaps]
    assert classes == [GapClass.OUTSIDE_INTERVAL, GapClass.OUTSIDE_INTERVAL]
    assert all(gap.fill_outcome is GapFillOutcome.NOT_REQUESTED for gap in rhob.gaps)
    assert all(is_structurally_non_fillable(gap_class) for gap_class in classes)
    assert _findings(result, BlockType.INFORMATIONAL, "RHOB") == []
    kinds = [(span.status, span.gap_class) for span in rhob.summaries]
    assert kinds == [
        (SampleStatus.ABSENT, GapClass.OUTSIDE_INTERVAL),
        (SampleStatus.MEASURED, None),
        (SampleStatus.ABSENT, GapClass.OUTSIDE_INTERVAL),
    ]


def test_legacy_depth_label_is_not_a_stage_4_class():
    assert legacy_depth_gap_label(1, 0.5) == "SMALL"
    assert legacy_depth_gap_label(100, 0.5) == "VERY_LARGE"
    assert "SMALL" not in {item.value for item in GapClass}


def test_runner_leaves_source_arrays_unchanged(tmp_path: Path):
    ingested, depth, identified, units, _result = _pipeline(
        tmp_path, _las([("GR", "GAPI")], [["40", "80"]])
    )
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)

    def stage0() -> RegisteredStage:
        contract = StageContract(
            0,
            "Ingest",
            StageRequires("Units stay as written.", "The file has been read.", ()),
            StageGuarantees("Unchanged.", "Unchanged.", "Unchanged."),
            ("The file cannot be read.",),
            (),
            False,
        )

        def run(_context: object) -> StageOutcome:
            return StageOutcome(StageStatus.COMPLETE, ())

        def unmet(_context: object) -> tuple[str, ...]:
            return ()

        return RegisteredStage(contract, run, unmet)

    runner = Runner(
        (
            stage0(),
            depth_stage(ingested),
            identify_stage(ingested, depth, tmp_path / "no-local-tables"),
            units_stage(ingested, identified),
            registered_stage(ingested, depth, identified, units),
        )
    )
    results = runner.run(ingested.dataset)
    assert [item.status for item in results] == [StageStatus.COMPLETE] * 5
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
    passed = _one(_result, "GR")
    assert passed.outcome is ValidationOutcome.PASSED
    assert passed.rule_set_version == "1"


def _density_body() -> str:
    null = "-999.25"
    present = ["2.3", "2.3", null, null, null, "2.4", "2.4"]
    correction = ["0.01", "0.01", null, null, null, "0.02", "0.02"]
    pef = ["3", "3", null, null, null, "4", "4"]
    gamma = ["80", "80", "80", "80", "80", "80", "80"]
    return _las(
        [("RHOB", "G/CC"), ("DRHO", "G/CC"), ("PEF", "B/E"), ("GR", "GAPI")],
        [present, correction, pef, gamma],
    )


def _fixture_report() -> dict[str, object]:
    """Per-fixture outcomes, gap classes, and samples each shipped rule flags."""

    report: dict[str, object] = {}
    for path in sorted(_FIXTURES.glob("*.las")):
        ingested = ingest(path, read_at=_WHEN)
        assert ingested.dataset is not None
        depth = classify(ingested)
        identified = identify(ingested, depth, local_tables_directory=_FIXTURES / "no-local-tables")
        units = interpret(ingested, identified)
        result = assess(ingested, depth, identified, units)
        outcomes: dict[str, int] = {}
        gaps: dict[str, int] = {}
        flags: dict[str, int] = {}
        for curve in result.curves:
            outcomes[curve.outcome.value] = outcomes.get(curve.outcome.value, 0) + 1
            for gap in curve.gaps:
                gaps[gap.classification.gap_class.value] = gaps.get(gap.classification.gap_class.value, 0) + 1
            for sample in curve.samples:
                for rule_id in sample.invalid_rules:
                    flags[rule_id] = flags.get(rule_id, 0) + 1
        for finding in result.findings:
            evidence = finding.evidence.as_dict()
            if "rule_id" not in evidence:
                continue
            rule_id = str(evidence["rule_id"])
            if finding.block_type is BlockType.INFORMATIONAL and evidence.get("log_level") is True:
                flags[rule_id] = flags.get(rule_id, 0) + int(evidence["outside_count"])
            elif finding.block_type is BlockType.AMBIGUOUS_INTERPRETATION and evidence.get("violation_effect") == "WARNING":
                flags[rule_id] = flags.get(rule_id, 0) + int(evidence["sample_count"])
        report[path.name] = {
            "outcomes": outcomes,
            "gaps": gaps,
            "flags": flags,
            "non_finite": sum(curve.non_finite_count for curve in result.curves),
        }
    return report


def test_fixtures_do_not_invent_od13_classes_and_match_the_recorded_flags():
    report = _fixture_report()
    for name, item in report.items():
        assert isinstance(item, dict)
        gaps = item["gaps"]
        assert isinstance(gaps, dict)
        assert "DROPOUT" not in gaps
        assert "TOOL_ABSENCE_EVIDENCED" not in gaps
        assert item["non_finite"] == 0, name
    assert report == _FIXTURE_FLAGS


# Recorded by running Stage 4 on every tests/fixtures/*.las file.
# Differences from build plan 1.0.1 (15 Sept) are explained in the P12 report.
_FIXTURE_FLAGS: dict[str, object] = {
    "159-19A_LFP.las": {
        "flags": {},
        "gaps": {"AMBIGUOUS": 109, "OUTSIDE_INTERVAL": 104},
        "non_finite": 0,
        "outcomes": {"NOT_ELIGIBLE": 171},
    },
    "30_3-5 S.las": {
        "flags": {"BULK_DENSITY": 4, "NEUTRON_POROSITY": 1, "PHOTOELECTRIC_FACTOR": 1},
        "gaps": {"AMBIGUOUS": 7, "OUTSIDE_INTERVAL": 24},
        "non_finite": 0,
        "outcomes": {"NOT_ELIGIBLE": 11, "PASSED": 2, "PLAUSIBILITY_WARNING": 3},
    },
    "34_10-19.las": {
        "flags": {"BULK_DENSITY": 119, "NEUTRON_POROSITY": 45, "SPONTANEOUS_POTENTIAL": 6703},
        "gaps": {"AMBIGUOUS": 21, "OUTSIDE_INTERVAL": 30},
        "non_finite": 0,
        "outcomes": {"NOT_ELIGIBLE": 12, "PASSED": 3, "PLAUSIBILITY_WARNING": 3},
    },
    "WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las": {
        "flags": {"RESISTIVITY_DEEP": 40},
        "gaps": {"AMBIGUOUS": 21, "OUTSIDE_INTERVAL": 42},
        "non_finite": 0,
        "outcomes": {"NOT_ELIGIBLE": 17, "PASSED": 5, "PLAUSIBILITY_WARNING": 1},
    },
    "new_file_null_minus_999.las": {
        "flags": {},
        "gaps": {},
        "non_finite": 0,
        "outcomes": {"NOT_ELIGIBLE": 1, "PASSED": 1},
    },
}
