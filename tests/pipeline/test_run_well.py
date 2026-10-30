"""run_well applies saved answers in dependency order and writes nothing. D98, D48, C5.

APPLE is an unrecognised curve whose unit is API. Shipped spellings give that
text two meanings, GAPI (gamma_api) and API (api_gravity). GAMMA_RAY_TOTAL
accepts gamma_api and count_rate. BULK_DENSITY accepts density.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polish.decisions.apply import apply_decision, listed_options
from polish.decisions.session import outstanding, save_session, session_path
from polish.model.mapping import roles_stated_by_family
from polish.model.states import CurveRole, UnitInterpretation, UserAction
from polish.provenance.describe import _continue
from polish.stages.s00_ingest import UnclassifiedIngest, ingest
from polish.tables.loader import load_shipped_tables
from polish.pipeline.run import run_well

_WHEN = datetime(2026, 10, 29, 18, 0, tzinfo=timezone.utc)
_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    tables = tmp_path / "tables"
    app = tmp_path / "app"
    tables.mkdir()
    app.mkdir()
    return tables, app


def _assert_empty(tables: Path, app: Path) -> None:
    assert list(tables.iterdir()) == []
    assert list(app.iterdir()) == []


def _las(specs: list[tuple[str, str]]) -> str:
    lines = ["DEPT.M : DEPTH"]
    for mnemonic, unit in specs:
        lines.append(f"{mnemonic}.{unit} : {mnemonic}")
    curves = "\n".join(lines) + "\n"
    width = 1 + len(specs)
    first = " ".join(["0.0"] + ["1.0"] * (width - 1))
    second = " ".join(["1.0"] + ["2.0"] * (width - 1))
    return (
        "~VERSION INFORMATION\n"
        "VERS. 2.0 : CWLS\n"
        "WRAP. NO : WRAP\n"
        "~WELL INFORMATION\n"
        "STRT.M 0.0 : START\n"
        "STOP.M 1.0 : STOP\n"
        "STEP.M 1.0 : STEP\n"
        "NULL. -999.25 : NULL\n"
        "~CURVE INFORMATION\n"
        f"{curves}"
        "~A\n"
        f"{first}\n"
        f"{second}\n"
    )


def _well(tmp_path: Path, specs: list[tuple[str, str]]) -> tuple[Path, Path, Path]:
    tables, app = _roots(tmp_path)
    path = tmp_path / "well.las"
    path.write_text(_las(specs), encoding="utf-8", newline="\n")
    return path, tables, app


def _run(path: Path, tables: Path, app: Path):
    return run_well(path, read_at=_WHEN, local_tables_directory=tables, app_data=app)


def _signature(result: object):
    if result is None:
        return None
    return (result.status, tuple(item.id for item in result.findings))


def _finding(result, check: str, mnemonic: str):
    found = [
        item
        for item in result.findings
        if item.check == check and item.evidence.as_dict().get("mnemonic") == mnemonic
    ]
    assert len(found) == 1, (check, mnemonic, [(item.check, item.evidence.as_dict().get("mnemonic")) for item in result.findings])
    return found[0]


def _curve(result, mnemonic: str):
    found = [item for item in result.curves if item.mnemonic == mnemonic]
    assert len(found) == 1, mnemonic
    return found[0]


def _ids(records) -> tuple[str, ...]:
    return tuple(item.decision_id for item in records)


def _outstanding_ids(session) -> set[str]:
    return {item.id for group in outstanding(session).values() for item in group}


def _reported(load, decision_id: str):
    return [item for item in load.reported if item.evidence.as_dict().get("decision_id") == decision_id]


def _stated_role(family_id: str) -> str:
    stated = roles_stated_by_family(load_shipped_tables(), family_id)
    assert len(stated) == 1, (family_id, stated)
    return stated[0].value


def _map(session, finding, family_id: str, role: str | None = None):
    chosen = _stated_role(family_id) if role is None else role
    return apply_decision(
        session,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason=f"it behaves as {family_id}",
        family_id=family_id,
        role=chosen,
    )


def _choose(session, finding, option: str, reason: str):
    return apply_decision(
        session,
        finding,
        UserAction.CHOOSE_OPTION,
        reason=reason,
        chosen_option=option,
    )


def _folder_snapshot(directory: Path) -> dict[str, int]:
    return {path.name: path.stat().st_mtime_ns for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.las")), ids=lambda item: item.name)
def test_a_fixture_with_no_session_matches_the_unanswered_stages(path: Path, tmp_path: Path):
    tables, app = _roots(tmp_path)
    before = _folder_snapshot(path.parent)
    assert session_path(path).is_file() is False
    try:
        ingested = ingest(path, read_at=_WHEN)
    except UnclassifiedIngest:
        with pytest.raises(UnclassifiedIngest):
            _run(path, tables, app)
    else:
        result = _run(path, tables, app)
        depth, identification, units, validity = _continue(ingested, tables)
        assert _signature(result.ingest) == _signature(ingested)
        assert _signature(result.depth) == _signature(depth)
        assert _signature(result.identification) == _signature(identification)
        assert _signature(result.units) == _signature(units)
        assert _signature(result.validity) == _signature(validity)
        assert result.applied == ()
        assert result.reasked == ()
    assert _folder_snapshot(path.parent) == before
    assert session_path(path).is_file() is False
    _assert_empty(tables, app)


def test_a_missing_file_is_refused(tmp_path: Path):
    tables, app = _roots(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_well(tmp_path / "missing.las", read_at=_WHEN, local_tables_directory=tables, app_data=app)
    _assert_empty(tables, app)


def test_a_saved_unit_choice_is_applied_and_its_question_is_not_outstanding(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API")])
    first = _run(path, tables, app)
    question = _finding(first.units, "unit_unresolved", "APPLE")
    assert "GAPI" in listed_options(question)
    saved = _choose(first.session_load.session, question, "GAPI", "gamma api units")
    save_session(saved)
    result = _run(path, tables, app)
    decision_id = saved.decisions[-1].decision_id
    assert _curve(result.units, "APPLE").decision_id == decision_id
    assert not any(item.check == "unit_unresolved" for item in result.units.findings)
    assert _ids(result.applied) == (decision_id,)
    assert result.reasked == ()
    assert _reported(result.session_load, decision_id) == []
    assert question.id not in _outstanding_ids(result.session_load.session)
    _assert_empty(tables, app)


def test_a_saved_mapping_is_applied_and_the_notice_is_not_raised(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API")])
    first = _run(path, tables, app)
    notice = _finding(first.identification, "mnemonic_unrecognised", "APPLE")
    saved = _map(first.session_load.session, notice, "BULK_DENSITY")
    save_session(saved)
    result = _run(path, tables, app)
    decision_id = saved.decisions[-1].decision_id
    apple = _curve(result.identification, "APPLE")
    assert apple.decision_id == decision_id
    assert apple.family_id == "BULK_DENSITY"
    assert apple.role is CurveRole(saved.decisions[-1].role)
    assert not any(
        item.check == "mnemonic_unrecognised" and item.evidence.as_dict().get("mnemonic") == "APPLE"
        for item in result.identification.findings
    )
    assert _ids(result.applied) == (decision_id,)
    assert result.reasked == ()
    _assert_empty(tables, app)


def test_a_saved_pass_choice_is_applied_and_the_question_is_not_raised(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("GR_MAIN", "GAPI"), ("GR_REF", "GAPI")])
    first = _run(path, tables, app)
    question = _finding_check(first.identification, "pass_default")
    assert "main pass" in listed_options(question)
    saved = _choose(first.session_load.session, question, "main pass", "use the main pass")
    save_session(saved)
    result = _run(path, tables, app)
    decision_id = saved.decisions[-1].decision_id
    assert len(result.identification.default_passes) == 1
    assert result.identification.default_passes[0].decision_id == decision_id
    assert not any(item.check == "pass_default" for item in result.identification.findings)
    assert _ids(result.applied) == (decision_id,)
    assert result.reasked == ()
    _assert_empty(tables, app)


def test_a_unit_question_asked_only_after_a_mapping_applies_both_answers(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API")])
    first = _run(path, tables, app)
    notice = _finding(first.identification, "mnemonic_unrecognised", "APPLE")
    mapped = _map(first.session_load.session, notice, "BULK_DENSITY")
    save_session(mapped)
    asked = _run(path, tables, app)
    question = _finding(asked.units, "unit_unresolved", "APPLE")
    assert question.evidence.as_dict()["family_id"] == "BULK_DENSITY"
    assert "GAPI" in listed_options(question)
    saved = _choose(asked.session_load.session, question, "GAPI", "gamma api on the mapped curve")
    save_session(saved)
    result = _run(path, tables, app)
    mapping_id = mapped.decisions[-1].decision_id
    unit_id = saved.decisions[-1].decision_id
    assert _ids(result.applied) == (mapping_id, unit_id)
    assert result.reasked == ()
    assert _curve(result.identification, "APPLE").decision_id == mapping_id
    assert _curve(result.identification, "APPLE").family_id == "BULK_DENSITY"
    apple = _curve(result.units, "APPLE")
    assert apple.decision_id == unit_id
    assert apple.outcome is UnitInterpretation.INCOMPATIBLE_WITH_FAMILY
    assert apple.canonical_unit == "GAPI"
    raised = [item.check for item in result.units.findings if item.evidence.as_dict().get("mnemonic") == "APPLE"]
    assert raised == ["unit_incompatible_with_family"]
    _assert_empty(tables, app)


def test_a_unit_choice_recorded_before_its_mapping_is_evidence_changed(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API")])
    first = _run(path, tables, app)
    question = _finding(first.units, "unit_unresolved", "APPLE")
    notice = _finding(first.identification, "mnemonic_unrecognised", "APPLE")
    chosen = _choose(first.session_load.session, question, "GAPI", "gamma api before the family")
    saved = _map(chosen, notice, "BULK_DENSITY")
    save_session(saved)
    result = _run(path, tables, app)
    unit_id = saved.decisions[0].decision_id
    mapping_id = saved.decisions[1].decision_id
    assert _ids(result.applied) == (mapping_id,)
    assert result.reasked == ()
    again = _finding(result.units, "unit_unresolved", "APPLE")
    assert again.evidence.as_dict()["family_id"] == "BULK_DENSITY"
    assert _curve(result.units, "APPLE").decision_id is None
    reported = _reported(result.session_load, unit_id)
    assert len(reported) == 1
    assert reported[0].check == "session_decision_evidence_changed"
    _assert_empty(tables, app)


def test_a_unit_choice_removed_by_a_mapping_is_unmatched(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API")])
    first = _run(path, tables, app)
    question = _finding(first.units, "unit_unresolved", "APPLE")
    notice = _finding(first.identification, "mnemonic_unrecognised", "APPLE")
    chosen = _choose(first.session_load.session, question, "GAPI", "gamma api before the family")
    saved = _map(chosen, notice, "GAMMA_RAY_TOTAL")
    save_session(saved)
    result = _run(path, tables, app)
    unit_id = saved.decisions[0].decision_id
    mapping_id = saved.decisions[1].decision_id
    assert _ids(result.applied) == (mapping_id,)
    assert unit_id not in _ids(result.applied)
    assert unit_id not in _ids(result.reasked)
    assert _curve(result.identification, "APPLE").family_id == "GAMMA_RAY_TOTAL"
    assert _curve(result.units, "APPLE").decision_id is None
    assert not any(
        item.check == "unit_unresolved" and item.evidence.as_dict().get("mnemonic") == "APPLE"
        for item in result.units.findings
    )
    reported = _reported(result.session_load, unit_id)
    assert len(reported) == 1
    assert reported[0].check == "session_decision_unmatched"
    _assert_empty(tables, app)


def test_a_mapping_whose_family_still_rejects_the_unit_is_reasked(tmp_path: Path):
    stated = roles_stated_by_family(load_shipped_tables(), "SONIC_COMPRESSIONAL")
    assert CurveRole.MEASURED_LOG in stated
    path, tables, app = _well(tmp_path, [("RHOB", "GAPI")])
    first = _run(path, tables, app)
    warning = _finding(first.units, "unit_incompatible_with_family", "RHOB")
    assert warning.evidence.as_dict()["family_id"] == "BULK_DENSITY"
    saved = _map(first.session_load.session, warning, "SONIC_COMPRESSIONAL", role=CurveRole.MEASURED_LOG.value)
    save_session(saved)
    result = _run(path, tables, app)
    decision_id = saved.decisions[-1].decision_id
    assert _ids(result.reasked) == (decision_id,)
    assert result.applied == ()
    rhob = _curve(result.identification, "RHOB")
    assert rhob.family_id == "SONIC_COMPRESSIONAL"
    assert rhob.decision_id == decision_id
    reported = _reported(result.session_load, decision_id)
    assert len(reported) == 1
    assert reported[0].check == "session_decision_evidence_changed"
    _assert_empty(tables, app)


def test_a_changed_file_applies_nothing_and_reports_the_hold(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API")])
    first = _run(path, tables, app)
    notice = _finding(first.identification, "mnemonic_unrecognised", "APPLE")
    saved = apply_decision(first.session_load.session, notice, UserAction.DEFER, reason="not now")
    save_session(saved)
    text = path.read_text(encoding="utf-8")
    edited = text.replace(": APPLE\n", ": APPLE edited\n", 1)
    assert edited != text
    path.write_text(edited, encoding="utf-8", newline="\n")
    result = _run(path, tables, app)
    ingested = ingest(path, read_at=_WHEN)
    depth, identification, units, validity = _continue(ingested, tables)
    assert _signature(result.ingest) == _signature(ingested)
    assert _signature(result.depth) == _signature(depth)
    assert _signature(result.identification) == _signature(identification)
    assert _signature(result.units) == _signature(units)
    assert _signature(result.validity) == _signature(validity)
    assert result.applied == ()
    assert result.reasked == ()
    assert any(item.check == "session_file_changed" for item in result.session_load.reported)
    assert session_path(path).is_file()
    assert list(path.parent.glob("*.polish-session.*.json")) == []
    _assert_empty(tables, app)


def test_defer_and_acknowledge_are_matched_and_not_applied(tmp_path: Path):
    path, tables, app = _well(tmp_path, [("APPLE", "API"), ("APPLE", "API")])
    first = _run(path, tables, app)
    duplicate = _finding(first.identification, "mnemonic_duplicated", "APPLE")
    warnings = [
        item
        for item in first.identification.findings
        if item.check == "mnemonic_unrecognised" and item.evidence.as_dict().get("mnemonic") == "APPLE"
    ]
    assert len(warnings) == 2
    warning = warnings[0]
    session = apply_decision(first.session_load.session, duplicate, UserAction.ACKNOWLEDGE)
    session = apply_decision(session, warning, UserAction.DEFER, reason="not now")
    save_session(session)
    result = _run(path, tables, app)
    matched = {item.decision_id for item in result.session_load.applied}
    assert session.decisions[0].decision_id in matched
    assert session.decisions[1].decision_id in matched
    assert result.applied == ()
    assert result.reasked == ()
    open_ids = _outstanding_ids(result.session_load.session)
    assert duplicate.id not in open_ids
    assert warning.id not in open_ids
    _assert_empty(tables, app)


def _finding_check(result, check: str):
    found = [item for item in result.findings if item.check == check]
    assert len(found) == 1, (check, [item.check for item in result.findings])
    return found[0]
