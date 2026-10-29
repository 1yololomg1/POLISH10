"""Decisions, the session file, and all-wells rules. D46, D47, D48, D63, D79, D86, D91, D92, UXC2, UXC4.

The real %APPDATA% directory is never written. Every local path is a temporary directory.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polish.decisions.apply import apply_decision, explicit_mappings, listed_options, pass_choices, unit_choices
from polish.decisions.local_store import tables_directory
from polish.model.mapping import ExplicitMapping, roles_stated_by_family
from polish.decisions.session import (
    SessionNotWritten,
    choose_changed_file,
    kept_session_path,
    load_session,
    open_session,
    outstanding,
    preferences_path,
    save_preferences,
    save_session,
    session_path,
)
from polish.model.findings import DecisionActor, DecisionScope, Evidence, FindingRecord, evidence_digest, finding_key
from polish.model.pass_choice import PassChoice
from polish.model.ids import CurveRef, DatasetId, FileHash
from polish.model.states import (
    USER_MAPPED_LABEL,
    BlockType,
    CurveRole,
    EvidenceTier,
    IssueDisposition,
    ProcessingEligibility,
    StageStatus,
    UserAction,
    permitted_actions,
)
from polish.stages.s00_ingest import ingest
from polish.stages.s01_depth import classify
from polish.stages.s02_identify import UNCHANGED_MNEMONICS, identify
from polish.stages.s03_units import interpret
from polish.tables.loader import (
    LOCAL_DECISIONS_FILE,
    LOCAL_MNEMONICS_FILE,
    load_local_decisions,
    load_local_mnemonics,
    load_shipped_tables,
)

_WHEN = datetime(2026, 10, 15, tzinfo=timezone.utc)
_FAMILY = "GAMMA_RAY_TOTAL"
_OTHER_FAMILY = "BULK_DENSITY"


def _file(directory: Path, name: str = "well.las", body: bytes = b"~V\nVERS. 2.0:\n") -> Path:
    path = directory / name
    path.write_bytes(body)
    return path


def _finding(
    block: BlockType,
    check: str,
    identity: tuple[tuple[str, str], ...] = (),
    fields: tuple[tuple[str, str | int | float | bool], ...] = (),
    *,
    message: str = "Recorded.",
    curve: CurveRef | None = None,
    stage: int = 2,
) -> FindingRecord:
    refs = () if curve is None else (curve,)
    return FindingRecord(
        id=finding_key(stage, check, identity),
        stage=stage,
        check=check,
        identity=identity,
        block_type=block,
        subject_refs=refs,
        evidence=Evidence(block.value, fields),
        message=message,
        created_at=_WHEN,
    )


def _ambiguous(mark: str = "amb") -> FindingRecord:
    return _finding(
        BlockType.AMBIGUOUS_INTERPRETATION,
        "suspicious_samples",
        (("column", "1"), ("mnemonic", "GR"), ("rule_id", mark)),
        (
            ("family_id", _FAMILY),
            ("mnemonic", "GR"),
            ("options", json.dumps(["KEEP", "EXCLUDE"])),
            ("rule_id", _FAMILY),
        ),
        message="Curve GR has samples outside the suspicious limit.",
    )


def _mappable(mnemonic: str = "APPLE", mark: str | None = None, curve: CurveRef | None = None) -> FindingRecord:
    identity: tuple[tuple[str, str], ...] = (("mnemonic", mnemonic),)
    if mark is not None:
        identity = (("decision_id", mark), ("mnemonic", mnemonic))
    return _finding(
        BlockType.RULE_DEPENDENT_WARNING,
        "mnemonic_unrecognised",
        identity,
        (("mnemonic", mnemonic), ("preserved", True)),
        message=f"Mnemonic {mnemonic} is not recognised.",
        curve=curve,
    )


def _noted(mnemonic: str = "NOTE") -> FindingRecord:
    return _finding(
        BlockType.INFORMATIONAL,
        "mnemonic_left_unchanged",
        (("mnemonic", mnemonic),),
        (("mnemonic", mnemonic),),
    )


def _for_block(block: BlockType) -> FindingRecord:
    if block is BlockType.AMBIGUOUS_INTERPRETATION:
        return _ambiguous(block.value)
    if block is BlockType.RULE_DEPENDENT_WARNING:
        return _mappable(mark=block.value)
    if block is BlockType.INFORMATIONAL:
        return _noted()
    return _finding(block, "structural_notice")


def _apply(session, finding, action, **kwargs):
    return apply_decision(session, finding, action, timestamp=_WHEN, app_data=kwargs.pop("app_data", None), **kwargs)


@pytest.mark.parametrize("block", list(BlockType))
@pytest.mark.parametrize("action", list(UserAction))
def test_every_block_type_and_action_is_permitted_or_refused(tmp_path: Path, block: BlockType, action: UserAction):
    finding = _for_block(block)
    session = open_session(_file(tmp_path, f"{block.value}-{action.value}.las"), (finding,))
    kwargs: dict[str, object] = {"reason": "because the log says so", "app_data": tmp_path / "app"}
    if action is UserAction.CHOOSE_OPTION:
        kwargs["chosen_option"] = "KEEP"
    if action is UserAction.MAP_EXPLICITLY:
        kwargs["family_id"] = _FAMILY
        kwargs["user_label"] = "APPLE"
    if action in permitted_actions(block):
        updated = _apply(session, finding, action, **kwargs)
        assert updated.decisions[-1].action is action
        assert updated.decisions[-1].finding_id == finding.id
    else:
        with pytest.raises(ValueError):
            _apply(session, finding, action, **kwargs)


@pytest.mark.parametrize(
    "action",
    [UserAction.DEFER, UserAction.CHOOSE_OPTION, UserAction.MAP_EXPLICITLY, UserAction.REJECT_PROPOSAL],
)
def test_an_empty_reason_is_refused(tmp_path: Path, action: UserAction):
    if action is UserAction.MAP_EXPLICITLY:
        finding = _mappable()
        extra = {"family_id": _FAMILY}
    elif action is UserAction.REJECT_PROPOSAL:
        finding = _ambiguous()
        extra = {}
    else:
        finding = _ambiguous()
        extra = {"chosen_option": "KEEP"} if action is UserAction.CHOOSE_OPTION else {}
    session = open_session(_file(tmp_path), (finding,))
    with pytest.raises(ValueError, match="requires a reason"):
        _apply(session, finding, action, reason="  ", app_data=tmp_path / "app", **extra)


def test_an_unlisted_option_is_refused(tmp_path: Path):
    finding = _ambiguous()
    session = open_session(_file(tmp_path), (finding,))
    with pytest.raises(ValueError, match="not an option"):
        _apply(session, finding, UserAction.CHOOSE_OPTION, reason="no", chosen_option="DROP")


def test_map_explicitly_requires_a_known_family_and_refuses_rm(tmp_path: Path):
    session = open_session(_file(tmp_path), (_mappable(),))
    with pytest.raises(ValueError, match="requires a family_id"):
        _apply(session, _mappable(), UserAction.MAP_EXPLICITLY, reason="named")
    with pytest.raises(ValueError, match="not in the shipped tables"):
        _apply(
            session,
            _mappable(),
            UserAction.MAP_EXPLICITLY,
            reason="named",
            family_id="NOT_A_FAMILY",
        )
    refused = _mappable("RM")
    with pytest.raises(ValueError, match="never mappable"):
        _apply(session, refused, UserAction.MAP_EXPLICITLY, reason="named", family_id=_FAMILY)
    assert "RM" in UNCHANGED_MNEMONICS


def test_superseding_keeps_the_earlier_decision(tmp_path: Path):
    finding = _noted()
    session = open_session(_file(tmp_path), (finding,))
    first = _apply(session, finding, UserAction.ACKNOWLEDGE)
    second = _apply(first, finding, UserAction.DEFER, reason="later, with a reason")
    assert len(second.decisions) == 2
    assert second.decisions[1].supersedes == second.decisions[0].decision_id
    assert second.decisions[0].disposition.issue_disposition is IssueDisposition.ACCEPTED
    assert second.decisions[1].disposition.issue_disposition is IssueDisposition.DEFERRED


def test_the_session_file_has_no_numeric_data_values(tmp_path: Path):
    finding = _ambiguous()
    session = open_session(_file(tmp_path), (finding,))
    saved = _apply(
        session,
        finding,
        UserAction.CHOOSE_OPTION,
        reason="keep the coal",
        chosen_option="KEEP",
        scope=DecisionScope.THIS_ITEM,
    )
    path = save_session(saved)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _assert_no_numbers(payload)
    text = path.read_text(encoding="utf-8")
    assert "KEEP" in text
    assert "depth" not in text.lower()


def test_a_different_hash_is_reported_and_not_applied(tmp_path: Path):
    path = _file(tmp_path)
    finding = _noted()
    session = _apply(open_session(path, (finding,)), finding, UserAction.ACKNOWLEDGE)
    save_session(session)
    path.write_bytes(path.read_bytes() + b"\n")
    loaded = load_session(path, (finding,), app_data=tmp_path / "app")
    assert loaded.applied == ()
    assert loaded.session.hold is True
    assert any(
        item.block_type is BlockType.AMBIGUOUS_INTERPRETATION
        and item.message
        == "this looks like the same well but the raw file has changed since these decisions were made"
        and json.loads(str(item.evidence.as_dict()["options"]))
        == ["USE_SAVED_DECISIONS", "START_FRESH"]
        for item in loaded.reported
    )
    open_ids = [item.id for group in outstanding(loaded.session).values() for item in group]
    assert finding.id in open_ids


def test_an_unwritable_directory_writes_nowhere_else(tmp_path: Path):
    path = _file(tmp_path)
    session = open_session(path)
    blocked = session_path(path)
    blocked.mkdir()
    with pytest.raises(SessionNotWritten) as raised:
        save_session(session)
    assert raised.value.finding.block_type is BlockType.INFORMATIONAL
    assert "not writable" in raised.value.finding.message
    assert [item for item in tmp_path.glob("*.json") if item.is_file()] == []
    alternative = tmp_path / "chosen.polish-session.json"
    written = save_session(session, alternative_path=alternative)
    assert written == alternative
    assert alternative.is_file()
    assert blocked.is_dir()


def test_preferences_never_contain_a_hash_or_a_depth(tmp_path: Path):
    save_preferences(tmp_path, {"note": "as recorded"})
    text = preferences_path(tmp_path).read_text(encoding="utf-8")
    assert "depth" not in text.lower()
    assert re.search(r"[0-9a-fA-F]{64}", text) is None
    assert "well" not in text.lower()
    before = preferences_path(tmp_path).read_bytes()
    with pytest.raises(ValueError, match="file hash"):
        save_preferences(tmp_path, {"note": "a" * 64})
    assert preferences_path(tmp_path).read_bytes() == before


def test_local_rule_is_not_a_side_effect_of_this_item(tmp_path: Path):
    app = tmp_path / "app"
    finding = _mappable()
    session = open_session(_file(tmp_path), (finding,))
    _apply(session, finding, UserAction.MAP_EXPLICITLY, reason="this well", family_id=_FAMILY, user_label="APPLE")
    assert not tables_directory(app).exists()


def test_a_shipped_mnemonic_cannot_be_saved_for_all_wells(tmp_path: Path):
    app = tmp_path / "app"
    finding = _mappable("GR")
    session = open_session(_file(tmp_path), (finding,))
    with pytest.raises(ValueError, match="shipped mnemonic table"):
        _apply(
            session,
            finding,
            UserAction.MAP_EXPLICITLY,
            reason="all wells",
            family_id=_FAMILY,
            scope=DecisionScope.LOCAL_RULE,
            app_data=app,
        )
    assert not (tables_directory(app) / LOCAL_MNEMONICS_FILE).exists()
    updated = _apply(
        session,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="this well only",
        family_id=_FAMILY,
        scope=DecisionScope.THIS_ITEM,
        app_data=app,
    )
    assert updated.decisions[-1].scope is DecisionScope.THIS_ITEM
    assert not (tables_directory(app) / LOCAL_MNEMONICS_FILE).exists()


def test_a_local_mapping_writes_one_row_and_leaves_the_other_byte_for_byte(tmp_path: Path):
    app = tmp_path / "app"
    directory = tables_directory(app)
    directory.mkdir(parents=True)
    pear = '{"mnemonic": "PEAR", "family": "GAMMA_RAY_TOTAL", "tier": "USER_DEFINED"}'
    original = '{\n  "kept": true,\n  "entries": [\n    ' + pear + "\n  ]\n}\n"
    path = directory / LOCAL_MNEMONICS_FILE
    path.write_text(original, encoding="utf-8", newline="\n")
    finding = _mappable()
    session = open_session(_file(tmp_path), (finding,))
    updated = _apply(
        session,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="behaves as gamma ray",
        family_id=_FAMILY,
        user_label="APPLE",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    text = path.read_text(encoding="utf-8")
    assert pear in text
    assert '"kept": true' in text
    families = frozenset(row.id for row in load_shipped_tables().families)
    rows = load_local_mnemonics(directory, families)
    apple = next(row for row in rows if row.mnemonic.upper() == "APPLE")
    assert apple.family_id == _FAMILY
    assert apple.decision_id == updated.decisions[-1].decision_id
    assert apple.user_label == "APPLE"
    assert apple.role is CurveRole.MEASURED_LOG
    assert len(rows) == 2


def test_saving_apple_again_with_the_same_family_does_not_write(tmp_path: Path):
    app = tmp_path / "app"
    finding = _mappable()
    session = open_session(_file(tmp_path), (finding,))
    once = _apply(
        session,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="first",
        family_id=_FAMILY,
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    path = tables_directory(app) / LOCAL_MNEMONICS_FILE
    before = path.read_bytes()
    twice = _apply(
        once,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="again",
        family_id=_FAMILY,
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    assert path.read_bytes() == before
    assert len(twice.decisions) == 2


def test_a_different_family_needs_confirmation_and_then_replaces_one_row(tmp_path: Path):
    app = tmp_path / "app"
    finding = _mappable()
    session = open_session(_file(tmp_path), (finding,))
    once = _apply(
        session,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="gamma",
        family_id=_FAMILY,
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    path = tables_directory(app) / LOCAL_MNEMONICS_FILE
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already saved for all wells as GAMMA_RAY_TOTAL"):
        _apply(
            once,
            finding,
            UserAction.MAP_EXPLICITLY,
            reason="density",
            family_id=_OTHER_FAMILY,
            scope=DecisionScope.LOCAL_RULE,
            app_data=app,
        )
    assert path.read_bytes() == before
    twice = _apply(
        once,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="density",
        family_id=_OTHER_FAMILY,
        scope=DecisionScope.LOCAL_RULE,
        confirm_saved_rule=True,
        app_data=app,
    )
    families = frozenset(row.id for row in load_shipped_tables().families)
    rows = load_local_mnemonics(tables_directory(app), families)
    apples = [row for row in rows if row.mnemonic.upper() == "APPLE"]
    assert len(apples) == 1
    assert apples[0].family_id == _OTHER_FAMILY
    assert twice.decisions[1].supersedes == twice.decisions[0].decision_id
    assert len(twice.decisions) == 2


def test_a_saved_keep_is_applied_on_a_second_well_as_default(tmp_path: Path):
    app = tmp_path / "app"
    first = _file(tmp_path, "one.las")
    finding = _ambiguous("on-one")
    session = _apply(
        open_session(first, (finding,)),
        finding,
        UserAction.CHOOSE_OPTION,
        reason="keep them",
        chosen_option="KEEP",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    assert load_local_decisions(tables_directory(app))[0].question_kind == _FAMILY
    second_finding = _ambiguous("on-two")
    second = _file(tmp_path, "two.las", b"~V\nVERS. 2.0:\nWRAP. NO:\n")
    loaded = load_session(second, (second_finding,), app_data=app)
    assert len(loaded.defaults_applied) == 1
    assert loaded.defaults_applied[0].actor is DecisionActor.DEFAULT_APPLIED
    assert loaded.defaults_applied[0].chosen_option == "KEEP"
    assert loaded.defaults_applied[0].finding_id == second_finding.id
    assert loaded.defaults_applied[0] in loaded.session.decisions
    assert session.decisions[0].decision_id != loaded.defaults_applied[0].decision_id


def test_no_session_does_not_copy_another_wells_session_decisions(tmp_path: Path):
    app = tmp_path / "app"
    path = _file(tmp_path, "only.las")
    finding = _noted()
    loaded = load_session(path, (finding,), app_data=app)
    assert loaded.applied == ()
    assert loaded.session.decisions == ()
    assert outstanding(loaded.session)[BlockType.INFORMATIONAL] == (finding,)


def test_apple_mapped_to_gamma_ray_keeps_its_name(tmp_path: Path):
    ingested, first = _identify(tmp_path, [("APPLE", "GAPI")])
    apple = next(item for item in first.curves if item.mnemonic == "APPLE")
    notice = next(item for item in first.findings if item.evidence.as_dict().get("mnemonic") == "APPLE")
    assert notice.block_type is BlockType.RULE_DEPENDENT_WARNING
    session = open_session(tmp_path / "well.las", first.findings)
    decided = _apply(
        session,
        notice,
        UserAction.MAP_EXPLICITLY,
        reason="it behaves as gamma ray",
        family_id=_FAMILY,
        user_label="APPLE",
    )
    again = identify(
        ingested,
        classify(ingested),
        local_tables_directory=tmp_path / "no-local",
        explicit_mappings=explicit_mappings(decided),
    )
    mapped = next(item for item in again.curves if item.mnemonic == "APPLE")
    assert mapped.curve == apple.curve
    assert mapped.evidence_tier is EvidenceTier.USER_DEFINED
    assert mapped.family_id == _FAMILY
    assert mapped.user_label == "APPLE"
    assert mapped.label == USER_MAPPED_LABEL
    assert mapped.processing_eligibility is ProcessingEligibility.ELIGIBLE
    assert mapped.mnemonic == "APPLE"
    assert mapped.role is CurveRole.MEASURED_LOG
    assert mapped.role is not CurveRole.UNKNOWN


def test_a_saved_row_with_no_family_stays_and_is_not_processed(tmp_path: Path):
    local = tmp_path / "local"
    local.mkdir()
    row = {"mnemonic": "APPLE", "user_label": "APPLE", "tier": "USER_DEFINED"}
    path = local / LOCAL_MNEMONICS_FILE
    path.write_text(json.dumps({"entries": [row]}), encoding="utf-8")
    before = path.read_bytes()
    _ingested, result = _identify(tmp_path, [("APPLE", "GAPI")], local=local)
    apple = next(item for item in result.curves if item.mnemonic == "APPLE")
    assert apple.evidence_tier is EvidenceTier.USER_DEFINED
    assert apple.user_label == "APPLE"
    assert apple.label == USER_MAPPED_LABEL
    assert apple.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY
    warnings = [
        item
        for item in result.findings
        if item.block_type is BlockType.RULE_DEPENDENT_WARNING and item.evidence.as_dict().get("mnemonic") == "APPLE"
    ]
    assert len(warnings) == 1
    assert "no family" in warnings[0].message
    families = frozenset(item.id for item in load_shipped_tables().families)
    loaded = load_local_mnemonics(local, families)
    assert len(loaded) == 1
    assert loaded[0].family_id is None
    assert path.read_bytes() == before


def test_later_scopes_are_refused(tmp_path: Path):
    finding = _ambiguous()
    session = open_session(_file(tmp_path), (finding,))
    for scope in (
        DecisionScope.SELECTED_ITEMS,
        DecisionScope.DETECTED_GROUP,
        DecisionScope.ALL_MATCHING_IN_FILE,
    ):
        with pytest.raises(ValueError, match="not available"):
            _apply(session, finding, UserAction.CHOOSE_OPTION, reason="no", chosen_option="KEEP", scope=scope)


def test_scope_tokens_are_gone_from_polish_and_tests():
    root = Path(__file__).resolve().parents[2]
    forbidden = ("THIS_" + "WELL", "ALL_" + "WELLS")
    hits: list[str] = []
    for folder in (root / "polish", root / "tests"):
        for path in folder.rglob("*"):
            if path.suffix not in {".py", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(root)} contains {token}")
    assert hits == []


def test_listed_options_are_the_finding_list():
    assert listed_options(_ambiguous()) == ("KEEP", "EXCLUDE")


def _first_entry_text(text: str) -> str:
    decoder = json.JSONDecoder()
    cursor = text.index("[") + 1
    while text[cursor] in " \t\r\n":
        cursor += 1
    _row, end = decoder.raw_decode(text, cursor)
    return text[cursor:end]


def _families_with_more_than_one_role() -> dict[str, tuple[CurveRole, ...]]:
    found: dict[str, list[CurveRole]] = {}
    for row in load_shipped_tables().mnemonics:
        if row.family_id is None or row.role is None or row.role is CurveRole.UNKNOWN:
            continue
        roles = found.setdefault(row.family_id, [])
        if row.role not in roles:
            roles.append(row.role)
    return {family: tuple(roles) for family, roles in found.items() if len(roles) > 1}


def recorded_current(path: Path) -> str:
    return FileHash.from_path(path).hex_digest


def _changed_file(directory: Path, name: str = "well.las"):
    app = directory / "app"
    source = _file(directory, "rule-source.las", b"~V\nVERS. 2.0:\nRULE\n")
    rule = _ambiguous("rule-on-source")
    _apply(
        open_session(source, (rule,)),
        rule,
        UserAction.CHOOSE_OPTION,
        reason="keep them",
        chosen_option="KEEP",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    path = _file(directory, name)
    finding = _noted()
    missing = _noted("GONE")
    session = _apply(open_session(path, (finding, missing)), finding, UserAction.ACKNOWLEDGE)
    session = _apply(session, missing, UserAction.ACKNOWLEDGE)
    save_session(session)
    original = session_path(path).read_bytes()
    recorded = json.loads(original.decode("utf-8"))["file_hash"]
    path.write_bytes(path.read_bytes() + b"\n")
    loaded = load_session(path, (finding, _ambiguous("rule-on-reload")), app_data=app)
    return path, finding, missing, loaded, original, recorded


def _identify(directory: Path, specs: list[tuple[str, str]], local: Path | None = None):
    curves = "\n".join(["DEPT.M : DEPTH"] + [f"{name}.{unit} : {name}" for name, unit in specs]) + "\n"
    body = (
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
        + " ".join(["0.0"] + ["1.0"] * len(specs))
        + "\n"
        + " ".join(["1.0"] + ["2.0"] * len(specs))
        + "\n"
    )
    path = directory / "well.las"
    path.write_text(body, encoding="utf-8", newline="\n")
    ingested = ingest(path, read_at=_WHEN)
    assert ingested.dataset is not None
    depth = classify(ingested)
    assert depth.status is StageStatus.COMPLETE
    result = identify(ingested, depth, local_tables_directory=local if local is not None else directory / "no-local")
    return ingested, result


def _assert_no_numbers(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise AssertionError(value)
    if isinstance(value, list):
        for item in value:
            _assert_no_numbers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_numbers(item)
        return
    raise AssertionError(type(value))


def test_real_appdata_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("APPDATA", raising=False)
    finding = _mappable()
    session = open_session(_file(tmp_path), (finding,))
    updated = _apply(
        session,
        finding,
        UserAction.MAP_EXPLICITLY,
        reason="injected",
        family_id=_FAMILY,
        scope=DecisionScope.LOCAL_RULE,
        app_data=tmp_path / "app",
    )
    assert os.environ.get("APPDATA") is None
    assert updated.decisions[-1].scope is DecisionScope.LOCAL_RULE
    assert FileHash.from_path(tmp_path / "well.las") == updated.file_hash
    assert isinstance(updated.decisions[-1].input_file_hash, FileHash)


def test_explicit_mapping_needs_a_curve_on_this_dataset():
    dataset_id = DatasetId.for_source(FileHash("c" * 64))
    mapping = ExplicitMapping(CurveRef(dataset_id, 3), _FAMILY, CurveRole.MEASURED_LOG, "APPLE")
    assert mapping.family_id == _FAMILY
    assert mapping.user_label == "APPLE"
    assert mapping.role is CurveRole.MEASURED_LOG


def test_a_saved_answer_changes_only_when_confirmed(tmp_path: Path):
    app = tmp_path / "app"
    path = _file(tmp_path)
    kept = _ambiguous("keep-me")
    other = _finding(
        BlockType.AMBIGUOUS_INTERPRETATION,
        "suspicious_samples",
        (("column", "2"), ("mnemonic", "RHOB"), ("rule_id", _OTHER_FAMILY)),
        (
            ("family_id", _OTHER_FAMILY),
            ("mnemonic", "RHOB"),
            ("options", json.dumps(["KEEP", "EXCLUDE"])),
            ("rule_id", _OTHER_FAMILY),
        ),
        message="Curve RHOB has samples outside the suspicious limit.",
    )
    session = open_session(path, (kept, other))
    session = _apply(
        session,
        other,
        UserAction.CHOOSE_OPTION,
        reason="exclude the other curve",
        chosen_option="EXCLUDE",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    decisions_path = tables_directory(app) / LOCAL_DECISIONS_FILE
    other_row = decisions_path.read_text(encoding="utf-8")
    once = _apply(
        session,
        kept,
        UserAction.CHOOSE_OPTION,
        reason="keep them",
        chosen_option="KEEP",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    other_entry = _first_entry_text(other_row)
    assert other_entry in decisions_path.read_text(encoding="utf-8")
    before = decisions_path.read_bytes()
    again = _apply(
        once,
        kept,
        UserAction.CHOOSE_OPTION,
        reason="keep them again",
        chosen_option="KEEP",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    assert decisions_path.read_bytes() == before
    assert len(again.decisions) == len(once.decisions) + 1
    with pytest.raises(ValueError, match="already saved for all wells as KEEP"):
        _apply(
            once,
            kept,
            UserAction.CHOOSE_OPTION,
            reason="exclude them",
            chosen_option="EXCLUDE",
            scope=DecisionScope.LOCAL_RULE,
            app_data=app,
        )
    assert decisions_path.read_bytes() == before
    untouched = tmp_path / "already-processed.polish-session.json"
    untouched.write_bytes(b"session-of-a-well-already-processed\n")
    changed = _apply(
        once,
        kept,
        UserAction.CHOOSE_OPTION,
        reason="exclude them",
        chosen_option="EXCLUDE",
        scope=DecisionScope.LOCAL_RULE,
        confirm_saved_rule=True,
        app_data=app,
    )
    assert untouched.read_bytes() == b"session-of-a-well-already-processed\n"
    assert other_entry in decisions_path.read_text(encoding="utf-8")
    rows = load_local_decisions(tables_directory(app))
    assert len(rows) == 2
    keep_rows = [row for row in rows if row.question_kind == _FAMILY]
    assert len(keep_rows) == 1
    assert keep_rows[0].chosen_option == "EXCLUDE"
    assert keep_rows[0].decision_id == changed.decisions[-1].decision_id
    assert changed.decisions[-1].supersedes == once.decisions[-1].decision_id
    assert any(item.decision_id == changed.decisions[-1].supersedes for item in changed.decisions)


def test_a_changed_file_keeps_the_old_session_and_then_chooses(tmp_path: Path):
    path, finding, missing, loaded, original, recorded = _changed_file(tmp_path)
    question = next(item for item in loaded.reported if item.block_type is BlockType.AMBIGUOUS_INTERPRETATION)
    assert json.loads(str(question.evidence.as_dict()["options"])) == ["USE_SAVED_DECISIONS", "START_FRESH"]
    archive = kept_session_path(path, recorded)
    assert not archive.exists()
    used = choose_changed_file(loaded.session, "USE_SAVED_DECISIONS", timestamp=_WHEN)
    assert archive.read_bytes() == original
    assert not session_path(path).exists()
    assert len(used.applied) == 1
    assert used.applied[0].finding_id == finding.id
    assert used.applied[0].actor is DecisionActor.USER
    assert used.applied[0].input_file_hash == loaded.session.file_hash
    assert used.applied[0].supersedes is not None
    assert any(item.decision_id == used.applied[0].supersedes for item in used.session.decisions)
    assert any(item.evidence.as_dict().get("finding_id") == missing.id for item in used.reported)
    assert all(item.finding_id != missing.id for item in used.session.decisions)
    assert used.defaults_applied
    assert all(item in used.session.decisions for item in used.defaults_applied)
    assert used.session.hold is False
    assert used.session.held_ids == frozenset()
    save_session(used.session)
    assert session_path(path).is_file()
    assert archive.read_bytes() == original
    assert json.loads(session_path(path).read_text(encoding="utf-8"))["file_hash"] == recorded_current(path)

    fresh_path, _fresh_finding, _fresh_missing, fresh_loaded, fresh_original, fresh_recorded = _changed_file(
        tmp_path, "fresh.las"
    )
    fresh = choose_changed_file(fresh_loaded.session, "START_FRESH", timestamp=_WHEN)
    fresh_archive = kept_session_path(fresh_path, fresh_recorded)
    assert fresh_archive.read_bytes() == fresh_original
    assert fresh.applied == ()
    assert all(item.finding_id != _noted().id for item in fresh.session.decisions)
    assert fresh.defaults_applied
    assert fresh.session.hold is False
    save_session(fresh.session)
    assert session_path(fresh_path).is_file()
    assert fresh_archive.read_bytes() == fresh_original
    assert json.loads(session_path(fresh_path).read_text(encoding="utf-8"))["file_hash"] == recorded_current(
        fresh_path
    )

    blocked_path, _blocked_finding, _blocked_missing, blocked, blocked_original, blocked_recorded = _changed_file(
        tmp_path, "blocked.las"
    )
    blocked_archive = kept_session_path(blocked_path, blocked_recorded)
    blocked_archive.write_bytes(b"already kept")
    with pytest.raises(SessionNotWritten, match="already exists"):
        choose_changed_file(blocked.session, "START_FRESH", timestamp=_WHEN)
    assert blocked_archive.read_bytes() == b"already kept"
    assert session_path(blocked_path).read_bytes() == blocked_original


def test_mapping_to_depth_requires_one_of_its_roles(tmp_path: Path):
    families = _families_with_more_than_one_role()
    assert list(families) == ["DEPTH"]
    stated = families["DEPTH"]
    assert stated == (CurveRole.MEASURED_LOG, CurveRole.GEOMETRIC_REFERENCE)
    ingested, first = _identify(tmp_path, [("APPLE", "M")])
    notice = next(item for item in first.findings if item.evidence.as_dict().get("mnemonic") == "APPLE")
    session = open_session(tmp_path / "well.las", first.findings)
    with pytest.raises(ValueError, match="requires a role"):
        _apply(
            session,
            notice,
            UserAction.MAP_EXPLICITLY,
            reason="it is a depth",
            family_id="DEPTH",
        )
    decided = _apply(
        session,
        notice,
        UserAction.MAP_EXPLICITLY,
        reason="it is a true vertical depth",
        family_id="DEPTH",
        role=CurveRole.GEOMETRIC_REFERENCE.value,
    )
    again = identify(
        ingested,
        classify(ingested),
        local_tables_directory=tmp_path / "no-local",
        explicit_mappings=explicit_mappings(decided),
    )
    mapped = next(item for item in again.curves if item.mnemonic == "APPLE")
    assert mapped.family_id == "DEPTH"
    assert mapped.role is CurveRole.GEOMETRIC_REFERENCE
    assert mapped.role is not CurveRole.UNKNOWN
    gamma = _apply(
        session,
        notice,
        UserAction.MAP_EXPLICITLY,
        reason="it behaves as gamma ray",
        family_id=_FAMILY,
    )
    assert gamma.decisions[-1].role == CurveRole.MEASURED_LOG.value
    gamma_curves = identify(
        ingested,
        classify(ingested),
        local_tables_directory=tmp_path / "no-local",
        explicit_mappings=explicit_mappings(gamma),
    )
    gamma_curve = next(item for item in gamma_curves.curves if item.mnemonic == "APPLE")
    assert gamma_curve.role is CurveRole.MEASURED_LOG
    assert gamma_curve.role is not CurveRole.UNKNOWN


def _unit_question(directory: Path):
    ingested, identified = _identify(directory, [("RHOB", "API"), ("GR", "GAPI")])
    units = interpret(ingested, identified)
    question = next(
        item
        for item in units.findings
        if item.block_type is BlockType.AMBIGUOUS_INTERPRETATION
        and item.evidence.as_dict().get("mnemonic") == "RHOB"
    )
    return ingested, question


def test_unit_choices_returns_only_a_stage_3_unit_answer(tmp_path: Path):
    _ingested, question = _unit_question(tmp_path)
    assert question.stage == 3
    assert "candidates" in question.evidence.as_dict()
    assert listed_options(question) == ("GAPI", "API")
    keep = _ambiguous("keep-1")
    session = open_session(_ingested.source_file.path, (question, keep))
    session = _apply(
        session,
        question,
        UserAction.CHOOSE_OPTION,
        reason="gamma api units",
        chosen_option="GAPI",
    )
    session = _apply(session, keep, UserAction.CHOOSE_OPTION, reason="keep them", chosen_option="KEEP")
    choices = unit_choices(session)
    assert len(choices) == 1
    assert choices[0].canonical_unit == "GAPI"
    assert choices[0].curve == question.subject_refs[0]
    assert choices[0].decision_id == session.decisions[0].decision_id

    not_user = replace(session.decisions[0], actor=DecisionActor.DEFAULT_APPLIED)
    assert unit_choices(replace(session, decisions=(not_user,))) == ()
    other = question.subject_refs[0]
    assert isinstance(other, CurveRef)
    two_curves = replace(question, subject_refs=(other, CurveRef(other.dataset_id, other.column_index + 1)))
    assert unit_choices(replace(session, findings=(two_curves, keep))) == ()
    bare = replace(
        question,
        evidence=Evidence(
            question.block_type.value,
            (("mnemonic", "RHOB"), ("outcome", "UNRECOGNISED"), ("family_id", "BULK_DENSITY")),
        ),
    )
    assert unit_choices(replace(session, findings=(bare, keep))) == ()
    wrong_stage = replace(question, stage=2, id=finding_key(2, question.check, question.identity))
    assert unit_choices(replace(session, findings=(wrong_stage, keep))) == ()


def test_a_superseded_or_held_unit_choice_is_not_returned(tmp_path: Path):
    _ingested, question = _unit_question(tmp_path)
    session = open_session(_ingested.source_file.path, (question,))
    first = _apply(session, question, UserAction.CHOOSE_OPTION, reason="gamma api", chosen_option="GAPI")
    second = _apply(first, question, UserAction.CHOOSE_OPTION, reason="gravity instead", chosen_option="API")
    choices = unit_choices(second)
    assert len(choices) == 1
    assert choices[0].canonical_unit == "API"
    assert choices[0].decision_id == second.decisions[1].decision_id
    assert choices[0].decision_id != second.decisions[0].decision_id
    held_ids = replace(second, held_ids=frozenset({second.decisions[1].decision_id}))
    assert unit_choices(held_ids) == ()
    assert unit_choices(replace(second, hold=True)) == ()


def test_a_unit_choice_outside_the_listed_candidates_is_refused(tmp_path: Path):
    _ingested, question = _unit_question(tmp_path)
    session = open_session(_ingested.source_file.path, (question,))
    decided = _apply(session, question, UserAction.CHOOSE_OPTION, reason="gamma api", chosen_option="GAPI")
    stale = replace(decided.decisions[0], chosen_option="G/CC")
    with pytest.raises(ValueError, match="not an option this finding lists"):
        unit_choices(replace(decided, decisions=(stale,)))


def test_a_local_rule_unit_choice_is_refused(tmp_path: Path):
    _ingested, question = _unit_question(tmp_path)
    session = open_session(_ingested.source_file.path, (question,))
    with pytest.raises(ValueError, match="this well only"):
        _apply(
            session,
            question,
            UserAction.CHOOSE_OPTION,
            reason="every well",
            chosen_option="GAPI",
            scope=DecisionScope.LOCAL_RULE,
        )


def _pass_finding(mark: str = "pass", family: str = "GAMMA_RAY_TOTAL") -> FindingRecord:
    return _finding(
        BlockType.AMBIGUOUS_INTERPRETATION,
        "pass_default",
        (("family_id", family), ("mnemonic", mark)),
        (
            ("family_id", family),
            ("base_mnemonic", "GR"),
            ("options", json.dumps(["main pass", "repeat pass"])),
            ("passes", json.dumps([{"curve": "curve:x#1", "mnemonic": "GR_MAIN", "pass": "main pass"}])),
        ),
        message="Which pass should be used by default?",
    )


def test_pass_choices_returns_only_a_stage_2_pass_answer(tmp_path: Path):
    finding = _pass_finding()
    keep = _ambiguous("keep-1")
    session = open_session(_file(tmp_path), (finding, keep))
    session = _apply(session, finding, UserAction.CHOOSE_OPTION, reason="main pass", chosen_option="main pass")
    session = _apply(session, keep, UserAction.CHOOSE_OPTION, reason="keep them", chosen_option="KEEP")
    choices = pass_choices(session)
    assert len(choices) == 1
    assert choices[0] == PassChoice("GAMMA_RAY_TOTAL", "main pass", session.decisions[0].decision_id)

    saved = replace(session.decisions[0], actor=DecisionActor.DEFAULT_APPLIED, scope=DecisionScope.LOCAL_RULE)
    assert pass_choices(replace(session, decisions=(saved, session.decisions[1]))) == (
        PassChoice("GAMMA_RAY_TOTAL", "main pass", saved.decision_id),
    )
    bare = replace(
        finding,
        evidence=Evidence(finding.block_type.value, (("family_id", "GAMMA_RAY_TOTAL"), ("options", json.dumps(["main pass"])))),
    )
    assert pass_choices(replace(session, findings=(bare, keep))) == ()
    shifted = replace(finding, stage=3, id=finding_key(3, finding.check, finding.identity))
    assert pass_choices(replace(session, findings=(shifted, keep))) == ()
    assert pass_choices(replace(session, hold=True)) == ()


def test_a_superseded_or_held_pass_choice_is_not_returned(tmp_path: Path):
    finding = _pass_finding()
    session = open_session(_file(tmp_path), (finding,))
    first = _apply(session, finding, UserAction.CHOOSE_OPTION, reason="main pass", chosen_option="main pass")
    second = _apply(first, finding, UserAction.CHOOSE_OPTION, reason="repeat pass", chosen_option="repeat pass")
    choices = pass_choices(second)
    assert len(choices) == 1
    assert choices[0].pass_name == "repeat pass"
    assert choices[0].decision_id == second.decisions[1].decision_id
    held = replace(second, held_ids=frozenset({second.decisions[1].decision_id}))
    assert pass_choices(held) == ()
    assert pass_choices(replace(second, hold=True)) == ()


def test_a_pass_choice_outside_the_listed_options_is_refused(tmp_path: Path):
    finding = _pass_finding()
    session = open_session(_file(tmp_path), (finding,))
    decided = _apply(session, finding, UserAction.CHOOSE_OPTION, reason="main pass", chosen_option="main pass")
    stale = replace(decided.decisions[0], chosen_option="unqualified")
    with pytest.raises(ValueError, match="not an option this finding lists"):
        pass_choices(replace(decided, decisions=(stale,)))


def test_two_in_force_pass_choices_for_one_family_are_refused(tmp_path: Path):
    one = _pass_finding("pass-a")
    two = _pass_finding("pass-b")
    session = open_session(_file(tmp_path), (one, two))
    session = _apply(session, one, UserAction.CHOOSE_OPTION, reason="main pass", chosen_option="main pass")
    session = _apply(session, two, UserAction.CHOOSE_OPTION, reason="repeat pass", chosen_option="repeat pass")
    with pytest.raises(ValueError, match="Two pass choices name GAMMA_RAY_TOTAL"):
        pass_choices(session)


def test_an_all_wells_pass_choice_is_the_default_on_the_next_well(tmp_path: Path):
    app = tmp_path / "app"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    ingested, plain = _identify(first, [("GR_MAIN", "GAPI"), ("GR_REF", "GAPI")])
    question = next(item for item in plain.findings if "passes" in item.evidence.as_dict())
    assert json.loads(str(question.evidence.as_dict()["options"])) == ["main pass", "repeat pass"]
    saved = _apply(
        open_session(ingested.source_file.path, plain.findings),
        question,
        UserAction.CHOOSE_OPTION,
        reason="use the main pass on every well",
        chosen_option="main pass",
        scope=DecisionScope.LOCAL_RULE,
        app_data=app,
    )
    on_this_well = pass_choices(saved)
    assert len(on_this_well) == 1
    assert on_this_well[0].pass_name == "main pass"
    assert saved.decisions[-1].actor is DecisionActor.USER
    assert saved.decisions[-1].scope is DecisionScope.LOCAL_RULE

    other, other_plain = _identify(second, [("GR_MAIN", "GAPI"), ("GR_REF", "GAPI")])
    loaded = load_session(other.source_file.path, other_plain.findings, app_data=app)
    assert len(loaded.defaults_applied) == 1
    assert loaded.defaults_applied[0].actor is DecisionActor.DEFAULT_APPLIED
    assert loaded.defaults_applied[0].scope is DecisionScope.LOCAL_RULE
    assert loaded.defaults_applied[0].chosen_option == "main pass"
    choices = pass_choices(loaded.session)
    assert choices == (
        PassChoice("GAMMA_RAY_TOTAL", "main pass", loaded.defaults_applied[0].decision_id),
    )
    chosen = identify(
        other,
        classify(other),
        local_tables_directory=second / "no-local",
        pass_choices=choices,
    )
    main = next(item for item in chosen.curves if item.mnemonic == "GR_MAIN")
    assert len(chosen.default_passes) == 1
    assert chosen.default_passes[0].pass_name == "main pass"
    assert chosen.default_passes[0].curves == (main.curve,)
    assert chosen.default_passes[0].decision_id == choices[0].decision_id
    assert chosen.curves == other_plain.curves
    assert [item for item in chosen.findings if "passes" in item.evidence.as_dict()] == []


@pytest.mark.parametrize(
    "module",
    [
        "polish.stages.s02_identify",
        "polish.stages.s03_units",
        "polish.decisions",
        "polish.decisions.apply",
        "polish.decisions.session",
        "polish.model.mapping",
        "polish.model.pass_choice",
        "polish.model.unit_choice",
    ],
)
def test_a_fresh_interpreter_imports_each_module(module: str):
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_roles_stated_by_family_is_the_shipped_table_order():
    shipped = load_shipped_tables()
    assert roles_stated_by_family(shipped, "DEPTH") == (
        CurveRole.MEASURED_LOG,
        CurveRole.GEOMETRIC_REFERENCE,
    )
    assert roles_stated_by_family(shipped, "GAMMA_RAY_TOTAL") == (CurveRole.MEASURED_LOG,)


def test_a_changed_file_rerecords_a_stated_depth_role_and_lists_another(tmp_path: Path):
    path = _file(tmp_path)
    good = _mappable("HOLE", "depth-good")
    bad = _mappable("PEAR", "depth-bad")
    session = open_session(path, (good, bad))
    session = _apply(
        session,
        good,
        UserAction.MAP_EXPLICITLY,
        reason="true vertical depth",
        family_id="DEPTH",
        role=CurveRole.GEOMETRIC_REFERENCE.value,
    )
    session = _apply(
        session,
        bad,
        UserAction.MAP_EXPLICITLY,
        reason="also a depth",
        family_id="DEPTH",
        role=CurveRole.GEOMETRIC_REFERENCE.value,
    )
    saved = save_session(session)
    payload = json.loads(saved.read_text(encoding="utf-8"))
    for item in payload["decisions"]:
        if item["finding_id"] == bad.id:
            item["role"] = CurveRole.DRILLING.value
    saved.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    path.write_bytes(path.read_bytes() + b"\n")
    loaded = load_session(path, (good, bad), app_data=tmp_path / "app")
    chosen = choose_changed_file(loaded.session, "USE_SAVED_DECISIONS", timestamp=_WHEN)
    assert len(chosen.applied) == 1
    assert chosen.applied[0].finding_id == good.id
    assert chosen.applied[0].role == CurveRole.GEOMETRIC_REFERENCE.value
    assert chosen.applied[0].family_id == "DEPTH"
    assert chosen.applied[0].actor is DecisionActor.USER
    assert chosen.applied[0].supersedes is not None
    listed = [item for item in chosen.reported if item.evidence.as_dict().get("finding_id") == bad.id]
    assert len(listed) == 1
    assert "the role is not one this family states" in listed[0].message


def _api_pair(directory: Path):
    """Two unrecognised curves, both unit API. Stage 3 asks GAPI or API for each. D93, D96."""

    ingested, identified = _identify(directory, [("A", "API"), ("B", "API")])
    units = interpret(ingested, identified)
    questions = {
        str(item.evidence.as_dict()["mnemonic"]): item
        for item in units.findings
        if item.check == "unit_unresolved"
    }
    return ingested, identified, units, questions


def test_a_unit_choice_does_not_renumber_the_deferred_curve(tmp_path: Path):
    app = tmp_path / "app"
    ingested, identified, units, questions = _api_pair(tmp_path)
    assert set(questions) == {"A", "B"}
    assert questions["A"].id != questions["B"].id
    assert listed_options(questions["A"]) == ("GAPI", "API")
    assert listed_options(questions["B"]) == ("GAPI", "API")
    session = open_session(ingested.source_file.path, units.findings)
    session = _apply(
        session,
        questions["A"],
        UserAction.CHOOSE_OPTION,
        reason="gamma api",
        chosen_option="GAPI",
    )
    session = _apply(session, questions["B"], UserAction.DEFER, reason="later")
    save_session(session)
    again = interpret(ingested, identified, unit_choices=unit_choices(session))
    unresolved = [item for item in again.findings if item.check == "unit_unresolved"]
    assert [str(item.evidence.as_dict()["mnemonic"]) for item in unresolved] == ["B"]
    assert unresolved[0].id == questions["B"].id
    assert questions["A"].id not in {item.id for item in again.findings}
    reloaded = load_session(ingested.source_file.path, again.findings, app_data=app)
    assert [item.finding_id for item in reloaded.applied] == [questions["B"].id]
    assert reloaded.applied[0].disposition.issue_disposition is IssueDisposition.DEFERRED
    assert questions["A"].id not in {item.finding_id for item in reloaded.applied}
    open_ids = [item.id for group in outstanding(reloaded.session).values() for item in group]
    assert questions["B"].id not in open_ids
    first = load_session(ingested.source_file.path, units.findings, app_data=app)
    assert {item.finding_id for item in first.applied} == {questions["A"].id, questions["B"].id}


def test_changed_evidence_holds_the_decision(tmp_path: Path):
    path = _file(tmp_path)
    finding = _ambiguous("held")
    session = _apply(open_session(path, (finding,)), finding, UserAction.DEFER, reason="not yet")
    save_session(session)
    fields = tuple(
        (key, "GR2") if key == "mnemonic" else (key, value) for key, value in finding.evidence.fields
    )
    changed = replace(finding, evidence=Evidence(finding.block_type.value, fields))
    assert changed.id == finding.id
    assert evidence_digest(changed) != session.decisions[0].evidence_digest
    loaded = load_session(path, (changed,), app_data=tmp_path / "app")
    assert loaded.applied == ()
    assert session.decisions[0].decision_id in loaded.session.held_ids
    reported = [item for item in loaded.reported if item.check == "session_decision_evidence_changed"]
    assert len(reported) == 1
    assert reported[0].identity == (("decision_id", session.decisions[0].decision_id),)
    assert reported[0].message == (
        f"The evidence behind decision {session.decisions[0].decision_id} has changed "
        "since it was recorded. It was not applied."
    )


def test_use_saved_decisions_rerecords_only_the_unchanged_unit_question(tmp_path: Path):
    ingested, _identified, units, questions = _api_pair(tmp_path)
    session = open_session(ingested.source_file.path, units.findings)
    session = _apply(session, questions["A"], UserAction.DEFER, reason="later")
    session = _apply(session, questions["B"], UserAction.DEFER, reason="later")
    saved = save_session(session)
    original = saved.read_bytes()
    path = ingested.source_file.path
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("B.API", "B.XYZ", 1), encoding="utf-8", newline="\n")
    again_ingested = ingest(path, read_at=_WHEN)
    again_depth = classify(again_ingested)
    again_identified = identify(again_ingested, again_depth, local_tables_directory=tmp_path / "no-local")
    again_units = interpret(again_ingested, again_identified)
    again = {
        str(item.evidence.as_dict()["mnemonic"]): item
        for item in again_units.findings
        if item.check == "unit_unresolved"
    }
    assert set(again) == {"A", "B"}
    assert again["A"].id == questions["A"].id
    assert again["B"].id == questions["B"].id
    assert evidence_digest(again["A"]) == evidence_digest(questions["A"])
    assert evidence_digest(again["B"]) != evidence_digest(questions["B"])
    loaded = load_session(path, again_units.findings, app_data=tmp_path / "app")
    assert loaded.session.hold is True
    chosen = choose_changed_file(loaded.session, "USE_SAVED_DECISIONS", timestamp=_WHEN)
    assert [item.finding_id for item in chosen.applied] == [questions["A"].id]
    assert chosen.applied[0].disposition.issue_disposition is IssueDisposition.DEFERRED
    assert chosen.applied[0].evidence_digest == evidence_digest(again["A"])
    changed = [item for item in chosen.reported if item.check == "session_decision_evidence_changed"]
    assert len(changed) == 1
    assert changed[0].evidence.as_dict()["finding_id"] == questions["B"].id
    assert all(item.finding_id != questions["B"].id for item in chosen.applied)
    recorded = json.loads(original.decode("utf-8"))["file_hash"]
    assert isinstance(recorded, str)
    assert kept_session_path(path, recorded).read_bytes() == original


def test_a_version_1_session_is_refused_and_a_saved_file_is_schema_2(tmp_path: Path):
    path = _file(tmp_path)
    session_path(path).write_text(
        json.dumps(
            {
                "schema_version": "1",
                "file_hash": "a" * 64,
                "input_name": path.name,
                "decisions": [],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="'1'") as raised:
        load_session(path, ())
    assert "'1'" in str(raised.value)
    finding = _noted()
    saved = save_session(_apply(open_session(path, (finding,)), finding, UserAction.ACKNOWLEDGE))
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2"
    assert payload["decisions"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["evidence_digest"]) for item in payload["decisions"])
    save_preferences(tmp_path / "prefs", {"note": "as recorded"})
    preferences = json.loads(preferences_path(tmp_path / "prefs").read_text(encoding="utf-8"))
    assert preferences["schema_version"] == "1"
