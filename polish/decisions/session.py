"""Session file and preferences. D46, D47, D48, D63, D96, OD-22.

Structure
---------
Session, SessionLoad, SessionNotWritten
    Decisions in memory, and the finding raised when the folder cannot be written.
session_path, save_session, load_session
    The file beside the input. Decisions only. No sample values.
outstanding
    Findings with no terminal disposition, grouped by block type.
save_preferences
    App-data preferences with no file hash, well name, or depth.

A hash match re-applies decisions that still match a finding. A decision that
does not match is reported and kept. The same filename with a different hash
raises one ambiguous finding, listing USE_SAVED_DECISIONS and START_FRESH,
and applies none of that file until the user chooses. The old session file
is renamed and kept before either choice takes effect. The only carry-over
from another well is local_decisions.json, applied as DEFAULT_APPLIED and
listed, and that carry-over stays after either choice.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polish
from polish.decisions.versions import versions_in_force
from polish.model.findings import (
    DecisionActor,
    DecisionRecord,
    DecisionScope,
    Evidence,
    FindingRecord,
    evidence_digest,
    finding_key,
)
from polish.model.ids import FileHash
from polish.model.mapping import roles_stated_by_family
from polish.model.states import BlockType, Disposition, IssueDisposition, UserAction
from polish.tables.loader import load_local_decisions, load_shipped_tables

_SESSION_SCHEMA_VERSION = "2"
_PREFERENCES_SCHEMA_VERSION = "1"
_SESSION_SUFFIX = ".polish-session.json"
_DECISION_KEYS = frozenset(
    {
        "decision_id",
        "finding_id",
        "action",
        "issue_disposition",
        "reason",
        "chosen_option",
        "scope",
        "actor",
        "timestamp",
        "app_version",
        "versions_in_force",
        "supersedes",
        "evidence_digest",
        "family_id",
        "user_label",
        "role",
    }
)
_FILE_KEYS = frozenset({"schema_version", "file_hash", "input_name", "decisions"})
_TERMINAL = frozenset(
    {
        IssueDisposition.RESOLVED,
        IssueDisposition.ACCEPTED,
        IssueDisposition.DEFERRED,
        IssueDisposition.DECLINED_BY_POLISH,
    }
)
_HASH = re.compile(r"^[0-9a-fA-F]{64}$")
_CHANGED_FILE_OPTIONS = ("USE_SAVED_DECISIONS", "START_FRESH")
_SESSION_CHECKS: tuple[tuple[str, BlockType], ...] = (
    ("session_file_changed", BlockType.AMBIGUOUS_INTERPRETATION),
    ("session_decision_unmatched", BlockType.INFORMATIONAL),
    ("session_kept_file_exists", BlockType.INFORMATIONAL),
    ("session_held", BlockType.AMBIGUOUS_INTERPRETATION),
    ("session_not_written", BlockType.INFORMATIONAL),
    ("session_decision_evidence_changed", BlockType.INFORMATIONAL),
)
CHECKS = frozenset(name for name, _block in _SESSION_CHECKS)
_CHECK_BLOCK = dict(_SESSION_CHECKS)


@dataclass(frozen=True)
class Session:
    """Decisions for one input file. D46, D47.

    ``findings`` are in memory so outstanding() can be answered. They are
    not written. ``held_ids`` are records kept in history and not in force:
    a hash mismatch holds every decision from the file, and a decision that
    no longer matches its finding is held the same way.
    """

    schema_version: str
    input_path: Path
    file_hash: FileHash
    decisions: tuple[DecisionRecord, ...]
    findings: tuple[FindingRecord, ...]
    held_ids: frozenset[str] = frozenset()
    hold: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != _SESSION_SCHEMA_VERSION:
            raise ValueError(f"Session schema version {_SESSION_SCHEMA_VERSION!r} is the one this code reads.")
        if not isinstance(self.input_path, Path):
            raise TypeError("input_path must be a Path")
        if not isinstance(self.file_hash, FileHash):
            raise TypeError("file_hash must be a FileHash")
        if type(self.decisions) is not tuple or not all(isinstance(item, DecisionRecord) for item in self.decisions):
            raise TypeError("decisions must be a tuple of DecisionRecord")
        if type(self.findings) is not tuple or not all(isinstance(item, FindingRecord) for item in self.findings):
            raise TypeError("findings must be a tuple of FindingRecord")
        if type(self.held_ids) is not frozenset or not all(isinstance(item, str) for item in self.held_ids):
            raise TypeError("held_ids must be a frozenset of str")
        if type(self.hold) is not bool:
            raise TypeError("hold must be a bool")


@dataclass(frozen=True)
class SessionLoad:
    """What loading a session did. D48.

    ``applied`` are the file's decisions that still match. ``reported`` are
    the findings the user must see: a hash mismatch, or a decision that no
    longer matches. ``defaults_applied`` are all-wells rules applied on this
    well. ``session`` keeps the history, including decisions that were not
    applied.
    """

    session: Session
    applied: tuple[DecisionRecord, ...]
    reported: tuple[FindingRecord, ...]
    defaults_applied: tuple[DecisionRecord, ...]


class SessionNotWritten(Exception):
    """The session file was not written. The finding says why. D47.

    Nothing is written to another path unless the caller passed one.
    """

    def __init__(self, finding: FindingRecord) -> None:
        if not isinstance(finding, FindingRecord):
            raise TypeError("finding must be a FindingRecord")
        super().__init__(finding.message)
        self.finding = finding


def session_path(input_path: Path) -> Path:
    """The session file beside the input: ``<stem>.polish-session.json``. D47."""

    if not isinstance(input_path, Path):
        raise TypeError("input_path must be a Path")
    return input_path.with_name(input_path.stem + _SESSION_SUFFIX)


def open_session(input_path: Path, findings: tuple[FindingRecord, ...] = ()) -> Session:
    """A session for a file that has no decisions yet. D46.

    Hashes the input. Does not read or write a session file.
    """

    path = Path(input_path)
    return Session(_SESSION_SCHEMA_VERSION, path, FileHash.from_path(path), (), findings, frozenset(), False)


def save_session(session: Session, *, alternative_path: Path | None = None) -> Path:
    """Write decisions beside the input. D47, D96.

    The directory is the input file's directory. When that directory cannot
    be written, a finding is raised and nothing is written, unless
    ``alternative_path`` was passed. The alternative is used only in that
    case. The file contains no sample values.

    Parameters
    ----------
    session:
        The decisions to write. Findings are not written.
    alternative_path:
        A path the user chose, used only when the folder beside the input
        cannot be written.

    Returns
    -------
    Path
        The file that was written.

    Raises
    ------
    SessionNotWritten
        The folder beside the input cannot be written and no alternative
        was passed, or the alternative could not be written either.
    """

    if not isinstance(session, Session):
        raise TypeError("session must be a Session")
    if session.hold:
        raise SessionNotWritten(_held_file())
    target = session_path(session.input_path)
    if not _can_write(target):
        if alternative_path is None:
            raise SessionNotWritten(_unwritable(target))
        target = alternative_path
        if not _can_write(target):
            raise SessionNotWritten(_unwritable(target))
    text = _encode(session)
    _reject_numbers(json.loads(text), "session")
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, target)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise SessionNotWritten(_unwritable(target)) from exc
    return target


def load_session(
    input_path: Path,
    findings: tuple[FindingRecord, ...],
    *,
    app_data: Path | None = None,
) -> SessionLoad:
    """Load the session beside ``input_path``, or report that there is none. D48, D63, D96, OD-22.

    No session file means no decision from any other well's session is
    applied. Entries in local_decisions.json are still applied to matching
    findings, with actor DEFAULT_APPLIED, and listed. Mapping rows are not
    applied here; Stage 2 reads those through its loader.

    A matching hash re-validates each current decision. One that no longer
    matches a finding is reported and kept. A different hash on this filename
    reports the ambiguous finding, with USE_SAVED_DECISIONS and
    START_FRESH, and applies none of the file. The file stays where it is
    until the user chooses.

    Parameters
    ----------
    input_path:
        The raw file. Its hash is compared with the session.
    findings:
        The findings just raised for this file.
    app_data:
        Injected app-data root for local_decisions.json. None uses %APPDATA%.

    Returns
    -------
    SessionLoad
        The session, what was applied, and what was reported.
    """

    path = Path(input_path)
    file_hash = FileHash.from_path(path)
    stored_path = session_path(path)
    reported: list[FindingRecord] = []
    applied: list[DecisionRecord] = []
    held: set[str] = set()
    records: tuple[DecisionRecord, ...] = ()
    hold = False
    if stored_path.is_file():
        payload = _read_session(stored_path)
        if payload["input_name"] != path.name:
            raise ValueError(
                f"Session file {stored_path.name} records {payload['input_name']!r}, not {path.name!r}."
            )
        records = payload["decisions"]
        if payload["file_hash"] != file_hash.hex_digest:
            hold = True
            held.update(item.decision_id for item in records)
            reported.append(_hash_mismatch(file_hash, payload["file_hash"]))
        else:
            by_id = {item.id: item for item in findings}
            from polish.decisions.apply import current_decisions

            for decision in current_decisions(records, frozenset()).values():
                finding = by_id.get(decision.finding_id)
                if finding is None:
                    held.add(decision.decision_id)
                    reported.append(_unmatched(decision, "no finding has this id"))
                    continue
                if evidence_digest(finding) != decision.evidence_digest:
                    held.add(decision.decision_id)
                    reported.append(_evidence_changed(decision))
                    continue
                problem = _revalidate(decision, finding)
                if problem is None:
                    applied.append(decision)
                else:
                    held.add(decision.decision_id)
                    reported.append(_unmatched(decision, problem))
    defaults = _defaults(findings, records, held, file_hash, app_data)
    session = Session(
        _SESSION_SCHEMA_VERSION,
        path,
        file_hash,
        records + tuple(defaults),
        findings + tuple(item for item in reported),
        frozenset(held),
        hold,
    )
    return SessionLoad(session, tuple(applied), tuple(reported), tuple(defaults))


def kept_session_path(input_path: Path, recorded_hash: str) -> Path:
    """The kept session beside the input. D48.

    The name is ``<stem>.polish-session.<first 12 hex digits of the recorded
    file hash>.json``. The recorded hash is the hash stored in the old
    session, not the hash of the file as it is now.
    """

    if not isinstance(input_path, Path):
        raise TypeError("input_path must be a Path")
    if not isinstance(recorded_hash, str) or len(recorded_hash) < 12:
        raise ValueError("A kept session name needs the recorded file hash.")
    prefix = recorded_hash[:12].lower()
    if re.fullmatch(r"[0-9a-f]{12}", prefix) is None:
        raise ValueError("A kept session name needs twelve hex digits of the recorded file hash.")
    return input_path.with_name(f"{input_path.stem}.polish-session.{prefix}.json")


def choose_changed_file(
    session: Session,
    option: str,
    *,
    timestamp: datetime | None = None,
) -> SessionLoad:
    """Answer the changed-file question. D48.

    Before either choice takes effect, the session file beside the input is
    renamed to ``kept_session_path``. If that name already exists, a finding
    is raised and nothing is written or renamed. USE_SAVED_DECISIONS
    re-validates each decision that was in force in the old file. One that
    still matches is recorded again as a new USER decision for the new file
    hash, superseding the held one. One that does not match is listed and
    left only in the kept file. START_FRESH applies no decision from the old
    session. After either choice the held state is clear, so save_session
    can write the new file hash. Rules already saved for all wells stay.

    Parameters
    ----------
    session:
        A session load left on hold by a hash mismatch.
    option:
        USE_SAVED_DECISIONS or START_FRESH.
    timestamp:
        When the new decisions are recorded. None means now, in UTC.

    Returns
    -------
    SessionLoad
        ``applied`` are the decisions recorded again. ``reported`` are the
        held decisions that no longer match.

    Raises
    ------
    SessionNotWritten
        The kept name already exists. The old session file is unchanged.
    ValueError
        There is no changed-file question, or the option is not one of the two.
    """

    if not isinstance(session, Session):
        raise TypeError("session must be a Session")
    if not isinstance(option, str):
        raise TypeError("option must be a str")
    if not session.hold:
        raise ValueError("There is no changed-file question on this session.")
    if option not in _CHANGED_FILE_OPTIONS:
        shown = ", ".join(_CHANGED_FILE_OPTIONS)
        raise ValueError(f"{option!r} is not one of the changed-file options ({shown}).")
    source = session_path(session.input_path)
    if not source.is_file():
        raise ValueError(f"The session file beside {session.input_path.name} is not there.")
    raw = source.read_bytes()
    payload = _read_session(source)
    recorded = payload["file_hash"]
    if not isinstance(recorded, str):
        raise ValueError("The session file has no recorded file hash.")
    archive = kept_session_path(session.input_path, recorded)
    if archive.exists():
        raise SessionNotWritten(_kept_exists(archive))
    file_records = payload["decisions"]
    if not isinstance(file_records, tuple):
        raise ValueError("The session file has no decisions.")
    when = datetime.now(timezone.utc) if timestamp is None else timestamp
    if not isinstance(when, datetime):
        raise TypeError("timestamp must be a datetime or None")
    _changed_file_finding(session)
    versions = versions_in_force()
    os.replace(source, archive)
    if archive.read_bytes() != raw:
        raise ValueError("The kept session file is not the file that was renamed.")
    fresh, reported, history = _replay(session, file_records, option, when, versions)
    file_ids = {item.decision_id for item in file_records}
    defaults = tuple(
        item
        for item in session.decisions
        if item.decision_id not in file_ids and item.actor is DecisionActor.DEFAULT_APPLIED
    )
    hash_finding = _changed_file_finding(session)
    choice = _choice_record(hash_finding, option, session.file_hash, when, versions)
    cleared = Session(
        session.schema_version,
        session.input_path,
        session.file_hash,
        history + fresh + defaults + (choice,),
        session.findings + reported,
        frozenset(),
        False,
    )
    return SessionLoad(cleared, fresh, reported, defaults)


def outstanding(session: Session) -> dict[BlockType, tuple[FindingRecord, ...]]:
    """Findings with no terminal disposition, grouped by block type. D79, D86.

    Held decisions are not a disposition. The groups follow BlockType order.
    A block type with nothing outstanding is absent.
    """

    if not isinstance(session, Session):
        raise TypeError("session must be a Session")
    from polish.decisions.apply import current_decisions

    current = current_decisions(session.decisions, session.held_ids)
    groups: dict[BlockType, list[FindingRecord]] = {}
    for finding in session.findings:
        decision = current.get(finding.id)
        if decision is not None and decision.disposition.issue_disposition in _TERMINAL:
            continue
        groups.setdefault(finding.block_type, []).append(finding)
    return {block: tuple(groups[block]) for block in BlockType if block in groups}


def preferences_path(app_data: Path) -> Path:
    """``<app-data>/POLISH/preferences.json``. D47. Does not create it."""

    if not isinstance(app_data, Path):
        raise TypeError("app_data must be a Path")
    return app_data / "POLISH" / "preferences.json"


def save_preferences(app_data: Path, values: dict[str, str]) -> None:
    """Write preferences that carry no well information. D47, D96.

    A file hash, a well name, or a depth is refused and nothing is written.
    ``app_data`` is the injected root. This does not read %APPDATA% itself.

    Parameters
    ----------
    app_data:
        The directory that stands in for %APPDATA%.
    values:
        Preference text. Keys and values are strings.

    Raises
    ------
    ValueError
        A key or value is a file hash, a well name, or a depth.
    """

    if not isinstance(values, dict):
        raise TypeError("preferences must be a dict of strings")
    for key, value in values.items():
        if not isinstance(key, str) or key.strip() == "":
            raise ValueError("A preference name must be a non-empty string.")
        if not isinstance(value, str):
            raise ValueError(f"Preference {key!r} must be text.")
        lowered = key.lower()
        if "hash" in lowered or "well" in lowered or "depth" in lowered:
            raise ValueError("Preferences cannot contain a file hash, a well name, or a depth.")
        if _HASH.fullmatch(value):
            raise ValueError("Preferences cannot contain a file hash.")
        if re.search(r"depth", value, re.IGNORECASE):
            raise ValueError("Preferences cannot contain a depth.")
    payload = {"schema_version": _PREFERENCES_SCHEMA_VERSION, "values": dict(sorted(values.items()))}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path = preferences_path(app_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _defaults(
    findings: tuple[FindingRecord, ...],
    records: tuple[DecisionRecord, ...],
    held: set[str],
    file_hash: FileHash,
    app_data: Path | None,
) -> list[DecisionRecord]:
    from polish.decisions.apply import current_decisions, listed_options
    from polish.decisions.local_store import tables_directory

    if app_data is None:
        raw = os.environ.get("APPDATA")
        if not isinstance(raw, str) or raw.strip() == "":
            return []
        directory = tables_directory(Path(raw))
    else:
        directory = tables_directory(app_data)
    saved = load_local_decisions(directory if directory.is_dir() else None)
    if not saved:
        return []
    in_force = current_decisions(records, frozenset(held))
    built: list[DecisionRecord] = []
    versions = versions_in_force()
    when = datetime.now(timezone.utc)
    for finding in findings:
        if finding.id in in_force:
            continue
        evidence = finding.evidence.as_dict()
        match_key = evidence.get("family_id")
        if not isinstance(match_key, str):
            continue
        options = listed_options(finding)
        for row in saved:
            if row.match_key != match_key:
                continue
            if row.question_kind == "pass_selection":
                if "passes" not in evidence:
                    continue
            elif evidence.get("rule_id") != row.question_kind:
                continue
            if row.chosen_option not in options:
                continue
            action = UserAction.CHOOSE_OPTION
            built.append(
                DecisionRecord(
                    decision_id=f"dec-default-{row.decision_id}-{finding.id}",
                    finding_id=finding.id,
                    disposition=Disposition(IssueDisposition.RESOLVED, row.reason),
                    action=action,
                    chosen_option=row.chosen_option,
                    scope=DecisionScope.LOCAL_RULE,
                    actor=DecisionActor.DEFAULT_APPLIED,
                    timestamp=when,
                    app_version=polish.__version__,
                    versions_in_force=versions,
                    input_file_hash=file_hash,
                    supersedes=None,
                    evidence_digest=evidence_digest(finding),
                    family_id=None,
                    user_label=None,
                )
            )
            break
    return built


def _revalidate(decision: DecisionRecord, finding: FindingRecord) -> str | None:
    from polish.decisions.apply import listed_options
    from polish.model.states import permitted_actions
    from polish.stages.s02_identify import UNCHANGED_MNEMONICS

    if decision.action is None or decision.action not in permitted_actions(finding.block_type):
        return "the action is not permitted for the finding"
    if decision.action is UserAction.CHOOSE_OPTION and decision.chosen_option not in listed_options(finding):
        return "the chosen option is not one the finding lists"
    if decision.action is UserAction.MAP_EXPLICITLY:
        mnemonic = finding.evidence.as_dict().get("mnemonic")
        if not isinstance(mnemonic, str) or mnemonic.upper() in UNCHANGED_MNEMONICS:
            return "the mnemonic is not mappable"
        shipped = load_shipped_tables()
        if decision.family_id is None or shipped.family(decision.family_id) is None:
            return "the family is not in the shipped tables"
        stated = tuple(role.value for role in roles_stated_by_family(shipped, decision.family_id))
        if decision.role not in stated:
            return "the role is not one this family states"
    return None


def _replay(
    session: Session,
    file_records: tuple[DecisionRecord, ...],
    option: str,
    when: datetime,
    versions: tuple[tuple[str, str], ...],
) -> tuple[tuple[DecisionRecord, ...], tuple[FindingRecord, ...], tuple[DecisionRecord, ...]]:
    if option == "START_FRESH":
        return (), (), ()
    from polish.decisions.apply import current_decisions

    by_id = {item.decision_id: item for item in file_records}
    findings = {item.id: item for item in session.findings}
    keep: set[str] = set()
    fresh: list[DecisionRecord] = []
    reported: list[FindingRecord] = []
    for decision in current_decisions(file_records, frozenset()).values():
        finding = findings.get(decision.finding_id)
        if finding is None:
            reported.append(_unmatched(decision, "no finding has this id"))
            continue
        if evidence_digest(finding) != decision.evidence_digest:
            reported.append(_evidence_changed(decision))
            continue
        problem = _revalidate(decision, finding)
        if problem is None:
            keep.update(_ancestor_ids(decision, by_id))
            fresh.append(_rerecord(decision, finding, session.file_hash, when, versions))
        else:
            reported.append(_unmatched(decision, problem))
    history = tuple(item for item in file_records if item.decision_id in keep)
    return tuple(fresh), tuple(reported), history


def _ancestor_ids(decision: DecisionRecord, by_id: dict[str, DecisionRecord]) -> set[str]:
    found: set[str] = set()
    cursor: DecisionRecord | None = decision
    while cursor is not None and cursor.decision_id not in found:
        found.add(cursor.decision_id)
        parent = cursor.supersedes
        cursor = None if parent is None else by_id.get(parent)
    return found


def _rerecord(
    decision: DecisionRecord,
    finding: FindingRecord,
    file_hash: FileHash,
    when: datetime,
    versions: tuple[tuple[str, str], ...],
) -> DecisionRecord:
    """A new USER decision for the same finding, on the file as it is now. D48, D96."""

    return DecisionRecord(
        decision_id=f"dec-{uuid.uuid4().hex}",
        finding_id=decision.finding_id,
        disposition=decision.disposition,
        action=decision.action,
        chosen_option=decision.chosen_option,
        scope=decision.scope,
        actor=DecisionActor.USER,
        timestamp=when,
        app_version=polish.__version__,
        versions_in_force=versions,
        input_file_hash=file_hash,
        supersedes=decision.decision_id,
        evidence_digest=evidence_digest(finding),
        family_id=decision.family_id,
        user_label=decision.user_label,
        role=decision.role,
    )


def _changed_file_finding(session: Session) -> FindingRecord:
    """The changed-file question on this session, found by its check. D48, D96."""

    found = [item for item in session.findings if item.check == "session_file_changed"]
    if len(found) != 1:
        raise ValueError("The changed-file question is not on this session.")
    return found[0]


def _choice_record(
    finding: FindingRecord,
    option: str,
    file_hash: FileHash,
    when: datetime,
    versions: tuple[tuple[str, str], ...],
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"dec-{uuid.uuid4().hex}",
        finding_id=finding.id,
        disposition=Disposition(IssueDisposition.RESOLVED, ""),
        action=UserAction.CHOOSE_OPTION,
        chosen_option=option,
        scope=DecisionScope.THIS_ITEM,
        actor=DecisionActor.USER,
        timestamp=when,
        app_version=polish.__version__,
        versions_in_force=versions,
        input_file_hash=file_hash,
        supersedes=None,
        evidence_digest=evidence_digest(finding),
    )


def _session_finding(
    check: str,
    identity: tuple[tuple[str, str], ...],
    message: str,
    evidence_fields: tuple[tuple[str, str | int | float | bool], ...] = (),
    subject_refs: tuple[FileHash, ...] = (),
) -> FindingRecord:
    """One stage-0 session finding. The id is its check and identity. D96."""

    if check not in CHECKS:
        raise ValueError(f"Check {check!r} is not a session check.")
    block = _CHECK_BLOCK[check]
    return FindingRecord(
        id=finding_key(0, check, identity),
        stage=0,
        check=check,
        identity=identity,
        block_type=block,
        subject_refs=subject_refs,
        evidence=Evidence(block.value, evidence_fields),
        message=message,
        created_at=datetime.now(timezone.utc),
    )


def _kept_exists(archive: Path) -> FindingRecord:
    return _session_finding(
        "session_kept_file_exists",
        (("file", archive.name),),
        f"A kept session file already exists at {archive.name}. Nothing was written.",
        (("reason", "a kept session file already exists"),),
    )


def _hash_mismatch(current: FileHash, stored: str) -> FindingRecord:
    """The file hash no longer matches. Identity is the check alone. D48, D96."""

    return _session_finding(
        "session_file_changed",
        (),
        "this looks like the same well but the raw file has changed since these decisions were made",
        (
            ("input_file_hash", current.hex_digest),
            ("options", json.dumps(list(_CHANGED_FILE_OPTIONS))),
            ("session_file_hash", stored),
        ),
        (current,),
    )


def _unmatched(decision: DecisionRecord, problem: str) -> FindingRecord:
    return _session_finding(
        "session_decision_unmatched",
        (("decision_id", decision.decision_id),),
        (
            f"Decision {decision.decision_id} no longer matches finding {decision.finding_id} "
            f"({problem}) and was not applied."
        ),
        (("decision_id", decision.decision_id), ("finding_id", decision.finding_id)),
        (decision.input_file_hash,),
    )


def _evidence_changed(decision: DecisionRecord) -> FindingRecord:
    """The finding is still there, but what was observed has changed. D96."""

    return _session_finding(
        "session_decision_evidence_changed",
        (("decision_id", decision.decision_id),),
        (
            f"The evidence behind decision {decision.decision_id} has changed "
            "since it was recorded. It was not applied."
        ),
        (("decision_id", decision.decision_id), ("finding_id", decision.finding_id)),
        (decision.input_file_hash,),
    )


def _held_file() -> FindingRecord:
    """Saving is refused while the changed-file question is unanswered. D48, D96."""

    return _session_finding(
        "session_held",
        (),
        "this looks like the same well but the raw file has changed since these decisions were made",
    )


def _unwritable(target: Path) -> FindingRecord:
    # Stage 0 pairs the session with the input file. The record needs a stage
    # number; the session is not a later stage's result. D47, D96.
    return _session_finding(
        "session_not_written",
        (),
        (
            f"The session could not be written to {target}. "
            "That directory is not writable. Nothing was written elsewhere."
        ),
        (("reason", "session directory is not writable"),),
    )


def _can_write(path: Path) -> bool:
    if path.exists() and not path.is_file():
        return False
    directory = path.parent
    if not directory.is_dir():
        return False
    probe = directory / ".polish-write-probe"
    try:
        probe.write_text("", encoding="utf-8")
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        return True
    return True


def _encode(session: Session) -> str:
    """Decisions only. Findings, depths, and sample values are not written. D47."""

    payload = {
        "schema_version": session.schema_version,
        "file_hash": session.file_hash.hex_digest,
        "input_name": session.input_path.name,
        "decisions": [_decision_json(item) for item in session.decisions],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _decision_json(record: DecisionRecord) -> dict[str, object]:
    versions = {name: version for name, version in record.versions_in_force}
    return {
        "decision_id": record.decision_id,
        "finding_id": record.finding_id,
        "action": None if record.action is None else record.action.value,
        "issue_disposition": record.disposition.issue_disposition.value,
        "reason": record.disposition.reason,
        "chosen_option": record.chosen_option,
        "scope": record.scope.value,
        "actor": record.actor.value,
        "timestamp": record.timestamp.isoformat(),
        "app_version": record.app_version,
        "versions_in_force": versions,
        "supersedes": record.supersedes,
        "evidence_digest": record.evidence_digest,
        "family_id": record.family_id,
        "user_label": record.user_label,
        "role": record.role,
    }


def _read_session(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _reject_numbers(payload, path.name)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object.")
    unknown = set(payload) - _FILE_KEYS
    if unknown:
        raise ValueError(f"{path.name} has unknown keys: {sorted(unknown)}.")
    version = payload.get("schema_version")
    if version != _SESSION_SCHEMA_VERSION:
        raise ValueError(
            f"{path.name} has session schema version {version!r}. "
            f"This code reads session schema {_SESSION_SCHEMA_VERSION!r}."
        )
    file_hash = payload.get("file_hash")
    input_name = payload.get("input_name")
    if not isinstance(file_hash, str) or FileHash(file_hash).hex_digest != file_hash:
        raise ValueError(f"{path.name} file_hash must be a SHA-256 hex digest.")
    if not isinstance(input_name, str) or input_name.strip() == "":
        raise ValueError(f"{path.name} input_name must be the input file name.")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError(f"{path.name} decisions must be a list.")
    decisions = tuple(_decode_decision(item, file_hash) for item in raw_decisions)
    return {"file_hash": file_hash, "input_name": input_name, "decisions": decisions}


def _decode_decision(item: object, file_hash: str) -> DecisionRecord:
    if not isinstance(item, dict):
        raise ValueError("A session decision must be an object.")
    unknown = set(item) - _DECISION_KEYS
    if unknown:
        raise ValueError(f"A session decision has unknown keys: {sorted(unknown)}.")
    missing = _DECISION_KEYS - set(item)
    if missing:
        raise ValueError(f"A session decision is missing {sorted(missing)}.")
    action_text = item["action"]
    action = None if action_text is None else UserAction(action_text)
    versions = item["versions_in_force"]
    if not isinstance(versions, dict) or not versions:
        raise ValueError("versions_in_force must be an object of table versions.")
    pairs: list[tuple[str, str]] = []
    for name, version in versions.items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError("A version name and its version are strings.")
        pairs.append((name, version))
    family = item["family_id"]
    label = item["user_label"]
    role = item["role"]
    chosen = item["chosen_option"]
    supersedes = item["supersedes"]
    digest = item["evidence_digest"]
    return DecisionRecord(
        decision_id=_text(item["decision_id"], "decision_id"),
        finding_id=_text(item["finding_id"], "finding_id"),
        disposition=Disposition(IssueDisposition(item["issue_disposition"]), _text_allow_empty(item["reason"])),
        action=action,
        chosen_option=None if chosen is None else _text(chosen, "chosen_option"),
        scope=DecisionScope(item["scope"]),
        actor=DecisionActor(item["actor"]),
        timestamp=datetime.fromisoformat(_text(item["timestamp"], "timestamp")),
        app_version=_text(item["app_version"], "app_version"),
        versions_in_force=tuple(pairs),
        input_file_hash=FileHash(file_hash),
        supersedes=None if supersedes is None else _text(supersedes, "supersedes"),
        evidence_digest=_text(digest, "evidence_digest"),
        family_id=None if family is None else _text(family, "family_id"),
        user_label=None if label is None else _text(label, "user_label"),
        role=None if role is None else _text(role, "role"),
    )


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{what} must be a non-empty string.")
    return value


def _text_allow_empty(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("reason must be a string.")
    return value


def _reject_numbers(value: object, where: str) -> None:
    """Refuse a JSON number. A session stores no sample values. D47."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise ValueError(f"{where} contains a number. A session stores no data values.")
    if isinstance(value, list):
        for item in value:
            _reject_numbers(item, where)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_numbers(item, where)
        return
    raise ValueError(f"{where} contains {type(value).__name__}, which is not session text.")
