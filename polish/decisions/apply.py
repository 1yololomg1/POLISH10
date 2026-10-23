"""Apply one decision to one finding. D79, D86, D91, D92, D93, UXC2, UXC4, OD-22.

Structure
---------
apply_decision
    Check the action, the scope, and the family, then append one record.
explicit_mappings
    THIS_ITEM mappings Stage 2 takes as an input.
unit_choices
    THIS_ITEM unit choices Stage 3 takes as an input.
listed_options
    The options a finding already lists.
current_decisions
    The records nothing else has superseded.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polish
from polish.decisions.local_store import (
    append_local_decision,
    saved_mnemonic,
    tables_directory,
    update_mnemonic,
)
from polish.model.mapping import ExplicitMapping, roles_stated_by_family
from polish.model.unit_choice import UnitChoice
from polish.decisions.versions import versions_in_force
from polish.model.findings import DecisionActor, DecisionRecord, DecisionScope, FindingRecord
from polish.model.ids import CurveRef
from polish.model.states import (
    BlockType,
    CurveRole,
    Disposition,
    IssueDisposition,
    UserAction,
    permitted_actions,
)
from polish.stages.s02_identify import UNCHANGED_MNEMONICS
from polish.tables.loader import ShippedTables, load_local_decisions, load_shipped_tables

# These three have no disposition. There is no function that clears them.
_NOT_DISPOSITIONABLE = frozenset(
    {
        BlockType.STRUCTURAL_DATA_FAILURE,
        BlockType.SCIENTIFIC_PRECONDITION_FAILURE,
        BlockType.EXPORT_CONFORMANCE_FAILURE,
    }
)
_REASON_REQUIRED = frozenset(
    {
        UserAction.CHOOSE_OPTION,
        UserAction.REJECT_PROPOSAL,
        UserAction.MAP_EXPLICITLY,
        UserAction.DEFER,
    }
)
_RESOLVED = frozenset(
    {UserAction.CHOOSE_OPTION, UserAction.REJECT_PROPOSAL, UserAction.MAP_EXPLICITLY}
)
# P15 names. A single decision does not use them.
_LATER_SCOPES = frozenset(
    {
        DecisionScope.SELECTED_ITEMS,
        DecisionScope.DETECTED_GROUP,
        DecisionScope.ALL_MATCHING_IN_FILE,
    }
)


def listed_options(finding: FindingRecord) -> tuple[str, ...]:
    """Return the options the finding already lists. D79, D92.

    ``options`` is the list when the finding has one. A pass question lists
    its pass names. A unit question lists its candidate canonical units.
    A finding that lists none returns an empty tuple. Nothing is added.
    """

    if not isinstance(finding, FindingRecord):
        raise TypeError("finding must be a FindingRecord")
    evidence = finding.evidence.as_dict()
    if "options" in evidence:
        return _string_list(evidence["options"], "options")
    if "passes" in evidence:
        payload = _json_list(evidence["passes"], "passes")
        names: list[str] = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("pass"), str):
                names.append(item["pass"])
        return tuple(dict.fromkeys(names))
    if "candidates" in evidence:
        payload = _json_list(evidence["candidates"], "candidates")
        units: list[str] = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("canonical_unit"), str):
                units.append(item["canonical_unit"])
        return tuple(dict.fromkeys(units))
    return ()


def current_decisions(decisions: tuple[DecisionRecord, ...], held_ids: frozenset[str]) -> dict[str, DecisionRecord]:
    """The decision in force for each finding. D79, UXC4.

    A record whose id is superseded, or held because it no longer matches,
    is history. It is not the decision in force.
    """

    superseded = {item.supersedes for item in decisions if item.supersedes is not None}
    current: dict[str, DecisionRecord] = {}
    for item in decisions:
        if item.decision_id in superseded or item.decision_id in held_ids:
            continue
        if item.finding_id in current:
            raise ValueError(f"Finding {item.finding_id} has two decisions that nothing supersedes.")
        current[item.finding_id] = item
    return current


def explicit_mappings(session: object) -> tuple[ExplicitMapping, ...]:
    """THIS_ITEM mappings for a later Stage 2 call. D91.

    LOCAL_RULE mappings are not returned. Stage 2 reads those from the
    local mnemonic table. A held session contributes none.

    Parameters
    ----------
    session:
        A Session. Typed as object here so this module can import Session
        without a cycle; the attribute check is the guard.
    """

    from polish.decisions.session import Session

    if not isinstance(session, Session):
        raise TypeError("session must be a Session")
    by_id = {finding.id: finding for finding in session.findings}
    built: list[ExplicitMapping] = []
    for decision in current_decisions(session.decisions, session.held_ids).values():
        if decision.action is not UserAction.MAP_EXPLICITLY or decision.scope is not DecisionScope.THIS_ITEM:
            continue
        if decision.actor is not DecisionActor.USER or not isinstance(decision.family_id, str):
            continue
        if decision.role is None:
            raise ValueError(f"Decision {decision.decision_id} maps a curve and has no role.")
        role = CurveRole(decision.role)
        if role is CurveRole.UNKNOWN:
            raise ValueError("A mapped curve cannot have role UNKNOWN.")
        finding = by_id.get(decision.finding_id)
        if finding is None:
            continue
        curves = [ref for ref in finding.subject_refs if isinstance(ref, CurveRef)]
        if len(curves) != 1:
            continue
        built.append(ExplicitMapping(curves[0], decision.family_id, role, decision.user_label))
    return tuple(built)


def unit_choices(session: object) -> tuple[UnitChoice, ...]:
    """THIS_ITEM unit choices for a later Stage 3 call. D93.

    A held session contributes none. LOCAL_RULE is not a unit choice.
    Stage 3 reads the returned records only when the caller passes them.

    Parameters
    ----------
    session:
        A Session. Typed as object here so this module can import Session
        without a cycle; the attribute check is the guard.
    """

    from polish.decisions.session import Session

    if not isinstance(session, Session):
        raise TypeError("session must be a Session")
    if session.hold:
        return ()
    by_id = {finding.id: finding for finding in session.findings}
    built: list[UnitChoice] = []
    for decision in current_decisions(session.decisions, session.held_ids).values():
        if decision.action is not UserAction.CHOOSE_OPTION or decision.scope is not DecisionScope.THIS_ITEM:
            continue
        if decision.actor is not DecisionActor.USER:
            continue
        finding = by_id.get(decision.finding_id)
        if finding is None or finding.stage != 3:
            continue
        if finding.block_type is not BlockType.AMBIGUOUS_INTERPRETATION:
            continue
        if "candidates" not in finding.evidence.as_dict():
            continue
        curves = [ref for ref in finding.subject_refs if isinstance(ref, CurveRef)]
        if len(curves) != 1 or len(finding.subject_refs) != 1:
            continue
        options = listed_options(finding)
        option = decision.chosen_option
        if not isinstance(option, str) or option not in options:
            shown = ", ".join(options) if options else "none"
            raise ValueError(
                f"Decision {decision.decision_id} chooses {option!r}, "
                f"which is not an option this finding lists ({shown})."
            )
        built.append(UnitChoice(curves[0], option, decision.decision_id))
    return tuple(built)


def apply_decision(
    session: object,
    finding: FindingRecord,
    action: UserAction,
    *,
    reason: str = "",
    chosen_option: str | None = None,
    scope: DecisionScope = DecisionScope.THIS_ITEM,
    family_id: str | None = None,
    user_label: str | None = None,
    role: str | None = None,
    confirm_saved_rule: bool = False,
    app_data: Path | None = None,
    timestamp: datetime | None = None,
) -> object:
    """Append one decision, or refuse it. D79, D86, D91, D92, UXC2, UXC4, OD-22.

    The action must be one ``permitted_actions`` allows for the finding's
    block type. STRUCTURAL_DATA_FAILURE, SCIENTIFIC_PRECONDITION_FAILURE and
    EXPORT_CONFORMANCE_FAILURE are refused.     MAP_EXPLICITLY requires a shipped
    family id and is refused for a leave-unchanged mnemonic. The role is the
    one every shipped mnemonic of that family states. When those mnemonics
    do not share one role, ``role`` is required and must be one of them.
    CHOOSE_OPTION
    must name an option the finding lists. CHOOSE_OPTION, REJECT_PROPOSAL,
    MAP_EXPLICITLY and DEFER require a reason, which is stored on the
    disposition.

    LOCAL_RULE writes only when this call asks for it. A mapping writes one
    mnemonic row. Any other saved rule writes one local-decision row.
    SELECTED_ITEMS, DETECTED_GROUP and ALL_MATCHING_IN_FILE are refused.

    Parameters
    ----------
    session:
        The session the decision is appended to. The previous records stay.
    finding:
        The one finding this decision is about.
    action:
        A user action.
    reason:
        The disposition reason. Required for the four actions named above.
    chosen_option:
        Required for CHOOSE_OPTION. For MAP_EXPLICITLY the family id is the choice.
    scope:
        THIS_ITEM, or LOCAL_RULE where that scope is allowed.
    family_id:
        Required for MAP_EXPLICITLY.
    user_label:
        Optional name kept on a mapping.
    role:
        Required for MAP_EXPLICITLY when the family's shipped mnemonics do
        not share one role. Ignored when they do, unless a different role
        is passed, which is refused.
    confirm_saved_rule:
        True only when the caller confirms replacing a saved all-wells rule.
    app_data:
        Injected app-data root. None uses %APPDATA%. Tests pass a temporary directory.
    timestamp:
        When the decision was made. None means now, in UTC.

    Returns
    -------
    Session
        A new session with the decision appended.

    Raises
    ------
    ValueError
        The action, scope, family, option, or saved rule is refused.
    """

    from polish.decisions.session import Session

    if not isinstance(session, Session):
        raise TypeError("session must be a Session")
    if not isinstance(finding, FindingRecord):
        raise TypeError("finding must be a FindingRecord")
    if not isinstance(action, UserAction):
        raise TypeError("action must be a UserAction")
    if not isinstance(scope, DecisionScope):
        raise TypeError("scope must be a DecisionScope")
    if not isinstance(reason, str):
        raise TypeError("reason must be a str")
    _refuse_undispositionable(finding.block_type)
    allowed = permitted_actions(finding.block_type)
    if action not in allowed:
        raise ValueError(
            f"{action.value} is not permitted for {finding.block_type.value}. "
            f"Permitted: {', '.join(sorted(item.value for item in allowed)) or 'none'}."
        )
    if action in _REASON_REQUIRED and reason.strip() == "":
        raise ValueError(f"{action.value} requires a reason.")
    _check_scope(finding, action, scope)
    family, mapped_role = _check_mapping(finding, action, scope, family_id, user_label, role)
    option = _check_option(finding, action, chosen_option, family)
    when = datetime.now(timezone.utc) if timestamp is None else timestamp
    if not isinstance(when, datetime):
        raise TypeError("timestamp must be a datetime or None")
    versions = versions_in_force()
    decision_id = f"dec-{uuid.uuid4().hex}"
    current = current_decisions(session.decisions, session.held_ids).get(finding.id)
    supersedes = None if current is None else current.decision_id
    if scope is DecisionScope.LOCAL_RULE:
        root = _app_data(app_data)
        directory = tables_directory(root)
        if action is UserAction.MAP_EXPLICITLY:
            supersedes = _write_mapping(
                directory,
                finding,
                decision_id,
                family,
                user_label,
                mapped_role,
                confirm_saved_rule,
                supersedes,
            )
        else:
            supersedes = _write_other(
                directory,
                finding,
                action,
                decision_id,
                option,
                reason,
                when,
                confirm_saved_rule,
                supersedes,
            )
    record = DecisionRecord(
        decision_id=decision_id,
        finding_id=finding.id,
        disposition=Disposition(_issue(action), reason),
        action=action,
        chosen_option=option,
        scope=scope,
        actor=DecisionActor.USER,
        timestamp=when,
        app_version=polish.__version__,
        versions_in_force=versions,
        input_file_hash=session.file_hash,
        supersedes=supersedes,
        family_id=family,
        user_label=user_label if action is UserAction.MAP_EXPLICITLY else None,
        role=None if mapped_role is None else mapped_role.value,
    )
    findings = session.findings
    if all(item.id != finding.id for item in findings):
        findings = findings + (finding,)
    return Session(
        schema_version=session.schema_version,
        input_path=session.input_path,
        file_hash=session.file_hash,
        decisions=session.decisions + (record,),
        findings=findings,
        held_ids=session.held_ids,
        hold=session.hold,
    )


def _refuse_undispositionable(block_type: BlockType) -> None:
    if block_type is BlockType.STRUCTURAL_DATA_FAILURE:
        raise ValueError("A structural data failure has no disposition.")
    if block_type is BlockType.SCIENTIFIC_PRECONDITION_FAILURE:
        raise ValueError("A scientific precondition failure has no disposition.")
    if block_type is BlockType.EXPORT_CONFORMANCE_FAILURE:
        raise ValueError("An export conformance failure has no disposition.")
    if block_type in _NOT_DISPOSITIONABLE:
        raise ValueError(f"{block_type.value} has no disposition.")


def _check_scope(finding: FindingRecord, action: UserAction, scope: DecisionScope) -> None:
    if scope is DecisionScope.THIS_ITEM:
        return
    if scope in _LATER_SCOPES:
        raise ValueError(f"{scope.value} is not available for a single decision.")
    if scope is not DecisionScope.LOCAL_RULE:
        raise ValueError(f"Unknown scope {scope.value}.")
    if _local_rule_allowed(finding, action):
        return
    raise ValueError("This decision can be recorded for this well only.")


def _local_rule_allowed(finding: FindingRecord, action: UserAction) -> bool:
    if action is UserAction.MAP_EXPLICITLY:
        return True
    evidence = finding.evidence.as_dict()
    options = set(listed_options(finding))
    if action is UserAction.CHOOSE_OPTION and {"KEEP", "EXCLUDE"} <= options:
        return True
    if action is UserAction.CHOOSE_OPTION and "passes" in evidence:
        return True
    return False


def _check_mapping(
    finding: FindingRecord,
    action: UserAction,
    scope: DecisionScope,
    family_id: str | None,
    user_label: str | None,
    role: str | None,
) -> tuple[str | None, CurveRole | None]:
    if action is not UserAction.MAP_EXPLICITLY:
        if family_id is not None:
            raise ValueError("family_id is only recorded for MAP_EXPLICITLY.")
        if role is not None:
            raise ValueError("role is only recorded for MAP_EXPLICITLY.")
        return None, None
    if user_label is not None and user_label.strip() == "":
        raise ValueError("user_label must be non-empty text or None")
    if family_id is None or family_id.strip() == "":
        raise ValueError("MAP_EXPLICITLY requires a family_id.")
    shipped = load_shipped_tables()
    if shipped.family(family_id) is None:
        raise ValueError(f"Family {family_id!r} is not in the shipped tables.")
    mnemonic = _mnemonic(finding)
    if mnemonic.upper() in UNCHANGED_MNEMONICS:
        raise ValueError(f"{mnemonic} is on the leave-unchanged list and is never mappable.")
    if scope is DecisionScope.LOCAL_RULE and shipped.mnemonic(mnemonic) is not None:
        raise ValueError(
            f"{mnemonic} is in the shipped mnemonic table. Saving it for all wells is refused, "
            "because a local row that clashes with a shipped one is applied to neither, "
            "and that would leave the curve unprocessed on every later well. This well is still allowed."
        )
    return family_id, _role_for_family(shipped, family_id, role)


def _check_option(
    finding: FindingRecord,
    action: UserAction,
    chosen_option: str | None,
    family_id: str | None,
) -> str | None:
    if action is UserAction.MAP_EXPLICITLY:
        if chosen_option is not None and chosen_option != family_id:
            raise ValueError("MAP_EXPLICITLY records the family id as the chosen option.")
        return family_id
    if action is not UserAction.CHOOSE_OPTION:
        if chosen_option is not None:
            raise ValueError(f"{action.value} does not take a chosen option.")
        return None
    if chosen_option is None or chosen_option.strip() == "":
        raise ValueError("CHOOSE_OPTION requires a chosen option.")
    options = listed_options(finding)
    if chosen_option not in options:
        shown = ", ".join(options) if options else "none"
        raise ValueError(f"{chosen_option!r} is not an option this finding lists ({shown}).")
    return chosen_option


def _role_for_family(shipped: ShippedTables, family_id: str, requested: str | None) -> CurveRole:
    """The role a mapping to this family uses. D91.

    ``shipped`` is the table load already done for this check. A family
    whose shipped mnemonics share one role uses that role. A family whose
    shipped mnemonics do not share one role requires ``requested`` to be
    one of those roles. UNKNOWN is never a mapping role.
    """

    if not isinstance(shipped, ShippedTables):
        raise TypeError("shipped must be the shipped tables")
    stated = list(roles_stated_by_family(shipped, family_id))
    if not stated:
        raise ValueError(f"Family {family_id!r} has no role in the shipped mnemonic table.")
    if len(stated) == 1:
        only = stated[0]
        if requested is not None and requested != only.value:
            raise ValueError(f"{family_id} takes the role {only.value}. {requested!r} is not that role.")
        return only
    if requested is None or requested.strip() == "":
        shown = ", ".join(item.value for item in stated)
        raise ValueError(
            f"MAP_EXPLICITLY to {family_id} requires a role chosen from that family's roles ({shown})."
        )
    match = [item for item in stated if item.value == requested]
    if not match:
        shown = ", ".join(item.value for item in stated)
        raise ValueError(f"Role {requested!r} is not one of {family_id}'s roles ({shown}).")
    return match[0]


def _mnemonic(finding: FindingRecord) -> str:
    value = finding.evidence.as_dict().get("mnemonic")
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("MAP_EXPLICITLY needs the curve's mnemonic on the finding.")
    return value


def _issue(action: UserAction) -> IssueDisposition:
    if action is UserAction.ACKNOWLEDGE:
        return IssueDisposition.ACCEPTED
    if action is UserAction.DEFER:
        return IssueDisposition.DEFERRED
    if action in _RESOLVED:
        return IssueDisposition.RESOLVED
    raise ValueError(f"{action.value} has no disposition result.")


def _write_mapping(
    directory: Path,
    finding: FindingRecord,
    decision_id: str,
    family_id: str | None,
    user_label: str | None,
    role: CurveRole | None,
    confirm: bool,
    supersedes: str | None,
) -> str | None:
    if not isinstance(family_id, str):
        raise ValueError("MAP_EXPLICITLY requires a family_id.")
    if not isinstance(role, CurveRole) or role is CurveRole.UNKNOWN:
        raise ValueError("MAP_EXPLICITLY requires a role, and that role is never UNKNOWN.")
    existing = saved_mnemonic(directory, _mnemonic(finding))
    previous = None
    if existing is not None:
        previous_value = existing.get("decision_id")
        if isinstance(previous_value, str) and previous_value.strip() != "":
            previous = previous_value
        saved_family = existing.get("family")
        changed = saved_family != family_id or existing.get("role") != role.value
        if changed and previous is not None and supersedes not in (None, previous):
            raise ValueError(
                "The saved rule was written by a different decision than the current one for this finding."
            )
    row: dict[str, str] = {
        "mnemonic": _mnemonic(finding),
        "family": family_id,
        "tier": "USER_DEFINED",
        "decision_id": decision_id,
        "role": role.value,
    }
    if user_label is not None:
        row["user_label"] = user_label
    written = update_mnemonic(directory, row, confirm=confirm)
    if written.status == "replaced" and written.previous_decision_id is not None:
        return written.previous_decision_id
    return supersedes


def _write_other(
    directory: Path,
    finding: FindingRecord,
    action: UserAction,
    decision_id: str,
    option: str | None,
    reason: str,
    when: datetime,
    confirm: bool,
    supersedes: str | None,
) -> str | None:
    evidence = finding.evidence.as_dict()
    if not isinstance(option, str):
        raise ValueError("A rule saved for all wells needs a chosen option.")
    if "passes" in evidence and action is UserAction.CHOOSE_OPTION:
        question_kind = "pass_selection"
    else:
        rule_id = evidence.get("rule_id")
        if not isinstance(rule_id, str) or rule_id.strip() == "":
            raise ValueError("A rule saved for all wells needs a rule id or a pass selection.")
        question_kind = rule_id
    match_key = evidence.get("family_id")
    if not isinstance(match_key, str) or match_key.strip() == "":
        raise ValueError("A rule saved for all wells needs a family id.")
    saved_rows = load_local_decisions(directory if directory.is_dir() else None)
    matched = [
        item
        for item in saved_rows
        if item.question_kind == question_kind and item.match_key == match_key
    ]
    if matched and matched[0].chosen_option != option and supersedes not in (None, matched[0].decision_id):
        raise ValueError(
            "The saved rule was written by a different decision than the current one for this finding."
        )
    written = append_local_decision(
        directory,
        {
            "decision_id": decision_id,
            "question_kind": question_kind,
            "match_key": match_key,
            "chosen_option": option,
            "reason": reason,
            "created": when.isoformat(),
        },
        confirm=confirm,
    )
    if written.status == "replaced" and written.previous_decision_id is not None:
        return written.previous_decision_id
    return supersedes


def _app_data(app_data: Path | None) -> Path:
    if app_data is not None:
        if not isinstance(app_data, Path):
            raise TypeError("app_data must be a Path or None")
        return app_data
    raw = os.environ.get("APPDATA")
    if not isinstance(raw, str) or raw.strip() == "":
        raise ValueError("APPDATA is not set, and no app-data directory was given.")
    return Path(raw)


def _string_list(value: object, what: str) -> tuple[str, ...]:
    payload = _json_list(value, what)
    if not all(isinstance(item, str) and item.strip() != "" for item in payload):
        raise ValueError(f"{what} must be a list of non-empty strings.")
    return tuple(payload)


def _json_list(value: object, what: str) -> list[object]:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be JSON text.")
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise ValueError(f"{what} must be a JSON list.")
    return payload
