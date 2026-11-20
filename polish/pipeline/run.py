"""Run Stages 0-4 of one well with the answers saved beside it. D98, D48, C5.

Structure
---------
WellRun
    The stage results that ran, the question set, and the answers a stage used.
StageNotice
    One progress call. It is not stored on the run.
run_well
    Read the file and the session. Write nothing. Report each stage as it
    starts and as its final result for this run is known.

Answers are applied in dependency order. A NULL choice is matched against
the Stage 0 run with no answers and applied before the depth-unit choice.
A depth-unit choice is matched against the Stage 1 run with no answers.
A mapping is matched against the
run that asks the unrecognised-name notice or the incompatible-unit warning.
A pass choice is matched against identification after those mappings. A unit
choice is matched against interpretation after the mappings and the pass
choices. A question that is not raised in its own kind's run is not extracted.
Runs are not merged. An answer a stage used stays linked to the finding it
answered, including the evidence fingerprint, so a later load can check it.
Answers re-run stages, so Stage 3 may start before Stage 2 finishes. The
progress hook does not promise an order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from polish.decisions.apply import depth_unit_choices, explicit_mappings, null_choices, pass_choices, unit_choices
from polish.decisions.session import SessionLoad, load_session
from polish.model.findings import DecisionRecord, FindingRecord
from polish.model.states import StageStatus, UnitInterpretation, ValidationOutcome
from polish.stages.s00_ingest import IngestResult, ingest
from polish.stages.s01_depth import DepthAxisResult, classify
from polish.stages.s02_identify import IdentificationResult, identify
from polish.stages.s03_units import UnitsResult, interpret
from polish.stages.s04_validity import ValidityResult, assess

_STAGE_COUNT = 5


@dataclass(frozen=True)
class WellRun:
    """Stages 0-4 of one well, with the answers this run used. D98, D48.

    ``depth``, ``identification``, ``units`` and ``validity`` are None when
    that stage did not run. ``findings`` is the question set: the findings
    the final stages raised, plus each applied answer's question as it was
    asked when that question is no longer raised. ``applied`` are the
    decisions a stage used, in order: NULL choices, depth-unit choices,
    mappings, pass choices, unit choices.
    ``reasked`` are decisions a stage used whose question is raised again.
    ``session_load`` is the load of that question set. This run does not
    answer the changed-file question.
    """

    path: Path
    ingest: IngestResult
    depth: DepthAxisResult | None
    identification: IdentificationResult | None
    units: UnitsResult | None
    validity: ValidityResult | None
    findings: tuple[FindingRecord, ...]
    session_load: SessionLoad
    applied: tuple[DecisionRecord, ...]
    reasked: tuple[DecisionRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        if not isinstance(self.ingest, IngestResult):
            raise TypeError("ingest must be an IngestResult")
        if self.depth is not None and not isinstance(self.depth, DepthAxisResult):
            raise TypeError("depth must be a DepthAxisResult or None")
        if self.identification is not None and not isinstance(self.identification, IdentificationResult):
            raise TypeError("identification must be an IdentificationResult or None")
        if self.units is not None and not isinstance(self.units, UnitsResult):
            raise TypeError("units must be a UnitsResult or None")
        if self.validity is not None and not isinstance(self.validity, ValidityResult):
            raise TypeError("validity must be a ValidityResult or None")
        if type(self.findings) is not tuple or any(not isinstance(item, FindingRecord) for item in self.findings):
            raise TypeError("findings must be a tuple of FindingRecord")
        if not isinstance(self.session_load, SessionLoad):
            raise TypeError("session_load must be a SessionLoad")
        if type(self.applied) is not tuple or any(not isinstance(item, DecisionRecord) for item in self.applied):
            raise TypeError("applied must be a tuple of DecisionRecord")
        if type(self.reasked) is not tuple or any(not isinstance(item, DecisionRecord) for item in self.reasked):
            raise TypeError("reasked must be a tuple of DecisionRecord")


@dataclass(frozen=True)
class StageNotice:
    """One progress call for one stage. D98.

    The hook is told when a stage first starts and when the result ``run_well``
    returns for that stage is known. A started call has no status. A finished
    call carries the status and one line of facts. The notice is not written
    and is not stored on the WellRun.
    """

    stage: int
    phase: str
    status: str | None
    summary: str

    def __post_init__(self) -> None:
        if type(self.stage) is not int or self.stage not in range(_STAGE_COUNT):
            raise ValueError("stage must be 0, 1, 2, 3, or 4.")
        if self.phase not in ("started", "finished"):
            raise ValueError("phase must be 'started' or 'finished'.")
        if self.phase == "started":
            if self.status is not None or self.summary != "":
                raise ValueError("A started notice has no status and no summary.")
        elif not isinstance(self.status, str) or self.status.strip() == "":
            raise ValueError("A finished notice names the status.")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a str.")


ProgressHook = Callable[[StageNotice], None]

_RESOLVED_UNITS = frozenset({UnitInterpretation.RECOGNISED, UnitInterpretation.INCOMPATIBLE_WITH_FAMILY})
_UNRESOLVED_UNITS = frozenset(
    {UnitInterpretation.ABSENT, UnitInterpretation.EMPTY, UnitInterpretation.UNRECOGNISED}
)


class _Reporter:
    """Sends each stage's start once and its final finish. Writes nothing."""

    def __init__(self, hook: ProgressHook | None) -> None:
        self._hook = hook
        self._started: set[int] = set()

    def start(self, stage: int) -> None:
        if stage in self._started:
            return
        self._started.add(stage)
        if self._hook is not None:
            self._hook(StageNotice(stage, "started", None, ""))

    def finish(self, stage: int, result: object) -> None:
        if self._hook is None:
            return
        status = getattr(result, "status", None)
        if not isinstance(status, StageStatus):
            raise TypeError("A finished stage has no status.")
        self._hook(StageNotice(stage, "finished", status.value, stage_line(stage, result)))


def stage_line(stage: int, result: object) -> str:
    """The one line of facts for a finished stage. D98.

    The progress hook and the window use this sentence. It is computed from
    the result ``run_well`` returns and is not stored on the run.
    """

    if stage == 0:
        return _stage0_line(result)
    if stage == 1:
        return _stage1_line(result)
    if stage == 2:
        return _stage2_line(result)
    if stage == 3:
        return _stage3_line(result)
    if stage == 4:
        return _stage4_line(result)
    raise ValueError(f"stage must be 0, 1, 2, 3, or 4, not {stage}.")


def stage_lines(run: WellRun) -> tuple[str, ...]:
    """One line for each stage this run produced, in stage order. D98."""

    lines = [stage_line(0, run.ingest)]
    for number, part in (
        (1, run.depth),
        (2, run.identification),
        (3, run.units),
        (4, run.validity),
    ):
        if part is not None:
            lines.append(stage_line(number, part))
    return tuple(lines)


def _count_word(count: int, singular: str) -> str:
    word = singular if count == 1 else f"{singular}s"
    return f"{count} {word}"


def _stage0_line(result: object) -> str:
    if not isinstance(result, IngestResult):
        raise TypeError("Stage 0 result must be an IngestResult.")
    dataset = result.dataset
    if dataset is not None:
        curves = len(dataset.curves)
        rows = int(dataset.grid.values.size)
    else:
        curves = len(result.audit.curves)
        rows = 0 if result.audit.depth_step_count is None else result.audit.depth_step_count
    return f"Stage 0: {_count_word(curves, 'curve')}, {_count_word(rows, 'row')}"


def _stage1_line(result: object) -> str:
    if not isinstance(result, DepthAxisResult):
        raise TypeError("Stage 1 result must be a DepthAxisResult.")
    direction = "not set" if result.direction is None else result.direction.value
    regularity = "not set" if result.regularity is None else result.regularity.value
    step = "not set" if result.declared_step is None else result.declared_step.text
    return f"Stage 1: {direction}, {regularity}, step {step}"


def _stage2_line(result: object) -> str:
    if not isinstance(result, IdentificationResult):
        raise TypeError("Stage 2 result must be an IdentificationResult.")
    groups = sum(1 for item in result.findings if item.check == "prefix_group")
    group_word = "prefix group" if groups == 1 else "prefix groups"
    return f"Stage 2: {_count_word(len(result.curves), 'curve')}, {groups} {group_word}"


def _stage3_line(result: object) -> str:
    if not isinstance(result, UnitsResult):
        raise TypeError("Stage 3 result must be a UnitsResult.")
    resolved = sum(1 for curve in result.curves if curve.outcome in _RESOLVED_UNITS)
    unresolved = sum(1 for curve in result.curves if curve.outcome in _UNRESOLVED_UNITS)
    return f"Stage 3: {_count_word(resolved, 'unit')} resolved, {unresolved} unresolved"


def _stage4_line(result: object) -> str:
    if not isinstance(result, ValidityResult):
        raise TypeError("Stage 4 result must be a ValidityResult.")
    parts: list[str] = []
    for outcome in ValidationOutcome:
        count = sum(1 for curve in result.curves if curve.outcome is outcome)
        if count:
            parts.append(f"{count} {outcome.value}")
    shown = ", ".join(parts) if parts else "no curves"
    return f"Stage 4: {shown}"


def run_well(
    path: Path | str,
    *,
    read_at: datetime | None = None,
    local_tables_directory: Path | None = None,
    app_data: Path | None = None,
    progress: ProgressHook | None = None,
) -> WellRun:
    """Run Stages 0-4 with the session saved beside ``path``. D98, D48, D66, D91, D93, D94, C5.

    Does not write a session file, a table, or a folder. Does not answer the
    changed-file question. A stage runs only when the stage before it is
    COMPLETE. Answers are applied only when the session is not on hold, and
    only from the run that raises that kind of question. A NULL choice is
    read against Stage 0 with no answers and applied before the depth-unit
    choice. A depth-unit choice is read against Stage 1 with no answers, then
    applied before later stages.

    ``progress``, when given, is called once when a stage first starts and
    once when the result this function returns for that stage is known. It
    writes nothing and does not change the WellRun. Stage 3 may start before
    Stage 2 finishes, because answers re-run those stages.

    Parameters
    ----------
    path:
        The LAS file. Opened read-only by Stage 0.
    read_at:
        When Stage 0 records the read. None lets Stage 0 choose.
    local_tables_directory:
        Local mnemonic tables, or None for the app-data path. A missing
        directory is empty and is not created.
    app_data:
        Injected app-data root for all-wells rules. None uses %APPDATA%.
        Tests pass a temporary directory.
    progress:
        Optional hook. None reports nothing.

    Returns
    -------
    WellRun
        The stages that ran, the question set, and the answers a stage used.

    Raises
    ------
    FileNotFoundError
        ``path`` is not a file. The hook is not called.
    UnclassifiedIngest
        Stage 0 cannot classify a condition. Propagated unchanged. The hook
        may already have been told that Stage 0 started.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Not a file: {file_path}")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be a callable or None.")
    reporter = _Reporter(progress)
    reporter.start(0)
    ingested = ingest(file_path, read_at=read_at)
    null_load = load_session(file_path, _collect(ingested), app_data=app_data)
    chosen_null = () if null_load.session.hold else null_choices(null_load.session)
    if chosen_null:
        ingested = ingest(file_path, read_at=read_at, null_choices=chosen_null)
    null_applied, null_reasked, null_restored = _null_used(ingested, null_load)
    reporter.finish(0, ingested)
    depth: DepthAxisResult | None = None
    identification: IdentificationResult | None = None
    units: UnitsResult | None = None
    validity: ValidityResult | None = None
    applied: tuple[DecisionRecord, ...] = null_applied
    reasked: tuple[DecisionRecord, ...] = null_reasked
    restored: tuple[FindingRecord, ...] = null_restored
    if ingested.status is StageStatus.COMPLETE and ingested.dataset is not None:
        reporter.start(1)
        depth = classify(ingested)
        depth_load = load_session(file_path, _collect(ingested, depth), app_data=app_data)
        if not depth_load.session.hold:
            choices = depth_unit_choices(depth_load.session)
            if choices:
                depth = classify(ingested, depth_unit_choices=choices)
        depth_applied, depth_reasked, depth_restored = _depth_used(depth, depth_load)
        applied = null_applied + depth_applied
        reasked = null_reasked + depth_reasked
        restored = null_restored + depth_restored
        reporter.finish(1, depth)
        if depth.status is StageStatus.COMPLETE:
            identification, units, validity, later_applied, later_reasked, later_restored = _with_answers(
                file_path,
                ingested,
                depth,
                local_tables_directory,
                app_data,
                reporter,
            )
            applied = null_applied + depth_applied + later_applied
            reasked = null_reasked + depth_reasked + later_reasked
            restored = null_restored + depth_restored + later_restored
    findings = _question_set((ingested, depth, identification, units, validity), restored)
    session_load = load_session(file_path, findings, app_data=app_data)
    return WellRun(
        file_path,
        ingested,
        depth,
        identification,
        units,
        validity,
        findings,
        session_load,
        applied,
        reasked,
    )


def _with_answers(
    path: Path,
    ingested: IngestResult,
    depth: DepthAxisResult,
    local_tables_directory: Path | None,
    app_data: Path | None,
    reporter: _Reporter,
) -> tuple[
    IdentificationResult | None,
    UnitsResult | None,
    ValidityResult | None,
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
    tuple[FindingRecord, ...],
]:
    """Apply mappings, then pass choices, then unit choices. D98, D48.

    A stage result already produced with no answers of that kind is reused
    when the extractor returns nothing. A session on hold (D48) applies none.
    Stage 3 is started before Stage 2 is finished, because a later answer can
    re-run identification. The hook does not promise that order to the caller
    beyond the calls it actually receives.
    """

    reporter.start(2)
    identified = identify(
        ingested,
        depth,
        local_tables_directory=local_tables_directory,
        explicit_mappings=(),
        pass_choices=(),
    )
    if identified.status is not StageStatus.COMPLETE:
        reporter.finish(2, identified)
        return identified, None, None, (), (), ()
    reporter.start(3)
    units = interpret(ingested, identified, unit_choices=())
    if units.status is not StageStatus.COMPLETE:
        reporter.finish(2, identified)
        reporter.finish(3, units)
        return identified, units, None, (), (), ()
    asked = load_session(path, _collect(ingested, depth, identified, units), app_data=app_data)
    if asked.session.hold:
        reporter.finish(2, identified)
        reporter.finish(3, units)
        reporter.start(4)
        validity = assess(ingested, depth, identified, units)
        reporter.finish(4, validity)
        return identified, units, validity, (), (), ()
    mappings = explicit_mappings(asked.session)
    mapped = identified
    if mappings:
        mapped = identify(
            ingested,
            depth,
            local_tables_directory=local_tables_directory,
            explicit_mappings=mappings,
            pass_choices=(),
        )
    passes_load = load_session(path, _collect(ingested, depth, mapped), app_data=app_data)
    passes = pass_choices(passes_load.session)
    chosen = mapped
    if passes:
        chosen = identify(
            ingested,
            depth,
            local_tables_directory=local_tables_directory,
            explicit_mappings=mappings,
            pass_choices=passes,
        )
    reporter.finish(2, chosen)
    opened = units if chosen is identified else interpret(ingested, chosen, unit_choices=())
    units_load = load_session(path, _collect(ingested, depth, chosen, opened), app_data=app_data)
    choices = unit_choices(units_load.session)
    interpreted = opened if not choices else interpret(ingested, chosen, unit_choices=choices)
    reporter.finish(3, interpreted)
    validity = None
    if interpreted.status is StageStatus.COMPLETE:
        reporter.start(4)
        validity = assess(ingested, depth, chosen, interpreted)
        reporter.finish(4, validity)
    applied, reasked, restored = _used(
        (ingested, depth, chosen, interpreted, validity),
        chosen,
        interpreted,
        asked,
        passes_load,
        units_load,
    )
    return chosen, interpreted, validity, applied, reasked, restored


def _null_used(
    ingested: IngestResult,
    load: SessionLoad | None,
) -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...], tuple[FindingRecord, ...]]:
    """The Stage 0 NULL answer this run used, if its question was consumed. D66, D98."""

    if load is None or ingested.null_decision_id is None:
        return (), (), ()
    record, finding = _answer(load, ingested.null_decision_id)
    if any(item.id == finding.id for item in ingested.findings):
        return (), (record,), ()
    return (record,), (), (finding,)


def _depth_used(
    depth: DepthAxisResult | None,
    load: SessionLoad | None,
) -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...], tuple[FindingRecord, ...]]:
    """The Stage 1 depth-unit answer this run used, if its question was consumed. D98."""

    if depth is None or load is None or depth.depth_unit_decision_id is None:
        return (), (), ()
    record, finding = _answer(load, depth.depth_unit_decision_id)
    if any(item.id == finding.id for item in depth.findings):
        return (), (record,), ()
    return (record,), (), (finding,)


def _used(
    stages: tuple[object, ...],
    identification: IdentificationResult,
    units: UnitsResult,
    mappings_load: SessionLoad,
    passes_load: SessionLoad,
    units_load: SessionLoad,
) -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...], tuple[FindingRecord, ...]]:
    """Split stage-used decisions into applied and reasked. D98.

    A decision whose question is still in the final stage findings was asked
    again. The others keep the finding from the load that extracted them.
    """

    present = {item.id for item in _collect(*stages)}
    applied: list[DecisionRecord] = []
    reasked: list[DecisionRecord] = []
    restored: list[FindingRecord] = []

    def take(load: SessionLoad, decision_id: str) -> None:
        record, finding = _answer(load, decision_id)
        if finding.id in present:
            reasked.append(record)
            return
        applied.append(record)
        restored.append(finding)

    for curve in identification.curves:
        if curve.decision_id is not None:
            take(mappings_load, curve.decision_id)
    for choice in identification.default_passes:
        take(passes_load, choice.decision_id)
    for curve in units.curves:
        if curve.decision_id is not None:
            take(units_load, curve.decision_id)
    return tuple(applied), tuple(reasked), tuple(restored)


def _answer(load: SessionLoad, decision_id: str) -> tuple[DecisionRecord, FindingRecord]:
    """The decision and the finding it answered, from the load that extracted it. D98."""

    records = [item for item in load.session.decisions if item.decision_id == decision_id]
    if len(records) != 1:
        raise ValueError(f"Decision {decision_id} is not in the session it was taken from.")
    record = records[0]
    findings = [item for item in load.session.findings if item.id == record.finding_id]
    if not findings:
        raise ValueError(
            f"Decision {decision_id} has no finding {record.finding_id} in the session it was taken from."
        )
    return record, findings[0]


def _collect(*results: object) -> tuple[FindingRecord, ...]:
    """Findings of the stage results that exist, in the order given."""

    found: list[FindingRecord] = []
    for result in results:
        if result is None:
            continue
        if not isinstance(
            result, (IngestResult, DepthAxisResult, IdentificationResult, UnitsResult, ValidityResult)
        ):
            raise TypeError("A stage result has an unexpected type.")
        found.extend(result.findings)
    return tuple(found)


def _question_set(
    stages: tuple[object, ...],
    restored: tuple[FindingRecord, ...],
) -> tuple[FindingRecord, ...]:
    """Final stage findings, with each consumed question placed at its stage. D98."""

    grouped: dict[int, list[FindingRecord]] = {stage: [] for stage in range(_STAGE_COUNT)}
    seen: set[str] = set()
    for finding in _collect(*stages):
        if finding.stage not in grouped:
            raise ValueError(f"Finding {finding.id} is stage {finding.stage}, outside Stages 0-4.")
        if finding.id in seen:
            raise ValueError(f"Finding {finding.id} is raised twice in this run.")
        seen.add(finding.id)
        grouped[finding.stage].append(finding)
    for finding in restored:
        if finding.id in seen:
            continue
        if finding.stage not in grouped:
            raise ValueError(f"Finding {finding.id} is stage {finding.stage}, outside Stages 0-4.")
        seen.add(finding.id)
        grouped[finding.stage].append(finding)
    ordered: list[FindingRecord] = []
    for stage in range(_STAGE_COUNT):
        ordered.extend(grouped[stage])
    return tuple(ordered)
