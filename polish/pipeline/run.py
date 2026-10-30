"""Run Stages 0-4 of one well with the answers saved beside it. D98, D48, C5.

Structure
---------
WellRun
    The stage results that ran, the question set, and the answers a stage used.
run_well
    Read the file and the session. Write nothing.

Answers are applied in dependency order. A mapping is matched against the
run that asks the unrecognised-name notice or the incompatible-unit warning.
A pass choice is matched against identification after those mappings. A unit
choice is matched against interpretation after the mappings and the pass
choices. A question that is not raised in its own kind's run is not extracted.
Runs are not merged. An answer a stage used stays linked to the finding it
answered, including the evidence fingerprint, so a later load can check it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from polish.decisions.apply import explicit_mappings, pass_choices, unit_choices
from polish.decisions.session import SessionLoad, load_session
from polish.model.findings import DecisionRecord, FindingRecord
from polish.model.states import StageStatus
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
    decisions a stage used, in order: mappings, pass choices, unit choices.
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


def run_well(
    path: Path | str,
    *,
    read_at: datetime | None = None,
    local_tables_directory: Path | None = None,
    app_data: Path | None = None,
) -> WellRun:
    """Run Stages 0-4 with the session saved beside ``path``. D98, D48, D91, D93, D94, C5.

    Does not write a session file, a table, or a folder. Does not answer the
    changed-file question. A stage runs only when the stage before it is
    COMPLETE. Answers are applied only when the session is not on hold, and
    only from the run that raises that kind of question.

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

    Returns
    -------
    WellRun
        The stages that ran, the question set, and the answers a stage used.

    Raises
    ------
    FileNotFoundError
        ``path`` is not a file.
    UnclassifiedIngest
        Stage 0 cannot classify a condition. Propagated unchanged.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Not a file: {file_path}")
    ingested = ingest(file_path, read_at=read_at)
    depth: DepthAxisResult | None = None
    identification: IdentificationResult | None = None
    units: UnitsResult | None = None
    validity: ValidityResult | None = None
    applied: tuple[DecisionRecord, ...] = ()
    reasked: tuple[DecisionRecord, ...] = ()
    restored: tuple[FindingRecord, ...] = ()
    if ingested.status is StageStatus.COMPLETE and ingested.dataset is not None:
        depth = classify(ingested)
        if depth.status is StageStatus.COMPLETE:
            identification, units, validity, applied, reasked, restored = _with_answers(
                file_path,
                ingested,
                depth,
                local_tables_directory,
                app_data,
            )
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
    """

    identified = identify(
        ingested,
        depth,
        local_tables_directory=local_tables_directory,
        explicit_mappings=(),
        pass_choices=(),
    )
    if identified.status is not StageStatus.COMPLETE:
        return identified, None, None, (), (), ()
    units = interpret(ingested, identified, unit_choices=())
    if units.status is not StageStatus.COMPLETE:
        return identified, units, None, (), (), ()
    asked = load_session(path, _collect(ingested, depth, identified, units), app_data=app_data)
    if asked.session.hold:
        validity = assess(ingested, depth, identified, units) if units.status is StageStatus.COMPLETE else None
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
    opened = units if chosen is identified else interpret(ingested, chosen, unit_choices=())
    units_load = load_session(path, _collect(ingested, depth, chosen, opened), app_data=app_data)
    choices = unit_choices(units_load.session)
    interpreted = opened if not choices else interpret(ingested, chosen, unit_choices=choices)
    validity = None
    if interpreted.status is StageStatus.COMPLETE:
        validity = assess(ingested, depth, chosen, interpreted)
    applied, reasked, restored = _used(
        (ingested, depth, chosen, interpreted, validity),
        chosen,
        interpreted,
        asked,
        passes_load,
        units_load,
    )
    return chosen, interpreted, validity, applied, reasked, restored


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
