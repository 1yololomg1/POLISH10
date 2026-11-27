"""Describe report: Stages 0-4 with no decisions. D95, D19, D22, D64, UXC1, UXC9, C6.

Structure
---------
build_report
    Run ingest, classify, identify, interpret, and assess. No session,
    no explicit mappings, no pass choices, no unit choices.
write_report
    Write ``<stem>.describe.json`` and ``<stem>.describe.md`` only into the
    directory the caller names. A failure leaves the previous file in place.
DescribeReport
    JSON document plus the Markdown rendered from that document.

A stage that refuses or does not complete is recorded. Later stages are
"not run" with that reason. Nothing is guessed. Glossary sentences are the
Meaning or preferred-wording column of docs/governing/state_glossary.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from polish import __version__
from polish.decisions.versions import versions_in_force
from polish.model.states import BlockType, StageStatus
from polish.stages.s00_ingest import IngestResult, UnclassifiedIngest, ingest
from polish.stages.s01_depth import DepthAxisResult, classify
from polish.stages.s02_identify import IdentificationResult, identify
from polish.stages.s03_units import UnitsResult, interpret
from polish.stages.s04_validity import ValidityResult, assess
from polish.tables.loader import default_local_tables_directory

SCHEMA_VERSION = "polish-describe/1"
NOTICE = (
    "This report shows POLISH's own reading with no decisions applied. "
    "Saved answers are not shown in this version."
)
OPEN_MARK = "Open: no decision recorded"

# Block types that are refusals. The others are open questions in this version.
_REFUSAL_BLOCKS = frozenset(
    {
        BlockType.STRUCTURAL_DATA_FAILURE,
        BlockType.SCIENTIFIC_PRECONDITION_FAILURE,
        BlockType.EXPORT_CONFORMANCE_FAILURE,
    }
)

# source_tier values the rule files use for a rule that is not verified.
_UNVERIFIED_TIERS = frozenset({"PARTIAL", "UNVER"})

_STAGE_NAMES = (
    (0, "Ingest"),
    (1, "Depth axis"),
    (2, "Identify"),
    (3, "Interpret units"),
    (4, "Assess validity"),
)

# Sentences from state_glossary.md. A code with no sentence is an error,
# not a sentence invented here.
_ROLE = {
    "INDEX": "The depth index of the file.",
    "MEASURED_LOG": "A measured log curve.",
    "DRILLING": "A drilling measurement carried in the file.",
    "ANCILLARY": "Coordinates, facies labels, or QC flags: loaded and displayed, excluded from processing.",
    "SYNTHETIC_SOURCE": "Vendor-computed: never treated as a measurement, never a derivation input.",
    "GEOMETRIC_REFERENCE": "Computed geometric reference that must track the index.",
    "UNKNOWN": "The role is not established.",
}
_TIER = {
    "CITED": "The identification is cited from a published or shipped mnemonic source.",
    "OBSERVED": "The identification was observed from the file, not only cited.",
    "USER_DEFINED": "User-mapped — not identified by POLISH",
    "PROPOSED": "POLISH proposed the identification and it is not yet a user mapping. Not eligible for processing.",
    "NO_MATCH": "No identification matched. Not eligible for processing.",
}
_ELIGIBILITY = {
    "ELIGIBLE": "The curve may enter automatic processing.",
    "NOT_ELIGIBLE_NO_FAMILY": "No family is established, so automatic range validation does not run.",
    "NOT_ELIGIBLE_LOW_TIER": (
        "The evidence tier is PROPOSED or NO_MATCH, so it does not unlock range validation or gap filling."
    ),
    "NOT_ELIGIBLE_UNIT_UNRESOLVED": "The unit is not resolved, so a range check would be a comparison without a unit.",
    "NOT_ELIGIBLE_ROLE": "The role excludes the curve from processing.",
}
_UNIT = {
    "RECOGNISED": "The declared unit is a known unit for this curve.",
    "ABSENT": "The file declares no unit.",
    "EMPTY": "A unit field is present and empty.",
    "UNRECOGNISED": "A unit text is present and is not a known unit.",
    "INCOMPATIBLE_WITH_FAMILY": (
        "The unit is recognised and its dimension is not one of the family's accepted dimensions."
    ),
}
_VALIDATION = {
    "PLAUSIBILITY_WARNING": "Plausibility warning",
    "HARD_VIOLATION_EXCLUDED": "Measured value excluded from processing",
    "NOT_ELIGIBLE": "Not automatically validated",
    "PENDING": "Validation pending",
    "NOT_APPLICABLE": "Validation method not applicable",
}
_GAP_CLASS = {
    "OUTSIDE_INTERVAL": "The gap lies outside the measured interval.",
    "TOOL_ABSENCE_EVIDENCED": (
        "An interior gap is shared across curves from the same tool. "
        "That is evidence consistent with tool absence, not proof."
    ),
    "DROPOUT": "An interior gap is shorter than the tool's vertical resolution.",
    "AMBIGUOUS": (
        "The gap is none of the classes above. Only this class needs a threshold, "
        "and the evidence and confidence are part of the record."
    ),
}
_FILL = {
    "NOT_REQUESTED": "Filling was not requested.",
}
_BLOCK = {
    "INFORMATIONAL": "Recorded — no action required.",
    "AMBIGUOUS_INTERPRETATION": "Your decision is required.",
    "RULE_DEPENDENT_WARNING": "Map explicitly to continue this analysis.",
    "SCIENTIFIC_PRECONDITION_FAILURE": "This method was not run.",
    "STRUCTURAL_DATA_FAILURE": "This file cannot proceed past Depth Axis.",
    "EXPORT_CONFORMANCE_FAILURE": "Export is blocked until this is corrected.",
}

_RULES_DIR = Path(__file__).resolve().parents[1] / "rules"


class OutputRefused(Exception):
    """The named output would replace the input, or an existing report, without permission. D19."""


class DescribeError(Exception):
    """The report cannot be built without inventing a fact."""


@dataclass(frozen=True)
class DescribeReport:
    """One describe result. Markdown is rendered from ``document``. D95.

    ``structural_refusal`` is true when a stage status is REFUSED_STRUCTURAL.
    The headline is ``document["result"]``.
    """

    document: dict[str, object]

    @property
    def structural_refusal(self) -> bool:
        """True when the headline is a structural refusal and the report was still built."""

        return self.document["structural_refusal"] is True

    def json_text(self) -> str:
        """JSON text for ``<stem>.describe.json``. NaN is an error, not a token."""

        return json.dumps(self.document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def markdown(self) -> str:
        """Markdown text for ``<stem>.describe.md``. The first two lines are fixed."""

        return _markdown(self.document)


def build_report(path: Path | str, *, local_tables_directory: Path | None = None) -> DescribeReport:
    """Run Stages 0-4 with no decisions and return one report. D95, D22, D64, C6.

    Does not read a session file. Does not write. ``local_tables_directory``
    None uses the app-data tables path when APPDATA is set. A missing
    directory is empty and is not created. Assess is called with its own
    defaults (no tool groups, no shared-gap definition).

    Parameters
    ----------
    path:
        LAS file. Opened read-only by Stage 0.
    local_tables_directory:
        Local mnemonic tables, or None for the default app-data path.

    Returns
    -------
    DescribeReport
        Stages that did not run are recorded with the reason. An
        UnclassifiedIngest is reported from its partial result.

    Raises
    ------
    FileNotFoundError
        ``path`` is not a file.
    DescribeError
        A glossary sentence or a PARTIAL rule marking is missing, or a
        passed validation has no rule-set version on the result.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Not a file: {file_path}")
    tables = local_tables_directory if local_tables_directory is not None else default_local_tables_directory()
    try:
        ingested = ingest(file_path)
    except UnclassifiedIngest as exc:
        return _assemble(exc.partial, None, None, None, None, tables, unclassified=str(exc))
    depth, identification, units, validity = _continue(ingested, tables)
    return _assemble(ingested, depth, identification, units, validity, tables, unclassified=None)


def write_report(
    report: DescribeReport,
    source: Path | str,
    directory: Path | str,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports into ``directory``. D19, D95, C5.

    Creates ``directory`` when it is missing. Refuses when either target is
    the input file. Refuses when a target already exists unless ``overwrite``
    is true. Each file is written to a temporary name in that directory and
    then replaced, so a failure does not leave a half-written report.

    Parameters
    ----------
    report:
        The report ``build_report`` returned.
    source:
        The input file. Compared as a resolved path. It is not opened.
    directory:
        The only directory written.
    overwrite:
        When true, an existing report file in ``directory`` is replaced.
        The input file is still refused.

    Returns
    -------
    tuple[Path, Path]
        The JSON path and the Markdown path.

    Raises
    ------
    OutputRefused
        A target is the input, or exists and ``overwrite`` is false.
    """

    if not isinstance(report, DescribeReport):
        raise TypeError("report must be a DescribeReport")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    input_path = Path(source).resolve()
    folder = Path(directory)
    if folder.exists() and folder.resolve() == input_path:
        raise OutputRefused(f"The output directory is the input file: {input_path}")
    stem = Path(source).stem
    if stem.strip() == "" or "/" in stem or "\\" in stem:
        raise OutputRefused(f"The input name cannot be used as a report name: {Path(source).name}")
    folder_resolved = folder.resolve()
    json_path = (folder_resolved / f"{stem}.describe.json").resolve()
    markdown_path = (folder_resolved / f"{stem}.describe.md").resolve()
    targets = (json_path, markdown_path)
    for target in targets:
        if target.parent != folder_resolved:
            raise OutputRefused(f"The report path leaves the output directory: {target}")
    for target in targets:
        if target == input_path:
            raise OutputRefused(f"The report path is the input file: {target}")
    if not overwrite:
        existing = [target for target in targets if target.exists()]
        if existing:
            names = ", ".join(str(target) for target in existing)
            raise OutputRefused(f"The report already exists: {names}")
    folder.mkdir(parents=True, exist_ok=True)
    _replace_pair(json_path, report.json_text(), markdown_path, report.markdown())
    return json_path, markdown_path


def _continue(
    ingested: IngestResult,
    tables: Path | None,
) -> tuple[DepthAxisResult | None, IdentificationResult | None, UnitsResult | None, ValidityResult | None]:
    """Run each later stage only when the previous one completed. No decisions."""

    if ingested.status is not StageStatus.COMPLETE or ingested.dataset is None:
        return None, None, None, None
    depth = classify(ingested)
    if depth.status is not StageStatus.COMPLETE:
        return depth, None, None, None
    identification = identify(
        ingested,
        depth,
        local_tables_directory=tables,
        explicit_mappings=(),
        pass_choices=(),
    )
    if identification.status is not StageStatus.COMPLETE:
        return depth, identification, None, None
    units = interpret(ingested, identification, unit_choices=())
    if units.status is not StageStatus.COMPLETE:
        return depth, identification, units, None
    validity = assess(ingested, depth, identification, units)
    return depth, identification, units, validity


def _assemble(
    ingested: IngestResult,
    depth: DepthAxisResult | None,
    identification: IdentificationResult | None,
    units: UnitsResult | None,
    validity: ValidityResult | None,
    tables: Path | None,
    *,
    unclassified: str | None,
) -> DescribeReport:
    """Build the document from the stage results that exist."""

    # Imported here: polish.cli imports this module to dispatch the command.
    from polish.cli import ingest_summary

    findings = _all_findings(ingested, depth, identification, units, validity)
    refusal_reason = _refusal_reason(ingested, depth, identification, units, validity)
    structural = refusal_reason is not None
    stages = _stage_rows(ingested, depth, identification, units, validity, unclassified, refusal_reason)
    curves, gap_inventory, non_eligibility = _curves(ingested, identification, units, validity)
    grouped = _grouped_findings(findings)
    refusals = [
        {
            "id": item["id"],
            "stage": item["stage"],
            "block_type": item["block_type"],
            "wording": _sentence(_BLOCK, item["block_type"], "block type"),
            "message": item["message"],
        }
        for item in findings
        if item["block_type"] in {block.value for block in _REFUSAL_BLOCKS}
    ]
    open_questions = len(findings) - len(refusals)
    if structural:
        result = f"Refused: {refusal_reason}"
    elif unclassified is not None:
        result = f"Unclassified: {unclassified}"
    else:
        curve_count = 0 if ingested.dataset is None else len(ingested.dataset.curves)
        result = (
            f"Read: {_counted(curve_count, 'curve', 'curves')}, "
            f"{_counted(open_questions, 'open question', 'open questions')}, "
            f"{_counted(len(refusals), 'refusal', 'refusals')}"
        )
    stage2_ran = identification is not None
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "result": result,
        "notice": NOTICE,
        "structural_refusal": structural,
        "unclassified": unclassified is not None,
        "unclassified_message": unclassified,
        "file": {
            "path": str(ingested.source_file.path),
            "sha256": ingested.source_file.file_hash.hex_digest,
            "size_bytes": ingested.source_file.size_bytes,
        },
        "stage0": ingest_summary(ingested),
        "stages": stages,
        "depth": None if depth is None else _depth_document(depth),
        "curves": curves,
        "gap_inventory": gap_inventory,
        "default_passes": [],
        "findings_by_block": grouped,
        "refusals": refusals,
        "non_eligibility": non_eligibility,
        "deferred": [],
        "versions": {
            "app": __version__,
            "in_force": [{"name": name, "version": version} for name, version in versions_in_force()],
        },
        "not_verified": _not_verified(),
        "local_tables": _local_tables(tables, stage2_ran),
        "not_available_from_the_engine": [],
    }
    return DescribeReport(document)


def _stage_rows(
    ingested: IngestResult,
    depth: DepthAxisResult | None,
    identification: IdentificationResult | None,
    units: UnitsResult | None,
    validity: ValidityResult | None,
    unclassified: str | None,
    refusal_reason: str | None,
) -> list[dict[str, object]]:
    """One row per stage. A stage that was not called is NOT_RUN with the reason."""

    ran = {
        0: ingested.status,
        1: None if depth is None else depth.status,
        2: None if identification is None else identification.status,
        3: None if units is None else units.status,
        4: None if validity is None else validity.status,
    }
    rows: list[dict[str, object]] = []
    stop_reason: str | None = None
    for number, name in _STAGE_NAMES:
        status = ran[number]
        if status is None:
            reason = stop_reason if stop_reason is not None else "The previous stage did not complete."
            rows.append({"stage": number, "name": name, "status": StageStatus.NOT_RUN.value, "reason": reason})
            continue
        rows.append({"stage": number, "name": name, "status": status.value, "reason": None})
        if number == 0 and status is StageStatus.COMPLETE and ingested.dataset is None:
            stop_reason = "Stage 0 returned no dataset."
            continue
        if status is not StageStatus.COMPLETE:
            if unclassified is not None and number == 0:
                stop_reason = f"Stage 0 is unclassified. {unclassified}"
            elif refusal_reason is not None:
                stop_reason = f"Stage {number} refused. {refusal_reason}"
            else:
                stop_reason = f"Stage {number} status is {status.value}."
    return rows


def _refusal_reason(
    ingested: IngestResult,
    depth: DepthAxisResult | None,
    identification: IdentificationResult | None,
    units: UnitsResult | None,
    validity: ValidityResult | None,
) -> str | None:
    """The first structural-refusal message, or None when no stage refused."""

    for result in (ingested, depth, identification, units, validity):
        if result is None or result.status is not StageStatus.REFUSED_STRUCTURAL:
            continue
        for finding in result.findings:
            if finding.block_type is BlockType.STRUCTURAL_DATA_FAILURE:
                return finding.message
        stage_number = result.findings[0].stage if result.findings else None
        if stage_number is None:
            return "A stage refused without a finding."
        return f"Stage {stage_number} refused."
    return None


def _all_findings(
    ingested: IngestResult,
    depth: DepthAxisResult | None,
    identification: IdentificationResult | None,
    units: UnitsResult | None,
    validity: ValidityResult | None,
) -> list[dict[str, object]]:
    """Every finding from the stages that ran, in stage order."""

    items: list[dict[str, object]] = []
    for result in (ingested, depth, identification, units, validity):
        if result is None:
            continue
        for finding in result.findings:
            items.append(
                {
                    "id": finding.id,
                    "stage": finding.stage,
                    "block_type": finding.block_type.value,
                    "subjects": [str(ref) for ref in finding.subject_refs],
                    "message": finding.message,
                    "open": OPEN_MARK,
                    "evidence": {key: value for key, value in finding.evidence.fields},
                }
            )
    return items


def _grouped_findings(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    """Findings grouped by block type, in glossary order. Empty groups are omitted."""

    groups: list[dict[str, object]] = []
    for block in BlockType:
        members = [item for item in findings if item["block_type"] == block.value]
        if not members:
            continue
        groups.append(
            {
                "block_type": block.value,
                "wording": _sentence(_BLOCK, block.value, "block type"),
                "findings": [
                    {
                        "id": item["id"],
                        "stage": item["stage"],
                        "subjects": item["subjects"],
                        "message": item["message"],
                        "open": item["open"],
                        "evidence": item["evidence"],
                    }
                    for item in members
                ],
            }
        )
    return groups


def _curves(
    ingested: IngestResult,
    identification: IdentificationResult | None,
    units: UnitsResult | None,
    validity: ValidityResult | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Per-curve rows, the gap inventory, and non-eligibility lines."""

    by_column_id = {} if identification is None else {item.curve.column_index: item for item in identification.curves}
    by_column_units = {} if units is None else {item.curve.column_index: item for item in units.curves}
    by_column_validity = {} if validity is None else {item.curve.column_index: item for item in validity.curves}
    rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    ineligible: list[dict[str, object]] = []
    if ingested.dataset is not None:
        if identification is not None and len(identification.curves) != len(ingested.dataset.curves):
            raise DescribeError("Stage 2 curve count does not match Stage 0.")
        if units is not None and len(units.curves) != len(ingested.dataset.curves):
            raise DescribeError("Stage 3 curve count does not match Stage 0.")
        if validity is not None and len(validity.curves) != len(ingested.dataset.curves):
            raise DescribeError("Stage 4 curve count does not match Stage 0.")
        sources = [
            (curve.ref.column_index, curve.original_mnemonic, curve.original_unit, curve.original_description)
            for curve in ingested.dataset.curves
        ]
    else:
        sources = [
            (curve.column_index, curve.mnemonic, curve.unit, curve.description) for curve in ingested.audit.curves
        ]
    for column, mnemonic, unit, description in sources:
        identified = by_column_id.get(column)
        interpreted = by_column_units.get(column)
        assessed = by_column_validity.get(column)
        identification_doc = None if identified is None else _identification_document(identified)
        units_doc = None if interpreted is None else _units_document(interpreted)
        validity_doc = None if assessed is None else _validity_document(assessed)
        curve_gaps = [] if assessed is None else [_gap_document(gap) for gap in assessed.gaps]
        gaps.extend(curve_gaps)
        rows.append(
            {
                "column": column,
                "mnemonic": mnemonic,
                "unit_as_written": unit,
                "description": description,
                "identification": identification_doc,
                "units": units_doc,
                "validity": validity_doc,
                "gaps": curve_gaps,
            }
        )
        if identification_doc is not None and identification_doc["eligibility"] != "ELIGIBLE":
            ineligible.append(
                _ineligible(
                    column,
                    mnemonic,
                    "identification",
                    identification_doc["eligibility"],
                    identification_doc["eligibility_wording"],
                )
            )
        if units_doc is not None and units_doc["eligibility"] != "ELIGIBLE":
            ineligible.append(
                _ineligible(
                    column,
                    mnemonic,
                    "units",
                    units_doc["eligibility"],
                    units_doc["eligibility_wording"],
                )
            )
        if validity_doc is not None and validity_doc["outcome"] == "NOT_ELIGIBLE":
            ineligible.append(
                _ineligible(
                    column,
                    mnemonic,
                    "validity",
                    validity_doc["outcome"],
                    validity_doc["wording"],
                    validity_doc["ineligibility_reason"],
                    validity_doc["eligibility_evidence"],
                )
            )
    return rows, gaps, ineligible


def _identification_document(item: object) -> dict[str, object]:
    role = item.role.value
    tier = item.evidence_tier.value
    eligibility = item.processing_eligibility.value
    return {
        "family_id": item.family_id,
        "qualifiers": [{"token": qualifier.token, "name": qualifier.name} for qualifier in item.qualifiers],
        "role": role,
        "role_wording": _sentence(_ROLE, role, "role"),
        "tier": tier,
        "tier_wording": _sentence(_TIER, tier, "tier"),
        "eligibility": eligibility,
        "eligibility_wording": _sentence(_ELIGIBILITY, eligibility, "eligibility"),
        "label": item.label,
        "user_label": item.user_label,
        "trace": [
            {"table": step.table, "entry": step.entry, "rule": step.rule, "applied": step.applied} for step in item.trace
        ],
    }


def _units_document(item: object) -> dict[str, object]:
    outcome = item.outcome.value
    eligibility = item.processing_eligibility.value
    return {
        "outcome": outcome,
        "outcome_wording": _sentence(_UNIT, outcome, "unit outcome"),
        "canonical_unit": item.canonical_unit,
        "dimension": item.dimension,
        "eligibility": eligibility,
        "eligibility_wording": _sentence(_ELIGIBILITY, eligibility, "eligibility"),
    }


def _validity_document(item: object) -> dict[str, object]:
    outcome = item.outcome.value
    if outcome == "PASSED":
        if not item.rule_set_version:
            raise DescribeError("A passed validation has no rule-set version on the result.")
        wording = f"Validated against rule set v{item.rule_set_version}"
    else:
        wording = _sentence(_VALIDATION, outcome, "validation outcome")
    return {
        "outcome": outcome,
        "wording": wording,
        "rule_set_version": item.rule_set_version,
        "ineligibility_reason": item.ineligibility_reason,
        "eligibility_evidence": item.eligibility_evidence,
        "family_rules_applied": list(item.family_rules_applied),
    }


def _gap_document(gap: object) -> dict[str, object]:
    gap_class = gap.classification.gap_class.value
    fill = gap.fill_outcome.value
    return {
        "column": gap.curve.column_index,
        "mnemonic": gap.mnemonic,
        "sample_start": gap.sample_start,
        "sample_end": gap.sample_end,
        "depth_at_start": float(gap.depth_at_start),
        "depth_at_end": float(gap.depth_at_end),
        "class": gap_class,
        "class_wording": _sentence(_GAP_CLASS, gap_class, "gap class"),
        "evidence": gap.classification.evidence,
        "confidence": gap.classification.confidence,
        "fill_outcome": fill,
        "fill_wording": _sentence(_FILL, fill, "gap fill"),
    }


def _depth_document(depth: DepthAxisResult) -> dict[str, object]:
    alias = None
    if depth.alias is not None:
        alias = {"source": str(depth.alias.source), "canonical_name": depth.alias.canonical_name}
    return {
        "status": depth.status.value,
        "direction": None if depth.direction is None else depth.direction.value,
        "regularity": None if depth.regularity is None else depth.regularity.value,
        "declared_step": _quantity(depth.declared_step),
        "observed_step": _quantity(depth.observed_step),
        "corrected_step": _quantity(depth.corrected_step),
        "recorded_strt": _quantity(depth.recorded_strt),
        "recorded_stop": _quantity(depth.recorded_stop),
        "displacement": _quantity(depth.displacement),
        "alias": alias,
        "primary_standard_step": _quantity(depth.primary_standard_step),
        "depth_unit": depth.depth_unit,
    }


def _quantity(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    return {"name": value.name, "text": value.text, "unit": value.unit}


def _local_tables(path: Path | None, stage2_ran: bool) -> dict[str, object]:
    """Whether the local tables directory was found, and whether Stage 2 read it."""

    if path is None:
        return {"path": None, "found": False, "read": False}
    found = path.is_dir()
    return {"path": str(path), "found": found, "read": found and stage2_ran}


def _not_verified() -> list[dict[str, object]]:
    """Rule files in force whose source_tier is PARTIAL or UNVER.

    Stage 0 loads null_comparison.json and null_sentinels.json itself. The
    other rule files are the loader's registry. This reads the JSON; it does
    not change a rule.
    """

    if not _RULES_DIR.is_dir():
        raise DescribeError(f"The rules directory is not present: {_RULES_DIR}")
    found: list[dict[str, object]] = []
    paths = sorted(_RULES_DIR.glob("*.json"))
    if not paths:
        raise DescribeError("No rule files were found.")
    marked = False
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise DescribeError(f"{path.name} is not a JSON object.")
        tier = payload.get("source_tier")
        if isinstance(tier, str):
            marked = True
            if tier.upper() in _UNVERIFIED_TIERS:
                evidence = payload.get("evidence")
                rule_id = payload.get("id")
                found.append(
                    {
                        "file": path.name,
                        "id": rule_id if isinstance(rule_id, str) else None,
                        "source_tier": tier,
                        "evidence": evidence if isinstance(evidence, str) else None,
                    }
                )
        rules = payload.get("rules")
        if isinstance(rules, list):
            for rule in rules:
                if isinstance(rule, dict) and isinstance(rule.get("source_tier"), str):
                    marked = True
                    rule_tier = rule["source_tier"]
                    if rule_tier.upper() in _UNVERIFIED_TIERS:
                        evidence = rule.get("evidence")
                        found.append(
                            {
                                "file": path.name,
                                "id": rule.get("id") if isinstance(rule.get("id"), str) else None,
                                "source_tier": rule_tier,
                                "evidence": evidence if isinstance(evidence, str) else None,
                            }
                        )
    if not marked:
        raise DescribeError("Rule files in force carry no PARTIAL or unverified marking.")
    return found


def _sentence(table: dict[str, str], code: str, field: str) -> str:
    sentence = table.get(code)
    if sentence is None:
        raise DescribeError(f"No glossary wording for {field} {code}.")
    return sentence


def _counted(count: int, singular: str, plural: str) -> str:
    word = singular if count == 1 else plural
    return f"{count} {word}"


def _ineligible(
    column: int,
    mnemonic: str,
    field: str,
    code: str,
    wording: str,
    reason: str | None = None,
    evidence: str | None = None,
) -> dict[str, object]:
    return {
        "column": column,
        "mnemonic": mnemonic,
        "field": field,
        "code": code,
        "wording": wording,
        "reason": reason,
        "evidence": evidence,
    }


def _replace_pair(json_path: Path, json_text: str, markdown_path: Path, markdown_text: str) -> None:
    """Write both temporary files, then replace. A failed write leaves both targets unchanged."""

    json_temporary = json_path.with_name(f".{json_path.name}.partial")
    markdown_temporary = markdown_path.with_name(f".{markdown_path.name}.partial")
    try:
        json_temporary.write_text(json_text, encoding="utf-8", newline="\n")
        markdown_temporary.write_text(markdown_text, encoding="utf-8", newline="\n")
        os.replace(json_temporary, json_path)
        os.replace(markdown_temporary, markdown_path)
    except Exception:
        for temporary in (json_temporary, markdown_temporary):
            if temporary.exists():
                temporary.unlink()
        raise


def _markdown(document: dict[str, object]) -> str:
    """Human report. The first line is the result. The second line is the notice."""

    lines: list[str] = [str(document["result"]), str(document["notice"]), ""]
    _file_section(lines, document)
    _stage0_section(lines, document)
    _depth_section(lines, document)
    _curve_section(lines, document)
    _units_section(lines, document)
    _validity_section(lines, document)
    _gap_section(lines, document)
    _passes_section(lines, document)
    _findings_section(lines, document)
    _refusal_section(lines, document)
    _deferred_section(lines, document)
    _version_section(lines, document)
    _unverified_section(lines, document)
    _tables_section(lines, document)
    _missing_section(lines, document)
    _run_section(lines, document)
    return "\n".join(lines) + "\n"


def _file_section(lines: list[str], document: dict[str, object]) -> None:
    file_info = document["file"]
    lines.append("## File identity")
    lines.append(f"- Path: {file_info['path']}")
    lines.append(f"- SHA-256: {file_info['sha256']}")
    lines.append(f"- Size: {file_info['size_bytes']} bytes")
    lines.append("")


def _stage0_section(lines: list[str], document: dict[str, object]) -> None:
    stage0 = document["stage0"]
    encoding = stage0["encoding"]
    lines.append("## Stage 0")
    lines.append(f"- Status: {stage0['status']}")
    name = "not recorded" if encoding["name"] is None else encoding["name"]
    lines.append(f"- Encoding: {name} ({encoding['method']})")
    lines.append(f"- VERS: {_shown(stage0['vers'])}")
    lines.append(f"- WRAP: {_shown(stage0['wrap'])}")
    null = stage0["null"]
    lines.append(f"- NULL: {_shown(null['text'])}")
    lines.append(f"- STRT: {_header_depth(stage0['strt'])}")
    lines.append(f"- STOP: {_header_depth(stage0['stop'])}")
    lines.append(f"- STEP: {_header_depth(stage0['step'])}")
    if document["unclassified"] is True:
        lines.append(f"- Unclassified: {document['unclassified_message']}")
    lines.append("")


def _header_depth(item: dict[str, object]) -> str:
    text = _shown(item["text"])
    unit = item["unit"]
    if unit:
        return f"{text} {unit}"
    return text


def _depth_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Depth axis")
    depth = document["depth"]
    if depth is None:
        lines.append(f"Not run. {_stage_reason(document, 1)}")
        lines.append("")
        return
    lines.append(f"- Status: {depth['status']}")
    lines.append(f"- Direction: {_shown(depth['direction'])}")
    lines.append(f"- Regularity: {_shown(depth['regularity'])}")
    lines.append(f"- Depth unit: {_shown(depth['depth_unit'])}")
    for key, label in (
        ("declared_step", "Declared step"),
        ("observed_step", "Observed step"),
        ("corrected_step", "Corrected step"),
        ("recorded_strt", "Recorded STRT"),
        ("recorded_stop", "Recorded STOP"),
        ("displacement", "Displacement"),
        ("primary_standard_step", "Primary standard step"),
    ):
        lines.append(f"- {label}: {_quantity_text(depth[key])}")
    alias = depth["alias"]
    if alias is None:
        lines.append("- Alias: not recorded")
    else:
        lines.append(f"- Alias: {alias['canonical_name']} from {alias['source']}")
    lines.append("")


def _curve_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Curves")
    curves = document["curves"]
    if not curves:
        lines.append("No curves were recorded.")
        lines.append("")
        return
    for curve in curves:
        lines.append(f"### Column {curve['column']} — {curve['mnemonic']}")
        lines.append(f"- Unit as written: {_shown(curve['unit_as_written'])}")
        description = curve["description"]
        if description:
            lines.append(f"- Description: {description}")
        identified = curve["identification"]
        if identified is None:
            lines.append("- Identification was not run.")
        else:
            family = "not recorded" if identified["family_id"] is None else identified["family_id"]
            lines.append(f"- Family: {family}")
            qualifiers = identified["qualifiers"]
            if qualifiers:
                rendered = ", ".join(f"{item['token']} ({item['name']})" for item in qualifiers)
                lines.append(f"- Qualifiers: {rendered}")
            else:
                lines.append("- Qualifiers: none recorded")
            lines.append(f"- Role: {identified['role_wording']}")
            lines.append(f"- Tier: {identified['tier_wording']}")
            lines.append(f"- Eligibility: {identified['eligibility_wording']}")
            if identified["label"]:
                lines.append(f"- Label: {identified['label']}")
            if identified["user_label"]:
                lines.append(f"- User label: {identified['user_label']}")
            trace = identified["trace"]
            if not trace:
                lines.append("- Match trace: none recorded")
            else:
                lines.append("- Match trace:")
                for step in trace:
                    applied = "applied" if step["applied"] else "not applied"
                    lines.append(f"  - {step['table']} / {step['entry']} / {step['rule']} / {applied}")
        lines.append("")


def _units_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Units")
    if _stage_status(document, 3) != "COMPLETE":
        lines.append(f"Not run. {_stage_reason(document, 3)}")
        lines.append("")
        return
    if not document["curves"]:
        lines.append("No curves were recorded.")
        lines.append("")
        return
    for curve in document["curves"]:
        units = curve["units"]
        lines.append(f"### {curve['mnemonic']}")
        lines.append(f"- Outcome: {units['outcome_wording']}")
        lines.append(f"- Canonical unit: {_shown(units['canonical_unit'])}")
        lines.append(f"- Dimension: {_shown(units['dimension'])}")
        lines.append(f"- Eligibility: {units['eligibility_wording']}")
        lines.append("")


def _validity_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Validity")
    if _stage_status(document, 4) != "COMPLETE":
        lines.append(f"Not run. {_stage_reason(document, 4)}")
        lines.append("")
        return
    if not document["curves"]:
        lines.append("No curves were recorded.")
        lines.append("")
        return
    for curve in document["curves"]:
        validity = curve["validity"]
        lines.append(f"### {curve['mnemonic']}")
        lines.append(f"- {validity['wording']}")
        if validity["ineligibility_reason"]:
            lines.append(f"- Reason: {validity['ineligibility_reason']}")
        if validity["eligibility_evidence"]:
            lines.append(f"- Evidence: {validity['eligibility_evidence']}")
        applied = validity["family_rules_applied"]
        if applied:
            lines.append("- Family rules applied: " + ", ".join(applied))
        lines.append("")


def _gap_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Gap inventory")
    gaps = document["gap_inventory"]
    if not gaps:
        if document["depth"] is None or _stage_status(document, 4) != "COMPLETE":
            lines.append(f"Not run. {_stage_reason(document, 4)}")
        else:
            lines.append("No gaps.")
        lines.append("")
        return
    for gap in gaps:
        lines.append(
            f"### {gap['mnemonic']} column {gap['column']}, samples {gap['sample_start']}–{gap['sample_end']}"
        )
        lines.append(f"- Class: {gap['class_wording']}")
        lines.append(f"- Evidence: {gap['evidence']}")
        lines.append(f"- Confidence: {gap['confidence']}")
        lines.append(f"- Fill: {gap['fill_wording']}")
        lines.append("")


def _passes_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Default passes")
    if document["default_passes"]:
        for item in document["default_passes"]:
            lines.append(f"- {item}")
    else:
        lines.append("No default pass is recorded. This version applies no pass choice.")
    lines.append("")


def _findings_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Findings")
    groups = document["findings_by_block"]
    if not groups:
        lines.append("No findings were recorded.")
        lines.append("")
        return
    for group in groups:
        lines.append(f"### {group['block_type']}")
        lines.append(group["wording"])
        for finding in group["findings"]:
            subjects = ", ".join(finding["subjects"]) if finding["subjects"] else "none recorded"
            lines.append(f"- {finding['id']} (stage {finding['stage']}). Subjects: {subjects}")
            lines.append(f"  {finding['message']}")
            lines.append(f"  {finding['open']}")
        lines.append("")


def _refusal_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Refusals and non-eligibility")
    refusals = document["refusals"]
    if not refusals:
        lines.append("No refusals")
    else:
        for item in refusals:
            lines.append(f"- {item['id']} (stage {item['stage']}). {item['wording']}")
            lines.append(f"  {item['message']}")
    ineligible = document["non_eligibility"]
    if ineligible:
        lines.append("")
        lines.append("### Non-eligibility")
        for item in ineligible:
            lines.append(f"- Column {item['column']} {item['mnemonic']}, {item['field']}: {item['wording']}")
            if item.get("reason"):
                lines.append(f"  Reason: {item['reason']}")
    lines.append("")


def _deferred_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Deferred")
    if document["deferred"]:
        for item in document["deferred"]:
            lines.append(f"- {item}")
    else:
        lines.append("No deferred items.")
    lines.append("")


def _version_section(lines: list[str], document: dict[str, object]) -> None:
    versions = document["versions"]
    lines.append("## Versions")
    lines.append(f"- App: {versions['app']}")
    for item in versions["in_force"]:
        lines.append(f"- {item['name']}: {item['version']}")
    lines.append("")


def _unverified_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Not verified")
    for item in document["not_verified"]:
        evidence = item["evidence"] if item["evidence"] else "no evidence text on the file"
        if item["id"]:
            lines.append(f"- {item['id']} ({item['file']}, {item['source_tier']}): {evidence}")
        else:
            lines.append(f"- {item['file']} ({item['source_tier']}): {evidence}")
    lines.append("")


def _tables_section(lines: list[str], document: dict[str, object]) -> None:
    tables = document["local_tables"]
    lines.append("## Local tables")
    lines.append(f"- Path: {_shown(tables['path'])}")
    lines.append(f"- Found: {'yes' if tables['found'] else 'no'}")
    lines.append(f"- Read: {'yes' if tables['read'] else 'no'}")
    lines.append("")


def _missing_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Not available from the engine")
    missing = document["not_available_from_the_engine"]
    if not missing:
        lines.append("None.")
    else:
        for item in missing:
            lines.append(f"- {item}")
    lines.append("")


def _run_section(lines: list[str], document: dict[str, object]) -> None:
    lines.append("## Stages")
    for row in document["stages"]:
        if row["reason"]:
            lines.append(f"- Stage {row['stage']} {row['name']}: {row['status']}. {row['reason']}")
        else:
            lines.append(f"- Stage {row['stage']} {row['name']}: {row['status']}")
    lines.append("")


def _stage_reason(document: dict[str, object], number: int) -> str:
    for row in document["stages"]:
        if row["stage"] == number:
            if row["reason"]:
                return str(row["reason"])
            return f"Stage {number} status is {row['status']}."
    return f"Stage {number} was not recorded."


def _stage_status(document: dict[str, object], number: int) -> str | None:
    for row in document["stages"]:
        if row["stage"] == number:
            return str(row["status"])
    return None


def _quantity_text(value: dict[str, str] | None) -> str:
    if value is None:
        return "not recorded"
    unit = f" {value['unit']}" if value["unit"] else ""
    return f"{value['text']}{unit}"


def _shown(value: object) -> str:
    if value is None or value == "":
        return "not recorded"
    return str(value)
