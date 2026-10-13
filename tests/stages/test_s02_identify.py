"""Stage 2 identification. D31–D42, C3, UXC1, UXC9.

Counts of migrated mnemonics and of the LFP prefix group are computed from
the legacy database and the golden facts. They are not typed in as literals.
"""

from __future__ import annotations

import ast
import datetime
import json
from pathlib import Path

import pytest

from core.curve_identification import build_mnemonic_database
from polish.model.ids import CurveRef
from polish.model.states import (
    USER_MAPPED_LABEL,
    BlockType,
    CurveRole,
    EvidenceTier,
    ProcessingEligibility,
    StageStatus,
)
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, Runner, StageOutcome
from polish.stages.s00_ingest import ingest
from polish.stages.s01_depth import CONTRACT as DEPTH_CONTRACT
from polish.stages.s01_depth import classify, registered_stage as depth_stage
from polish.stages.s02_identify import (
    CONTRACT,
    UNCHANGED_MNEMONICS,
    identify,
    processing_eligibility_for,
    registered_stage,
)
from polish.tables.loader import load_shipped_tables, load_unit_spellings

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"
_GOLDEN = _REPO / "tests" / "golden"
_WHEN = datetime.datetime(2026, 10, 9, tzinfo=datetime.timezone.utc)

_REVIEW = (
    "Legacy names reviewed and stated correct by Alvaro Chaveste-Fernandez "
    "on 9 Oct 2026, from first-degree sources (chartbooks); not re-verified against them."
)


def _write(directory: Path, body: str) -> Path:
    path = directory / "well.las"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _las(curves: str, data: str) -> str:
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
        f"{data}"
    )


def _curve_block(specs: list[tuple[str, str]]) -> str:
    lines = ["DEPT.M : DEPTH"]
    for mnemonic, unit in specs:
        lines.append(f"{mnemonic}.{unit} : {mnemonic}")
    return "\n".join(lines) + "\n"


def _data(width: int) -> str:
    first = " ".join(["0.0"] + ["1.0"] * (width - 1))
    second = " ".join(["1.0"] + ["2.0"] * (width - 1))
    return first + "\n" + second + "\n"


def _identify(directory: Path, specs: list[tuple[str, str]], local: Path | None = None):
    if local is None:
        local = directory / "no-local-tables"
    body = _las(_curve_block(specs), _data(1 + len(specs)))
    ingested = ingest(_write(directory, body), read_at=_WHEN)
    assert ingested.dataset is not None
    before_values = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)
    before_names = tuple(curve.original_mnemonic for curve in ingested.dataset.curves)
    depth = classify(ingested)
    assert depth.status is StageStatus.COMPLETE
    result = identify(ingested, depth, local_tables_directory=local)
    assert result.status is StageStatus.COMPLETE
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before_values
    assert tuple(curve.original_mnemonic for curve in ingested.dataset.curves) == before_names
    return ingested, result


def _one(result, mnemonic: str):
    matches = [item for item in result.curves if item.mnemonic == mnemonic]
    assert len(matches) == 1, mnemonic
    return matches[0]


def _of_type(result, block: BlockType):
    return [finding for finding in result.findings if finding.block_type is block]


def _local(directory: Path, entries: list[dict[str, object]]) -> Path:
    path = directory / "local-tables"
    path.mkdir()
    (path / "mnemonics.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def test_contract_is_read_only_and_depends_on_stages_0_and_1():
    assert CONTRACT.stage_number == 2
    assert CONTRACT.alters_data is False
    assert CONTRACT.depends_on_stages == (0, 1)
    assert DEPTH_CONTRACT.stage_number == 1


def test_main_and_repeat_qualifiers_and_the_index_role(tmp_path: Path):
    ingested, result = _identify(tmp_path, [("GR_MAIN", "GAPI"), ("GR_REF", "GAPI")])
    main = _one(result, "GR_MAIN")
    repeat = _one(result, "GR_REF")
    index = _one(result, "DEPT")
    assert main.family_id == "GAMMA_RAY_TOTAL"
    assert [(item.token, item.name) for item in main.qualifiers] == [("_MAIN", "main pass")]
    assert any(step.applied and step.entry == "GR" and step.rule == "qualifier_split" for step in main.trace)
    assert main.evidence_tier is EvidenceTier.CITED
    assert main.role is CurveRole.MEASURED_LOG
    assert repeat.family_id == "GAMMA_RAY_TOTAL"
    assert [(item.token, item.name) for item in repeat.qualifiers] == [("_REF", "repeat pass")]
    assert index.role is CurveRole.INDEX
    assert index.family_id == "DEPTH"
    assert index.curve.column_index == 0
    assert ingested.dataset is not None
    assert ingested.dataset.curves[1].original_mnemonic == "GR_MAIN"
    assert ingested.dataset.curves[2].original_mnemonic == "GR_REF"


def test_at90_is_cited_deep_resistivity_with_no_invented_qualifier(tmp_path: Path):
    _, result = _identify(tmp_path, [("AT90", "OHMM")])
    at90 = _one(result, "AT90")
    assert at90.family_id == "RESISTIVITY_DEEP"
    assert at90.evidence_tier is EvidenceTier.CITED
    assert at90.qualifiers == ()
    assert at90.role is CurveRole.MEASURED_LOG
    assert at90.processing_eligibility is ProcessingEligibility.ELIGIBLE
    assert any(step.applied and step.rule == "exact" and step.entry == "AT90" for step in at90.trace)
    assert at90.suggestions == ()


def test_lfp_dt_synt_is_not_stripped_before_a_prefix_decision(tmp_path: Path):
    _, result = _identify(tmp_path, [("LFP_DT_SYNT", "US/F")])
    synt = _one(result, "LFP_DT_SYNT")
    assert synt.role is CurveRole.SYNTHETIC_SOURCE
    assert synt.family_id is None
    assert synt.evidence_tier is EvidenceTier.NO_MATCH
    assert synt.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER
    assert synt.suggestions == ()
    assert synt.prefix_proposal is not None
    assert synt.prefix_proposal.prefix == "LFP_"
    assert synt.prefix_proposal.match.mnemonic == "DT"
    assert synt.prefix_proposal.match.family_id == "SONIC_COMPRESSIONAL"
    assert all(not step.applied or step.rule != "prefix_strip_proposal" for step in synt.trace)
    findings = [item for item in _of_type(result, BlockType.AMBIGUOUS_INTERPRETATION) if "prefix" in item.evidence.as_dict()]
    assert len(findings) == 1
    evidence = findings[0].evidence.as_dict()
    assert evidence["prefix"] == "LFP_"
    assert evidence["table_version"] == "1"
    members = json.loads(str(evidence["members"]))
    stripped = json.loads(str(evidence["stripped_matches"]))
    assert len(members) == len(stripped) == 1
    assert stripped[0]["mnemonic"] == "LFP_DT_SYNT"
    assert stripped[0]["would_match_mnemonic"] == "DT"
    assert synt.curve in findings[0].subject_refs


def test_duplicate_gr_curves_are_identified_independently(tmp_path: Path):
    _, result = _identify(tmp_path, [("GR", "GAPI"), ("GR", "GAPI")])
    grs = [item for item in result.curves if item.mnemonic == "GR"]
    assert len(grs) == 2
    assert grs[0].curve != grs[1].curve
    assert all(item.family_id == "GAMMA_RAY_TOTAL" for item in grs)
    assert all(item.evidence_tier is EvidenceTier.CITED for item in grs)
    informational = _of_type(result, BlockType.INFORMATIONAL)
    assert len(informational) == 1
    assert "GR" in informational[0].message
    refs = [ref for ref in informational[0].subject_refs if isinstance(ref, CurveRef)]
    assert {ref.column_index for ref in refs} == {1, 2}
    assert _of_type(result, BlockType.AMBIGUOUS_INTERPRETATION) == []


def test_unknown_mnemonic_is_preserved_and_not_eligible(tmp_path: Path):
    ingested, result = _identify(tmp_path, [("ZZTOP", "UNK")])
    unknown = _one(result, "ZZTOP")
    assert unknown.role is CurveRole.UNKNOWN
    assert unknown.family_id is None
    assert unknown.evidence_tier is EvidenceTier.NO_MATCH
    assert unknown.processing_eligibility is not ProcessingEligibility.ELIGIBLE
    assert unknown.label is None
    assert ingested.dataset is not None
    assert ingested.dataset.curves[1].original_mnemonic == "ZZTOP"
    notices = _of_type(result, BlockType.INFORMATIONAL)
    assert len(notices) == 1
    assert notices[0].evidence.as_dict()["mnemonic"] == "ZZTOP"
    assert notices[0].evidence.as_dict()["preserved"] is True


def test_item_8b_mnemonics_are_not_mapped_and_get_no_suggestion(tmp_path: Path):
    expected = frozenset(
        {
            "RM",
            "RD",
            "RACEHM",
            "RACELM",
            "RPCEHM",
            "RPCELM",
            "ABDCQF01",
            "ABDCQF02",
            "ABDCQF03",
            "ABDCQF04",
            "NBGRCFM",
            "ROP",
        }
    )
    assert UNCHANGED_MNEMONICS == expected
    specs = [("RM", "OHMM"), ("RD", "OHMM"), ("ROP", "M/H")]
    specs.extend((name, "UNK") for name in sorted(UNCHANGED_MNEMONICS) if name not in {"RM", "RD", "ROP"})
    _, result = _identify(tmp_path, specs)
    assert {item.mnemonic for item in result.curves if item.mnemonic != "DEPT"} == UNCHANGED_MNEMONICS
    for mnemonic in UNCHANGED_MNEMONICS:
        ident = _one(result, mnemonic)
        assert ident.role is CurveRole.UNKNOWN
        assert ident.family_id is None
        assert ident.evidence_tier is EvidenceTier.NO_MATCH
        assert ident.processing_eligibility is not ProcessingEligibility.ELIGIBLE
        assert ident.suggestions == ()
        assert ident.mnemonic == mnemonic
    notices = _of_type(result, BlockType.INFORMATIONAL)
    assert len(notices) == len(UNCHANGED_MNEMONICS)
    assert all(notice.evidence.as_dict().get("suggested_family") is None for notice in notices)
    assert all("suggested_family" not in notice.evidence.as_dict() for notice in notices)


def test_main_and_repeat_passes_ask_once_and_choose_no_default(tmp_path: Path):
    ingested, result = _identify(tmp_path, [("GR", "GAPI"), ("GR_MAIN", "GAPI"), ("GR_REF", "GAPI")])
    ambiguous = _of_type(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert len(ambiguous) == 1
    evidence = ambiguous[0].evidence.as_dict()
    assert evidence["family_id"] == "GAMMA_RAY_TOTAL"
    assert json.loads(str(evidence["scope_options"])) == ["THIS_WELL", "ALL_WELLS"]
    assert "default" not in evidence
    assert "chosen" not in evidence
    assert "No default is chosen" in ambiguous[0].message
    passes = json.loads(str(evidence["passes"]))
    assert {item["mnemonic"] for item in passes} == {"GR", "GR_MAIN", "GR_REF"}
    assert ingested.dataset is not None
    assert [curve.original_mnemonic for curve in ingested.dataset.curves] == ["DEPT", "GR", "GR_MAIN", "GR_REF"]
    for mnemonic in ("GR", "GR_MAIN", "GR_REF"):
        assert _one(result, mnemonic).family_id == "GAMMA_RAY_TOTAL"


def test_similarity_suggests_without_identifying(tmp_path: Path):
    _, result = _identify(tmp_path, [("AT91", "OHMM"), ("QQ", "UNK"), ("THORX", "PPM")])
    near = _one(result, "AT91")
    assert near.family_id is None
    assert near.evidence_tier is EvidenceTier.NO_MATCH
    assert near.role is CurveRole.UNKNOWN
    assert near.suggestions
    assert near.suggestions[0].mnemonic == "AT90"
    assert near.suggestions[0].family_id == "RESISTIVITY_DEEP"
    assert near.suggestions[0].method == "similarity"
    assert near.suggestions[0].tier is EvidenceTier.PROPOSED
    assert isinstance(near.suggestions[0].score, float)
    assert near.suggestions[0].score >= 0.7
    notice = next(item for item in _of_type(result, BlockType.INFORMATIONAL) if item.evidence.as_dict().get("mnemonic") == "AT91")
    evidence = notice.evidence.as_dict()
    assert evidence["suggested_family"] == "RESISTIVITY_DEEP"
    assert evidence["method"] == "similarity"
    assert evidence["suggestion_tier"] == "PROPOSED"
    assert isinstance(evidence["similarity_score"], float)
    short = _one(result, "QQ")
    assert len(short.mnemonic) <= 2
    assert short.suggestions == ()
    assert short.family_id is None
    short_notice = next(item for item in result.findings if item.evidence.as_dict().get("mnemonic") == "QQ")
    assert "suggested_family" not in short_notice.evidence.as_dict()
    contained = _one(result, "THORX")
    assert "THOR" in contained.mnemonic
    assert len(contained.mnemonic) > 2
    assert contained.suggestions == ()
    assert contained.family_id is None


def test_synt_on_a_known_mnemonic_is_a_synthetic_source(tmp_path: Path):
    _, result = _identify(tmp_path, [("DT_SYNT", "US/F")])
    synt = _one(result, "DT_SYNT")
    assert synt.family_id == "SONIC_COMPRESSIONAL"
    assert synt.role is CurveRole.SYNTHETIC_SOURCE
    assert synt.evidence_tier is EvidenceTier.CITED
    assert synt.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_ROLE
    assert [(item.token, item.name) for item in synt.qualifiers] == [("_SYNT", "synthetic source")]


def test_tvd_that_is_not_the_index_is_a_geometric_reference(tmp_path: Path):
    _, result = _identify(tmp_path, [("TVD", "M"), ("TVDEPTH", "M"), ("TVDSS", "M")])
    assert _one(result, "DEPT").role is CurveRole.INDEX
    assert _one(result, "TVD").role is CurveRole.GEOMETRIC_REFERENCE
    assert _one(result, "TVDEPTH").role is CurveRole.GEOMETRIC_REFERENCE
    assert _one(result, "TVDSS").role is CurveRole.GEOMETRIC_REFERENCE
    assert _one(result, "TVD").family_id == "DEPTH"


def test_short_mnemonics_match_the_whole_token_only(tmp_path: Path):
    _, result = _identify(tmp_path, [("K", "PERCENT"), ("XK", "UNK"), ("GR_K", "UNK"), ("GR_FOO", "UNK")])
    assert _one(result, "K").family_id == "POTASSIUM"
    assert _one(result, "XK").family_id is None
    assert _one(result, "XK").suggestions == ()
    assert _one(result, "GR_K").family_id is None
    assert _one(result, "GR_FOO").family_id is None
    assert _one(result, "GR_FOO").suggestions == ()


def test_local_entry_is_user_defined_and_eligible(tmp_path: Path):
    local = _local(
        tmp_path,
        [
            {
                "mnemonic": "SHOP_GR",
                "family": "GAMMA_RAY_TOTAL",
                "role": "MEASURED_LOG",
                "user_label": "shop gamma",
                "unit": "GAPI",
            }
        ],
    )
    _, result = _identify(tmp_path, [("SHOP_GR", "GAPI")], local=local)
    shop = _one(result, "SHOP_GR")
    assert shop.evidence_tier is EvidenceTier.USER_DEFINED
    assert shop.family_id == "GAMMA_RAY_TOTAL"
    assert shop.role is CurveRole.MEASURED_LOG
    assert shop.processing_eligibility is ProcessingEligibility.ELIGIBLE
    assert shop.label == USER_MAPPED_LABEL
    assert shop.user_label == "shop gamma"
    assert any(step.applied and step.table == "local_mnemonics" for step in shop.trace)


def test_family_less_local_entry_loads(tmp_path: Path):
    local = _local(tmp_path, [{"mnemonic": "SHOP_FLAG", "user_label": "shop flag"}])
    _, result = _identify(tmp_path, [("SHOP_FLAG", "UNK")], local=local)
    flag = _one(result, "SHOP_FLAG")
    assert flag.family_id is None
    assert flag.evidence_tier is EvidenceTier.USER_DEFINED
    assert flag.label == USER_MAPPED_LABEL
    assert flag.user_label == "shop flag"
    assert flag.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY


def test_missing_local_directory_is_not_created(tmp_path: Path):
    missing = tmp_path / "absent-local-tables"
    assert not missing.exists()
    _, result = _identify(tmp_path, [("GR", "GAPI")], local=missing)
    assert not missing.exists()
    assert _one(result, "GR").family_id == "GAMMA_RAY_TOTAL"


def test_local_and_shipped_mnemonic_clash_applies_neither(tmp_path: Path):
    local = _local(
        tmp_path,
        [{"mnemonic": "GR", "family": "GAMMA_RAY_TOTAL", "role": "MEASURED_LOG", "user_label": "my gr"}],
    )
    _, result = _identify(tmp_path, [("GR", "GAPI")], local=local)
    gr = _one(result, "GR")
    assert gr.family_id is None
    assert gr.evidence_tier is EvidenceTier.NO_MATCH
    assert gr.role is CurveRole.UNKNOWN
    assert gr.label is None
    assert gr.processing_eligibility is not ProcessingEligibility.ELIGIBLE
    clashes = _of_type(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert len(clashes) == 1
    evidence = clashes[0].evidence.as_dict()
    shipped = json.loads(str(evidence["shipped_entry"]))
    local_entry = json.loads(str(evidence["local_entry"]))
    assert shipped["table"] == "shipped_mnemonics"
    assert shipped["mnemonic"] == "GR"
    assert local_entry["table"] == "local_mnemonics"
    assert local_entry["mnemonic"] == "GR"
    assert "Neither entry is applied" in clashes[0].message
    assert _of_type(result, BlockType.INFORMATIONAL) == []


def test_local_clash_matches_mnemonic_without_regard_to_case(tmp_path: Path):
    local = _local(tmp_path, [{"mnemonic": "gr", "family": "GAMMA_RAY_TOTAL", "role": "MEASURED_LOG"}])
    _, result = _identify(tmp_path, [("GR", "GAPI")], local=local)
    assert _one(result, "GR").family_id is None
    assert len(_of_type(result, BlockType.AMBIGUOUS_INTERPRETATION)) == 1


def test_local_family_must_be_a_shipped_family_id(tmp_path: Path):
    local = _local(tmp_path, [{"mnemonic": "SHOP_GR", "family": "NOT_A_FAMILY"}])
    with pytest.raises(ValueError, match="unknown family"):
        _identify(tmp_path, [("SHOP_GR", "GAPI")], local=local)


@pytest.mark.parametrize(
    ("tier", "eligible"),
    [
        (EvidenceTier.CITED, True),
        (EvidenceTier.OBSERVED, True),
        (EvidenceTier.USER_DEFINED, True),
        (EvidenceTier.PROPOSED, False),
        (EvidenceTier.NO_MATCH, False),
    ],
)
def test_d34_tier_eligibility(tier: EvidenceTier, eligible: bool):
    result = processing_eligibility_for(tier, "GAMMA_RAY_TOTAL", CurveRole.MEASURED_LOG)
    if eligible:
        assert result is ProcessingEligibility.ELIGIBLE
    else:
        assert result is ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER
    assert result is not ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED


def test_synthetic_role_is_not_eligible_when_the_tier_would_otherwise_pass():
    result = processing_eligibility_for(
        EvidenceTier.CITED,
        "SONIC_COMPRESSIONAL",
        CurveRole.SYNTHETIC_SOURCE,
    )
    assert result is ProcessingEligibility.NOT_ELIGIBLE_ROLE


def test_migrated_mnemonics_are_cited_once_and_families_carry_no_range():
    legacy = build_mnemonic_database()
    expected = [mnemonic for entry in legacy.values() for mnemonic in entry["mnemonics"]]
    tables = load_shipped_tables()
    loaded = [row.mnemonic for row in tables.mnemonics]
    assert len(loaded) == len(expected)
    assert len(loaded) == len({item.upper() for item in expected})
    assert sorted(item.upper() for item in loaded) == sorted(item.upper() for item in expected)
    assert len(tables.families) == len(legacy)
    assert tables.review_note == _REVIEW
    assert all(row.source_tier == "CITED" for row in tables.mnemonics)
    assert all(family.source_tier == "CITED" for family in tables.families)
    assert UNCHANGED_MNEMONICS.isdisjoint({row.mnemonic.upper() for row in tables.mnemonics})
    for family_id, entry in legacy.items():
        family = tables.family(family_id)
        assert family is not None
        assert family.source == entry["source"]
        assert family.display_name == entry["description"]
        assert family.legacy_group == entry["curve_family"]
        assert family.tool_group is None
        if family_id.startswith("RESISTIVITY_"):
            assert family.interpolation_domain == "LOG"
        else:
            assert family.interpolation_domain is None
        for mnemonic in entry["mnemonics"]:
            row = tables.mnemonic(mnemonic)
            assert row is not None
            assert row.source == entry["source"]
            assert row.family_id == family_id
            assert row.qualifiers == ()
    raw = json.loads((_REPO / "polish" / "tables" / "families.json").read_text(encoding="utf-8"))
    for entry in raw["entries"]:
        assert "range" not in entry
        assert "units" not in entry
        assert "unit" not in entry
    bulk = next(entry for entry in raw["entries"] if entry["id"] == "BULK_DENSITY")
    assert bulk["accepted_dimensions"] == ["density"]
    assert bulk["accepted_dimensions_note"] is None
    assert "unit_dimension" not in bulk
    assert "range" not in bulk
    imaging = tables.family("FORMATION_RESISTIVITY_IMAGING")
    assert imaging is not None
    assert imaging.interpolation_domain is None
    assert imaging.display_log_scale is True
    permeability = tables.family("NMR_PERMEABILITY")
    assert permeability is not None
    assert permeability.display_log_scale is True
    assert permeability.interpolation_domain is None
    gamma = tables.family("GAMMA_RAY_TOTAL")
    assert gamma is not None
    assert gamma.accepted_dimensions == ("gamma_api", "count_rate")
    assert gamma.accepted_dimensions_note is not None


def test_every_family_accepts_at_least_one_dimension():
    tables = load_shipped_tables()
    assert tables.families_version == "2"
    for family in tables.families:
        assert len(family.accepted_dimensions) >= 1
        assert len(set(family.accepted_dimensions)) == len(family.accepted_dimensions)
        assert "unit_dimension" not in family.__dict__


def test_family_dimensions_are_in_the_spellings_table():
    tables = load_shipped_tables()
    names = {row.dimension for row in load_unit_spellings().spellings}
    for family in tables.families:
        for name in family.accepted_dimensions:
            assert name in names


def test_lfp_prefix_group_size_matches_golden_facts(tmp_path: Path):
    facts = json.loads((_GOLDEN / "159-19A_LFP.facts.json").read_text(encoding="utf-8"))
    expected = sum(1 for curve in facts["curves"] if str(curve["mnemonic"]).startswith("LFP_"))
    missing = tmp_path / "absent-local-tables"
    ingested = ingest(_FIXTURES / "159-19A_LFP.las", read_at=_WHEN)
    assert ingested.dataset is not None
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)
    depth = classify(ingested)
    assert depth.status is StageStatus.COMPLETE
    result = identify(ingested, depth, local_tables_directory=missing)
    assert not missing.exists()
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
    findings = [item for item in result.findings if item.evidence.as_dict().get("prefix") == "LFP_"]
    assert len(findings) == 1
    members = json.loads(str(findings[0].evidence.as_dict()["members"]))
    assert len(members) == expected
    stripped = json.loads(str(findings[0].evidence.as_dict()["stripped_matches"]))
    synt = [row for row in stripped if row["mnemonic"] == "LFP_DT_SYNT"]
    assert len(synt) == 1
    assert synt[0]["would_match_mnemonic"] == "DT"


def test_display_log_scale_is_not_read_by_interpolation_code():
    offenders: list[str] = []
    for path in (_REPO / "polish").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "interpolat" not in node.name.lower():
                continue
            segment = ast.get_source_segment(text, node) or ""
            if "display_log_scale" in segment:
                offenders.append(f"{path.relative_to(_REPO)}:{node.name}")
    assert offenders == []
    stage = (_REPO / "polish" / "stages" / "s02_identify.py").read_text(encoding="utf-8")
    assert "display_log_scale" not in stage


def test_runner_leaves_source_arrays_unchanged(tmp_path: Path):
    ingested = ingest(_write(tmp_path, _las(_curve_block([("GR", "GAPI")]), _data(2))), read_at=_WHEN)
    assert ingested.dataset is not None
    depth = classify(ingested)
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

    missing = tmp_path / "absent-local-tables"
    runner = Runner((stage0(), depth_stage(ingested), registered_stage(ingested, depth, missing)))
    results = runner.run(ingested.dataset)
    assert [item.status for item in results] == [
        StageStatus.COMPLETE,
        StageStatus.COMPLETE,
        StageStatus.COMPLETE,
    ]
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
    assert not missing.exists()
