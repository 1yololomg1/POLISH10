"""A finding's id is its check and what it is about. D96.

Stages 0-4 on the fixtures, with no decisions. A second run repeats the ids
and the evidence digests. The ledger refuses an unknown check and a repeated id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polish.model.findings import Evidence, FindingRecord, evidence_digest, finding_key
from polish.model.ids import FileHash
from polish.model.states import BlockType
from polish.stages.s00_ingest import _Ledger, ingest
from polish.stages.s01_depth import classify
from polish.stages.s02_identify import identify
from polish.stages.s03_units import interpret
from polish.stages.s04_validity import assess

_WHEN = datetime(2026, 10, 28, tzinfo=timezone.utc)
_FIXTURES = sorted(Path(__file__).resolve().parents[1].joinpath("fixtures").glob("*.las"))


def _run(path: Path) -> dict[int, tuple]:
    ingested = ingest(path, read_at=_WHEN)
    assert ingested.dataset is not None, path.name
    depth = classify(ingested)
    identified = identify(ingested, depth, local_tables_directory=path.parent / "no-local-tables")
    units = interpret(ingested, identified)
    validity = assess(ingested, depth, identified, units)
    return {
        0: ingested.findings,
        1: depth.findings,
        2: identified.findings,
        3: units.findings,
        4: validity.findings,
    }


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda item: item.name)
def test_fixture_finding_ids_and_digests_are_stable(path: Path):
    first = _run(path)
    second = _run(path)
    for stage in range(5):
        findings = first[stage]
        again = second[stage]
        ids = [item.id for item in findings]
        assert len(ids) == len(set(ids)), path.name
        assert ids == [item.id for item in again]
        for item, other in zip(findings, again, strict=True):
            assert item.stage == stage
            assert item.id == finding_key(item.stage, item.check, item.identity)
            assert evidence_digest(item) == evidence_digest(other)


def test_the_ledger_refuses_an_undeclared_check_and_a_repeated_id():
    ledger = _Ledger(_WHEN, FileHash("ab" * 32))
    with pytest.raises(ValueError, match="not declared"):
        ledger.add("not_a_check", (), "A condition was seen.")
    ledger.add("ascii_absent", (), "No ~A section.")
    with pytest.raises(ValueError, match="already"):
        ledger.add("ascii_absent", (), "No ~A section, stated again.")


def test_a_finding_refuses_an_id_that_is_not_its_key():
    with pytest.raises(ValueError, match="key"):
        FindingRecord(
            id="s00.ascii_absent.0000000000000000",
            stage=0,
            check="ascii_absent",
            identity=(),
            block_type=BlockType.STRUCTURAL_DATA_FAILURE,
            subject_refs=(),
            evidence=Evidence(BlockType.STRUCTURAL_DATA_FAILURE.value),
            message="No ~A section.",
            created_at=_WHEN,
        )


def test_a_finding_refuses_an_identity_name_that_is_not_allowed():
    with pytest.raises(ValueError, match="not allowed"):
        FindingRecord(
            id="not-used",
            stage=0,
            check="ascii_absent",
            identity=(("depth", "7"),),
            block_type=BlockType.STRUCTURAL_DATA_FAILURE,
            subject_refs=(),
            evidence=Evidence(BlockType.STRUCTURAL_DATA_FAILURE.value),
            message="No ~A section.",
            created_at=_WHEN,
        )
