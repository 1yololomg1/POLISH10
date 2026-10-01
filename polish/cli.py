"""Command line for POLISH. Stage 0 ingest writes nothing.

Implements D64 and C6 for the read-only ingest command. ``python -m polish
ingest FILE --json`` prints the ledger and a dataset summary to stdout.

Structure
---------
main
    Parse arguments and dispatch.
ingest_summary
    JSON-ready view of an IngestResult. Arrays are not included.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polish.model.states import StageStatus
from polish.stages.s00_ingest import IngestResult, UnclassifiedIngest, ingest


def ingest_summary(result: IngestResult) -> dict:
    """Ledger and dataset summary. No sample arrays. D64, C6."""

    dataset_summary = None
    if result.dataset is not None:
        dataset_summary = {
            "curve_count": len(result.dataset.curves),
            "row_count": int(result.dataset.grid.values.size),
            "direction": result.dataset.grid.direction.value,
            "regularity": result.dataset.grid.regularity.value,
            "declared_step": result.dataset.grid.declared_step,
            "depth_unit": result.dataset.grid.unit,
            "curves": [
                {
                    "column_index": curve.ref.column_index,
                    "mnemonic": curve.original_mnemonic,
                    "unit": curve.original_unit,
                    "description": curve.original_description,
                    "null_count": len(result.null_sample_indexes[index]),
                }
                for index, curve in enumerate(result.dataset.curves)
            ],
        }
    audit = result.audit
    return {
        "path": str(result.source_file.path),
        "sha256": result.source_file.file_hash.hex_digest,
        "status": result.status.value,
        "encoding": {"name": audit.encoding_name, "method": audit.encoding_method},
        "line_endings": list(audit.line_endings),
        "vers": audit.vers_text,
        "wrap": audit.wrap_text,
        "null": {"text": audit.null_text, "number": audit.null_number},
        "strt": {"text": audit.strt_text, "unit": audit.strt_unit, "number": audit.strt_number},
        "stop": {"text": audit.stop_text, "unit": audit.stop_unit, "number": audit.stop_number},
        "step": {"text": audit.step_text, "unit": audit.step_unit, "number": audit.step_number},
        "duplicate_mnemonics": list(audit.duplicate_mnemonics),
        "ascii_is_last_section": audit.ascii_is_last,
        "ledger": [
            {"id": finding.id, "stage": finding.stage, "block_type": finding.block_type.value, "message": finding.message}
            for finding in result.findings
        ],
        "dataset": dataset_summary,
    }


def _print_summary(result: IngestResult) -> None:
    json.dump(ingest_summary(result), sys.stdout, indent=2)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the POLISH command line. Ingest prints JSON and writes no file.

    Implements D64, C6. A structural refusal exits 1. An unclassified
    condition exits 2 and is not given a block type.
    """

    parser = argparse.ArgumentParser(prog="polish")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="Read a LAS file and print the audit ledger.")
    ingest_parser.add_argument("file", type=Path)
    ingest_parser.add_argument("--json", action="store_true", help="Print the ledger and dataset summary as JSON.")
    args = parser.parse_args(argv)
    if args.command != "ingest":
        parser.error(f"Unknown command {args.command}")
        return 2
    if not args.json:
        parser.error("ingest requires --json")
    try:
        result = ingest(args.file)
    except UnclassifiedIngest as exc:
        print(str(exc), file=sys.stderr)
        _print_summary(exc.partial)
        return 2
    _print_summary(result)
    if result.status is StageStatus.REFUSED_STRUCTURAL:
        return 1
    return 0
