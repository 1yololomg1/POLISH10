"""Command line for POLISH.

Implements D64 and C6 for the read-only ingest command, and D95, D19, D22,
UXC1, and UXC9 for describe. ``python -m polish ingest FILE --json`` prints
the ledger and a dataset summary. ``python -m polish describe FILE --out DIR``
writes the describe report into that directory and nowhere else.

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
            "regularity": (
                None
                if result.dataset.grid.regularity is None
                else result.dataset.grid.regularity.value
            ),
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
        "duplicate_sections": list(audit.duplicate_section_letters),
        "absent_sections": list(audit.absent_section_letters),
        "preamble_lines": list(audit.preamble_lines),
        "repeated_well_mnemonics": list(audit.repeated_well_mnemonics),
        "repeated_version_mnemonics": list(audit.repeated_version_mnemonics),
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
    """Run the POLISH command line.

    Implements D64, C6 for ingest, and D95, D19 for describe. Ingest: a
    structural refusal exits 1, and an unclassified condition exits 2 and is
    not given a block type. Describe: 0 when the report is written, 2 when
    that report is a structural refusal, 1 when the path or the output is
    refused or an unexpected error is reported on stderr.
    """

    parser = argparse.ArgumentParser(prog="polish")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="Read a LAS file and print the audit ledger.")
    ingest_parser.add_argument("file", type=Path)
    ingest_parser.add_argument("--json", action="store_true", help="Print the ledger and dataset summary as JSON.")
    describe_parser = sub.add_parser("describe", help="Write a read-only describe report. No session is read.")
    describe_parser.add_argument("file", type=Path)
    describe_parser.add_argument("--out", type=Path, required=True, help="Directory for the two report files.")
    describe_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing describe report in the output directory.",
    )
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _ingest(parser, args)
    if args.command == "describe":
        return _describe(args)
    parser.error(f"Unknown command {args.command}")
    return 2


def _ingest(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Ingest exit codes are unchanged: 1 structural, 2 unclassified, 0 otherwise."""

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


def _describe(args: argparse.Namespace) -> int:
    """Write the describe report. Exit 2 still writes the report. D95, D19."""

    from polish.provenance.describe import OutputRefused, build_report, write_report

    if not args.file.is_file():
        print(f"Not a file: {args.file}", file=sys.stderr)
        return 1
    try:
        report = build_report(args.file)
        json_path, markdown_path = write_report(report, args.file, args.out, overwrite=args.overwrite)
    except OutputRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json_path)
    print(markdown_path)
    if report.structural_refusal:
        return 2
    return 0
