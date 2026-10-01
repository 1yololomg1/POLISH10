"""Stage 0: read a LAS file and record an audit ledger. D64, D65, D66, D68.

The file is opened read-only. Pass 1 reads the bytes and the text without a
parser library. Pass 2 may call lasio and compare. The dataset is built from
pass 1. A pass-2 result never overwrites a pass-1 sample.

Implements D64, D65 (PARTIAL), D66 (PARTIAL, F19), D68 (PARTIAL), D30, D46,
D17, C3, C6, and contracts section 6. D65, D66 and D68 are corroborated by
other implementations. They are not verified against the CWLS specification
(OD-24).

Structure
---------
CONTRACT
    Stage 0 contract. alters_data is False. depends_on_stages is empty.
ingest
    Read one file. Returns an IngestResult.
TextAudit
    Facts pass 1 recorded that are not findings: encoding, line endings,
    sections, VERS, WRAP, well header numbers, curve text, duplicate mnemonics.
IngestResult
    Source file, dataset or none, findings, audit, and null sample indexes.
UnclassifiedIngest
    Raised when a condition would need a block type the approved ledger
    mapping does not name. The partial result is attached. It is not classified.

Ledger mapping, approved with this stage
-----------------------------------------
undeclared NULL → AMBIGUOUS_INTERPRETATION
undeclared sentinel candidate → RULE_DEPENDENT_WARNING
header line not splitting cleanly → INFORMATIONAL
unrecognised section → INFORMATIONAL
LAS 3.0 feature detected → RULE_DEPENDENT_WARNING
column count irreconcilable after WRAP/comments/blanks → STRUCTURAL_DATA_FAILURE
no ~A section or no data rows → STRUCTURAL_DATA_FAILURE
encoding undeterminable → STRUCTURAL_DATA_FAILURE
pass-1/pass-2 disagreement → INFORMATIONAL, unless it changes a value, then
STRUCTURAL_DATA_FAILURE

NULL matching uses polish/rules/null_comparison.json. Sentinel candidates
live in polish/rules/null_sentinels.json. Stage 4 owns sample status, so a
null here is a NaN plus an index, not a SampleStatus.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy

from polish.model.dataset import (
    Curve,
    Dataset,
    DepthDirection,
    DepthGrid,
    DepthRegularity,
    SourceFile,
)
from polish.model.findings import Evidence, FindingRecord
from polish.model.ids import CurveRef, DatasetId, FileHash
from polish.model.states import BlockType, StageStatus
from polish.stages.contract import StageContract, StageGuarantees, StageRequires

# LAS 2.0 section letters. PARTIAL (D65): the first character after "~" is the
# section. The titles these letters stand for are the ones lasio and the
# legacy compliance checker both treat as the 2.0 set. Not CWLS-verified.
_KNOWN_SECTION_LETTERS = frozenset({"V", "W", "C", "P", "O", "A"})
_TRACKED_ABSENCES = ("V", "W", "C", "A")

# Harvested from lasio.reader.SectionParser.las3_section_indicators. A title
# containing one of these is reported as a LAS 3.0 feature and its body is
# not read as LAS 2.0 curves or data (D68). Not CWLS-verified (OD-24).
_LAS3_TITLE_MARKERS = ("_DATA", "_PARAMETER", "_DEFINITION")

_LASIO_SECTION_NAME = {
    "V": "Version",
    "W": "Well",
    "C": "Curves",
    "P": "Parameter",
    "O": "Other",
}

_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


def _load_null_comparison() -> tuple[float, float]:
    """Relative and absolute tolerances for a declared NULL. D66.

    Both numbers come from the rules file. A missing key is an error.
    There is no stand-in tolerance.
    """

    payload = json.loads((_RULES_DIR / "null_comparison.json").read_text(encoding="utf-8"))
    rel_tol = payload["rel_tol"]
    abs_tol = payload["abs_tol"]
    if isinstance(rel_tol, bool) or not isinstance(rel_tol, (int, float)):
        raise ValueError("null_comparison.json rel_tol must be a real number")
    if isinstance(abs_tol, bool) or not isinstance(abs_tol, (int, float)):
        raise ValueError("null_comparison.json abs_tol must be a real number")
    return float(rel_tol), float(abs_tol)


def _load_sentinels() -> tuple[tuple[str, float], ...]:
    """Candidate spellings and values for the undeclared-sentinel warning. D66, F19."""

    payload = json.loads((_RULES_DIR / "null_sentinels.json").read_text(encoding="utf-8"))
    rows = payload["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("null_sentinels.json candidates must be a non-empty list")
    loaded: list[tuple[str, float]] = []
    for row in rows:
        text = row["text"]
        value = row["value"]
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("null sentinel text must be a non-empty string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"null sentinel {text!r} value must be a real number")
        loaded.append((text, float(value)))
    return tuple(loaded)


NULL_REL_TOL, NULL_ABS_TOL = _load_null_comparison()
SENTINEL_CANDIDATES = _load_sentinels()

CONTRACT = StageContract(
    stage_number=0,
    name="Ingest",
    requires=StageRequires(
        units="Units are whatever the file declares. An absent unit stays absent (C1, C3).",
        completeness="The file may contain the declared NULL. Sample status is not known at ingest.",
        preconditions=(),
    ),
    guarantees=StageGuarantees(
        row_count="The number of depth steps read from ~A. Ingest does not add or drop a step.",
        units="Unchanged from the curve-section text.",
        validity="Declared NULL samples are NaN in the array and listed by index. Sample status is not set (Stage 4).",
    ),
    declines_when=(
        "The file encoding cannot be determined.",
        "The file has no ~A section or the data section has no rows.",
        "The data column count cannot be reconciled with the ~C count after WRAP, comments, and blank lines.",
        "VERS has major version 3. LAS 3.0 is reported and is not parsed as LAS 2.0.",
    ),
    depends_on_stages=(),
    alters_data=False,
)


class UnclassifiedIngest(Exception):
    """A condition the approved ledger mapping does not name.

    ``partial`` is the audit and any findings already recorded. This exception
    does not choose a block type.
    """

    def __init__(self, message: str, partial: IngestResult) -> None:
        super().__init__(message)
        self.partial = partial


@dataclass(frozen=True)
class SectionMark:
    """One section marker. ``letter`` is the first character after ``~``. D65."""

    lineno: int
    letter: str
    title: str
    las3_feature: bool


@dataclass(frozen=True)
class ParsedHeaderLine:
    """One header line that split on the first period and the last colon. D65."""

    lineno: int
    section: str
    mnemonic: str
    unit: str
    value_text: str
    description: str


@dataclass(frozen=True)
class ParsedCurve:
    """Curve text as split from ~C. Identity is the column index, not the mnemonic. D30."""

    column_index: int
    mnemonic: str
    unit: str
    description: str


@dataclass(frozen=True)
class TextAudit:
    """Pass-1 facts that are recorded whether or not they are findings."""

    encoding_name: str | None
    encoding_method: str
    line_endings: tuple[str, ...]
    sections: tuple[SectionMark, ...]
    duplicate_section_letters: tuple[str, ...]
    absent_section_letters: tuple[str, ...]
    ascii_is_last: bool | None
    vers_text: str | None
    vers_major: int | None
    wrap_text: str | None
    wrapped: bool | None
    null_text: str | None
    null_number: float | None
    strt_text: str | None
    strt_unit: str | None
    strt_number: float | None
    stop_text: str | None
    stop_unit: str | None
    stop_number: float | None
    step_text: str | None
    step_unit: str | None
    step_number: float | None
    header_lines: tuple[ParsedHeaderLine, ...]
    curves: tuple[ParsedCurve, ...]
    duplicate_mnemonics: tuple[str, ...]
    repeated_well_mnemonics: tuple[str, ...]
    preamble_lines: tuple[int, ...]
    column_mismatches: tuple[tuple[int, int, int], ...]
    data_token_count: int
    depth_step_count: int | None


@dataclass(frozen=True)
class IngestResult:
    """What Stage 0 returns. ``dataset`` is None when no source frame was built."""

    source_file: SourceFile
    dataset: Dataset | None
    findings: tuple[FindingRecord, ...]
    audit: TextAudit
    null_sample_indexes: tuple[tuple[int, ...], ...]
    status: StageStatus


@dataclass
class _Ledger:
    """Approved findings only. Identifiers are stable within one ingest."""

    read_at: datetime
    file_hash: FileHash
    items: list[FindingRecord]

    def __init__(self, read_at: datetime, file_hash: FileHash) -> None:
        self.read_at = read_at
        self.file_hash = file_hash
        self.items = []

    def add(self, block_type: BlockType, message: str, column: int | None = None) -> None:
        if message.strip() == "":
            raise ValueError("A ledger entry needs a message.")
        number = len(self.items) + 1
        refs: list[FileHash | CurveRef] = [self.file_hash]
        if column is not None:
            refs.append(CurveRef(DatasetId.for_source(self.file_hash), column))
        self.items.append(
            FindingRecord(
                id=f"s00-{number:04d}",
                stage=0,
                block_type=block_type,
                subject_refs=tuple(refs),
                evidence=Evidence(block_type.value),
                message=message,
                created_at=self.read_at,
            )
        )


@dataclass
class _Pass1:
    """Working state for the text walk. Not part of the public result."""

    audit: TextAudit
    rows: list[list[float]] | None
    non_numeric: tuple[tuple[int, str], ...]
    header_for_lasio: tuple[tuple[int, str, str, bool], ...]


def _read_bytes(path: Path) -> bytes:
    """Read the whole file. The mode is binary read. D17, C5."""

    with path.open("rb") as handle:
        return handle.read()


def _detect_encoding(raw: bytes) -> tuple[str | None, str, str | None]:
    """Detect an encoding from the bytes. Never substitutes another encoding.

    A byte-order mark selects UTF-8, UTF-16, or UTF-32. With no mark, the
    file is UTF-8 only when every byte decodes as UTF-8. Anything else is
    undeterminable: latin-1 and ``errors='ignore'`` are not used.

    Returns (encoding name or None, how it was detected, decoded text or None).
    """

    if raw.startswith(b"\xff\xfe\x00\x00") or raw.startswith(b"\x00\x00\xfe\xff"):
        name, method, codec = "utf-32", "UTF-32 byte-order mark", "utf-32"
    elif raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        name, method, codec = "utf-16", "UTF-16 byte-order mark", "utf-16"
    elif raw.startswith(b"\xef\xbb\xbf"):
        name, method, codec = "utf-8", "UTF-8 byte-order mark EF BB BF", "utf-8-sig"
    else:
        name, method, codec = "utf-8", "strict UTF-8 decode of the whole file; no byte-order mark", "utf-8"
    try:
        text = raw.decode(codec)
    except UnicodeError:
        return None, (
            "Encoding undeterminable: no recognised byte-order mark, and the "
            "bytes are not valid UTF-8. No other encoding was assumed."
        ), None
    return name, method, text


def _line_endings(text: str) -> tuple[str, ...]:
    """Which of CRLF, LF, and CR occur. A mix is recorded, not classified."""

    crlf = text.count("\r\n")
    without_crlf = text.replace("\r\n", "")
    found: list[str] = []
    if crlf:
        found.append("CRLF")
    if "\n" in without_crlf:
        found.append("LF")
    if "\r" in without_crlf:
        found.append("CR")
    if not found:
        found.append("NONE")
    return tuple(found)


def _blank_or_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def _section_letter(line: str) -> str | None:
    """First character after ``~``, uppercased, or None when this is not a marker.

    PARTIAL, D65. A leading space on the line is indentation, not the
    character after the tilde. ``~CURVE`` and ``~C`` are both ``C``.
    ``~ ASCII`` is a space, which is not a letter.
    """

    stripped = line.strip()
    if not stripped.startswith("~"):
        return None
    if len(stripped) < 2:
        return ""
    return stripped[1].upper()


def _las3_title(title: str) -> bool:
    """True when the section title carries a LAS 3.0 marker. D68, PARTIAL."""

    upper = title.upper()
    return any(marker in upper for marker in _LAS3_TITLE_MARKERS)


def _major_version(vers_text: str | None) -> int | None:
    """Major version from the VERS value text, or None when it is not a finite number."""

    if vers_text is None:
        return None
    try:
        number = float(vers_text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(math.floor(number))


def split_header_line(line: str) -> tuple[str, str, str, str] | None:
    """Split one header line on the first period and the last colon. D65, PARTIAL.

    Returns (mnemonic, unit, value text, description) or None when the line
    does not have both delimiters or the mnemonic is empty. The mnemonic is
    the text before the first period with surrounding spaces removed. The
    unit is the run of non-space characters immediately after that period,
    kept character for character. The description is the text after the last
    colon, including spaces. The value text is the remainder, with surrounding
    spaces removed so it can be read as a number.

    A colon inside the description is part of the value, because the split
    is the last colon. This is the PARTIAL D65 rule. It is not the first-colon
    split the golden-facts extractor uses.
    """

    dot = line.find(".")
    if dot < 0:
        return None
    mnemonic = line[:dot].strip()
    if mnemonic == "":
        return None
    after = line[dot + 1 :]
    colon = after.rfind(":")
    if colon < 0:
        return None
    middle = after[:colon]
    description = after[colon + 1 :]
    unit_end = 0
    while unit_end < len(middle) and not middle[unit_end].isspace():
        unit_end += 1
    unit = middle[:unit_end]
    value_text = middle[unit_end:].strip()
    return mnemonic, unit, value_text, description


def _parse_finite(text: str) -> float | None:
    """A finite float, or None. Scientific notation is accepted. D66, F19."""

    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _is_close(left: float, right: float) -> bool:
    """Declared-NULL comparison. The tolerances are the rules file, not a default."""

    return math.isclose(left, right, rel_tol=NULL_REL_TOL, abs_tol=NULL_ABS_TOL)


def _empty_audit(encoding_name: str | None, encoding_method: str) -> TextAudit:
    return TextAudit(
        encoding_name=encoding_name,
        encoding_method=encoding_method,
        line_endings=(),
        sections=(),
        duplicate_section_letters=(),
        absent_section_letters=_TRACKED_ABSENCES,
        ascii_is_last=None,
        vers_text=None,
        vers_major=None,
        wrap_text=None,
        wrapped=None,
        null_text=None,
        null_number=None,
        strt_text=None,
        strt_unit=None,
        strt_number=None,
        stop_text=None,
        stop_unit=None,
        stop_number=None,
        step_text=None,
        step_unit=None,
        step_number=None,
        header_lines=(),
        curves=(),
        duplicate_mnemonics=(),
        repeated_well_mnemonics=(),
        preamble_lines=(),
        column_mismatches=(),
        data_token_count=0,
        depth_step_count=None,
    )


def _well_slot(mnemonic: str) -> str | None:
    upper = mnemonic.upper()
    if upper in {"NULL", "STRT", "STOP", "STEP"}:
        return upper
    return None


def _pass1(text: str, ledger: _Ledger) -> _Pass1:
    """Walk the text. Parser libraries are not used. D64, D65, D66, D68."""

    endings = _line_endings(text)
    sections: list[SectionMark] = []
    letter_counts: Counter[str] = Counter()
    header_lines: list[ParsedHeaderLine] = []
    curves: list[ParsedCurve] = []
    header_for_lasio: list[tuple[int, str, str, bool]] = []
    data_lines: list[tuple[int, str]] = []
    preamble_lines: list[int] = []
    collecting_data = False
    vers_text: str | None = None
    wrap_text: str | None = None
    well: dict[str, tuple[str, str, float | None]] = {}
    repeated_well: list[str] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        letter = _section_letter(line)
        if letter is not None:
            title = line.strip()
            las3 = _las3_title(title)
            sections.append(SectionMark(lineno, letter, title, las3))
            if letter == "":
                ledger.add(
                    BlockType.INFORMATIONAL,
                    f"Line {lineno}: section marker {title!r} has no character after '~'.",
                )
            elif letter not in _KNOWN_SECTION_LETTERS:
                ledger.add(
                    BlockType.INFORMATIONAL,
                    f"Line {lineno}: unrecognised section {title!r} (first character {letter!r}).",
                )
            if las3:
                ledger.add(
                    BlockType.RULE_DEPENDENT_WARNING,
                    (
                        f"Line {lineno}: LAS 3.0 feature in section title {title!r}. "
                        "The section was not parsed as LAS 2.0."
                    ),
                )
            letter_counts[letter] += 1
            collecting_data = letter == "A" and not las3
            if collecting_data:
                data_lines = []
            continue
        if collecting_data:
            data_lines.append((lineno, line))
            continue
        if not sections:
            if not _blank_or_comment(line):
                preamble_lines.append(lineno)
            continue
        if _blank_or_comment(line):
            continue
        current = sections[-1]
        if current.las3_feature or current.letter not in _KNOWN_SECTION_LETTERS or current.letter == "A":
            continue
        parsed = split_header_line(line)
        split_failed = parsed is None
        header_for_lasio.append((lineno, current.letter, line, split_failed))
        if parsed is None:
            ledger.add(
                BlockType.INFORMATIONAL,
                (
                    f"Line {lineno}: header line in section {current.letter} does not split "
                    f"on the first period and the last colon: {line.strip()!r}."
                ),
            )
            continue
        mnemonic, unit, value_text, description = parsed
        header_lines.append(
            ParsedHeaderLine(lineno, current.letter, mnemonic, unit, value_text, description)
        )
        if current.letter == "V" and mnemonic.upper() == "VERS" and vers_text is None:
            vers_text = value_text
        elif current.letter == "V" and mnemonic.upper() == "WRAP" and wrap_text is None:
            wrap_text = value_text
        elif current.letter == "W":
            slot = _well_slot(mnemonic)
            if slot is not None:
                if slot in well:
                    repeated_well.append(slot)
                else:
                    well[slot] = (unit, value_text, _parse_finite(value_text))
        elif current.letter == "C":
            curves.append(ParsedCurve(len(curves), mnemonic, unit, description))

    vers_major = _major_version(vers_text)
    if vers_major == 3:
        ledger.add(
            BlockType.RULE_DEPENDENT_WARNING,
            (
                f"LAS 3.0 feature: VERS value {vers_text!r} has major version 3. "
                "The file was not parsed as LAS 2.0."
            ),
        )

    wrapped: bool | None
    if wrap_text is None:
        wrapped = None
    else:
        flag = wrap_text.strip().upper()
        if flag == "YES":
            wrapped = True
        elif flag == "NO":
            wrapped = False
        else:
            wrapped = None

    duplicate_letters = tuple(letter for letter, count in letter_counts.items() if count > 1 and letter != "")
    absent = tuple(letter for letter in _TRACKED_ABSENCES if letter_counts[letter] == 0)
    ascii_is_last = (sections[-1].letter == "A") if sections else None

    mnemonic_counts: Counter[str] = Counter(curve.mnemonic for curve in curves)
    duplicate_mnemonics = tuple(name for name, count in mnemonic_counts.items() if count > 1)

    null_text = well["NULL"][1] if "NULL" in well else None
    null_number = well["NULL"][2] if "NULL" in well else None
    if "NULL" not in well:
        ledger.add(
            BlockType.AMBIGUOUS_INTERPRETATION,
            "NULL is not declared in the well section. No null value was assumed, and no sample was converted.",
        )
    # A NULL line whose value is not a finite number stays declared and unused.
    # That is not the undeclared-NULL condition, and no number is assumed.

    strt = well.get("STRT", (None, None, None))
    stop = well.get("STOP", (None, None, None))
    step = well.get("STEP", (None, None, None))

    rows: list[list[float]] | None = None
    mismatches: list[tuple[int, int, int]] = []
    non_numeric: list[tuple[int, str]] = []
    token_count = 0
    depth_steps: int | None = None
    parse_data = vers_major != 3

    if parse_data and "A" not in letter_counts:
        ledger.add(BlockType.STRUCTURAL_DATA_FAILURE, "No ~A section.")
    elif parse_data:
        content = [(lineno, line) for lineno, line in data_lines if not _blank_or_comment(line)]
        n_curves = len(curves)
        if wrapped is None:
            token_count = sum(len(line.split()) for _, line in content)
        elif wrapped:
            tokens: list[tuple[int, str]] = []
            for lineno, line in content:
                for token in line.split():
                    tokens.append((lineno, token))
            token_count = len(tokens)
            if n_curves == 0:
                if token_count:
                    ledger.add(
                        BlockType.STRUCTURAL_DATA_FAILURE,
                        (
                            f"Column count irreconcilable after WRAP=YES, comments, and blank lines: "
                            f"~C count is 0 and the data section has {token_count} tokens."
                        ),
                    )
            elif token_count % n_curves != 0:
                leftover = token_count % n_curves
                complete = token_count // n_curves
                ledger.add(
                    BlockType.STRUCTURAL_DATA_FAILURE,
                    (
                        f"Column count irreconcilable after WRAP=YES, comments, and blank lines: "
                        f"~C count is {n_curves}, data tokens are {token_count}, "
                        f"{leftover} left over after {complete} complete steps."
                    ),
                )
            else:
                built, bad = _rows_from_tokens(tokens, n_curves)
                non_numeric.extend(bad)
                if not bad:
                    rows = built
                    depth_steps = len(built)
        else:
            parsed_rows: list[list[float]] = []
            bad_line = False
            for lineno, line in content:
                parts = line.split()
                token_count += len(parts)
                if len(parts) != n_curves:
                    mismatches.append((lineno, len(parts), n_curves))
                    bad_line = True
                    continue
                values, bad = _floats_on_line(lineno, parts)
                non_numeric.extend(bad)
                if bad:
                    bad_line = True
                elif not bad_line:
                    parsed_rows.append(values)
            if mismatches:
                shown = ", ".join(
                    f"line {lineno} has {actual} tokens where ~C count is {expected}"
                    for lineno, actual, expected in mismatches[:10]
                )
                extra = ""
                if len(mismatches) > 10:
                    extra = f" {len(mismatches) - 10} further lines are in the audit."
                ledger.add(
                    BlockType.STRUCTURAL_DATA_FAILURE,
                    (
                        f"Column count irreconcilable after WRAP=NO, comments, and blank lines: "
                        f"{len(mismatches)} data line(s) differ from ~C count {n_curves}. {shown}.{extra}"
                    ),
                )
            elif not bad_line:
                rows = parsed_rows
                depth_steps = len(parsed_rows)
        if rows is not None and depth_steps == 0:
            ledger.add(
                BlockType.STRUCTURAL_DATA_FAILURE,
                "The ~A section has no data rows after comments and blank lines.",
            )
            rows = None
            depth_steps = 0
        elif content == [] and not any(
            item.block_type is BlockType.STRUCTURAL_DATA_FAILURE for item in ledger.items
        ):
            ledger.add(
                BlockType.STRUCTURAL_DATA_FAILURE,
                "The ~A section has no data rows after comments and blank lines.",
            )
            depth_steps = 0

    audit = TextAudit(
        encoding_name=None,
        encoding_method="",
        line_endings=endings,
        sections=tuple(sections),
        duplicate_section_letters=duplicate_letters,
        absent_section_letters=absent,
        ascii_is_last=ascii_is_last,
        vers_text=vers_text,
        vers_major=vers_major,
        wrap_text=wrap_text,
        wrapped=wrapped,
        null_text=null_text,
        null_number=null_number,
        strt_text=strt[1],
        strt_unit=strt[0],
        strt_number=strt[2],
        stop_text=stop[1],
        stop_unit=stop[0],
        stop_number=stop[2],
        step_text=step[1],
        step_unit=step[0],
        step_number=step[2],
        header_lines=tuple(header_lines),
        curves=tuple(curves),
        duplicate_mnemonics=duplicate_mnemonics,
        repeated_well_mnemonics=tuple(repeated_well),
        preamble_lines=tuple(preamble_lines),
        column_mismatches=tuple(mismatches),
        data_token_count=token_count,
        depth_step_count=depth_steps,
    )
    return _Pass1(audit, rows, tuple(non_numeric), tuple(header_for_lasio))


def _floats_on_line(lineno: int, parts: list[str]) -> tuple[list[float], list[tuple[int, str]]]:
    values: list[float] = []
    bad: list[tuple[int, str]] = []
    for token in parts:
        number = _parse_finite(token)
        if number is None:
            bad.append((lineno, token))
        else:
            values.append(number)
    if bad:
        return [], bad
    return values, []


def _rows_from_tokens(
    tokens: list[tuple[int, str]], n_curves: int
) -> tuple[list[list[float]], list[tuple[int, str]]]:
    rows: list[list[float]] = []
    bad: list[tuple[int, str]] = []
    for start in range(0, len(tokens), n_curves):
        chunk = tokens[start : start + n_curves]
        values, chunk_bad = _floats_on_line(chunk[0][0], [token for _, token in chunk])
        if chunk_bad:
            bad.extend(chunk_bad)
        elif not bad:
            rows.append(values)
    if bad:
        return [], bad
    return rows, []


def _status(findings: tuple[FindingRecord, ...]) -> StageStatus:
    if any(item.block_type is BlockType.STRUCTURAL_DATA_FAILURE for item in findings):
        return StageStatus.REFUSED_STRUCTURAL
    return StageStatus.COMPLETE


def _result(
    source: SourceFile,
    dataset: Dataset | None,
    ledger: _Ledger,
    audit: TextAudit,
    null_indexes: tuple[tuple[int, ...], ...],
) -> IngestResult:
    findings = tuple(ledger.items)
    return IngestResult(source, dataset, findings, audit, null_indexes, _status(findings))


def _direction(depths: numpy.ndarray) -> DepthDirection | None:
    """Strict pairwise direction. No tolerance. D88's test, recorded not reversed."""

    diffs = numpy.diff(depths)
    if diffs.size == 0:
        return None
    if bool(numpy.all(diffs > 0)):
        return DepthDirection.ASCENDING
    if bool(numpy.all(diffs < 0)):
        return DepthDirection.DESCENDING
    return None


def _regularity(depths: numpy.ndarray, step: float) -> DepthRegularity:
    """REGULAR only when every difference equals the declared STEP exactly.

    This is float64 equality, not the Stage 1 rounding rule. OD-10 is not
    applied here. A grid that is regular only after rounding is IRREGULAR.
    """

    diffs = numpy.diff(depths)
    if diffs.size and bool(numpy.all(diffs == step)):
        return DepthRegularity.REGULAR
    return DepthRegularity.IRREGULAR


def _apply_nulls(
    rows: list[list[float]], null_number: float | None
) -> tuple[numpy.ndarray, tuple[tuple[int, ...], ...]]:
    """Copy rows to float64 and replace declared-NULL matches with NaN. D66.

    Samples that are not the declared NULL stay numbers, including values that
    merely coincide with a sentinel candidate.
    """

    matrix = numpy.array(rows, dtype=numpy.float64)
    indexes: list[tuple[int, ...]] = []
    if null_number is None:
        return matrix, tuple(() for _ in range(matrix.shape[1]))
    for column in range(matrix.shape[1]):
        mask = numpy.isclose(matrix[:, column], null_number, rtol=NULL_REL_TOL, atol=NULL_ABS_TOL)
        found = tuple(int(index) for index in numpy.flatnonzero(mask))
        matrix[mask, column] = numpy.nan
        indexes.append(found)
    return matrix, tuple(indexes)


def _sentinel_warnings(matrix: numpy.ndarray, curves: tuple[ParsedCurve, ...], null_number: float | None, ledger: _Ledger) -> None:
    """Warn on sentinel candidates that are not the declared NULL. D66, F19.

    The values are not changed. Scientific notation is not a signal: only the
    numeric candidates in the rules file are checked.
    """

    grouped: list[tuple[list[str], float]] = []
    for text, value in SENTINEL_CANDIDATES:
        for labels, existing in grouped:
            if existing == value:
                labels.append(text)
                break
        else:
            grouped.append(([text], value))
    for labels, sentinel in grouped:
        if null_number is not None and _is_close(sentinel, null_number):
            continue
        spelling = labels[0] if len(labels) == 1 else f"{labels[0]} (also written {', '.join(labels[1:])})"
        for column, curve in enumerate(curves):
            series = matrix[:, column]
            finite = numpy.isfinite(series)
            if not bool(numpy.any(finite)):
                continue
            hits = numpy.isclose(series, sentinel, rtol=NULL_REL_TOL, atol=NULL_ABS_TOL) & finite
            count = int(numpy.count_nonzero(hits))
            if count == 0:
                continue
            ledger.add(
                BlockType.RULE_DEPENDENT_WARNING,
                (
                    f"Column {column} mnemonic {curve.mnemonic!r} has {count} sample(s) equal to "
                    f"undeclared sentinel candidate {spelling}. The samples were not converted."
                ),
                column=column,
            )


def _build_dataset(
    source: SourceFile,
    audit: TextAudit,
    matrix: numpy.ndarray,
) -> Dataset:
    """Source dataset from pass-1 columns. Arrays are copied read-only. D17, D30, C5, C6."""

    dataset_id = DatasetId.for_source(source.file_hash)
    if audit.step_number is None:
        raise UnclassifiedIngest(
            "STEP is not a finite number. DepthGrid.declared_step has no approved stand-in, "
            "and this is not an approved ledger condition.",
            _placeholder_partial(source, audit),
        )
    depths = matrix[:, 0]
    direction = _direction(depths)
    if direction is None:
        raise UnclassifiedIngest(
            "The index curve is not strictly monotonic, or it has fewer than 2 samples. "
            "Stage 1 owns that finding (P09). It is not in the Stage 0 ledger mapping.",
            _placeholder_partial(source, audit),
        )
    grid = DepthGrid(
        depths,
        direction,
        _regularity(depths, audit.step_number),
        audit.step_number,
        audit.curves[0].unit,
    )
    curves = tuple(
        Curve(
            CurveRef(dataset_id, curve.column_index),
            curve.mnemonic,
            curve.unit,
            curve.description,
            matrix[:, curve.column_index],
        )
        for curve in audit.curves
    )
    return Dataset(dataset_id, None, source, grid, curves, ())


def _placeholder_partial(source: SourceFile, audit: TextAudit) -> IngestResult:
    """Used only so UnclassifiedIngest can carry the audit before findings are sealed.

    Callers replace this with the real result before raising.
    """

    return IngestResult(source, None, (), audit, (), StageStatus.COMPLETE)


def _compare_pass2(
    path: Path,
    audit: TextAudit,
    matrix: numpy.ndarray | None,
    header_for_lasio: tuple[tuple[int, str, str, bool], ...],
    ledger: _Ledger,
) -> None:
    """Read with lasio and record every disagreement with pass 1. D64.

    ``ignore_header_errors=True`` is used only after each header line lasio
    would reject has already been recorded. The rejection check calls
    ``lasio.reader.read_header_line``, which is the call ``ignore_header_errors``
    wraps in lasio.reader.parse_header_items_section. Mnemonic case is
    ``preserve`` so lasio's default uppercasing is not a false disagreement.
    """

    if audit.vers_major == 3:
        return
    import lasio
    from lasio.exceptions import LASDataError, LASHeaderError
    from lasio.reader import read_header_line

    captured = {lineno for lineno, _letter, _line, failed in header_for_lasio if failed}
    lasio_rejects: list[tuple[int, str]] = []
    for lineno, letter, line, _failed in header_for_lasio:
        section_name = _LASIO_SECTION_NAME.get(letter, letter)
        try:
            read_header_line(line.strip(), section_name=section_name)
        except (AttributeError, ValueError, TypeError, IndexError, re.error):
            lasio_rejects.append((lineno, line.strip()))
    for lineno, text in lasio_rejects:
        if lineno in captured:
            continue
        ledger.add(
            BlockType.INFORMATIONAL,
            (
                f"Pass 2 cannot read header line {lineno}, which pass 1 split: {text!r}. "
                "No sample value was changed."
            ),
        )
        captured.add(lineno)
    use_ignore = bool(lasio_rejects)
    try:
        parsed = lasio.read(str(path), ignore_header_errors=use_ignore, mnemonic_case="preserve")
    except LASHeaderError as exc:
        ledger.add(
            BlockType.INFORMATIONAL,
            (
                "Pass 2 stopped on a header error that was not suppressed "
                f"({exc}). ignore_header_errors was not used to hide an unrecorded line. "
                "No sample value was changed."
            ),
        )
        return
    except (LASDataError, UnicodeError, OSError, ValueError) as exc:
        ledger.add(
            BlockType.INFORMATIONAL,
            f"Pass 2 did not return a parse ({type(exc).__name__}: {exc}). No sample value was changed.",
        )
        return
    _compare_header_numbers(audit, parsed, ledger)
    _compare_curves(audit, matrix, parsed, ledger)


def _compare_header_numbers(audit: TextAudit, parsed: object, ledger: _Ledger) -> None:
    """VERS, WRAP, NULL, STRT, STOP, STEP. A numeric change is structural. D64."""

    version = _item_value(parsed, "version", "VERS")
    wrap = _item_value(parsed, "version", "WRAP")
    _compare_one_header("VERS", audit.vers_text, version, ledger)
    _compare_one_header("WRAP", audit.wrap_text, wrap, ledger)
    _compare_one_header("NULL", audit.null_number, _item_value(parsed, "well", "NULL"), ledger)
    _compare_one_header("STRT", audit.strt_number, _item_value(parsed, "well", "STRT"), ledger)
    _compare_one_header("STOP", audit.stop_number, _item_value(parsed, "well", "STOP"), ledger)
    _compare_one_header("STEP", audit.step_number, _item_value(parsed, "well", "STEP"), ledger)


def _item_value(parsed: object, section_name: str, mnemonic: str) -> object:
    section = getattr(parsed, section_name, None)
    if section is None:
        return None
    try:
        item = section[mnemonic]
    except KeyError:
        return None
    return getattr(item, "value", None)


def _as_real(value: object) -> float | None:
    """A finite real from a Python or numpy number. Bool is not a number."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    item = getattr(value, "item", None)
    if not callable(item):
        return None
    try:
        raw = item()
    except (ValueError, TypeError):
        return None
    return _as_real(raw)


def _compare_one_header(label: str, ours: object, theirs: object, ledger: _Ledger) -> None:
    if ours is None and theirs is None:
        return
    our_number = _as_real(ours)
    their_number = _as_real(theirs)
    if our_number is not None and their_number is not None:
        if _is_close(our_number, their_number):
            return
        ledger.add(
            BlockType.STRUCTURAL_DATA_FAILURE,
            f"Pass 1 {label} value {our_number} disagrees with pass 2 value {their_number}.",
        )
        return
    if isinstance(ours, str) and their_number is not None:
        parsed = _parse_finite(ours)
        if parsed is not None and _is_close(parsed, their_number):
            return
    if ours is not None and theirs is not None and str(ours).strip() == str(theirs).strip():
        return
    changes_value = our_number is not None or their_number is not None
    block = BlockType.STRUCTURAL_DATA_FAILURE if changes_value else BlockType.INFORMATIONAL
    tail = "This changes a recorded value." if changes_value else "No sample value was changed."
    ledger.add(block, f"Pass 1 {label} {ours!r} disagrees with pass 2 {theirs!r}. {tail}")


def _compare_curves(audit: TextAudit, matrix: numpy.ndarray | None, parsed: object, ledger: _Ledger) -> None:
    las_curves = list(getattr(parsed, "curves"))
    if len(las_curves) != len(audit.curves):
        ledger.add(
            BlockType.STRUCTURAL_DATA_FAILURE,
            (
                f"Pass 1 curve count {len(audit.curves)} disagrees with pass 2 curve count {len(las_curves)}. "
                "This changes which values are in the file."
            ),
        )
        return
    whitespace_columns: list[int] = []
    for index, (ours, theirs) in enumerate(zip(audit.curves, las_curves, strict=True)):
        their_name = str(getattr(theirs, "mnemonic", ""))
        their_unit = str(getattr(theirs, "unit", ""))
        their_desc = str(getattr(theirs, "descr", ""))
        if their_name.strip() != ours.mnemonic.strip():
            ledger.add(
                BlockType.INFORMATIONAL,
                (
                    f"Pass 1 mnemonic {ours.mnemonic!r} at column {index} disagrees with "
                    f"pass 2 mnemonic {their_name!r}. No sample value was changed."
                ),
                column=index,
            )
        if their_unit.strip() != ours.unit.strip():
            ledger.add(
                BlockType.INFORMATIONAL,
                (
                    f"Pass 1 unit {ours.unit!r} at column {index} disagrees with "
                    f"pass 2 unit {their_unit!r}. No sample value was changed."
                ),
                column=index,
            )
        if their_desc.strip() != ours.description.strip():
            ledger.add(
                BlockType.INFORMATIONAL,
                (
                    f"Pass 1 description {ours.description!r} at column {index} disagrees with "
                    f"pass 2 description {their_desc!r}. No sample value was changed."
                ),
                column=index,
            )
        elif their_desc != ours.description:
            whitespace_columns.append(index)
        if matrix is None:
            continue
        their_values = numpy.asarray(getattr(theirs, "data"), dtype=numpy.float64)
        our_values = matrix[:, index]
        if their_values.shape != our_values.shape:
            ledger.add(
                BlockType.STRUCTURAL_DATA_FAILURE,
                (
                    f"Pass 1 column {index} length {our_values.shape[0]} disagrees with "
                    f"pass 2 length {their_values.shape[0]}."
                ),
                column=index,
            )
            continue
        nan_ours = numpy.isnan(our_values)
        nan_theirs = numpy.isnan(their_values)
        numeric = ~nan_ours & ~nan_theirs
        differ = (nan_ours != nan_theirs) | (
            numeric & ~numpy.isclose(our_values, their_values, rtol=NULL_REL_TOL, atol=NULL_ABS_TOL)
        )
        count = int(numpy.count_nonzero(differ))
        if count == 0:
            continue
        first = int(numpy.flatnonzero(differ)[0])
        ledger.add(
            BlockType.STRUCTURAL_DATA_FAILURE,
            (
                f"Pass 1 and pass 2 disagree on {count} sample(s) in column {index} "
                f"mnemonic {ours.mnemonic!r}. First index {first}: "
                f"pass 1 {our_values[first]!r}, pass 2 {their_values[first]!r}."
            ),
            column=index,
        )
    if whitespace_columns:
        listed = ", ".join(str(index) for index in whitespace_columns)
        ledger.add(
            BlockType.INFORMATIONAL,
            (
                f"Pass 2 description text differs from the file text by surrounding whitespace "
                f"only, on column(s) {listed}. No sample value was changed."
            ),
        )


def _with_encoding(audit: TextAudit, name: str | None, method: str) -> TextAudit:
    return TextAudit(
        encoding_name=name,
        encoding_method=method,
        line_endings=audit.line_endings,
        sections=audit.sections,
        duplicate_section_letters=audit.duplicate_section_letters,
        absent_section_letters=audit.absent_section_letters,
        ascii_is_last=audit.ascii_is_last,
        vers_text=audit.vers_text,
        vers_major=audit.vers_major,
        wrap_text=audit.wrap_text,
        wrapped=audit.wrapped,
        null_text=audit.null_text,
        null_number=audit.null_number,
        strt_text=audit.strt_text,
        strt_unit=audit.strt_unit,
        strt_number=audit.strt_number,
        stop_text=audit.stop_text,
        stop_unit=audit.stop_unit,
        stop_number=audit.stop_number,
        step_text=audit.step_text,
        step_unit=audit.step_unit,
        step_number=audit.step_number,
        header_lines=audit.header_lines,
        curves=audit.curves,
        duplicate_mnemonics=audit.duplicate_mnemonics,
        repeated_well_mnemonics=audit.repeated_well_mnemonics,
        preamble_lines=audit.preamble_lines,
        column_mismatches=audit.column_mismatches,
        data_token_count=audit.data_token_count,
        depth_step_count=audit.depth_step_count,
    )


def ingest(path: Path | str, *, read_at: datetime | None = None) -> IngestResult:
    """Read one LAS file into a source dataset and an audit ledger.

    Implements D64, D65 (PARTIAL), D66 (PARTIAL, F19), D68 (PARTIAL), D30,
    D46, D17, C3, C6, contracts section 6.

    Parameters
    ----------
    path:
        Input LAS file. Opened read-only. The bytes are not written.
    read_at:
        Timestamp stored on the source file and the findings. When omitted,
        the clock is UTC now. Callers pass a fixed time in tests.

    Returns
    -------
    IngestResult
        The dataset is None when a structural finding, or a LAS 3.0 version,
        means no source frame was built. Null samples are NaN, and their
        indexes are on ``null_sample_indexes``.

    Raises
    ------
    UnclassifiedIngest
        The text has a condition the approved mapping does not name, and a
        dataset cannot be built without classifying it.
    """

    file_path = Path(path)
    if read_at is None:
        read_at = datetime.now(timezone.utc)
    if not isinstance(read_at, datetime):
        raise TypeError("read_at must be a datetime or None")
    raw = _read_bytes(file_path)
    source = SourceFile(
        path=file_path,
        file_hash=FileHash(hashlib.sha256(raw).hexdigest()),
        size_bytes=len(raw),
        read_at=read_at,
    )
    ledger = _Ledger(read_at, source.file_hash)
    encoding_name, encoding_method, text = _detect_encoding(raw)
    if text is None:
        ledger.add(BlockType.STRUCTURAL_DATA_FAILURE, encoding_method)
        audit = _empty_audit(None, encoding_method)
        return _result(source, None, ledger, audit, ())

    passed = _pass1(text, ledger)
    audit = _with_encoding(passed.audit, encoding_name, encoding_method)
    structural = any(item.block_type is BlockType.STRUCTURAL_DATA_FAILURE for item in ledger.items)
    las3_version = audit.vers_major == 3
    matrix: numpy.ndarray | None = None
    null_indexes: tuple[tuple[int, ...], ...] = ()
    dataset: Dataset | None = None

    if (
        not structural
        and not las3_version
        and passed.rows is None
        and audit.wrapped is None
        and audit.data_token_count > 0
        and "A" not in audit.absent_section_letters
    ):
        partial = _result(source, None, ledger, audit, ())
        raise UnclassifiedIngest(
            "WRAP is not YES or NO, so the column layout was not assumed. "
            "This is not an approved ledger condition.",
            partial,
        )
    if passed.non_numeric and not structural and not las3_version:
        shown = ", ".join(f"line {lineno} token {token!r}" for lineno, token in passed.non_numeric[:5])
        partial = _result(source, None, ledger, audit, ())
        raise UnclassifiedIngest(
            "A data token is not a finite number (" + shown + "). "
            "Converting it to NaN would invent a null. This is not an approved ledger condition.",
            partial,
        )
    if passed.rows is not None and not structural and not las3_version:
        if audit.wrapped is None:
            partial = _result(source, None, ledger, audit, ())
            raise UnclassifiedIngest(
                "WRAP is not YES or NO, so the column layout was not assumed. "
                "This is not an approved ledger condition.",
                partial,
            )
        matrix, null_indexes = _apply_nulls(passed.rows, audit.null_number)
        _sentinel_warnings(matrix, audit.curves, audit.null_number, ledger)
        try:
            dataset = _build_dataset(source, audit, matrix)
        except UnclassifiedIngest as exc:
            partial = _result(source, None, ledger, audit, null_indexes)
            raise UnclassifiedIngest(str(exc), partial) from exc

    _compare_pass2(file_path, audit, matrix, passed.header_for_lasio, ledger)
    return _result(source, dataset, ledger, audit, null_indexes)
