"""Independent reader of raw LAS facts. D27.

Golden expectations for Stages 0–4 come from the files themselves. This module
does not import polish, lasio, or the legacy application, and it does not read
legacy pipeline output.

These facts are measurements, not an approval. A person still has to review
the JSON before any stage treats it as expected behaviour.

Structure
---------
FactsError
    Raised when a file does not contain a required field, or when the data
    section does not form complete depth steps. Nothing is filled in.
parse_header_line
    LAS 2.0 line: mnemonic, unit, value text, description.
is_declared_null
    Relative comparison against the declared NULL. rel_tol is 1e-6 and
    abs_tol is 0, which is the tolerance this prompt set. It is not a
    petrophysical cutoff.
extract_facts
    SHA-256, VERS, WRAP, NULL, STRT, STOP, STEP, every ~C line, depth-step
    count, and per-curve null count, depth span, and min/max.
canonical_json
    Stable text for the committed facts files.
main
    Writes one facts file per fixture, skipping any name that contains
    KEOUGH (D5 is deferred; that well is not a public fixture).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Streaming read size for the file hash. Not a petrophysical threshold.
_HASH_CHUNK_BYTES = 65536

# Prompt: a value equals the declared NULL within a relative tolerance of 1e-6.
# abs_tol is 0 because the prompt states a relative tolerance only.
NULL_REL_TOL = 1e-6
NULL_ABS_TOL = 0.0

_REQUIRED_WELL = ("STRT", "STOP", "STEP", "NULL")


class FactsError(Exception):
    """The LAS text cannot be recorded without inventing a missing fact."""


def is_keough(path: Path) -> bool:
    """True when the file name is the deferred KEOUGH well. D5 is not in this pass."""
    return "keough" in path.name.lower()


def is_declared_null(value: float, null: float) -> bool:
    """Whether ``value`` matches the declared NULL.

    Uses ``math.isclose(value, null, rel_tol=1e-6, abs_tol=0.0)``, so
    ``abs(value - null) <= 1e-6 * max(abs(value), abs(null))``.
    A NULL of 0 therefore matches only 0, because the relative band collapses.
    """
    return math.isclose(value, null, rel_tol=NULL_REL_TOL, abs_tol=NULL_ABS_TOL)


def parse_header_line(line: str, lineno: int) -> tuple[str, str, str, str]:
    """Split one LAS 2.0 header line.

    The first ``.`` ends the mnemonic. The unit is the following run of
    non-space characters, kept as written (``M``, ``m``, ``kPa.s/m``, ``_``,
    or empty). The last ``:`` on the line ends the value. The description is
    everything after that colon, including spaces. A description that itself
    contains a colon is stored as the text to the right of the last one.
    Mnemonic and value have their surrounding alignment spaces removed; the
    characters that remain are the file's own text.

    Returns (mnemonic, unit, value_text, description).
    """
    if "." not in line:
        raise FactsError(f"line {lineno}: no '.' in header line")
    left, rest = line.split(".", 1)
    mnemonic = left.strip()
    if mnemonic == "":
        raise FactsError(f"line {lineno}: empty mnemonic")
    unit_end = 0
    while unit_end < len(rest) and not rest[unit_end].isspace():
        unit_end += 1
    unit = rest[:unit_end]
    after_unit = rest[unit_end:]
    # D65 / CWLS LAS 2.0 (Feb 2017): the delimiter is the last colon on the line.
    colon = after_unit.rfind(":")
    if colon < 0:
        raise FactsError(f"line {lineno}: no ':' in header line")
    value_text = after_unit[:colon].strip()
    description = after_unit[colon + 1 :]
    return mnemonic, unit, value_text, description


def _section_letter(line: str, lineno: int) -> str | None:
    """LAS section id: the first letter after ``~``, uppercased.

    Returns None when the line is not a section marker.
    """
    stripped = line.lstrip()
    if not stripped.startswith("~"):
        return None
    for char in stripped[1:]:
        if char.isalpha():
            return char.upper()
        if char.isspace():
            continue
        raise FactsError(f"line {lineno}: section marker has no letter")
    raise FactsError(f"line {lineno}: section marker has no letter")


def _parse_number(text: str, lineno: int, mnemonic: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise FactsError(f"line {lineno}: {mnemonic} value {text!r} is not a number") from exc
    if math.isnan(value) or math.isinf(value):
        raise FactsError(f"line {lineno}: {mnemonic} value {text!r} is not finite")
    return value


def _tokens(line: str, lineno: int) -> list[float]:
    values: list[float] = []
    for token in line.split():
        try:
            value = float(token)
        except ValueError as exc:
            raise FactsError(f"line {lineno}: data token {token!r} is not a number") from exc
        if math.isnan(value) or math.isinf(value):
            raise FactsError(f"line {lineno}: data token {token!r} is not finite")
        values.append(value)
    return values


def _sha256(path: Path) -> tuple[str, bytes]:
    """SHA-256 of the file bytes, read in chunks. The file is not written."""
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        while True:
            block = handle.read(_HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
    return digest.hexdigest(), b"".join(chunks)


def _blank_or_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


class _CurveAccum:
    """Running null count and non-null span for one curve. No stored samples."""

    def __init__(self) -> None:
        self.null_count = 0
        self.first_non_null_depth: float | None = None
        self.last_non_null_depth: float | None = None
        self.min_non_null: float | None = None
        self.max_non_null: float | None = None

    def observe(self, value: float, depth: float, null: float) -> None:
        if is_declared_null(value, null):
            self.null_count += 1
            return
        if self.first_non_null_depth is None:
            self.first_non_null_depth = depth
        self.last_non_null_depth = depth
        if self.min_non_null is None or value < self.min_non_null:
            self.min_non_null = value
        if self.max_non_null is None or value > self.max_non_null:
            self.max_non_null = value


def _consume_steps(
    data_lines: list[tuple[int, str]],
    n_curves: int,
    wrapped: bool,
    null: float,
) -> tuple[int, list[_CurveAccum]]:
    """Count depth steps and update per-curve facts.

    WRAP NO: each data line is one depth step and must hold exactly one value
    per curve. A short line is not joined to the next line.

    WRAP YES: one depth step may occupy several lines. Values are taken in
    order until each step has one value per curve. Leftover values are refused.
    """
    accums = [_CurveAccum() for _ in range(n_curves)]
    step_count = 0

    def take(step: list[float]) -> None:
        nonlocal step_count
        # Curve 0 is the LAS index channel. Its value is the depth for this step.
        depth = step[0]
        for index, value in enumerate(step):
            accums[index].observe(value, depth, null)
        step_count += 1

    if not wrapped:
        for lineno, line in data_lines:
            if _blank_or_comment(line):
                continue
            values = _tokens(line, lineno)
            if len(values) != n_curves:
                raise FactsError(
                    f"line {lineno}: WRAP NO line has {len(values)} values, expected {n_curves}"
                )
            take(values)
        return step_count, accums

    pending: list[float] = []
    for lineno, line in data_lines:
        if _blank_or_comment(line):
            continue
        pending.extend(_tokens(line, lineno))
        while len(pending) >= n_curves:
            take(pending[:n_curves])
            del pending[:n_curves]
    if pending:
        raise FactsError(
            f"{len(pending)} data values left over after {step_count} depth steps "
            f"(each step needs {n_curves})"
        )
    return step_count, accums


def extract_facts(path: Path) -> dict:
    """Record raw-file facts for one LAS file. D27.

    The file is opened for reading only. Text is UTF-8. A leading byte-order
    mark is accepted as part of UTF-8; any other encoding fails the read.
    Depth for a non-null sample is the value of curve index 0 on that step,
    which is the LAS index channel.
    """
    sha256, raw = _sha256(path)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise FactsError(f"{path.name}: file is not UTF-8") from exc

    section: str | None = None
    seen: set[str] = set()
    vers_text: str | None = None
    wrap_text: str | None = None
    well: dict[str, tuple[str, str, float]] = {}
    curves: list[dict] = []
    data_lines: list[tuple[int, str]] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        letter = _section_letter(line, lineno)
        if letter is not None:
            if letter in seen:
                raise FactsError(f"line {lineno}: section ~{letter} appears twice")
            seen.add(letter)
            section = letter
            continue
        if section is None:
            if _blank_or_comment(line):
                continue
            raise FactsError(f"line {lineno}: text before the first section")
        if section == "A":
            data_lines.append((lineno, line))
            continue
        if _blank_or_comment(line):
            continue
        if section not in {"V", "W", "C"}:
            continue
        mnemonic, unit, value_text, description = parse_header_line(line, lineno)
        if section == "V" and mnemonic == "VERS":
            if vers_text is not None:
                raise FactsError(f"line {lineno}: VERS appears twice")
            vers_text = value_text
        elif section == "V" and mnemonic == "WRAP":
            if wrap_text is not None:
                raise FactsError(f"line {lineno}: WRAP appears twice")
            wrap_text = value_text
        elif section == "W" and mnemonic in _REQUIRED_WELL:
            if mnemonic in well:
                raise FactsError(f"line {lineno}: {mnemonic} appears twice")
            well[mnemonic] = (unit, value_text, _parse_number(value_text, lineno, mnemonic))
        elif section == "C":
            curves.append(
                {
                    "index": len(curves),
                    "mnemonic": mnemonic,
                    "unit": unit,
                    "description": description,
                    "line": line,
                }
            )

    if vers_text is None:
        raise FactsError(f"{path.name}: VERS is not in the version section")
    if wrap_text is None:
        raise FactsError(f"{path.name}: WRAP is not in the version section")
    wrap_flag = wrap_text.strip().upper()
    if wrap_flag not in {"YES", "NO"}:
        raise FactsError(f"{path.name}: WRAP value {wrap_text!r} is not YES or NO")
    for mnemonic in _REQUIRED_WELL:
        if mnemonic not in well:
            raise FactsError(f"{path.name}: {mnemonic} is not in the well section")
    if not curves:
        raise FactsError(f"{path.name}: the curve section is empty")
    if "A" not in seen:
        raise FactsError(f"{path.name}: no data section")

    _null_unit, null_text, null_number = well["NULL"]
    step_count, accums = _consume_steps(data_lines, len(curves), wrap_flag == "YES", null_number)

    curve_facts: list[dict] = []
    for curve, accum in zip(curves, accums, strict=True):
        curve_facts.append(
            {
                **curve,
                "null_count": accum.null_count,
                "first_non_null_depth": accum.first_non_null_depth,
                "last_non_null_depth": accum.last_non_null_depth,
                "min_non_null": accum.min_non_null,
                "max_non_null": accum.max_non_null,
            }
        )

    def _depth_header(mnemonic: str) -> dict:
        unit, text, number = well[mnemonic]
        return {"text": text, "unit": unit, "number": number}

    return {
        "fixture": path.name,
        "sha256": sha256,
        "vers": {"text": vers_text},
        "wrap": {"text": wrap_text, "wrapped": wrap_flag == "YES"},
        "null": {"text": null_text, "number": null_number},
        "strt": _depth_header("STRT"),
        "stop": _depth_header("STOP"),
        "step": _depth_header("STEP"),
        "null_comparison": {
            "rel_tol": NULL_REL_TOL,
            "abs_tol": NULL_ABS_TOL,
            "rule": "math.isclose(value, null, rel_tol=1e-6, abs_tol=0.0)",
        },
        "depth_channel": {
            "curve_index": 0,
            "meaning": "LAS index channel, the first curve in the curve section",
        },
        "depth_step_count": step_count,
        "curves": curve_facts,
    }


def canonical_json(facts: dict) -> str:
    """Pretty JSON, UTF-8, with a trailing newline. Key order is the dict order."""
    return json.dumps(facts, indent=2, allow_nan=False) + "\n"


def facts_path_for(fixture: Path, directory: Path) -> Path:
    """``<stem>.facts.json`` beside the other golden files."""
    return directory / f"{fixture.stem}.facts.json"


def iter_fixtures(fixtures_dir: Path) -> list[Path]:
    """Every ``*.las`` fixture except KEOUGH, sorted by file name."""
    return sorted(path for path in fixtures_dir.glob("*.las") if not is_keough(path))


def prefix_counts(facts: dict) -> list[tuple[str, int]]:
    """Reading aid: mnemonic text through the first underscore, or no underscore.

    This is not stored in the facts file. It is only for the human check.
    """
    counts: Counter[str] = Counter()
    for curve in facts["curves"]:
        mnemonic = curve["mnemonic"]
        mark = mnemonic.find("_")
        bucket = mnemonic[: mark + 1] if mark >= 0 else "(no underscore)"
        counts[bucket] += 1
    return counts.most_common()


def write_fixture_facts(fixture: Path, directory: Path) -> Path:
    """Extract one fixture and write its facts file. Returns the JSON path."""
    destination = facts_path_for(fixture, directory)
    destination.write_text(canonical_json(extract_facts(fixture)), encoding="utf-8", newline="\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    """Write facts for every non-KEOUGH fixture and print headline numbers."""
    parser = argparse.ArgumentParser(description="Write raw LAS facts for the fixtures.")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args(argv)
    fixtures = iter_fixtures(args.fixtures)
    if not fixtures:
        raise FactsError(f"no LAS fixtures in {args.fixtures}")
    args.out.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        destination = write_fixture_facts(fixture, args.out)
        facts = json.loads(destination.read_text(encoding="utf-8"))
        lfp = sum(1 for curve in facts["curves"] if curve["mnemonic"].startswith("LFP_"))
        prefixes = prefix_counts(facts)
        top = prefixes[0]
        print(f"file: {facts['fixture']}")
        print(f"  curves: {len(facts['curves'])}")
        print(f"  depth_steps: {facts['depth_step_count']}")
        print(f"  NULL: {facts['null']['text']} ({facts['null']['number']})")
        print(
            f"  STRT: {facts['strt']['text']} {facts['strt']['unit']}"
            f"  STOP: {facts['stop']['text']} {facts['stop']['unit']}"
            f"  STEP: {facts['step']['text']} {facts['step']['unit']}"
        )
        print(f"  WRAP: {facts['wrap']['text']}")
        print(f"  most_common_prefix: {top[0]!r} count {top[1]}")
        print(f"  prefix_counts: {prefixes}")
        print(f"  mnemonics_starting_LFP_: {lfp}")
        print(f"  wrote: {destination.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
