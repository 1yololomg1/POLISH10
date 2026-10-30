"""Governance checks. Each one reads the files at test time. D97.

Checks
------
a. State enum names and glossary tokens.
b. Block-type action matrix against permitted_actions.
c. D, F, UXC and C citations in polish/.
d. Register rows D1 through the last row, plus struck D20a.
e. The Count section against those rows.
f. The current D1–Dn range outside fenced prompt text.
g. The governing-document list in the rules file and the README.
h. Public docstrings in polish/.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from polish.model.states import BlockType, permitted_actions

_REPO = Path(__file__).resolve().parents[2]
_STATUS_WORDS = frozenset({"REVISE", "NEW", "RESOLVED", "OPEN", "PARTIAL", "STRUCT", "UNVER", "CONFLICT"})
_ROW_CITE = re.compile(r"\b(?:D\d+[a-z]?|C\d+|UXC\d+|F\d+)\b")
_GLOSSARY_TOKEN = re.compile(r"[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+")
_D_RANGE = re.compile(r"D1[\u2013\u2014-]D(\d+)")
_CITATION = re.compile(r"\b(D\d+[a-z]?|UXC\d+|F\d+|C\d+)\b")
_NO_ACTIONS = frozenset({"view only", "fix upstream only"})
_GOVERNING = (
    "POLISH_pipeline_contracts.md",
    "POLISH_decision_register.md",
    "POLISH_design_brief.md",
    "POLISH_UIUX_cohesion_supplement.md",
    "POLISH_build_plan.md",
    "state_glossary.md",
)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _fail(path: Path, item: str, detail: str) -> None:
    raise AssertionError(f"{path.as_posix()}: {item}: {detail}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_cells(line: str) -> list[str]:
    """Split a markdown row on unescaped pipes. D89's decision cell contains one."""

    parts: list[str] = []
    buf: list[str] = []
    index = 0
    while index < len(line):
        if line[index] == "\\" and index + 1 < len(line) and line[index + 1] == "|":
            buf.append("|")
            index += 2
            continue
        if line[index] == "|":
            parts.append("".join(buf).strip())
            buf = []
            index += 1
            continue
        buf.append(line[index])
        index += 1
    if buf:
        parts.append("".join(buf).strip())
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _first_word(cell: str) -> str:
    cleaned = cell.replace("*", "").replace("_", "")
    cleaned = cleaned.replace("\u2014", " ").replace("\u2013", " ")
    words = cleaned.split()
    return words[0] if words else ""


def _status_key(cell: str) -> str:
    word = _first_word(cell)
    if word in _STATUS_WORDS:
        return word
    return "not flagged"


def _is_enum_base(base: ast.expr) -> bool:
    if isinstance(base, ast.Name):
        return base.id == "Enum"
    return isinstance(base, ast.Attribute) and base.attr == "Enum"


def _enum_members(states_path: Path) -> tuple[set[str], set[str]]:
    """Member names and string values of every Enum class in states.py."""

    try:
        tree = ast.parse(_read(states_path), filename=str(states_path))
    except (OSError, SyntaxError) as exc:
        _fail(states_path, "states.py", f"could not be read ({exc})")
    names: set[str] = set()
    values: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not any(_is_enum_base(base) for base in node.bases):
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name) or target.id.startswith("_"):
                continue
            names.add(target.id)
            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                values.add(item.value.value)
    if not names:
        _fail(states_path, "Enum", "no enum members were found")
    return names, values


def _decision_rows(register_text: str, register_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in register_text.splitlines():
        if not line.startswith("| D"):
            continue
        cells = _split_cells(line)
        if not cells or re.fullmatch(r"D\d+[a-z]?", cells[0]) is None:
            continue
        if len(cells) != 4:
            _fail(register_path, cells[0], f"decision row has {len(cells)} cells")
        rows.append(cells)
    if not rows:
        _fail(register_path, "decision row", "none were found")
    return rows


def _section(text: str, heading: str) -> str:
    """Text from ``heading`` up to the next heading of the same or higher level."""

    if text.startswith(heading):
        start = 0
    else:
        found = text.find("\n" + heading)
        if found < 0:
            return ""
        start = found + 1
    level = len(heading) - len(heading.lstrip("#"))
    rest = text[start + len(heading) :]
    next_heading = re.search(rf"\n#{{1,{level}}} ", rest)
    if next_heading is None:
        return text[start:]
    return text[start : start + len(heading) + next_heading.start()]


def _outside_prompt_text(text: str) -> list[tuple[int, str]]:
    """Lines that are not inside a fenced prompt (a ```text or ````text block)."""

    outside: list[tuple[int, str]] = []
    in_prompt = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not in_prompt and (stripped.startswith("````text") or stripped.startswith("```text")):
            in_prompt = True
            continue
        if in_prompt and stripped in {"````", "```"}:
            in_prompt = False
            continue
        if not in_prompt:
            outside.append((number, line))
    return outside


def _d_ranges(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for match in _D_RANGE.finditer(text):
        found.append((match.group(0), int(match.group(1))))
    return found


def _cell_count(cell: str) -> int | None:
    bold = re.search(r"\*\*(\d+)\*\*", cell)
    if bold is not None:
        return int(bold.group(1))
    stripped = cell.strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return None


def check_glossary_names(root: Path) -> None:
    """Enum member names appear in the glossary, and glossary tokens are those names or values."""

    glossary_path = root / "docs" / "governing" / "state_glossary.md"
    states_path = root / "polish" / "model" / "states.py"
    glossary = _read(glossary_path)
    names, values = _enum_members(states_path)
    allowed = names | values
    for name in sorted(names):
        if re.search(rf"\b{re.escape(name)}\b", glossary) is None:
            _fail(glossary_path, name, "enum member is not in the glossary")
    for match in _GLOSSARY_TOKEN.finditer(glossary):
        token = match.group(0)
        if token not in allowed:
            _fail(glossary_path, token, "token is not a state enum member or value")


def check_action_matrix(root: Path) -> None:
    """The glossary action table equals permitted_actions for every block type."""

    glossary_path = root / "docs" / "governing" / "state_glossary.md"
    section = _section(_read(glossary_path), "## Block type, actions, disposition, export")
    if section == "":
        _fail(glossary_path, "Block type, actions, disposition, export", "section is missing")
    seen: dict[str, set[str]] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = _split_cells(line)
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        block_match = re.search(r"`([^`]+)`", cells[0])
        if block_match is None:
            continue
        block_name = block_match.group(1)
        action_cell = cells[1].strip()
        if action_cell.lower() in _NO_ACTIONS:
            actions: set[str] = set()
        else:
            actions = set(re.findall(r"`([A-Z][A-Z0-9_]*)`", cells[1]))
        seen[block_name] = actions
    for block_type in BlockType:
        if block_type.name not in seen:
            _fail(glossary_path, block_type.name, "block type has no action-matrix row")
        permitted = {action.name for action in permitted_actions(block_type)}
        if seen[block_type.name] != permitted:
            got = ", ".join(sorted(seen[block_type.name])) or "(none)"
            expected = ", ".join(sorted(permitted)) or "(none)"
            _fail(glossary_path, block_type.name, f"actions {got} are not the permitted actions {expected}")
    for block_name in sorted(seen):
        if block_name not in BlockType.__members__:
            _fail(glossary_path, block_name, "action-matrix row is not a BlockType")


def check_cited_rows(root: Path) -> None:
    """Every cited D row is unstruck, and F, UXC and C citations stay in range."""

    register_path = root / "docs" / "governing" / "POLISH_decision_register.md"
    rows = _decision_rows(_read(register_path), register_path)
    unstruck = {cells[0] for cells in rows if cells[0] != "D20a"}
    polish = root / "polish"
    paths = sorted(polish.rglob("*.py")) + sorted(polish.rglob("*.json"))
    if not paths:
        _fail(polish, "polish", "no Python or JSON files were found")
    for path in paths:
        text = _read(path)
        for match in _CITATION.finditer(text):
            token = match.group(1)
            if token.startswith("D"):
                if token == "D20a" or token not in unstruck:
                    reason = "cited decision is struck" if token == "D20a" else "cited decision is not an unstruck register row"
                    _fail(path, token, reason)
            elif token.startswith("UXC"):
                number = int(token[3:])
                if number < 1 or number > 10:
                    _fail(path, token, "UXC citation is outside UXC1-UXC10")
            elif token.startswith("F"):
                number = int(token[1:])
                if number < 1 or number > 19:
                    _fail(path, token, "F citation is outside F1-F19")
            else:
                number = int(token[1:])
                if number < 1 or number > 7:
                    _fail(path, token, "C citation is outside C1-C7")


def check_register_continuity(root: Path) -> None:
    """D1 through the last row each appear once. D20a is the struck row, once."""

    register_path = root / "docs" / "governing" / "POLISH_decision_register.md"
    rows = _decision_rows(_read(register_path), register_path)
    counts: dict[str, int] = {}
    for cells in rows:
        counts[cells[0]] = counts.get(cells[0], 0) + 1
    numbers = [int(re.fullmatch(r"D(\d+)[a-z]?", row_id).group(1)) for row_id in counts]
    last = max(numbers)
    expected = {f"D{number}" for number in range(1, last + 1)}
    expected.add("D20a")
    for row_id in sorted(expected):
        found = counts.get(row_id, 0)
        if found != 1:
            _fail(register_path, row_id, f"row appears {found} times")
    for row_id in sorted(set(counts) - expected):
        _fail(register_path, row_id, "row is outside D1 through the last row, and is not struck D20a")
    struck = next(cells for cells in rows if cells[0] == "D20a")
    if _first_word(struck[2]) != "STRUCK":
        _fail(register_path, "D20a", "struck row source is not STRUCK")


def check_count_section(root: Path) -> None:
    """The Count tables equal the kept rows. D20a is excluded."""

    register_path = root / "docs" / "governing" / "POLISH_decision_register.md"
    text = _read(register_path)
    rows = _decision_rows(text, register_path)
    kept = [cells for cells in rows if cells[0] != "D20a"]
    sources: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for cells in kept:
        source = _first_word(cells[2])
        status = _status_key(cells[3])
        sources[source] = sources.get(source, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
    count_text = _section(text, "## Count")
    if count_text == "":
        _fail(register_path, "Count", "section is missing")
    total_row = _labelled_row(count_text, "Total register entries")
    total = _cell_count(total_row[1]) if total_row is not None else None
    if total != len(kept):
        shown = "missing" if total is None else str(total)
        _fail(register_path, "Total register entries", f"count {shown} is not the kept-row total {len(kept)}")
    _compare_bucket(register_path, count_text, "### By source", sources, _first_word)
    _compare_bucket(register_path, count_text, "### By review status", statuses, _status_key)


def _labelled_row(text: str, label: str) -> list[str] | None:
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = _split_cells(line)
        if cells and label in cells[0]:
            return cells
    return None


def _compare_bucket(
    register_path: Path,
    count_text: str,
    heading: str,
    expected: dict[str, int],
    key_of,
) -> None:
    section = _section(count_text, heading)
    if section == "":
        _fail(register_path, heading, "section is missing")
    published: dict[str, int] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = _split_cells(line)
        if len(cells) < 2 or cells[0] in {"Source", "Status"} or set(cells[0]) <= set("-: "):
            continue
        number = _cell_count(cells[1])
        if number is None:
            continue
        published[key_of(cells[0])] = number
    for key, number in sorted(expected.items()):
        if published.get(key) != number:
            _fail(register_path, key, f"{heading} count {published.get(key)!r} is not {number}")
    for key, number in sorted(published.items()):
        if expected.get(key, 0) != number:
            _fail(register_path, key, f"{heading} count {number} is not {expected.get(key, 0)}")


def check_d_range(root: Path) -> None:
    """D1 through the last register row is the only D1–Dn range outside fenced prompts."""

    register_path = root / "docs" / "governing" / "POLISH_decision_register.md"
    rows = _decision_rows(_read(register_path), register_path)
    last = max(int(re.fullmatch(r"D(\d+)[a-z]?", cells[0]).group(1)) for cells in rows)
    title = _read(register_path).splitlines()[0]
    _require_only_current_range(register_path, "title", title, last)
    readme = root / "docs" / "governing" / "README.md"
    _require_only_current_range(readme, "README.md", _read(readme), last)
    rules = root / ".cursor" / "rules" / "polish-rebuild.mdc"
    _require_only_current_range(rules, "polish-rebuild.mdc", _read(rules), last)
    plan = root / "docs" / "governing" / "POLISH_build_plan.md"
    outside = "\n".join(line for _number, line in _outside_prompt_text(_read(plan)))
    _require_only_current_range(plan, "POLISH_build_plan.md", outside, last)


def _require_only_current_range(path: Path, item: str, text: str, last: int) -> None:
    ranges = _d_ranges(text)
    if not ranges:
        _fail(path, item, f"D1-D{last} range is missing")
    for written, number in ranges:
        if number != last:
            _fail(path, written, f"D-range is not D1-D{last}")


def check_governing_documents(root: Path) -> None:
    """The rules file and the README each name every governing document."""

    rules = root / ".cursor" / "rules" / "polish-rebuild.mdc"
    readme = root / "docs" / "governing" / "README.md"
    for path in (rules, readme):
        text = _read(path)
        for name in _GOVERNING:
            if name not in text:
                _fail(path, name, "governing document is not named")


def check_public_docstrings(root: Path) -> None:
    """Public classes and methods have docstrings. Public functions name a governing row."""

    polish = root / "polish"
    paths = sorted(polish.rglob("*.py"))
    if not paths:
        _fail(polish, "polish", "no Python files were found")
    for path in paths:
        tree = ast.parse(_read(path), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                _require_docstring(path, node.name, node)
                if _ROW_CITE.search(ast.get_docstring(node) or "") is None:
                    _fail(path, node.name, "public function docstring does not name a D, C, UXC or F row")
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                _require_docstring(path, node.name, node)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                        _require_docstring(path, f"{node.name}.{item.name}", item)


def _require_docstring(path: Path, item: str, node: ast.AST) -> None:
    if ast.get_docstring(node) is None:
        _fail(path, item, "public symbol has no docstring")


def test_glossary_lists_every_state_member_and_only_those_tokens() -> None:
    """Glossary names match the state enums. D97."""

    check_glossary_names(_REPO)


def test_block_type_actions_match_permitted_actions() -> None:
    """The glossary action matrix matches permitted_actions. D97, D86, UXC2."""

    check_action_matrix(_REPO)


def test_cited_rows_are_register_rows() -> None:
    """Citations in polish/ name unstruck rows inside the allowed ranges. D97."""

    check_cited_rows(_REPO)


def test_register_rows_run_from_d1_without_gaps() -> None:
    """D1 through the last row each appear once, and D20a is struck. D97."""

    check_register_continuity(_REPO)


def test_count_section_matches_register_rows() -> None:
    """The Count section is the parsed rows, with D20a excluded. D97."""

    check_count_section(_REPO)


def test_current_d_range_is_the_only_range_outside_prompts() -> None:
    """Outside fenced prompts, the only D1–Dn range is the current register. D97."""

    check_d_range(_REPO)


def test_governing_documents_are_named() -> None:
    """The rules file and the README name each governing document. D97."""

    check_governing_documents(_REPO)


def test_public_symbols_document_their_rows() -> None:
    """Public polish symbols have docstrings, and functions name a row. D97."""

    check_public_docstrings(_REPO)


def _copy(root: Path, relative: str) -> Path:
    source = _REPO / relative
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def _rewrite(path: Path, text: str) -> None:
    newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", newline).encode("utf-8"))


def test_glossary_check_fails_when_a_member_is_absent(tmp_path: Path) -> None:
    glossary = _copy(tmp_path, "docs/governing/state_glossary.md")
    _copy(tmp_path, "polish/model/states.py")
    _rewrite(glossary, re.sub(r"\bINDEX\b", "DEPTHROLE", _read(glossary)))
    with pytest.raises(AssertionError, match=r"state_glossary\.md: INDEX: enum member is not in the glossary"):
        check_glossary_names(tmp_path)


def test_glossary_check_fails_when_a_token_is_not_an_enum(tmp_path: Path) -> None:
    glossary = _copy(tmp_path, "docs/governing/state_glossary.md")
    _copy(tmp_path, "polish/model/states.py")
    _rewrite(glossary, _read(glossary) + "\nNOT_A_REAL_STATE\n")
    with pytest.raises(AssertionError, match=r"NOT_A_REAL_STATE: token is not a state enum member or value"):
        check_glossary_names(tmp_path)


def test_action_matrix_fails_when_an_action_is_removed(tmp_path: Path) -> None:
    glossary = _copy(tmp_path, "docs/governing/state_glossary.md")
    text = _read(glossary).replace("`ACKNOWLEDGE` · `DEFER` (reason)", "`ACKNOWLEDGE` (reason)", 1)
    _rewrite(glossary, text)
    with pytest.raises(AssertionError, match=r"state_glossary\.md: INFORMATIONAL: actions"):
        check_action_matrix(tmp_path)


def test_citation_check_fails_when_d20a_is_cited(tmp_path: Path) -> None:
    _copy(tmp_path, "docs/governing/POLISH_decision_register.md")
    cited = tmp_path / "polish" / "cited.py"
    cited.parent.mkdir(parents=True)
    cited.write_text("# D20a\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"polish/cited\.py: D20a: cited decision is struck"):
        check_cited_rows(tmp_path)


def test_continuity_check_fails_when_a_row_is_repeated(tmp_path: Path) -> None:
    register = _copy(tmp_path, "docs/governing/POLISH_decision_register.md")
    text = _read(register)
    row = next(line for line in text.splitlines() if line.startswith("| D1 |"))
    _rewrite(register, text.replace(row, row + "\n" + row, 1))
    with pytest.raises(AssertionError, match=r"POLISH_decision_register\.md: D1: row appears 2 times"):
        check_register_continuity(tmp_path)


def test_count_check_fails_when_the_total_is_wrong(tmp_path: Path) -> None:
    register = _copy(tmp_path, "docs/governing/POLISH_decision_register.md")
    _rewrite(register, _read(register).replace("**Total register entries** | **98**", "**Total register entries** | **90**", 1))
    with pytest.raises(AssertionError, match=r"Total register entries: count 90 is not the kept-row total"):
        check_count_section(tmp_path)


def test_d_range_check_fails_when_another_range_remains(tmp_path: Path) -> None:
    _copy(tmp_path, "docs/governing/POLISH_decision_register.md")
    _copy(tmp_path, "docs/governing/POLISH_build_plan.md")
    _copy(tmp_path, ".cursor/rules/polish-rebuild.mdc")
    readme = _copy(tmp_path, "docs/governing/README.md")
    _rewrite(readme, _read(readme).replace("D1\u2013D98", "D1\u2013D90", 1))
    with pytest.raises(AssertionError, match=r"README\.md: D1.D90: D-range is not D1-D98"):
        check_d_range(tmp_path)


def test_document_list_fails_when_the_glossary_is_unnamed(tmp_path: Path) -> None:
    _copy(tmp_path, "docs/governing/README.md")
    rules = _copy(tmp_path, ".cursor/rules/polish-rebuild.mdc")
    _rewrite(rules, _read(rules).replace("state_glossary.md", "state_words.md", 1))
    with pytest.raises(AssertionError, match=r"polish-rebuild\.mdc: state_glossary\.md: governing document is not named"):
        check_governing_documents(tmp_path)


def test_docstring_check_fails_when_a_function_names_no_row(tmp_path: Path) -> None:
    orphan = tmp_path / "polish" / "orphan.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text('def orphan():\n    """No governing row is named here."""\n    return None\n', encoding="utf-8")
    with pytest.raises(AssertionError, match=r"polish/orphan\.py: orphan: public function docstring does not name"):
        check_public_docstrings(tmp_path)
