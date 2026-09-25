"""Import lists for the architecture guards. Tests only."""

from __future__ import annotations

import ast
from pathlib import Path


def imported_names(root: Path) -> list[tuple[Path, int, str]]:
    """Every imported module name under ``root``, as (file, line, dotted name)."""

    found: list[tuple[Path, int, str]] = []
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        found.append((path, node.lineno, node.module))
                    else:
                        found.append((path, node.lineno, f"{node.module}.{alias.name}"))
    return found
