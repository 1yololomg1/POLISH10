"""Network imports are absent, and sockets raise at runtime. D14."""

import socket
from pathlib import Path

import pytest

from tests.arch.import_walk import imported_names

_REPO = Path(__file__).resolve().parents[2]
_ROOTS = ("polish", "polish_ui")

# Qualified names, matched as the whole import or as a prefix.
_BANNED_PREFIXES = (
    "socket",
    "requests",
    "urllib.request",
    "http.client",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "webbrowser",
    "PySide6.QtNetwork",
)


def _banned(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _BANNED_PREFIXES)


def test_polish_sources_do_not_import_network_modules():
    offenders: list[str] = []
    for folder in _ROOTS:
        root = _REPO / folder
        for path, line, name in imported_names(root):
            if _banned(name):
                offenders.append(f"{path.relative_to(_REPO)}:{line} imports {name}")
    assert offenders == []


def test_runtime_socket_constructors_raise():
    with pytest.raises(RuntimeError, match="D14"):
        socket.socket()
    with pytest.raises(RuntimeError, match="D14"):
        socket.create_connection(("example.invalid", 80))
