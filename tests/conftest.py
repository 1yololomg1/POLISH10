"""Block network sockets for every test under tests/. D14.

No extra dependency. ``socket.socket`` and ``socket.create_connection`` raise
for the duration of each test.
"""

import os
import socket

import pytest

# tests/ui runs Qt without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import order matters on Windows. With scikit-learn 1.5.0 and PySide6 6.11.2,
# importing sklearn (the legacy tests do) and then PySide6.QtWidgets ends the
# interpreter with an access violation; Qt first, then sklearn, works
# (observed 12 Nov 2026). conftest is imported before any test module, so Qt
# is loaded here when it is installed. polish/ imports neither package.
try:
    import PySide6.QtWidgets  # noqa: F401
except ImportError:
    pass


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Network access is not permitted (D14).")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
