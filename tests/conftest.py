"""Block network sockets for every test under tests/. D14.

No extra dependency. ``socket.socket`` and ``socket.create_connection`` raise
for the duration of each test.
"""

import socket

import pytest


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Network access is not permitted (D14).")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
