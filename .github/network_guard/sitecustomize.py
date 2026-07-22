"""Fail closed if a deterministic G0 subprocess attempts outbound networking."""

import socket
from typing import NoReturn


def _blocked(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("outbound network is disabled for deterministic G0")


socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.socket.sendto = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked
socket.gethostbyname_ex = _blocked
socket.gethostbyaddr = _blocked
