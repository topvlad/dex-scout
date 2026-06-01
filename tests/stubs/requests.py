"""Minimal local requests fallback for test environments without requests installed."""

from __future__ import annotations

from typing import Any


class RequestException(Exception):
    pass


class Timeout(RequestException, TimeoutError):
    pass


class _Exceptions:
    RequestException = RequestException
    Timeout = Timeout


exceptions = _Exceptions()


class Response:
    def __init__(self, json_data: Any | None = None, status_code: int = 200, text: str = "") -> None:
        self._json_data = {} if json_data is None else json_data
        self.status_code = status_code
        self.text = text
        self.content = text.encode()
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RequestException(f"HTTP {self.status_code}")


def request(*args: Any, **kwargs: Any) -> Response:
    return Response()


def get(*args: Any, **kwargs: Any) -> Response:
    return request(*args, **kwargs)


def post(*args: Any, **kwargs: Any) -> Response:
    return request(*args, **kwargs)


def put(*args: Any, **kwargs: Any) -> Response:
    return request(*args, **kwargs)


def delete(*args: Any, **kwargs: Any) -> Response:
    return request(*args, **kwargs)


class Session:
    def request(self, *args: Any, **kwargs: Any) -> Response:
        return request(*args, **kwargs)
    def get(self, *args: Any, **kwargs: Any) -> Response:
        return request(*args, **kwargs)
    def post(self, *args: Any, **kwargs: Any) -> Response:
        return request(*args, **kwargs)
