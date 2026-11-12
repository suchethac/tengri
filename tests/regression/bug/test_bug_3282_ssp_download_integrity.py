# SPDX-License-Identifier: BSD-3-Clause
"""SSP and template downloads must match a pinned length and sha256 (#3282).

Before the fix, a cache hit in :func:`tengri.download_ssp` counted as valid
whenever the file existed with size > 0, the downloader never compared the
bytes received with ``Content-Length``, and no digest was recorded anywhere. A
truncated or substituted grid was therefore trusted forever.

These tests serve every HTTP request from an in-memory fake, so no network is
touched. The pin table is replaced per test; the real table is covered by
``test_pin_table_covers_every_known_ssp``.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request

import pytest

import tengri._data_setup as ds
from tengri.config.exceptions import TengriIOError

pytestmark = pytest.mark.regression_bug

NAME = "fsps_prsc_miles_chabrier"
FILE = "fsps_prsc_miles_chabrier.h5"
GOOD = b"tengri-ssp-body-" * 64
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()


class _Response:
    """Minimal stand-in for the object ``urllib.request.urlopen`` returns."""

    def __init__(self, body: bytes, content_length: int | None):
        self._body = io.BytesIO(body)
        self.status = 200
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


@pytest.fixture
def served(monkeypatch):
    """Serve ``state['body']`` for GET and an empty body for HEAD."""
    state = {"body": GOOD, "content_length": len(GOOD), "requests": 0}

    def fake_urlopen(req, timeout=None):
        state["requests"] += 1
        method = req.get_method() if hasattr(req, "get_method") else "GET"
        body = b"" if method == "HEAD" else state["body"]
        return _Response(body, state["content_length"])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return state


@pytest.fixture
def pinned(monkeypatch):
    """Pin ``FILE`` to the length and digest of ``GOOD``."""
    monkeypatch.setattr(ds, "_DATA_PINS", {FILE: (len(GOOD), GOOD_SHA)}, raising=False)


def _leftovers(directory) -> list:
    return sorted(p.name for p in directory.iterdir())


def test_truncated_download_raises_and_leaves_no_file(tmp_path, served, pinned):
    """Content-Length says 1000 bytes, only 400 arrive: refuse, delete the partial."""
    served["body"] = GOOD[:400]
    served["content_length"] = len(GOOD)

    with pytest.raises(TengriIOError, match="truncated"):
        ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert _leftovers(tmp_path) == []


def test_short_body_without_content_length_fails_pinned_length(tmp_path, served, pinned):
    """No Content-Length header: the pinned byte length still catches a short file."""
    served["body"] = GOOD[:400]
    served["content_length"] = None

    with pytest.raises(TengriIOError, match="bytes"):
        ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert _leftovers(tmp_path) == []


def test_wrong_digest_raises_and_leaves_no_file(tmp_path, served, monkeypatch):
    """Right length, wrong bytes: the sha256 pin refuses the download."""
    monkeypatch.setattr(ds, "_DATA_PINS", {FILE: (len(GOOD), "0" * 64)}, raising=False)

    with pytest.raises(TengriIOError, match="sha256"):
        ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert _leftovers(tmp_path) == []


def test_cached_file_with_wrong_size_is_not_valid(tmp_path, served, pinned):
    """A 10-byte cached copy is not the pinned grid: it is replaced, not returned."""
    (tmp_path / FILE).write_bytes(GOOD[:10])

    path = ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert served["requests"] > 0, "a mis-sized cache hit must trigger a fetch"
    assert path.read_bytes() == GOOD


def test_cached_file_with_wrong_digest_is_not_valid(tmp_path, served, pinned):
    """Right size, wrong bytes: the cache hit is checked against the digest too."""
    (tmp_path / FILE).write_bytes(b"X" * len(GOOD))

    path = ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert served["requests"] > 0
    assert path.read_bytes() == GOOD


def test_good_download_passes_and_good_cache_is_reused(tmp_path, served, pinned):
    """A download matching its pin is accepted; the next call reuses it without a fetch."""
    path = ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert path == tmp_path / FILE
    assert path.read_bytes() == GOOD
    assert _leftovers(tmp_path) == [FILE]

    fetches = served["requests"]
    again = ds.download_ssp(NAME, dest=tmp_path, progress=False)

    assert again == path
    assert served["requests"] == fetches, "a verified cache hit must not refetch"


def test_unpinned_entry_warns_loudly(tmp_path, served, monkeypatch):
    """An entry with no digest is still downloaded, but the gap is announced."""
    monkeypatch.setattr(ds, "_DATA_PINS", {FILE: (len(GOOD), None)}, raising=False)

    with pytest.warns(UserWarning, match="sha256"):
        ds.download_ssp(NAME, dest=tmp_path, progress=False)


def test_pin_table_covers_every_known_ssp():
    """Every catalog filename has a pin, and every digest is None or 64 hex chars."""
    pins = ds._DATA_PINS
    missing = sorted(ds.KNOWN_SSP_FILENAMES - set(pins))
    assert missing == []
    for filename, (length, digest) in pins.items():
        assert length > 0, filename
        assert digest is None or len(digest) == 64, filename
