# SPDX-License-Identifier: BSD-3-Clause
"""`_KNOWN_SSPS` must describe the catalog it claims to index.

The dict is hand-maintained, and it had drifted in exactly one direction: 17
hosted grids were absent and no entry was phantom. That asymmetry names the
cause — grids were added to the server and only the Chabrier ones were
transcribed — and it says which failure to expect next time.

The two directions fail differently, so they are separate tests:

* **Hosted but unregistered** is a *discoverability* failure. The grid is still
  downloadable by filename (``_resolve_ssp_filename`` passes a bare ``.h5``
  through), but ``download_ssp("fsps_mist_miles_kroupa")`` raises KeyError for a
  file sitting right there, and ``list_known_ssps()`` under-reports the catalog.

* **Registered but not hosted** is worse: the menu advertises a download that
  404s. Nothing was in this state, and the test exists to keep it that way.

Both require the network, so both skip when the catalog is unreachable — a
guard that turns an offline CI runner red teaches people to delete it. Skipping
is honest here because the invariant is about the *live* catalog: there is
nothing local to check it against.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request

import pytest

from tengri._data_setup import _KNOWN_SSPS, SSP_BASE_URL

pytestmark = pytest.mark.contract

_TIMEOUT_S = 30


@pytest.fixture(scope="module")
def hosted_filenames() -> set[str]:
    """Every ``.h5`` the catalog index lists, or skip if it is unreachable."""
    try:
        with urllib.request.urlopen(SSP_BASE_URL, timeout=_TIMEOUT_S) as fh:
            html = fh.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"SSP catalog unreachable ({type(exc).__name__}): {exc}")

    names = set(re.findall(r'href="([^"]+\.h5)"', html))
    if not names:
        pytest.skip(
            "catalog index returned no .h5 links — the page layout changed, "
            "so this test cannot tell drift from a parsing failure"
        )
    return names


def test_no_registered_ssp_is_missing_from_the_catalog(hosted_filenames):
    """An entry with no file behind it advertises a download that 404s."""
    phantom = sorted(set(_KNOWN_SSPS.values()) - hosted_filenames)
    assert not phantom, (
        "_KNOWN_SSPS names grids the catalog does not serve, so "
        "download_ssp() on them fails with an HTTP error rather than a clear "
        "message:\n  " + "\n  ".join(phantom)
    )


def test_no_hosted_ssp_is_missing_from_the_registry(hosted_filenames):
    """A hosted grid absent from the dict is unreachable by its short name.

    The filename spelling still works, so this is discoverability rather than
    breakage — but ``list_known_ssps()`` is how a user finds out what exists,
    and a menu that under-reports is the reason nobody noticed for months.
    """
    unregistered = sorted(hosted_filenames - set(_KNOWN_SSPS.values()))
    assert not unregistered, (
        "the catalog serves grids _KNOWN_SSPS does not list, so "
        "list_known_ssps() under-reports and the short-name form raises "
        "KeyError for them:\n  "
        + "\n  ".join(unregistered)
        + "\n\nAdd them to _KNOWN_SSPS in src/tengri/_data_setup.py."
    )


def test_every_registry_key_matches_its_filename():
    """The key is the filename minus ``.h5`` -- no exceptions, so no lookup table.

    Checked offline and separately from the catalog tests: this is an internal
    consistency property, and a mismatch here would make the two network tests
    above compare the wrong things.
    """
    mismatched = {k: v for k, v in _KNOWN_SSPS.items() if v != f"{k}.h5"}
    assert not mismatched, f"keys that are not their filename minus '.h5': {mismatched}"
