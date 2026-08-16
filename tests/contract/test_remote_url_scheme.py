# SPDX-License-Identifier: BSD-3-Clause
"""``require_remote_url`` admits only ordinary remote URLs.

``urllib.request.urlopen`` accepts ``file://``, and on most builds ``ftp://``
and other handlers too, so a download helper that forwards an unchecked string
doubles as a local-file reader: a catalog entry naming
``file:///etc/passwd`` would be opened and written to the destination path as
though it had been fetched over the network.

Every URL tengri fetches is an https one from the public data mirror, so the
restriction costs nothing. These tests pin that it stays in place — the guard
is one line, and one line is easy to delete while "simplifying".
"""

import pytest

pytestmark = pytest.mark.contract

from tengri._data_setup import require_remote_url

#: Schemes urlopen would happily handle and that must be refused.
REFUSED = [
    "file:///etc/passwd",
    "file://localhost/etc/passwd",
    "ftp://example.invalid/grid.h5",
    "data:text/plain;base64,aGk=",
    "/etc/passwd",  # bare path: parses to an empty scheme
    "",
]

ACCEPTED = [
    "https://halos.as.arizona.edu/suchethacooray/ssp-spectra/",
    "http://example.invalid/grid.h5",
    "HTTPS://EXAMPLE.INVALID/Grid.h5",  # scheme comparison is case-insensitive
]


@pytest.mark.parametrize("url", REFUSED)
def test_non_http_schemes_are_refused(url):
    with pytest.raises(ValueError, match="only http and https"):
        require_remote_url(url)


@pytest.mark.parametrize("url", ACCEPTED)
def test_http_urls_pass_through_unchanged(url):
    assert require_remote_url(url) == url


def test_the_error_names_the_offending_url():
    """The message has to identify which URL was refused.

    A download helper may be called in a loop over a catalog; "refused a
    URL" without saying which one leaves the caller to guess.
    """
    with pytest.raises(ValueError, match=r"grid\.h5"):
        require_remote_url("ftp://example.invalid/grid.h5")
