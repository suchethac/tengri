# SPDX-License-Identifier: BSD-3-Clause
"""``Photometry([...names...])`` must name the remedy, not leak an AttributeError (#1735).

``Photometry`` is public, its first positional parameter is ``filters``, and
``tengri.list_filters()`` hands out exactly the strings a new user then passes
to it. So ``Photometry(["sdss_g", ...])`` is the obvious first guess.

Before this fix the strings were accepted, stored, and only failed partway
through ``__post_init__`` with::

    AttributeError: 'str' object has no attribute 'name'

which names neither filters, nor ``from_names``, nor what was expected. The
failure was not at the boundary, so the traceback pointed at a line the caller
did not write.

The bar is set elsewhere in the library: ``load_ssp`` on a missing grid lists
the search paths and names ``download_ssp`` and ``$TENGRI_DATA_DIR``. This
module pins that a caller who makes the natural mistake is told the next step.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.config.exceptions import ConfigError, TengriError

pytestmark = pytest.mark.regression_bug

_NAMES = ["sdss_g", "sdss_r", "sdss_i"]


def test_a_list_of_names_is_refused_at_the_boundary():
    """The old failure was an AttributeError from deep inside ``__post_init__``."""
    with pytest.raises(ConfigError) as excinfo:
        tengri.Photometry(_NAMES)

    message = str(excinfo.value)
    assert "from_names" in message, f"error does not name the remedy: {message}"
    assert "list_filters" in message, f"error does not say where names come from: {message}"
    assert "FilterCurve" in message, f"error does not say what was expected: {message}"


def test_the_message_hands_back_a_call_the_user_can_paste():
    """Naming the remedy generically is weaker than naming it with their own bands."""
    with pytest.raises(ConfigError) as excinfo:
        tengri.Photometry(_NAMES)

    message = str(excinfo.value)
    for name in _NAMES:
        assert repr(name) in message, f"{name!r} missing from: {message}"


def test_it_is_a_tengri_error_not_an_attributeerror():
    """Catchable as ``TengriError``; an AttributeError leak is not a contract."""
    with pytest.raises(TengriError):
        tengri.Photometry(_NAMES)

    with pytest.raises(ConfigError):
        tengri.Photometry(_NAMES)


def test_a_bare_string_is_refused_too():
    """``Photometry("sdss_g")`` — a str is a sequence, so it passes the length check.

    Without its own branch this reported one failure per *letter*, which is a
    worse message than the one the fix replaces.
    """
    with pytest.raises(ConfigError, match="bare string"):
        tengri.Photometry("sdss_g")


def test_a_wrong_type_is_named_with_its_index():
    """Anything not a FilterCurve is refused, saying which element and what type."""
    with pytest.raises(ConfigError) as excinfo:
        tengri.Photometry([1, 2, 3])

    message = str(excinfo.value)
    assert "int" in message
    assert "index 0" in message


def test_an_empty_list_keeps_its_own_message():
    """The pre-existing empty check must not be shadowed by the new one."""
    with pytest.raises(ValueError, match="at least one filter"):
        tengri.Photometry([])


# ── the working path must be untouched ────────────────────────────


def test_from_names_still_builds():
    """The guard must reject only what was already broken."""
    phot = tengri.Photometry.from_names(_NAMES)
    assert phot.names == tuple(_NAMES)
    assert phot.n_filters == len(_NAMES)


def test_filter_curves_are_still_accepted_positionally():
    """Passing real FilterCurve objects — the documented use — keeps working."""
    built = tengri.Photometry.from_names(_NAMES)
    direct = tengri.Photometry(built.filters)
    assert direct.names == tuple(_NAMES)
