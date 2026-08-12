# SPDX-License-Identifier: BSD-3-Clause
"""#1735: Photometry(filters=...) rejects non-FilterCurve entries with an
actionable error instead of an inscrutable AttributeError."""

import jax.numpy as jnp
import pytest

from tengri.observation import Photometry
from tengri.observation.photometry import FilterCurve


def _make_curve(name: str) -> FilterCurve:
    wave = jnp.array([3000.0, 4000.0, 5000.0, 6000.0, 7000.0])
    trans = jnp.array([0.0, 0.8, 1.0, 0.8, 0.0])
    return FilterCurve(wave=wave, trans=trans, name=name)


def test_strings_raise_actionable_typeerror():
    """Passing filter names as bare strings must fail loudly with a remedy."""
    with pytest.raises(TypeError) as excinfo:
        Photometry(["sdss_g", "sdss_r"])
    msg = str(excinfo.value)
    assert "FilterCurve" in msg
    assert "str" in msg
    assert "from_names" in msg
    assert "list_filters" in msg


def test_mixed_entries_raise_typeerror():
    """A list mixing curves and strings must also be rejected."""
    with pytest.raises(TypeError) as excinfo:
        Photometry([_make_curve("sdss_g"), "sdss_r"])
    assert "from_names" in str(excinfo.value)


def test_empty_filters_still_raise_valueerror():
    """The pre-existing empty-list guard is unchanged."""
    with pytest.raises(ValueError, match="at least one filter"):
        Photometry([])


def test_filtercurve_objects_construct():
    """Legal input: FilterCurve objects still build with derived fields."""
    phot = Photometry([_make_curve("sdss_g"), _make_curve("sdss_r")])
    assert phot.n_filters == 2
    assert phot.names == ("sdss_g", "sdss_r")
    assert len(phot.filter_waves) == 2
    assert len(phot.filter_trans) == 2


def test_explicit_names_accepted_with_curves():
    """Names may be supplied explicitly alongside FilterCurve objects."""
    phot = Photometry(
        filters=[_make_curve("g"), _make_curve("r")],
        names=("sdss_g", "sdss_r"),
    )
    assert phot.names == ("sdss_g", "sdss_r")


def test_from_names_still_constructs(monkeypatch):
    """from_names (the blessed path) is unaffected by the new guard."""
    from tengri.observation import filters as filters_module

    def fake_load_filter_set(names, cache_dir=None):
        return (
            [_make_curve(n).wave for n in names],
            [_make_curve(n).trans for n in names],
            [_make_curve(n) for n in names],
        )

    monkeypatch.setattr(filters_module, "load_filter_set", fake_load_filter_set)
    phot = Photometry.from_names(["sdss_g", "sdss_r"])
    assert phot.names == ("sdss_g", "sdss_r")
    assert phot.n_filters == 2
