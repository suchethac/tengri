# SPDX-License-Identifier: BSD-3-Clause
"""Regression: hardening round 6 (resample redshift decision, SKIRTOR load except).

Item 1 (``resample="auto"`` redshift). The decision is taken at a reference
redshift. With a free redshift and no finite lower bound it used to fall back
to z=0 silently, which point-samples or integrates the spectrum at the wrong
rest frame. The contract now is: fixed z uses the fixed value, a free z with a
finite ``lo`` uses ``lo``, and anything else raises ``ParameterError`` under
``resample="auto"`` only. Explicit modes never consult the reference, so they
are not refused.

Item 3 (SKIRTOR template load). ``build`` used to swallow every exception from
the SKIRTOR template load. Only the missing-file case may be silent: the lazy
``apply()`` path raises the clear not-found error itself. A corrupt grid must
propagate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression_bug

from tengri.config.exceptions import ParameterError
from tengri.forward.sed_model import SEDModel


def _redshift_model(distribution, *, fixed_z=None, resample="auto", spectroscopy=True):
    """Stand-in carrying only the attributes ``_resample_z_ref`` reads."""
    spec = SimpleNamespace(get_distribution=lambda name: distribution)
    spectro = SimpleNamespace(resample=resample) if spectroscopy else None
    return SimpleNamespace(
        spec=spec,
        observation=SimpleNamespace(spectroscopy=spectro),
        _z_fixed=fixed_z,
    )


def _free(lo):
    return SimpleNamespace(is_fixed=False, lo=lo, hi=3.0)


def _fixed(value):
    return SimpleNamespace(is_fixed=True, value=value, lo=value, hi=value)


def _ref(model):
    return SEDModel._resample_z_ref(model)


def test_free_unbounded_redshift_under_auto_raises():
    """A free z with no lower bound must not silently decide at z=0."""
    model = _redshift_model(_free(lo=None))
    with pytest.raises(ParameterError, match="bounded redshift"):
        _ref(model)


def test_free_non_finite_lower_bound_under_auto_raises():
    """An infinite ``lo`` is no reference redshift either."""
    model = _redshift_model(_free(lo=float("-inf")))
    with pytest.raises(ParameterError, match='resample="auto"'):
        _ref(model)


def test_missing_distribution_without_fixed_z_under_auto_raises():
    """No redshift distribution and no fixed z leaves nothing to decide at."""

    def _missing(name):
        raise KeyError(name)

    model = _redshift_model(None)
    model.spec = SimpleNamespace(get_distribution=_missing)
    with pytest.raises(ParameterError, match="bounded redshift"):
        _ref(model)


def test_free_bounded_redshift_uses_its_lower_bound():
    """A finite prior lower bound is the reference, clipped at zero."""
    assert _ref(_redshift_model(_free(lo=0.4))) == pytest.approx(0.4)
    assert _ref(_redshift_model(_free(lo=-0.2))) == 0.0


def test_fixed_redshift_uses_the_fixed_value():
    """Fixed z keeps its value; the free-z guard never fires for it."""
    assert _ref(_redshift_model(_fixed(0.3), fixed_z=0.3)) == pytest.approx(0.3)


def test_explicit_resample_mode_is_not_refused():
    """With ``resample='point'`` or ``'conserving'`` the reference is unused, so no raise."""
    for mode in ("point", "conserving"):
        model = _redshift_model(_free(lo=None), resample=mode)
        assert _ref(model) == 0.0


def test_no_spectroscopy_is_not_refused():
    """A photometry-only model has no resample decision to make."""
    model = _redshift_model(_free(lo=None), spectroscopy=False)
    assert _ref(model) == 0.0


def test_skirtor_template_load_propagates_non_missing_errors(monkeypatch):
    """A corrupt SKIRTOR grid is an error, not a silent fallback to lazy loading."""
    import tengri.components.agn.skirtor as skirtor

    def _corrupt():
        raise RuntimeError("corrupt SKIRTOR grid")

    monkeypatch.setattr(skirtor, "_load_skirtor_default_grid", _corrupt)
    from tengri.components.agn.component import AGNSEDComponent

    component = AGNSEDComponent.__new__(AGNSEDComponent)
    object.__setattr__(component, "config", SimpleNamespace(model="skirtor"))
    with pytest.raises(RuntimeError, match="corrupt SKIRTOR grid"):
        AGNSEDComponent.precompute(component, wave_grid=None)
