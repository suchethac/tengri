# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #3430: emission-line output units are self-describing.

``pred.lines`` headline fields are in L_sun (#1206 section A), the catalog
``pred.lines.all_lums`` / ``.get()`` and the ``log_<line>`` properties are in
erg/s, and the ``EmissionLines`` NamedTuple from ``predict_emission_lines`` is
in erg/s. The units were right; the docstrings were not. ``EmissionLines`` was
documented as the return type of ``Prediction.lines``, and its example printed
``pred.lines.halpha`` as erg/s.

These tests pin the unit claims against the returned values for the three
photoionization backends, and the docstring claim against the object model.
"""

from __future__ import annotations

import math
import os
import warnings

import jax
import pytest

import tengri
from tengri import DEFAULT, Fixed
from tengri.forward.prediction import EmissionLines
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_HALPHA_A = 6564.61
_CLOUDY_GRID = os.path.join(os.environ.get("TENGRI_DATA_DIR", "data"), "cloudy_grid_mist.h5")


def test_emission_lines_docstring_does_not_claim_prediction_lines():
    """``EmissionLines`` is the erg/s NamedTuple, not the ``Prediction.lines`` accessor.

    ``Prediction.lines`` is ``LineProperties``, whose headline fields are L_sun.
    """
    doc = EmissionLines.__doc__ or ""
    assert "Returned by :attr:`Prediction.lines`" not in doc
    assert "pred.lines  # LineProperties" not in doc
    assert "[erg/s]" in doc


def _sfh_and_dust():
    return {
        "sfh": {
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
        },
        "dust_attenuation": {
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "law": "calzetti",
            "tau_diff": 0.5,
            "tau_bc": 0.1,
        },
    }


def _build(backend: str):
    ssp = tengri.load_ssp()
    if backend == "cloudy_grid":
        if not os.path.exists(_CLOUDY_GRID):
            pytest.skip(f"shipped CloudyGrid file not present: {_CLOUDY_GRID}")
        neb = {"type": "cloudy", "all_params": Fixed(DEFAULT), "grid": _CLOUDY_GRID}
    else:
        neb = {"type": backend, "all_params": Fixed(DEFAULT)}
    return tengri.SEDModel.build(ssp, neb=neb, redshift=Fixed(0.05), **_sfh_and_dust())


@pytest.mark.parametrize("backend", ["cue", "cloudy_grid", "gutkin16"])
def test_units_agree_across_accessors(backend):
    """Headline L_sun, log_halpha erg/s, catalog erg/s and predict_emission_lines erg/s agree.

    The conversion between any L_sun and erg/s pair is exactly ``L_SUN``.
    """
    model = _build(backend)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    pred = model.predict(params)

    halpha_lsun = float(pred.lines.halpha)
    log_halpha_erg = float(pred.properties["log_halpha"])
    assert math.isfinite(halpha_lsun) and halpha_lsun > 0.0
    # pred.lines headline is L_sun; the log property is dex re erg/s.
    assert 10.0**log_halpha_erg == pytest.approx(halpha_lsun * L_SUN, rel=1e-6)

    # The erg/s catalog accessor reads the same line as the log property.
    catalog_halpha_erg = float(pred.lines.get(_HALPHA_A))
    if math.isfinite(catalog_halpha_erg):
        assert catalog_halpha_erg == pytest.approx(10.0**log_halpha_erg, rel=1e-6)

    # The deprecated NamedTuple surface is erg/s, as its docstring says.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = model.predict_emission_lines(params)
    assert float(legacy.halpha) == pytest.approx(10.0**log_halpha_erg, rel=1e-6)
