# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1889: LineProperties parity with EmissionLines (#1889).

When ``Prediction.lines`` (new surface) was added to provide parity with the
deprecated ``predict_emission_lines()`` surface, the new LineProperties object
had to expose ``all_waves``, ``all_lums``, and ``.get()`` method to match
the EmissionLines NamedTuple contract.

This test verifies that both surfaces return identical results for:
- Headline fields (halpha, hbeta, oiii_5007, etc.)
- Full catalog arrays (all_waves, all_lums)
- Nearest-wavelength lookup via .get(wavelength)
- Tolerance boundary behavior
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import FIXED, Fixed

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp_bare():
    """The bare-stellar SSP grid for nebular models (Cue needs it)."""
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


@pytest.fixture
def model_with_cue(ssp_bare):
    """Simplest model with Cue nebular backend for line emission."""
    return tengri.SEDModel.build(
        ssp_bare,
        sfh={
            "type": "const",
            "*": FIXED,
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
        },
        dust={
            "type": "two_component",
            "*": FIXED,
            "law_bc": "calzetti",
            "tau_diff": 0.5,
            "tau_bc": 0.1,
            "slope": -0.7,
        },
        neb={"type": "cue", "*": FIXED},
        redshift=Fixed(0.05),
    )


def test_lineproperties_parity_headline_fields(model_with_cue):
    """Headline line fields (halpha, hbeta, etc.) are equal across surfaces."""
    params = dict(model_with_cue.spec.sample(jax.random.PRNGKey(0)))

    # Old surface: deprecated predict_emission_lines
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_lines = model_with_cue.predict_emission_lines(params)

    # New surface: pred.lines
    pred = model_with_cue.predict(params)
    new_lines = pred.lines

    # Check all headline fields match
    for field in [
        "lya",
        "civ_1549",
        "oii",
        "hbeta",
        "oiii_4959",
        "oiii_5007",
        "nii_6548",
        "halpha",
        "nii_6584",
        "sii_6717",
        "sii_6731",
    ]:
        old_val = float(getattr(old_lines, field))
        new_val = float(getattr(new_lines, field))
        assert jnp.allclose(jnp.asarray(old_val), jnp.asarray(new_val), rtol=1e-6, atol=1e-20), (
            f"Field {field}: old={old_val}, new={new_val}"
        )


def test_lineproperties_parity_full_catalog(model_with_cue):
    """Full catalog arrays (all_waves, all_lums) are equal and nonempty."""
    params = dict(model_with_cue.spec.sample(jax.random.PRNGKey(0)))

    # Old surface
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_lines = model_with_cue.predict_emission_lines(params)

    # New surface
    pred = model_with_cue.predict(params)
    new_lines = pred.lines

    # Both should have nonempty catalogs for Cue
    assert old_lines.all_waves.size > 0, "Old surface should have nonempty all_waves"
    assert old_lines.all_lums.size > 0, "Old surface should have nonempty all_lums"
    assert new_lines.all_waves.size > 0, "New surface should have nonempty all_waves"
    assert new_lines.all_lums.size > 0, "New surface should have nonempty all_lums"

    # Wavelengths and luminosities should be identical
    assert jnp.allclose(old_lines.all_waves, new_lines.all_waves), "all_waves mismatch"
    assert jnp.allclose(old_lines.all_lums, new_lines.all_lums), "all_lums mismatch"


def test_lineproperties_parity_get_exact_match(model_with_cue):
    """The .get() method returns exact matches when available."""
    params = dict(model_with_cue.spec.sample(jax.random.PRNGKey(0)))

    # Old surface
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_lines = model_with_cue.predict_emission_lines(params)

    # New surface
    pred = model_with_cue.predict(params)
    new_lines = pred.lines

    # Both should match halpha via .get() (H-alpha is at 6564.61 Å)
    halpha_wavelength = 6564.61
    old_halpha_via_get = float(old_lines.get(halpha_wavelength))
    new_halpha_via_get = float(new_lines.get(halpha_wavelength))
    old_halpha_direct = float(old_lines.halpha)
    new_halpha_direct = float(new_lines.halpha)

    # .get() should find the nearest match (H-alpha)
    assert jnp.allclose(jnp.asarray(old_halpha_via_get), jnp.asarray(new_halpha_via_get)), (
        f"old .get(6564.61) != new .get(6564.61): {old_halpha_via_get} vs {new_halpha_via_get}"
    )

    # Both surfaces should match on direct field
    assert jnp.allclose(jnp.asarray(old_halpha_direct), jnp.asarray(new_halpha_direct)), (
        f"old.halpha != new.halpha: {old_halpha_direct} vs {new_halpha_direct}"
    )


def test_lineproperties_parity_get_out_of_tolerance(model_with_cue):
    """The .get() method returns NaN when no line is within tolerance."""
    params = dict(model_with_cue.spec.sample(jax.random.PRNGKey(0)))

    # Old surface
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_lines = model_with_cue.predict_emission_lines(params)

    # New surface
    pred = model_with_cue.predict(params)
    new_lines = pred.lines

    # Query a wavelength far from any line (10000 Å should be isolated)
    far_wavelength = 10000.0
    old_far = old_lines.get(far_wavelength, tol_aa=1.0)
    new_far = new_lines.get(far_wavelength, tol_aa=1.0)

    # Both should return NaN
    assert bool(jnp.isnan(old_far)), f"Old surface should return NaN, got {old_far}"
    assert bool(jnp.isnan(new_far)), f"New surface should return NaN, got {new_far}"


def test_lineproperties_parity_get_tolerance_boundary(model_with_cue):
    """Tolerance boundary behavior is identical across surfaces."""
    params = dict(model_with_cue.spec.sample(jax.random.PRNGKey(0)))

    # Old surface
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_lines = model_with_cue.predict_emission_lines(params)

    # New surface
    pred = model_with_cue.predict(params)
    new_lines = pred.lines

    # Use H-alpha at 6564.61 Å
    halpha_wl = 6564.61
    query_wl = 6566.0  # 1.39 Å away
    tol = 2.0  # Should be within tolerance

    old_result_tight = old_lines.get(query_wl, tol_aa=1.0)
    new_result_tight = new_lines.get(query_wl, tol_aa=1.0)
    old_result_loose = old_lines.get(query_wl, tol_aa=2.0)
    new_result_loose = new_lines.get(query_wl, tol_aa=2.0)

    # Tight tolerance should give NaN for both
    assert bool(jnp.isnan(old_result_tight)), "Old surface tight tolerance should be NaN"
    assert bool(jnp.isnan(new_result_tight)), "New surface tight tolerance should be NaN"

    # Loose tolerance should give a value for both
    assert not bool(jnp.isnan(old_result_loose)), "Old surface loose tolerance should find a line"
    assert not bool(jnp.isnan(new_result_loose)), "New surface loose tolerance should find a line"

    # And they should be equal
    assert jnp.allclose(jnp.asarray(old_result_loose), jnp.asarray(new_result_loose)), (
        f"Loose tolerance results differ: old={old_result_loose}, new={new_result_loose}"
    )
