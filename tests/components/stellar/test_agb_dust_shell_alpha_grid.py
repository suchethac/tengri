# SPDX-License-Identifier: BSD-3-Clause
"""AGB dust-shell weighting on alpha-enhanced 4-D SSP grids (#2534).

An alpha-enhanced library carries ``ssp_flux`` of shape
``(n_met, n_alpha, n_age, n_wave)``. The shell ratio depends on metallicity,
age and wavelength and is taken independent of [alpha/Fe] (an assumption of
the template, computed at solar-scaled abundances), so it must multiply every
alpha slice alike. Two grids are used: ``n_alpha == n_met``, where a ratio
broadcast without an explicit alpha axis would multiply along the wrong axis
and return wrong numbers, and ``n_alpha != n_met``, where it would raise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.components.stellar.agb_dust_shell import (
    agb_dust_ratio,
    align_ratio_to_cube,
    bake_agb_dust_shell,
    load_agb_dust_shell_template,
    resample_agb_dust_shell,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData, interpolate_alpha_only

pytestmark = pytest.mark.bounds

_N_MET = 3
_ALPHA_GRIDS = {3: [-0.2, 0.0, 0.4], 5: [-0.2, 0.0, 0.2, 0.4, 0.6]}


def _grid(n_alpha: int) -> SSPData:
    """Synthetic 4-D grid, tagged as a MIST library so the weighting accepts it."""
    n_age, n_wave = 8, 120
    rng = np.random.default_rng(7)
    alpha_fe = jnp.asarray(_ALPHA_GRIDS[n_alpha])
    flux = rng.uniform(0.5, 1.5, (_N_MET, n_alpha, n_age, n_wave)) * 1e-3
    return SSPData(
        ssp_wave=jnp.linspace(1000.0, 30000.0, n_wave),
        ssp_flux=jnp.asarray(flux),
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.0, n_age),
        ssp_lgmet=jnp.asarray([-2.0, -1.5, -1.0]),
        ssp_alpha_fe=alpha_fe,
        ssp_mass_remaining=None,
        source="fsps_mist_synthetic_alpha",
    )


def _ratio(ssp: SSPData, weight: float):
    resampled = resample_agb_dust_shell(
        load_agb_dust_shell_template(),
        np.asarray(ssp.ssp_lgmet),
        np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp.ssp_wave),
    )
    return np.asarray(agb_dust_ratio(resampled, weight))


def _model(ssp: SSPData, **agb):
    spec = Parameters(
        met_mode="delta",
        met_alpha_fe=Fixed(0.1),
        mean_sfh_type="const",
        sfh_const_log_total_mass=Fixed(0.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_diff=Fixed(0.0),
        dust_tau_bc=Fixed(0.0),
        redshift=Fixed(0.05),
        **agb,
    )
    return SEDModel(spec, ssp)


def _sed(model, params=None):
    params = model.spec.sample(jax.random.PRNGKey(0)) if params is None else params
    return np.asarray(model.predict(dict(params)).rest_sed())


@pytest.mark.parametrize("n_alpha", [3, 5])
def test_baked_cube_is_the_cube_times_the_ratio_in_every_alpha_slice(n_alpha):
    ssp = _grid(n_alpha)
    ratio = _ratio(ssp, 2.0)
    assert not np.allclose(ratio, 1.0)
    baked = np.asarray(bake_agb_dust_shell(ssp, 2.0).ssp_flux)
    cube = np.asarray(ssp.ssp_flux)
    for k in range(n_alpha):
        np.testing.assert_allclose(baked[:, k], cube[:, k] * ratio, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("n_alpha", [3, 5])
def test_free_weight_equals_fixed_weight_on_a_4d_grid(n_alpha):
    ssp = _grid(n_alpha)
    fixed = _model(ssp, agb_dust=True, agb_dust_weight=Fixed(2.0))
    free = _model(ssp, agb_dust=True, agb_dust_weight=Uniform(0.0, 3.0))
    sed_fixed = _sed(fixed)
    sed_free = _sed(free, {**free.spec.sample(jax.random.PRNGKey(0)), "agb_dust_weight": 2.0})
    np.testing.assert_allclose(sed_free, sed_fixed, rtol=1e-5)
    assert np.max(np.abs(sed_free / _sed(_model(ssp)) - 1.0)) > 1e-3


@pytest.mark.parametrize("n_alpha", [3, 5])
def test_default_weight_is_bit_identical_to_the_plain_4d_model(n_alpha):
    ssp = _grid(n_alpha)
    plain = _sed(_model(ssp))
    with_default = _sed(_model(ssp, agb_dust=True, agb_dust_weight=Fixed(1.0)))
    assert np.array_equal(plain, with_default)


@pytest.mark.parametrize("n_alpha", [3, 5])
def test_ratio_before_or_after_the_alpha_collapse_gives_the_same_spectrum(n_alpha):
    """The ratio does not depend on alpha, so multiplying the 4-D cube and then
    collapsing it equals collapsing and then multiplying."""
    ssp = _grid(n_alpha)
    ratio = jnp.asarray(_ratio(ssp, 0.4))
    for alpha in (-0.1, 0.0, 0.25):
        before = interpolate_alpha_only(
            ssp.ssp_flux * align_ratio_to_cube(ratio, ssp.ssp_flux), ssp.ssp_alpha_fe, alpha
        )
        after = interpolate_alpha_only(ssp.ssp_flux, ssp.ssp_alpha_fe, alpha) * ratio
        np.testing.assert_allclose(np.asarray(before), np.asarray(after), rtol=1e-12)


def test_ratio_that_does_not_match_the_cube_is_refused():
    ssp = _grid(5)
    with pytest.raises(ValueError, match="does not match"):
        align_ratio_to_cube(jnp.ones((_N_MET + 1, 8, 120)), ssp.ssp_flux)
