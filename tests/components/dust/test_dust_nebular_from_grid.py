# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: a dust screen fed the per-Q_H nebular grid reproduces the screened continuum.

With ``nebular_from_grid`` set, both attenuators read five grid channels instead
of ``sed_nebular``. This file pins, on hand-made synthetic state (no SSP):

* the flag swaps the declared inputs and ``materialized()`` undoes it;
* ``nebular_screen_transmission`` is bit-for-bit the ratio ``apply()`` applies to
  the nebular continuum (one screen, two consumers);
* the published nebular photometry, observed and rest-band twins, is the
  K-point sum ``sum_k Phi_k T(lambda_k)`` with ``T`` the continuum screen;
* the nebular half of the energy balance is the bilinear bracket of the per-Q_H
  absorbed-power grid on the stellar LUT's tau axes (deliberately unequal, so a
  transposed contraction cannot pass), fed into the same closing ``log10_add``;
* at ``tau = 0`` the nebular term vanishes and the stellar term stands alone.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.component import (
    DustAttenuationSEDComponent,
    DustAttenuationSEDComponentConfig,
)
from tengri.components.dust.energy_balance_precompute import (
    EnergyBalanceLUT,
    lut_l_absorbed_stellar_log10,
)
from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig
from tengri.protocols.component import ForwardState
from tengri.utils.scale import log10_add

pytestmark = pytest.mark.conservation

_GRID_KEYS = {
    "nebular_phot_lnu_subband_precomp",
    "nebular_subband_waves_rest_precomp",
    "nebular_restband_lnu_subband_precomp",
    "nebular_restband_subband_waves_precomp",
    "nebular_eb_absorbed_per_qh_grid_precomp",
}
_N_MET, _N_AGE, _N_FILTER, _K = 3, 4, 3, 4
_LOG_NION = 50.0
_TAU_A_GRID = np.linspace(0.0, 2.0, 5)
_TAU_B_GRID = np.linspace(0.0, 3.0, 4)  # spacing 1.0 vs 0.5 on the other axis
_LAM_OBS = np.linspace(1500.0, 9000.0, _N_FILTER * _K).reshape(_N_FILTER, _K)
_LAM_REST = np.linspace(1200.0, 7000.0, _N_FILTER * _K).reshape(_N_FILTER, _K)
_PHI_OBS = np.linspace(1.0, 2.0, _N_FILTER * _K).reshape(_N_FILTER, _K) * 1.0e27
_PHI_REST = np.linspace(2.0, 3.0, _N_FILTER * _K).reshape(_N_FILTER, _K) * 1.0e27
_PARAMS = {
    "dust_tau_bc": 0.7,
    "dust_tau_diff": 0.4,
    "dust_tau_v": 0.6,
    "dust_f_obscuration": 0.2,
    "dust_slope": -1.2,
}
_TWO = "two"
_ONE = "one"


def _component(kind: str, *, flagged: bool):
    if kind == _TWO:
        comp = DustSEDComponent(
            config=DustSEDComponentConfig(
                law_bc="calzetti", law_diff="power_law", nebular_screen="birth_cloud"
            )
        )
    else:
        comp = DustAttenuationSEDComponent(
            config=DustAttenuationSEDComponentConfig(law="calzetti")
        )
    return dataclasses.replace(comp, nebular_from_grid=flagged)


def _lut(kind: str) -> EnergyBalanceLUT:
    rng = np.random.default_rng(5)
    tau_a = _TAU_A_GRID if kind == _TWO else np.zeros(1)
    b = rng.random((_N_MET, _N_AGE)) + 0.5
    # attenuated <= intrinsic, so the stellar absorbed term is positive
    g = b[..., None, None] * rng.uniform(0.2, 0.9, (_N_MET, _N_AGE, tau_a.size, _TAU_B_GRID.size))
    return EnergyBalanceLUT(
        B=jnp.asarray(b),
        G=jnp.asarray(g),
        tau_bc_grid=jnp.asarray(tau_a),
        tau_diff_grid=jnp.asarray(_TAU_B_GRID),
    )


def _grid_channels(kind: str) -> dict:
    tau_a = _TAU_A_GRID if kind == _TWO else np.zeros(1)
    return {
        "nebular_phot_lnu_subband_precomp": jnp.asarray(_PHI_OBS),
        "nebular_subband_waves_rest_precomp": jnp.asarray(_LAM_OBS),
        "nebular_restband_lnu_subband_precomp": jnp.asarray(_PHI_REST),
        "nebular_restband_subband_waves_precomp": jnp.asarray(_LAM_REST),
        # per unit Q_H, bilinear-exact for the bracket
        "nebular_eb_absorbed_per_qh_grid_precomp": jnp.asarray(
            np.outer(tau_a + 1.0, _TAU_B_GRID) * 1.0e-11
        ),
    }


def _state(wave, *, derived=None) -> ForwardState:
    wave = jnp.asarray(wave)
    lnu_age = jnp.ones((_N_AGE, wave.shape[0])) * 1.0e27
    base = {
        "lnu_age": lnu_age,
        "ssp_ages_yr": jnp.asarray([1.0e6, 1.0e7, 1.0e8, 1.0e10]),
        "log_nion": jnp.asarray(_LOG_NION),
    }
    base.update(derived or {})
    return ForwardState(wave=wave, sed_intrinsic=jnp.sum(lnu_age, axis=0), derived=base)


def _flagged_state(kind: str, *, with_bands: bool = True) -> ForwardState:
    wave = jnp.logspace(np.log10(600.0), np.log10(30000.0), 200)
    derived = _grid_channels(kind)
    derived.update(
        {
            "joint_weights": jnp.asarray(np.random.default_rng(3).random((_N_MET, _N_AGE))),
            "log_stellar_mass_scale": jnp.asarray(40.0),
        }
    )
    if with_bands:
        derived.update(
            {
                "filter_eff_waves": jnp.asarray([2300.0, 4800.0, 7600.0]),
                "filter_restband_eff_waves": jnp.asarray([2000.0, 4000.0, 6000.0]),
                "phot_filter_waves_padded": jnp.asarray(
                    np.tile(np.linspace(1000, 9000, 20), (3, 1))
                ),
                "phot_filter_trans_padded": jnp.ones((3, 20)),
            }
        )
    return _state(wave, derived=derived)


def _apply(comp, kind, state, params=None, *, with_lut=True):
    tdata = {"dust_ir": {"energy_balance_lut": _lut(kind)}} if with_lut else None
    return comp.apply(state, dict(params or _PARAMS) | {"redshift": 0.5}, template_data=tdata)


def _exact_screen(kind: str, wavelength: np.ndarray, params=None) -> np.ndarray:
    """The screen the exact path applies to the continuum, read from an unflagged apply()."""
    wavelength = np.asarray(wavelength).reshape(-1)
    state = _state(wavelength, derived={"sed_nebular": jnp.ones(wavelength.shape)})
    out = _apply(_component(kind, flagged=False), kind, state, params, with_lut=False)
    key = "sed_nebular" if kind == _TWO else "dust_attenuation_factor"
    return np.asarray(out.derived[key])


@pytest.mark.parametrize("kind", [_TWO, _ONE])
def test_flag_swaps_the_declared_inputs(kind):
    plain = _component(kind, flagged=False)
    fast = _component(kind, flagged=True)
    plain_names = {k.name for k in plain.optional_inputs()}
    fast_names = {k.name for k in fast.optional_inputs()}
    assert "sed_nebular" in plain_names and not (_GRID_KEYS & plain_names)
    assert "sed_nebular" not in fast_names
    assert fast_names >= _GRID_KEYS
    assert {"line_waves", "log_line_lums"} <= fast_names
    assert fast.materialized().nebular_from_grid is False
    assert plain.materialized() is plain


@pytest.mark.parametrize("kind", [_TWO, _ONE])
def test_screen_method_is_the_screen_apply_puts_on_the_continuum(kind):
    wave = np.logspace(np.log10(700.0), np.log10(25000.0), 64)
    got = np.asarray(
        _component(kind, flagged=False).nebular_screen_transmission(
            _PARAMS | {"redshift": 0.5}, jnp.asarray(wave)
        )
    )
    np.testing.assert_allclose(got, _exact_screen(kind, wave), rtol=0.0, atol=1e-15)
    assert 0.0 < got.min() < got.max() <= 1.0


@pytest.mark.parametrize("kind", [_TWO, _ONE])
def test_grid_photometry_is_the_screened_subband_sum(kind):
    out = _apply(_component(kind, flagged=True), kind, _flagged_state(kind))
    for phi, lam, key in (
        (_PHI_OBS, _LAM_OBS, "nebular_phot_lnu_attenuated_precomp"),
        (_PHI_REST, _LAM_REST, "nebular_restband_lnu_attenuated_precomp"),
    ):
        t = _exact_screen(kind, lam).reshape(lam.shape)
        want = np.sum(phi * t, axis=-1)
        np.testing.assert_allclose(np.asarray(out.derived[key]), want, rtol=1e-13)
        assert not np.allclose(want, phi.sum(-1))  # the screen is not unity


@pytest.mark.parametrize("kind", [_TWO, _ONE])
def test_grid_energy_balance_is_the_bilinear_bracket(kind):
    params = dict(_PARAMS, dust_tau_bc=0.35, dust_tau_diff=1.3, dust_tau_v=1.3)
    state = _flagged_state(kind, with_bands=False)
    out = _apply(_component(kind, flagged=True), kind, state, params)

    lut = _lut(kind)
    tau_a = params["dust_tau_bc"] if kind == _TWO else 0.0
    tau_b = params["dust_tau_diff"] if kind == _TWO else params["dust_tau_v"]
    log_stellar, sign = lut_l_absorbed_stellar_log10(
        lut,
        state.derived["joint_weights"],
        state.derived["log_stellar_mass_scale"],
        jnp.asarray(tau_a),
        jnp.asarray(tau_b),
    )
    # grid = outer(a + 1, b) * 1e-11 is bilinear: the bracket reproduces it exactly
    per_qh = (tau_a + 1.0) * tau_b * 1.0e-11 if kind == _TWO else 1.0 * tau_b * 1.0e-11
    want = log10_add(log_stellar, _LOG_NION + np.log10(per_qh), sign_a=sign, sign_b=1.0)
    np.testing.assert_allclose(
        np.asarray(out.derived["log_L_absorbed"]), np.asarray(want), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out.derived["L_absorbed"]), 10.0 ** np.asarray(want), rtol=1e-11
    )
    # the nebular term is a real share of the total, so a wrong contraction shows
    assert float(want) - float(log_stellar) > 1e-3


@pytest.mark.parametrize("kind", [_TWO, _ONE])
def test_tau_zero_leaves_the_stellar_term_alone(kind):
    params = dict(_PARAMS, dust_tau_bc=0.0, dust_tau_diff=0.0, dust_tau_v=0.0)
    state = _flagged_state(kind, with_bands=False)
    out = _apply(_component(kind, flagged=True), kind, state, params)
    log_stellar, _ = lut_l_absorbed_stellar_log10(
        _lut(kind),
        state.derived["joint_weights"],
        state.derived["log_stellar_mass_scale"],
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    # the grid's tau_b = 0 column is exactly zero, so only the stellar term remains
    np.testing.assert_allclose(
        np.asarray(out.derived["log_L_absorbed"]), np.asarray(log_stellar), rtol=0.0, atol=1e-12
    )
