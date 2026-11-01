# SPDX-License-Identifier: BSD-3-Clause
"""#2320: ``agn_norm='conserving'`` beside an active fracAGN sets only the covering factor.

Owner ruling (2026-10-31, option 1): under fracAGN (``agn_ir_frac = f > 0``) the
``'conserving'`` policy keeps the AGN power at ``agn_lum_ratio x 10**agn_log_lbol L_sun``
and fracAGN sets only ``agn_torus_frac``. The torus re-emits that share of the accretion
power and the disc keeps the rest, so the energy ledger closes on ``agn_log_lbol``. Before
the ruling the build was refused (R67, ``_validate_fracagn_requires_cigale_joint``).

The mapping (``AGNSEDComponent.apply`` Notes), total dust:

    t = min(1, L_absorbed f / ((1 - f) agn_lum_ratio L_bol)),

so the AGN dust is the fraction ``f`` of ``AGN dust + L_absorbed``. With
``ir_frac_window`` ``t`` solves ``W_agn(t) = H f/(1 - f)`` on the affine window light.
Checked here: the build is admitted, the realized fraction is ``f`` (total dust and two
windows), the ledger closes (``agn_ledger_closed == 1``, C2 to 1e-6), the covering factor
saturates at 1 with the ledger still closed, gradients in ``agn_ir_frac`` and
``agn_log_lbol`` are finite and live, and ``'cigale_joint'`` is not clipped (its mapping is
the CIGALE ``skirtor2016`` one, unchanged).

The synthetic SSP's flux scale is arbitrary: at ``log_total_mass = -7`` it absorbs
``L_absorbed ~ 1.3e43 erg/s``, the order of an ``agn_log_lbol = 10`` AGN, which puts ``t``
inside ``(0, 1)``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.agn._ir_frac_window import _window_q
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_LOG_LBOL = 10.0  # the registry default of agn_log_lbol
_LEDGER = (
    "log_L_agn_disc",
    "log_L_agn_torus",
    "log_L_agn_lines",
    "log_L_agn_polar_dust",
    "log_L_agn_disc_unemitted",
    "log_L_agn_torus_unemitted",
    "log_L_agn_lines_unemitted",
)
#: The ledger's C2 tolerance (tests/contract/test_agn_composable_ledger_closes.py).
_RTOL = 1.0e-6


def _build(
    ssp,
    *,
    norm="conserving",
    ir_frac=0.3,
    log_total_mass=-7.0,
    window=None,
    obs=None,
    log_lbol=None,
    screen="none",
):
    agn = {
        "type": "composable",
        "norm": norm,
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "agn_ir_frac": ir_frac if not isinstance(ir_frac, float) else Fixed(ir_frac),
        "all_params": Fixed(DEFAULT),
    }
    if window is not None:
        agn["ir_frac_window"] = window
    if log_lbol is not None:
        agn["agn_log_lbol"] = log_lbol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(log_total_mass),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
                "agn_screen": screen,
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.05),
        )


def _ledger(model):
    props = model.predict_properties({}, names=(*_LEDGER, "agn_ledger_closed"))
    return {k: float(v) for k, v in props.items()}


def _dust_fraction(model):
    """AGN dust (angle-integrated torus + polar dust) over AGN dust + L_absorbed."""
    state = model.predict_state({})
    ledger = _ledger(model)
    agn_dust = 10.0 ** ledger["log_L_agn_torus"] + 10.0 ** ledger["log_L_agn_polar_dust"]
    return agn_dust / (agn_dust + 10.0 ** float(state.derived["log_L_absorbed"]))


def _window_fraction(model, window):
    """AGN share of the light in the window, from the emitted SEDs [dimensionless]."""
    state = model.predict_state({})
    wave = np.asarray(state.wave)
    agn = np.asarray(state.derived["sed_agn"])
    host = np.asarray(state.sed_intrinsic) - agn
    lo, hi = window[0] * 1.0e4, window[1] * 1.0e4
    if lo == hi:
        w_agn = float(np.interp(lo, wave, agn / wave**2))
        w_host = float(np.interp(lo, wave, host / wave**2))
    else:
        w_agn = float(_window_q(wave, agn, lo, hi))
        w_host = float(_window_q(wave, host, lo, hi))
    return w_agn / (w_agn + w_host)


def _torus_frac(model):
    return float(model.predict_state({}).derived["agn_torus_frac_effective"])


@pytest.mark.parametrize("ir_frac", [Fixed(0.3), FREE], ids=["fixed", "free"])
def test_conserving_with_active_fracagn_builds(synthetic_ssp_wide, ir_frac):
    """The refusal is gone: Fixed and FREE fracAGN both build under 'conserving'."""
    model = _build(synthetic_ssp_wide, ir_frac=ir_frac)
    assert model.spec.agn_norm == "conserving"


def test_screened_agn_with_fracagn_is_refused(synthetic_ssp_wide):
    """A screened AGN runs before the dust, so the absorbed power fracAGN reads does not exist
    yet: refused, instead of falling back to the user ``agn_torus_frac``."""
    from tengri.config.exceptions import ConfigError

    with pytest.raises(ConfigError, match="dust_agn_screen='diffuse'"):
        _build(synthetic_ssp_wide, screen="diffuse")
    # Without the coupling the screened conserving AGN is legal.
    model = _build(synthetic_ssp_wide, ir_frac=Fixed(0.0), screen="diffuse")
    assert model.spec.agn_norm == "conserving"


@pytest.mark.parametrize("frac", [0.1, 0.3])
def test_total_dust_fraction_is_realized(synthetic_ssp_wide, frac):
    """The AGN dust is the fraction ``f`` of the total dust (measured 2.6e-8 relative)."""
    model = _build(synthetic_ssp_wide, ir_frac=frac)
    assert 0.0 < _torus_frac(model) < 1.0
    assert _dust_fraction(model) == pytest.approx(frac, rel=_RTOL)


@pytest.mark.parametrize("window", [(5.0, 20.0), (10.0, 10.0)], ids=["band", "mono"])
def test_window_fraction_is_realized(synthetic_ssp_wide, window):
    """With a window, ``f`` is the AGN share of the window light, from the emitted SED."""
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=-6.0, window=window)
    assert 0.0 < _torus_frac(model) < 1.0
    assert _window_fraction(model, window) == pytest.approx(0.3, rel=_RTOL)


def test_ledger_closes_on_agn_log_lbol(synthetic_ssp_wide):
    """``agn_ledger_closed == 1`` and the pieces plus unemitted are ``L_bol`` to 1e-6 (C2)."""
    ledger = _ledger(_build(synthetic_ssp_wide, ir_frac=0.3))
    assert ledger["agn_ledger_closed"] == 1.0
    total = sum(10.0 ** ledger[k] for k in _LEDGER)
    assert total / (10.0**_LOG_LBOL * L_SUN) - 1.0 == pytest.approx(0.0, abs=_RTOL)


def test_covering_factor_saturates_and_the_ledger_still_closes(synthetic_ssp_wide):
    """A host that asks for more AGN dust than the accretion power has: ``t = 1``, no more.

    The torus then re-emits the whole power, the realized fraction falls below ``f``, and the
    pieces still add up to ``L_bol``.
    """
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=-5.0)
    assert _torus_frac(model) == 1.0
    assert _dust_fraction(model) < 0.3
    ledger = _ledger(model)
    assert ledger["agn_ledger_closed"] == 1.0
    total = sum(10.0 ** ledger[k] for k in _LEDGER)
    assert total / (10.0**_LOG_LBOL * L_SUN) - 1.0 == pytest.approx(0.0, abs=_RTOL)


def test_gradients_are_finite_and_live(synthetic_ssp_wide, synthetic_tophat_obs):
    """d(photometry)/d(agn_ir_frac, agn_log_lbol): finite and nonzero."""
    model = _build(
        synthetic_ssp_wide,
        ir_frac=FREE,
        obs=synthetic_tophat_obs,
        log_lbol=FREE,
    )
    params = {"agn_ir_frac": jnp.asarray(0.3), "agn_log_lbol": jnp.asarray(_LOG_LBOL)}

    def objective(p):
        return jnp.sum(jnp.log10(model.predict_photometry(p)))

    grads = jax.grad(objective)(params)
    for name, g in grads.items():
        assert np.isfinite(float(g)), name
        assert float(g) != 0.0, name


def test_cigale_joint_is_not_clipped(synthetic_ssp_wide):
    """'cigale_joint' keeps CIGALE's ``agn_power / L_bol`` unclipped: the covering-factor bound
    is the conserving policy's alone.

    At ``log_total_mass = -5`` the CIGALE mapping asks for ``t > 1``; 'cigale_joint' passes it
    through (the torus then carries ``L_absorbed f/(1-f)``) while 'conserving' stops at 1.
    """
    joint = _torus_frac(_build(synthetic_ssp_wide, norm="cigale_joint", log_total_mass=-5.0))
    conserving = _torus_frac(_build(synthetic_ssp_wide, log_total_mass=-5.0))
    assert joint > 1.0
    assert conserving == 1.0
    state = _build(synthetic_ssp_wide, norm="cigale_joint", log_total_mass=-5.0).predict_state({})
    expected = 10.0 ** (
        float(state.derived["log_L_absorbed"]) + np.log10(0.3 / 0.7) - _LOG_LBOL - np.log10(L_SUN)
    )
    assert joint == pytest.approx(expected, rel=1e-12)
