#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""``met_alpha_fe`` is a real axis on a 4-D grid and is refused on a 3-D one (#3411).

``met_alpha_fe`` changes the SED only through bilinear interpolation across a
4-D SSP grid whose fourth axis is [alpha/Fe]. A 3-D grid has no such axis, and
``met_logzsol`` is the total metallicity there, so a non-zero ``met_alpha_fe``
would have no effect. The builder therefore refuses it on a 3-D grid, free or
fixed, and the tests below check that refusal by name. The forward-model
tests run on a synthetic 4-D grid with a Mg b-like trough that deepens with
[alpha/Fe], so the parameter is exercised on the path it actually moves.

Three things can go wrong here and none of them raises on a 4-D grid:

1. the parameter is **inert**: declared free, accepted by the builder, and
   never read, so a fit samples it against a flat likelihood;
2. the parameter moves the prediction but carries **no usable gradient**, so
   NUTS, HMC and MAP cannot move it even though a grid search could;
3. ``Fixed(x)`` and a free parameter evaluated at ``x`` **disagree**, which
   would mean the pinned and sampled paths are different models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
ALPHA_LO = -0.2
ALPHA_HI = 0.6
PROBE = 0.25


@pytest.fixture(scope="module")
def ssp4d():
    """A 3 Z x 5 [alpha/Fe] x 20 age x 300 wavelength grid with a Mg b-like trough."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(7)
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    cubes = [base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES]
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.stack(cubes, axis=1),
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))


def _model(ssp, observation, alpha_spec):
    """One model in which [alpha/Fe] is the only thing that varies."""
    met = {"logzsol": Fixed(-0.3)}
    if alpha_spec is not None:
        met["alpha_fe"] = alpha_spec
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met=met,
        redshift=Fixed(0.1),
        neb={"type": "ssp"},
    )


# ── 3-D grid: the parameter has no effect, so it is refused ─────────────────


def test_free_alpha_fe_is_refused_on_a_3d_grid(ssp_data_wne, observation):
    with pytest.raises(tengri.ParameterError, match="met_alpha_fe"):
        _model(ssp_data_wne, observation, Uniform(ALPHA_LO, ALPHA_HI))


def test_fixed_nonzero_alpha_fe_is_refused_on_a_3d_grid(ssp_data_wne, observation):
    with pytest.raises(tengri.ParameterError, match="no \\[alpha/Fe\\] axis"):
        _model(ssp_data_wne, observation, Fixed(0.6))


def test_fixed_zero_alpha_fe_builds_and_equals_the_model_without_it(ssp_data_wne, observation):
    """``Fixed(0.0)`` is the declared default and must not change the model."""
    with_zero = _model(ssp_data_wne, observation, Fixed(0.0))
    without = _model(ssp_data_wne, observation, None)
    assert np.array_equal(
        np.asarray(with_zero.predict_photometry({}), dtype=np.float64),
        np.asarray(without.predict_photometry({}), dtype=np.float64),
    )


# ── 4-D grid: the parameter is a real axis ──────────────────────────────────


def test_a_free_alpha_fe_is_declared_free_on_a_4d_grid(ssp4d, observation):
    model = _model(ssp4d, observation, Uniform(ALPHA_LO, ALPHA_HI))
    assert "met_alpha_fe" in model.spec.free_params


def test_a_free_alpha_fe_is_not_inert_on_a_4d_grid(ssp4d, observation):
    """A parameter that changes nothing gives back its own prior."""
    model = _model(ssp4d, observation, Uniform(ALPHA_LO, ALPHA_HI))
    lo = np.asarray(model.predict_photometry({"met_alpha_fe": ALPHA_LO}), dtype=np.float64)
    hi = np.asarray(model.predict_photometry({"met_alpha_fe": ALPHA_HI}), dtype=np.float64)

    relative = np.max(np.abs(hi - lo) / np.abs(lo))
    assert relative > 1e-3, (
        f"[alpha/Fe] moved across its whole prior and the photometry changed by "
        f"{relative:.3e}; the parameter is inert and a fit would return the prior"
    )


def test_the_alpha_fe_gradient_agrees_with_a_finite_difference_on_a_4d_grid(ssp4d, observation):
    """Nonzero is not enough: a wrong gradient is nonzero too."""
    model = _model(ssp4d, observation, Uniform(ALPHA_LO, ALPHA_HI))

    def total(alpha):
        return jnp.sum(model.predict_photometry({"met_alpha_fe": alpha}))

    analytic = float(jax.grad(total)(PROBE))
    step = 0.01
    numeric = (float(total(PROBE + step)) - float(total(PROBE - step))) / (2.0 * step)

    assert np.isfinite(analytic), "the [alpha/Fe] gradient is not finite"
    assert analytic != 0.0, (
        "the [alpha/Fe] gradient is identically zero: the parameter is severed from "
        "the photometry, and a sampler could not move it"
    )
    assert abs(numeric) > 0.0, "the finite difference is zero, so this check is vacuous"

    # Compare the ratio: the fluxes are tiny, so an absolute tolerance would
    # swallow a severed gradient (see the git history of this file for the
    # mutation that showed it).
    ratio = analytic / numeric
    assert ratio == pytest.approx(1.0, rel=1e-4), (
        f"autodiff gradient {analytic:.6e} disagrees with the central finite "
        f"difference {numeric:.6e} (ratio {ratio:.6f}); a sampler would be led "
        "by the wrong slope"
    )


def test_fixed_alpha_fe_matches_the_free_model_at_the_same_value_on_a_4d_grid(ssp4d, observation):
    """The pinned and sampled paths must be one model, not two."""
    pinned = _model(ssp4d, observation, Fixed(PROBE))
    free = _model(ssp4d, observation, Uniform(ALPHA_LO, ALPHA_HI))

    assert "met_alpha_fe" not in pinned.spec.free_params

    from_pinned = np.asarray(pinned.predict_photometry({}), dtype=np.float64)
    from_free = np.asarray(free.predict_photometry({"met_alpha_fe": PROBE}), dtype=np.float64)
    np.testing.assert_allclose(
        from_pinned,
        from_free,
        rtol=0.0,
        atol=0.0,
        err_msg=(
            "Fixed(alpha) and a free alpha evaluated at the same value give "
            "different photometry, so pinning it selects a different model"
        ),
    )
