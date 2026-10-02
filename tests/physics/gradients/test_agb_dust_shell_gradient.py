# SPDX-License-Identifier: BSD-3-Clause
"""Gradient of the photometry with respect to the AGB dust-shell weight (#2534).

``agb_dust_weight`` enters through :func:`agb_dust_ratio`, linear in the weight
between the template's stored weight nodes. The gradient is therefore the
slope of the bracketing segment: it matches a central finite difference
strictly inside a segment and is discontinuous at a stored node. At the
reference weight ``w = 1`` the ratio peaks (it is 1 there by construction and
below 1 on both sides at 10 um), so the gradient changes sign across that
node; tests evaluate at weights between nodes, and the kink is asserted
separately.

``jax.grad`` through ``SEDModel.predict_photometry`` is finite and non-zero for
a population with TP-AGB stars, and the ratio is exactly flat in ``w`` for a
population too young to have any (3 Myr).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, SEDModel, Uniform, load_ssp
from tengri.components.stellar.agb_dust_shell import (
    agb_dust_ratio,
    load_agb_dust_shell_template,
    resample_agb_dust_shell,
)
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.gradient

_SSP = "fsps_mist_c3k_a_chabrier"
_TEN_MICRON_BAND = 4


def fd_grad(f, x: float, eps: float = 1e-3) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def _observation() -> Observation:
    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (4800.0, 8000.0, 22000.0, 50000.0, 100000.0))
    return Observation(photometry=Photometry(filters=curves))


@pytest.fixture(scope="module")
def model_and_params():
    model = SEDModel.build(
        ssp_data=load_ssp(_SSP),
        observation=_observation(),
        sfh={"type": "const"},
        redshift=Fixed(0.0),
        agb_dust={"type": "fsps_shell", "weight": Uniform(0.0, 3.0)},
    )
    return model, model.spec.sample(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def resampled():
    ssp = load_ssp(_SSP)
    template = load_agb_dust_shell_template()
    return resample_agb_dust_shell(
        template,
        np.asarray(ssp.ssp_lgmet),
        np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp.ssp_wave),
    ), ssp


def _ten_micron_flux(model, params):
    def flux(w):
        return model.predict_photometry({**params, "agb_dust_weight": w})[_TEN_MICRON_BAND]

    return flux


@pytest.mark.parametrize("w0", [0.4, 1.1, 1.8])
def test_photometry_grad_matches_finite_difference_between_nodes(model_and_params, w0):
    """End to end through ``predict_photometry`` at weights strictly inside a
    segment of the stored ladder (0.3125-0.4375, 1-1.25, 1.625-2)."""
    model, params = model_and_params
    flux = _ten_micron_flux(model, params)
    grad = float(jax.grad(flux)(w0))
    fd = fd_grad(flux, w0)
    assert jnp.isfinite(grad)
    assert jnp.isfinite(fd)
    assert grad != 0.0
    assert fd != 0.0
    assert abs(grad - fd) / abs(fd) < 1e-2, f"grad={grad}, finite-diff={fd}"


def test_photometry_grad_changes_sign_across_the_reference_weight(model_and_params):
    """The stored node at w=1 is a kink: R peaks at 1 there, so the 10 um flux
    rises into it and falls away from it."""
    model, params = model_and_params
    flux = _ten_micron_flux(model, params)
    below = float(jax.grad(flux)(1.0 - 1e-4))
    above = float(jax.grad(flux)(1.0 + 1e-4))
    assert below > 0.0 > above


def test_grad_wrt_weight_is_zero_for_3myr_population(resampled):
    """3 Myr: no star has reached the TP-AGB phase, so R(w) = 1 at every
    wavelength and the gradient of the ratio vanishes."""
    resampled_grid, ssp = resampled
    met_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - (-1.848))))
    age_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(0.003))))
    wave_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_wave) - 1.0e5)))

    def ratio_at_10um(w):
        return agb_dust_ratio(resampled_grid, w)[met_idx, age_idx, wave_idx]

    grad = float(jax.grad(ratio_at_10um)(0.4))
    # grad-assert: finite-only — the claim on this gradient is that it is ~0
    # (asserted next via an explicit magnitude bound), the opposite of "non-zero".
    assert jnp.isfinite(grad)
    assert abs(grad) < 1e-9, f"expected ~0 gradient for a pre-TP-AGB population, got {grad}"
