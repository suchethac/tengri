# SPDX-License-Identifier: BSD-3-Clause
"""#2485: a single-screen dust model forfeited the dust-emission band response.

``dust_attenuation={'type': 'single_component', ...}`` has one free attenuation
parameter, ``dust_tau_v``, and that name was absent from the allowlist the band
response gates on. ``shape_free`` therefore went True and the per-filter
response was never built, so every gradient evaluation paid the dense per-call
filter integral instead. The model stayed correct, which is why nothing failed:
two of the six paper-1 configurations were in this state.

``dust_tau_v`` is an optical depth. It scales how much energy is absorbed, hence
the ``L_ir`` amplitude, but leaves the emission template's spectral *shape*
alone, so the response stays a build-time constant and admitting it is exact.

**The allowlist it was added to is deliberately not the shared one.**
``_EB_ATTEN_FREE_OK`` is also read by the energy-balance LUT, whose builder
takes required ``tau_bc_grid``/``tau_diff_grid`` axes and has no ``tau_v`` axis.
Admitting ``dust_tau_v`` there would let a LUT be built on the wrong axes while
the runtime varies ``tau_v`` -- a silently wrong ``L_absorbed``, worse than the
slow path this fixes. ``test_the_energy_balance_lut_was_not_enabled_too`` pins
that separation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.regression_bug


def _single_screen(ssp, obs):
    """The configuration that forfeited: one Calzetti screen, tau_v free."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def _two_component(ssp, obs):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 3.0),
            "other_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def _params(model):
    """One deterministic in-prior draw. A fixed key so both arms see the same point."""
    import jax

    return model.spec.sample(jax.random.PRNGKey(0))


def test_the_single_screen_model_builds_its_band_response(synthetic_ssp, synthetic_tophat_obs):
    """The defect. Fails before ``dust_tau_v`` joins the band-response allowlist."""
    model = _single_screen(synthetic_ssp, synthetic_tophat_obs)

    response = model._dust_band_response_cache
    assert response is not None, (
        "the band response was not built, so this model pays the dense per-call "
        "filter integral on every gradient evaluation"
    )
    assert np.all(np.isfinite(response))


def test_the_band_response_does_not_change_the_photometry(
    synthetic_ssp, synthetic_tophat_obs, monkeypatch
):
    """The gate that matters: this optimization is exact, or it is a bug.

    The comparison arm is built with ``dust_tau_v`` removed from the allowlist,
    which is precisely the pre-fix state, so this measures the fix itself rather
    than two unrelated code paths. ``array_equal``, not ``allclose``: an exact
    factorization of a linear integral has no error budget to spend.
    """
    fast = _single_screen(synthetic_ssp, synthetic_tophat_obs)
    assert fast._dust_band_response_cache is not None, "fast arm did not take the fast path"
    phot_fast = np.asarray(fast.predict_photometry(_params(fast)))

    monkeypatch.setattr(
        SEDModel,
        "_BAND_RESPONSE_ATTEN_FREE_OK",
        frozenset(SEDModel._BAND_RESPONSE_ATTEN_FREE_OK - {"dust_tau_v"}),
    )
    exact = _single_screen(synthetic_ssp, synthetic_tophat_obs)
    assert exact._dust_band_response_cache is None, (
        "control arm still took the fast path, so this test compares a path with "
        "itself and cannot observe a difference"
    )
    phot_exact = np.asarray(exact.predict_photometry(_params(exact)))

    assert np.array_equal(phot_fast, phot_exact), (
        "the band response changed the photometry; it is advertised as exact.\n"
        f"  max |diff|     = {np.max(np.abs(phot_fast - phot_exact)):.6e}\n"
        f"  max rel. diff  = {np.max(np.abs(phot_fast - phot_exact) / np.abs(phot_exact)):.6e}"
    )


def test_the_energy_balance_lut_was_not_enabled_too(synthetic_ssp, synthetic_tophat_obs):
    """The degenerate two-component LUT structure.

    Single-screen ``dust_tau_v`` is now enabled for the energy-balance LUT
    by mapping it to the degenerate two-component geometry:
    tau_bc = [0.0] (birth-cloud disabled) and tau_diff = tau_v.
    This test verifies the LUT is built with this degenerate axis structure,
    not that it is absent.
    """
    model = _single_screen(synthetic_ssp, synthetic_tophat_obs)

    lut = model._energy_balance_lut_cache
    assert lut is not None, "Energy balance LUT was not built for single-screen"

    # Verify degenerate structure: tau_bc pinned to [0.0], tau_diff spans tau_v prior
    assert lut.tau_bc_grid.shape[0] == 1, "tau_bc axis should have exactly one point"
    assert np.isclose(lut.tau_bc_grid[0], 0.0), "tau_bc should be pinned to 0.0"
    assert lut.tau_diff_grid.shape[0] > 1, "tau_diff should have multiple points"
    # tau_v prior is Uniform(0.0, 3.0)
    assert np.isclose(lut.tau_diff_grid.min(), 0.0), "tau_diff grid minimum should be 0.0"
    assert np.isclose(lut.tau_diff_grid.max(), 3.0), "tau_diff grid maximum should be 3.0"


def test_the_two_component_path_is_unchanged(synthetic_ssp, synthetic_tophat_obs):
    """The four configurations that were already fast must stay fast."""
    model = _two_component(synthetic_ssp, synthetic_tophat_obs)

    assert model._dust_band_response_cache is not None
    assert model._energy_balance_lut_cache is not None
