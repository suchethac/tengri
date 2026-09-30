# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the nebular/shock emission-line LSF kernel (#2519).

The default spectroscopy path used to convolve the whole summed SED
(stellar continuum plus nebular lines) with a single kernel,
sqrt(sigma_v^2 + sigma_inst^2 - sigma_lib(lambda)^2). Nebular and shock
emission are painted at their own intrinsic velocity width
(``neb_eline_sigma_kms``) when the rest-frame SED is built and were never
broadened by the SSP template library, so that kernel over-broadened them
by the stellar velocity dispersion and, wherever the instrument resolved
the library better than the library's own resolution, "deconvolved" a
resolution they never had. ``Observation.predict`` now splits the
additive rest-frame SED (``_split_stellar_and_instrument_only_sed`` in
``src/tengri/observation/observation.py``) and gives the nebular+shock
piece the instrument kernel only.

Measurement note: a real Cue/CloudyGrid spectrum places every one of ~138
lines with the same intrinsic width, and [N II] 6548/6584 sit only
16.6/20.0 A (rest) from H-alpha -- close enough that their own
instrument-broadened wings contaminate a naive windowed second moment
around H-alpha at the percent level. The sigma_v/library-curve
*invariance* tests below are immune to this (the contamination is the
same constant bias at every sigma_v), so they measure directly on a real
predicted spectrum. The absolute sqrt(sigma_gas^2 + sigma_inst^2) formula
check instead isolates H-alpha with a clean, single-line synthetic probe
built from the exact same rendering primitive
(``place_line_profiles_velocity``) the real backends use, put through the
exact same kernel (``apply_lsf`` with ``sigma_lib_kms=0, sigma_v_kms=0``)
``Observation.predict`` now applies to the nebular/shock group.
"""

from __future__ import annotations

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Uniform, load_ssp_data
from tengri.components.nebular._shared import place_line_profiles_velocity
from tengri.observation.observation import _split_stellar_and_instrument_only_sed
from tengri.observation.spectroscopy import Spectroscopy
from tengri.observation.spectrum import apply_lsf, project_spectrum

pytestmark = pytest.mark.regression_bug

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # dimensionless, ~2.3548
_HALPHA_REST = 6564.61
_MILES_BARE = "data/fsps_prsc_miles_chabrier.h5"


def _resolution_for_sigma_inst(sigma_inst_kms: float) -> float:
    """Invert apply_lsf's sigma_inst_kms = C_KM_S / (FWHM_TO_SIGMA * resolution)."""
    return _C_KMS / (_FWHM_TO_SIGMA * sigma_inst_kms)


def _second_moment_kms(wave_obs: np.ndarray, flux: np.ndarray, center: float) -> float:
    """Continuum-subtracted velocity-space second moment about ``center``."""
    f = np.clip(np.asarray(flux) - np.min(flux), 0.0, None)
    lnwave = np.log(np.asarray(wave_obs))
    var = np.sum(f * (lnwave - np.log(center)) ** 2) / np.sum(f)
    return float(np.sqrt(var) * _C_KMS)


def test_split_conserves_total_sed():
    """_split_stellar_and_instrument_only_sed's two pieces always sum to the input.

    The stellar piece is defined as the residual (``sed_spec -
    sed_instrument_only``), so this holds by construction for any
    ``state.derived`` content -- verified here for a representative case
    (#2519).
    """
    from tengri.protocols.component import ForwardState

    n = 32
    wave = jnp.linspace(3000.0, 9000.0, n)
    sed_total = jnp.full(n, 5.0)
    nebular = jnp.full(n, 1.5)
    shock = jnp.full(n, 0.5)
    fake_state = ForwardState(
        wave=wave,
        sed_intrinsic=sed_total,
        derived={"sed_nebular": nebular, "sed_shock": shock},
    )
    igm_trans = jnp.linspace(0.8, 1.0, n)
    sed_spec = sed_total * igm_trans
    sed_stellar, sed_instrument_only = _split_stellar_and_instrument_only_sed(
        fake_state, sed_spec, igm_trans
    )
    chex.assert_trees_all_close(sed_stellar + sed_instrument_only, sed_spec, atol=1e-12)
    chex.assert_trees_all_close(sed_instrument_only, (nebular + shock) * igm_trans, atol=1e-12)


def test_halpha_kernel_matches_instrument_only_formula():
    """H-alpha's width equals sqrt(sigma_gas^2 + sigma_inst^2), the instrument
    kernel alone -- independent of sigma_v and sigma_lib, which never reach it.

    Isolated single-line probe (see module docstring); the kernel applied is
    exactly ``apply_lsf(..., sigma_lib_kms=0.0, sigma_v_kms=0.0)``, the call
    ``Observation.predict`` makes for the nebular/shock group regardless of
    the model's actual sigma_v or library curve (#2519).
    """
    wave_rest = jnp.linspace(6500.0, 6630.0, 20000)
    sigma_gas = 100.0
    line_sed = place_line_profiles_velocity(
        jnp.array([_HALPHA_REST]), jnp.array([1e40]), wave_rest, sigma_gas
    )
    for z in (0.0, 0.05):
        wave_obs = wave_rest * (1.0 + z)
        center_obs = _HALPHA_REST * (1.0 + z)
        for sigma_inst_kms in (50.0, 150.0):
            resolution = _resolution_for_sigma_inst(sigma_inst_kms)
            out = apply_lsf(
                line_sed, wave_obs, resolution=resolution, sigma_lib_kms=0.0, sigma_v_kms=0.0
            )
            measured = _second_moment_kms(wave_obs, out, center_obs)
            expected = float(np.sqrt(sigma_gas**2 + sigma_inst_kms**2))
            assert measured == pytest.approx(expected, rel=2e-2), (z, sigma_inst_kms)


def _ssp_or_skip(path):
    from pathlib import Path

    if not Path(path).is_file():
        pytest.skip(f"missing SSP grid {path}")
    return load_ssp_data(path)


def _cue_or_skip():
    from pathlib import Path

    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _cloudy_grid_or_skip():
    from pathlib import Path

    path = "data/cloudy_grid_prsc.h5"
    if not Path(path).is_file():
        pytest.skip(f"CloudyGrid data ({path}) not present")
    return path


_NEB_BACKENDS = {
    "cue": (lambda: {"type": "cue", "all_params": Fixed(DEFAULT)}, _cue_or_skip),
    "cloudy": (
        lambda: {"type": "cloudy", "grid": _cloudy_grid_or_skip(), "all_params": Fixed(DEFAULT)},
        lambda: None,
    ),
}


def _build_model_free_sigma_v(z, wave_obs, resolution, ssp, neb):
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=resolution, sigma_lib_kms=70.0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb=neb,
            redshift=Fixed(z),
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        model = SEDModel(merged, ssp, observation=obs)
    return model


@pytest.mark.parametrize("neb_backend", ["cue", "cloudy"])
@pytest.mark.parametrize("sigma_inst_kms", [50.0, 150.0])
def test_halpha_width_invariant_to_sigma_v_and_library_curve(sigma_inst_kms, neb_backend):
    """On a real Cue- or CloudyGrid-backend spectrum, H-alpha's measured width
    does not move with sigma_v or with the library curve (only the stellar
    continuum does).

    Contamination from nearby lines ([N II] 6548/6584) is a fixed additive
    bias at every sigma_v/library setting (#2519 module docstring), so
    comparing measurements against EACH OTHER, rather than against an
    absolute formula, is immune to it: any residual sigma_v/library
    dependence left by a regression would show up as spread across this
    parametrization.
    """
    neb_factory, skip_check = _NEB_BACKENDS[neb_backend]
    skip_check()
    ssp = _ssp_or_skip(_MILES_BARE)
    neb = neb_factory()
    z = 0.05
    halpha_obs = _HALPHA_REST * (1.0 + z)
    wave_obs = jnp.linspace(halpha_obs - 9.0, halpha_obs + 9.0, 2000)
    resolution = _resolution_for_sigma_inst(sigma_inst_kms)
    model = _build_model_free_sigma_v(z, wave_obs, resolution, ssp, neb)
    p0 = dict(model.spec.sample(jax.random.PRNGKey(1)))
    p0["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p0["dust_tau_bc"] = jnp.asarray(0.0)
    p0["dust_tau_diff"] = jnp.asarray(0.0)

    widths = {}
    for sigma_v in (0.0, 100.0, 300.0):
        p = dict(p0)
        p["sigma_v_kms"] = jnp.asarray(sigma_v)
        flux = np.asarray(model.predict_spectrum(p))
        widths[sigma_v] = _second_moment_kms(np.asarray(wave_obs), flux, halpha_obs)

    mean_width = float(np.mean(list(widths.values())))
    for sigma_v, w in widths.items():
        assert w == pytest.approx(mean_width, rel=2e-2), (sigma_v, widths)

    # Library curve: MILES curve vs. the curve stripped away (flat sigma_lib
    # fallback) must also leave H-alpha's width unchanged -- the whole point
    # of #2519 is that sigma_lib never reaches the line kernel either way.
    model_flat = SEDModel(
        model.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0)),
        ssp._replace(ssp_resolution_kms=None),
        observation=model.observation,
    )
    flux_flat = np.asarray(model_flat.predict_spectrum({**p0, "sigma_v_kms": jnp.asarray(0.0)}))
    width_flat = _second_moment_kms(np.asarray(wave_obs), flux_flat, halpha_obs)
    assert width_flat == pytest.approx(widths[0.0], rel=2e-2)


def test_stellar_feature_still_gets_full_kernel_sigma_v_changes_it():
    """A stellar absorption feature's width DOES move with sigma_v -- the
    complement of the H-alpha invariance above, confirming the split routes
    the stellar continuum through the unchanged full kernel (#2519).

    Synthetic probe, mirroring
    ``test_ssp_resolution_wiring.test_apply_lsf_reproduces_fsps_kernel_width``:
    ``apply_lsf`` is exactly what ``_split_stellar_and_instrument_only_sed``'s
    stellar piece is convolved with in ``Observation.predict``.
    """
    n = 8192
    wave = jnp.exp(jnp.linspace(np.log(4000.0), np.log(9000.0), n))
    center = 5175.0  # Mg b, a real MILES-range stellar absorption feature
    sigma_lib_kms = 60.0
    lnwave = jnp.log(wave)
    probe = 1.0 - 0.3 * jnp.exp(-0.5 * ((lnwave - jnp.log(center)) / (20.0 / _C_KMS)) ** 2)

    sigma_inst_kms = 150.0
    resolution = _resolution_for_sigma_inst(sigma_inst_kms)

    widths = {}
    for sigma_v in (0.0, 300.0):
        out = apply_lsf(
            probe,
            wave,
            resolution=resolution,
            sigma_lib_kms=sigma_lib_kms,
            sigma_v_kms=sigma_v,
        )
        w = np.clip(1.0 - np.asarray(out), 0.0, None)
        lnw = np.asarray(lnwave)
        mean = np.sum(w * lnw) / np.sum(w)
        var = np.sum(w * (lnw - mean) ** 2) / np.sum(w)
        widths[sigma_v] = float(np.sqrt(var) * _C_KMS)

    expected_0 = np.sqrt(max(sigma_inst_kms**2 - sigma_lib_kms**2, 0.0))
    expected_300 = np.sqrt(max(sigma_inst_kms**2 - sigma_lib_kms**2, 0.0) + 300.0**2)
    assert widths[0.0] < widths[300.0]
    assert widths[0.0] == pytest.approx(expected_0, rel=5e-2)
    assert widths[300.0] == pytest.approx(expected_300, rel=5e-2)


def test_bit_identical_at_zero_sigma_v_and_zero_sigma_lib():
    """sigma_v=0 and sigma_lib=0 must reproduce the pre-#2519 single-kernel
    result exactly: both kernels reduce to sigma_inst alone, so splitting
    the SED and convolving each piece separately is, by linearity of
    convolution, identical to convolving the sum once.
    """
    n = 4096
    wave_rest = jnp.exp(jnp.linspace(jnp.log(3000.0), jnp.log(9000.0), n))
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    sed_stellar = 1.0 + 0.1 * jax.random.normal(k1, (n,))
    sed_instrument_only = 0.2 + 0.05 * jax.random.normal(k2, (n,))
    sed_total = sed_stellar + sed_instrument_only
    z = 0.1
    wave_obs = wave_rest * (1.0 + z)
    dl_cm = jnp.asarray(1.0e27)
    resolution = 2000.0

    combined = project_spectrum(
        sed_total,
        wave_rest,
        wave_obs,
        z,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=0.0,
        sigma_v_kms=0.0,
        cal_coeffs=None,
    )
    flux_stellar = project_spectrum(
        sed_stellar,
        wave_rest,
        wave_obs,
        z,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=0.0,
        sigma_v_kms=0.0,
        cal_coeffs=None,
    )
    flux_instrument_only = project_spectrum(
        sed_instrument_only,
        wave_rest,
        wave_obs,
        z,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=0.0,
        sigma_v_kms=0.0,
        cal_coeffs=None,
    )
    split_sum = flux_stellar + flux_instrument_only
    chex.assert_trees_all_close(split_sum, combined, atol=1e-10, rtol=1e-10)


def test_integrated_halpha_flux_unchanged_across_sigma_v():
    """The split changes H-alpha's shape, not its integrated flux: Gaussian
    convolution with any normalized kernel conserves the integral, so the
    windowed flux is unchanged whether sigma_v broadens the (unaffected)
    stellar continuum or not (#2519).
    """
    _cue_or_skip()
    ssp = _ssp_or_skip(_MILES_BARE)
    z = 0.05
    halpha_obs = _HALPHA_REST * (1.0 + z)
    wave_obs = jnp.linspace(halpha_obs - 9.0, halpha_obs + 9.0, 2000)
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    model = _build_model_free_sigma_v(z, wave_obs, 5000.0, ssp, neb)
    p0 = dict(model.spec.sample(jax.random.PRNGKey(2)))
    p0["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p0["dust_tau_bc"] = jnp.asarray(0.0)
    p0["dust_tau_diff"] = jnp.asarray(0.0)

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    fluxes = {}
    for sigma_v in (0.0, 300.0):
        p = dict(p0)
        p["sigma_v_kms"] = jnp.asarray(sigma_v)
        flux = np.asarray(model.predict_spectrum(p))
        cont = np.min(flux)
        fluxes[sigma_v] = trapz(flux - cont, np.asarray(wave_obs))

    assert fluxes[300.0] == pytest.approx(fluxes[0.0], rel=1e-4)


def test_mutation_line_routed_back_through_stellar_kernel_would_fail_invariance():
    """Sanity-checks the invariance test's own discriminating power: a
    version of the kernel that gives H-alpha the full (sigma_v-dependent)
    kernel instead of the instrument-only one measurably fails the same
    assertion the real fix passes.

    This is a permanent, fast in-process guard, run every time; the
    orchestrator's own source-level mutation (routing
    ``Observation.predict``'s line group through the stellar kernel and
    re-running the suite) is the authoritative regression check and is done
    by hand, not left running in CI (see the worklog for its verbatim
    FAILED lines).
    """
    wave_rest = jnp.linspace(6500.0, 6630.0, 20000)
    sigma_gas = 100.0
    line_sed = place_line_profiles_velocity(
        jnp.array([_HALPHA_REST]), jnp.array([1e40]), wave_rest, sigma_gas
    )
    sigma_inst_kms = 150.0
    resolution = _resolution_for_sigma_inst(sigma_inst_kms)
    wave_obs = wave_rest  # z=0

    widths = {}
    for sigma_v in (0.0, 300.0):
        # The mutation: apply the FULL (sigma_v-dependent) kernel to the line,
        # exactly what Observation.predict did before #2519.
        out = apply_lsf(
            line_sed, wave_obs, resolution=resolution, sigma_lib_kms=0.0, sigma_v_kms=sigma_v
        )
        widths[sigma_v] = _second_moment_kms(np.asarray(wave_obs), out, _HALPHA_REST)

    # Under the mutation, widths at sigma_v=0 and sigma_v=300 differ by far
    # more than the 2% invariance bar the fixed code satisfies.
    rel_spread = abs(widths[300.0] - widths[0.0]) / widths[0.0]
    assert rel_spread > 0.02, "mutation should fail the invariance bar the fix satisfies"
