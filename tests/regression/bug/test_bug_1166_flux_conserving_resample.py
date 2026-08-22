# SPDX-License-Identifier: BSD-3-Clause
r"""Point interpolation aliases the model at low resolution (#1166).

`compute_spectrum` resamples the model onto observed pixels by point sampling at
each pixel center (`jnp.interp`). That is unbiased only when the model grid is much
finer than the pixel spacing. For low-resolution spectroscopy (NIRSpec PRISM,
R≲500) a pixel spans one or more model bins, so point sampling **aliases** the
sub-pixel structure and biases the integrated continuum by up to ~0.7% — a
systematic, not noise. `SpectrumPrecomp` inherits the same bias (its build-time
`_vectorized_interp` is also a point sample).

The fix is the flux-conserving bin integral of SpectRes (Carnall 2017), selected via
`Spectroscopy(resample="conserving")` or `"auto"` (which turns it on only when the
observed pixels actually under-sample the model grid). At DESI/SDSS resolution
(R≳2000) point sampling is already unbiased, so the default stays `"point"` and
those results do not move.

The evaluation that motivated this (see the issue thread) found: SpectRes buys **no
speed** here (SpectrumPrecomp already runs the CSP at pixel centers; the model grid
is at the Nyquist floor), and its **noise machinery is irrelevant** — tengri
resamples the *model*, which Carnall explicitly recommends precisely to avoid the
error covariances that resampling *data* would introduce. Only the flux-conserving
weights matter, and only at low R.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Spectroscopy, Uniform
from tengri.observation.spectrum import compute_spectrum, compute_spectrum_conserving
from tengri.units import lnu_to_fnu

pytestmark = pytest.mark.conservation

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def ssp():
    # wNE (nebular-baked): the sharp emission lines are what make point
    # interpolation drift at low R, which is the aliasing bias this test
    # measures. The bare-stellar default has no such lines and the drift falls
    # below the vacuity guard (~0.03%), so the test would be toothless.
    return tengri.load_ssp("prsc_miles_chabrier_wNE")


@pytest.fixture(scope="module")
def rest_sed(ssp):
    """A realistic rest-frame model SED on the native (fine) grid."""
    m = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["des_g"])),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_diff": 0.3,
        },
        redshift=Fixed(0.1),
    )
    pred = m.predict(dict(m.spec.sample(KEY)))
    return np.asarray(m.wavelengths), np.asarray(pred.rest_sed())


def _band_edges(wo):
    e = np.empty(wo.size + 1)
    e[1:-1] = 0.5 * (wo[1:] + wo[:-1])
    e[0] = wo[0] - 0.5 * (wo[1] - wo[0])
    e[-1] = wo[-1] + 0.5 * (wo[-1] - wo[-2])
    return e


_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _integrated_flux_error(wave_rest, sed_rest, wo, resampler):
    """|Σ f_j·Δλ_j / ∫model dλ − 1| over the exact bin coverage, scale-free.

    The reference integrates the (linearly interpolated) model densely over the
    exact bin span [edges[0], edges[-1]] so the comparison is not confused by
    partial edge bins at the model-grid boundary.
    """
    z, dl = 0.0, 3.086e19
    f = np.asarray(
        resampler(jnp.asarray(sed_rest), jnp.asarray(wave_rest), jnp.asarray(wo), z, dl)
    )
    edges = _band_edges(wo)
    binned = np.sum(f * np.diff(edges))
    scale = float(lnu_to_fnu(1.0, dl, z))
    dense = np.linspace(edges[0], edges[-1], 40000)
    true_integral = _trapezoid(np.interp(dense, wave_rest, sed_rest), dense) * scale
    return abs(binned / true_integral - 1.0)


# ── 1. the fix: flux conservation at low resolution ───────────────────────────


def test_conserving_preserves_integrated_flux_at_low_resolution(rest_sed):
    """At R≈150 the flux-conserving resample conserves the band-integrated flux to
    machine precision, while point interpolation drifts by ~1% — the aliasing bias.

    Neuter-check: make ``compute_spectrum_conserving`` fall back to point
    interpolation and the conserving assertion goes red.
    """
    wave_rest, sed_rest = rest_sed
    wo = np.geomspace(4200.0, 8300.0, 200)  # ~R150, coarse vs the ~0.9 Å model grid

    err_point = _integrated_flux_error(wave_rest, sed_rest, wo, compute_spectrum)
    err_cons = _integrated_flux_error(wave_rest, sed_rest, wo, compute_spectrum_conserving)

    assert err_cons < 1e-3, f"flux-conserving resample did not conserve flux: {err_cons:.2%}"
    assert err_point > 5e-3, (
        f"point interp did not drift at low R (test toothless): {err_point:.2%}"
    )
    assert err_cons < err_point, "flux-conserving is not better than point at low R"


def test_point_and_conserving_agree_on_a_fine_grid(rest_sed):
    """At DESI-like sampling (pixels finer than the model grid) the two resamplers
    agree — the reason the default stays point and DESI results do not move."""
    wave_rest, sed_rest = rest_sed
    wo = np.arange(4200.0, 8300.0, 0.8)  # 0.8 Å, finer than the ~0.9 Å model grid
    z, dl = 0.0, 3.086e19
    fp = np.asarray(
        compute_spectrum(jnp.asarray(sed_rest), jnp.asarray(wave_rest), jnp.asarray(wo), z, dl)
    )
    fc = np.asarray(
        compute_spectrum_conserving(
            jnp.asarray(sed_rest), jnp.asarray(wave_rest), jnp.asarray(wo), z, dl
        )
    )
    # Per-pixel differences peak at absorption-line cores even on a fine grid (both
    # resamplers are valid there) — the meaningful agreement is aggregate: small rms
    # and near-perfect integrated flux.
    rel = (fc - fp) / np.where(np.abs(fp) > 0, fp, 1.0)
    assert np.sqrt(np.mean(rel**2)) < 0.01, "rms disagreement too large on a fine grid"
    dlam = np.diff(_band_edges(wo))
    assert abs(np.sum(fc * dlam) / np.sum(fp * dlam) - 1.0) < 1e-3, "integrated flux disagrees"


# ── 2. the auto policy: on only when pixels under-sample the model ─────────────


def test_auto_selects_conserving_only_for_coarse_pixels(rest_sed):
    """``resample='auto'`` must pick the flux-conserving path for coarse (prism)
    pixels and leave fine (DESI) pixels on point interpolation."""
    wave_rest, _ = rest_sed
    prism = Spectroscopy(
        wave_obs=jnp.asarray(np.geomspace(4000.0, 8500.0, 250)), resolution=200.0, resample="auto"
    )
    desi = Spectroscopy(
        wave_obs=jnp.asarray(np.arange(4000.0, 8500.0, 0.8)), resolution=2500.0, resample="auto"
    )
    assert prism.resolve_conserving(wave_rest) is True, "auto missed coarse prism pixels"
    assert desi.resolve_conserving(wave_rest) is False, "auto over-triggered on fine DESI pixels"


def test_explicit_modes_override_the_grid_ratio(rest_sed):
    """Explicit modes are honored regardless of grid ratio; default is point."""
    wave_rest, _ = rest_sed
    wo = jnp.asarray(np.arange(4000.0, 8500.0, 0.8))
    assert Spectroscopy(wave_obs=wo).resample == "point"
    assert Spectroscopy(wave_obs=wo, resample="conserving").resolve_conserving(wave_rest) is True
    assert Spectroscopy(wave_obs=wo, resample="point").resolve_conserving(wave_rest) is False


def test_invalid_resample_mode_raises():
    with pytest.raises(ValueError, match="resample must be one of"):
        Spectroscopy(wave_obs=jnp.linspace(4000.0, 8000.0, 100), resample="bogus")


# ── 3. end-to-end through the model, and JIT/grad safety ──────────────────────


def _model(ssp, resample, wave_obs, z=0.1, free_z=False):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(
            photometry=Photometry.from_names(["des_g"]),
            spectroscopy=Spectroscopy(
                wave_obs=jnp.asarray(wave_obs), resolution=1500.0, resample=resample
            ),
        ),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_diff": 0.3,
        },
        redshift=(Uniform(0.05, 0.5) if free_z else Fixed(z)),
    )


def test_resample_mode_reaches_predict_spectrum(ssp):
    """The three modes must all trace and return finite spectra; on coarse pixels
    conserving must actually change the result vs point (not a silent no-op)."""
    wo = np.geomspace(4000.0, 8500.0, 250)  # coarse
    p = dict(_model(ssp, "point", wo).spec.sample(KEY))
    f_point = np.asarray(_model(ssp, "point", wo).predict_spectrum(p))
    f_cons = np.asarray(_model(ssp, "conserving", wo).predict_spectrum(p))
    f_auto = np.asarray(_model(ssp, "auto", wo).predict_spectrum(p))

    for f in (f_point, f_cons, f_auto):
        assert np.all(np.isfinite(f)) and np.all(f >= 0.0)
    assert np.max(np.abs(f_cons / np.where(f_point > 0, f_point, 1) - 1.0)) > 1e-3, (
        "conserving did not change the coarse-pixel spectrum — the mode is a no-op"
    )
    # auto sees coarse pixels here, so it must match the conserving path exactly.
    np.testing.assert_allclose(f_auto, f_cons, rtol=1e-6)


def test_conserving_models_do_not_collide_in_the_compile_cache(ssp):
    """``resample`` must be part of ``compile_signature`` (#1166).

    The kernel is fetched from a structural cache keyed on ``compile_signature``.
    If ``resample`` is absent, a conserving model built after a point model
    silently reuses the point kernel — the mode works everywhere *except* the
    cached path inference uses. Two models differing only in ``resample`` must
    therefore get distinct signatures.

    Neuter-check: drop ``spec_resample_conserving`` from ``compile_signature`` and
    this (and ``test_resample_mode_reaches_predict_spectrum``) go red.
    """
    wo = np.geomspace(4000.0, 8500.0, 250)
    sig_point = _model(ssp, "point", wo).compile_signature()
    sig_cons = _model(ssp, "conserving", wo).compile_signature()
    assert sig_point != sig_cons, "point and conserving models share a compile signature"


def test_spectrum_precomp_warns_that_it_ignores_conserving(ssp):
    """SpectrumPrecomp point-interpolates its LUT, so pairing it with a conserving
    resample must warn rather than silently drop the request (#1166)."""
    from tengri import SpectrumPrecomp

    wo = np.geomspace(4000.0, 8500.0, 250)  # coarse → auto/conserving would apply
    with pytest.warns(UserWarning, match="does not apply it"):
        SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(["des_g"]),
                spectroscopy=Spectroscopy(
                    wave_obs=jnp.asarray(wo), resolution=300.0, resample="conserving"
                ),
            ),
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": FIXED,
                "tau_diff": 0.3,
            },
            redshift=Fixed(0.1),
            approx=SpectrumPrecomp(),
        )


@pytest.mark.gradient
def test_gradient_through_conserving_is_finite(ssp):
    """The flux-conserving resample must be gradient-safe (it feeds the likelihood)."""
    wo = np.geomspace(4000.0, 8500.0, 200)
    m = _model(ssp, "conserving", wo)
    p = dict(m.spec.sample(KEY))

    def loss(logm):
        return jnp.sum(m.predict_spectrum({**p, "sfh_dpl_log_total_mass": logm}))

    g = float(jax.grad(loss)(10.0))
    assert np.isfinite(g) and g != 0.0
