# SPDX-License-Identifier: BSD-3-Clause
r"""Spectral resampling is ``"auto"`` by default and every path honors it (#2530).

A pixel records the mean of the spectrum over its edges. Point-sampling the model
at the pixel center equals that mean only when the model is smooth across the
pixel; for a line narrower than the pixel it is wrong by tens of per cent (a
Gaussian of sigma = 1 Angstrom in 2 Angstrom pixels reads +14 % at the center,
and an unresolved line +166 %). ``Spectroscopy(resample="auto")`` integrates
over each pixel where the pixels are wider than the model grid and point-samples
otherwise; it is the default, it is decided in the model's rest frame at the
fixed redshift (the lowest prior redshift when free), and one function decides
for every path that turns a model into pixels.

Expected values below are closed-form (error-function pixel integrals of a
Gaussian line on a power-law continuum, the integral of a power law) or follow
from the decision rule written out independently in numpy.
"""

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, Spectroscopy
from tengri.observation.spectroscopy import Spectroscopy as _Spectroscopy
from tengri.observation.spectrum import compute_spectrum, compute_spectrum_conserving
from tengri.units import lnu_to_fnu

pytestmark = pytest.mark.conservation

KEY = jax.random.PRNGKey(0)
DL = 3.086e19
LAM0, ALPHA = 5000.0, -2.0


def _edges(w):
    return np.concatenate(
        [[w[0] - 0.5 * (w[1] - w[0])], 0.5 * (w[1:] + w[:-1]), [w[-1] + 0.5 * (w[-1] - w[-2])]]
    )


def _powerlaw_integral(a, b):
    return LAM0 / (ALPHA + 1) * ((b / LAM0) ** (ALPHA + 1) - (a / LAM0) ** (ALPHA + 1))


def _gauss_integral(a, b, mu, s, amp):
    return (
        amp
        * s
        * np.sqrt(np.pi / 2)
        * (erf((b - mu) / (np.sqrt(2) * s)) - erf((a - mu) / (np.sqrt(2) * s)))
    )


def _expected_decision(wave_obs, wave_model, z):
    """Independent statement of the rule: some pixel is wider than the model interval under it."""
    w = np.asarray(wave_obs) / (1.0 + z)
    e = _edges(w)
    widths = np.diff(e)
    for center, width in zip(w, widths):
        k = int(np.searchsorted(wave_model, center, side="right")) - 1
        if 0 <= k < len(wave_model) - 1 and width > wave_model[k + 1] - wave_model[k]:
            return True
    return False


# ── 1. the default and the closed-form pixel integrals ─────────────────────────


def test_default_resample_is_auto():
    """The default mode is ``"auto"`` (#2530)."""
    assert Spectroscopy(wave_obs=jnp.linspace(4000.0, 8000.0, 100)).resample == "auto"


@pytest.mark.parametrize("z", [0.0, 2.0])
@pytest.mark.parametrize(
    "pix, sigma_rest",
    [(2.0, 1.0), (2.0, 0.3), (100.0, 40.0), (100.0, 1.0)],
    ids=["2A-resolved", "2A-narrow", "prism-resolved", "prism-unresolved"],
)
def test_pixel_integral_of_a_gaussian_line_on_a_power_law(pix, sigma_rest, z):
    """The bin integral reproduces the erf pixel integral; point sampling does not.

    A Gaussian line of known width on a power-law continuum has the mean over
    each pixel in closed form. The model is evaluated on a grid fine enough that
    its own linear interpolation is not the error. Conserving agrees to 1e-3 of
    the line peak and recovers the line flux to 1e-3 at every pixel phase; a
    line narrower than the pixel is mis-measured by tens of per cent when
    point-sampled.
    """
    mu_obs = 6000.0 if pix < 10 else 20000.0
    mu = mu_obs / (1.0 + z)
    amp = 20.0
    scale = float(lnu_to_fnu(1.0, DL, z))
    worst_point, worst_cons, flux_cons, flux_point = 0.0, 0.0, [], []
    for phase in np.linspace(-0.5, 0.5, 9):
        center = mu_obs + phase * pix
        wave = center + pix * np.arange(-8, 9)
        wr = np.arange(
            mu - 40 * sigma_rest - 12 * pix, mu + 40 * sigma_rest + 12 * pix, sigma_rest / 25.0
        )
        sed = (wr / LAM0) ** ALPHA + amp * np.exp(-0.5 * ((wr - mu) / sigma_rest) ** 2)
        e = _edges(wave) / (1.0 + z)
        truth = (
            _powerlaw_integral(e[:-1], e[1:]) + _gauss_integral(e[:-1], e[1:], mu, sigma_rest, amp)
        ) / np.diff(e)
        cont = (_powerlaw_integral(e[:-1], e[1:])) / np.diff(e)
        got = {}
        for name, fn in (("point", compute_spectrum), ("cons", compute_spectrum_conserving)):
            got[name] = (
                np.asarray(fn(jnp.asarray(sed), jnp.asarray(wr), jnp.asarray(wave), z, DL)) / scale
            )
        inner = slice(2, -2)
        peak = truth[inner].max()
        worst_point = max(worst_point, np.max(np.abs(got["point"] - truth)[inner]) / peak)
        worst_cons = max(worst_cons, np.max(np.abs(got["cons"] - truth)[inner]) / peak)
        line_true = np.sum((truth - cont)[inner] * np.diff(e)[inner])
        flux_cons.append(np.sum((got["cons"] - cont)[inner] * np.diff(e)[inner]) / line_true)
        flux_point.append(np.sum((got["point"] - cont)[inner] * np.diff(e)[inner]) / line_true)
    assert worst_cons < 1e-3, f"pixel integral off the closed form by {worst_cons:.2e} of the peak"
    np.testing.assert_allclose(flux_cons, 1.0, atol=1e-3)
    if sigma_rest / (pix / (1.0 + z)) < 1.0:
        assert worst_point > 0.05, (
            "point sampling was expected to fail on a line narrower than a pixel"
        )


@pytest.mark.parametrize("pix", [2.0, 100.0])
def test_flux_conservation_of_a_power_law(pix):
    """Sum of F_lambda x pixel width equals the integral of the power law over the pixel span."""
    wave = np.arange(4000.0, 4000.0 + 400 * pix, pix)
    wr = np.geomspace(wave[0] * 0.9, wave[-1] * 1.1, 200000)
    sed = (wr / LAM0) ** ALPHA
    scale = float(lnu_to_fnu(1.0, DL, 0.0))
    got = (
        np.asarray(
            compute_spectrum_conserving(
                jnp.asarray(sed), jnp.asarray(wr), jnp.asarray(wave), 0.0, DL
            )
        )
        / scale
    )
    e = _edges(wave)
    np.testing.assert_allclose(
        np.sum(got * np.diff(e)), _powerlaw_integral(e[0], e[-1]), rtol=1e-8
    )


def test_finer_pixels_than_the_model_agree_with_point_sampling():
    """Where pixels are finer than the model, the integral and the point sample coincide.

    A smooth line resolved by the model (sigma = 6 Angstrom, 1 Angstrom nodes)
    sampled by 0.3 Angstrom pixels: ``"auto"`` resolves to point sampling and
    the bin integral differs from it by less than 1e-3 of the peak, the
    curvature of the model across a pixel.
    """
    wr = np.arange(4900.0, 5100.0, 1.0)
    sed = np.exp(-0.5 * ((wr - 5000.0) / 6.0) ** 2) + 0.1
    wave = np.arange(4980.0, 5020.0, 0.3)
    assert Spectroscopy(wave_obs=jnp.asarray(wave)).resolve_conserving(wr) is False
    a = compute_spectrum(jnp.asarray(sed), jnp.asarray(wr), jnp.asarray(wave), 0.0, DL)
    b = compute_spectrum_conserving(jnp.asarray(sed), jnp.asarray(wr), jnp.asarray(wave), 0.0, DL)
    assert float(jnp.max(jnp.abs(a - b)) / jnp.max(a)) < 1e-3


# ── 2. the decision ────────────────────────────────────────────────────────────


def test_decision_is_per_region_not_a_median():
    """A coarse-pixel region is detected although most of the grid is finer.

    The model is 0.5 Angstrom spaced over 4000-4100 and 10 Angstrom spaced over
    4100-9000 (the median spacing is 10); 2 Angstrom pixels are four times
    wider than the model in the first region, where a point sample misses
    structure, and finer than it everywhere else.
    """
    wr = np.concatenate([np.arange(4000.0, 4100.0, 0.5), np.arange(4100.0, 9000.0, 10.0)])
    wave = np.arange(4000.0, 9000.0, 2.0)
    assert np.median(np.diff(wave)) < np.median(np.diff(wr))
    assert Spectroscopy(wave_obs=jnp.asarray(wave), resample="auto").resolve_conserving(wr) is True
    only_coarse = np.arange(4100.0, 9000.0, 2.0)
    assert (
        Spectroscopy(wave_obs=jnp.asarray(only_coarse), resample="auto").resolve_conserving(wr)
        is False
    )


def test_decision_is_made_in_the_rest_frame():
    """The same observed pixels are coarse at z = 0 and finer than the model at z = 6.

    2 Angstrom observed pixels against a 0.9 Angstrom model: 2 Angstrom rest
    pixels at z = 0, 0.29 Angstrom at z = 6.
    """
    wr = np.arange(3000.0, 9000.0, 0.9)
    spec = Spectroscopy(wave_obs=jnp.asarray(np.arange(4000.0, 6000.0, 2.0)), resample="auto")
    assert spec.resolve_conserving(wr, 0.0) is True
    assert spec.resolve_conserving(wr, 6.0) is False


def test_explicit_modes_override_the_grid_ratio():
    """``"point"`` and ``"conserving"`` are honored whatever the grids are."""
    wr = np.arange(3000.0, 9000.0, 0.9)
    wo = jnp.asarray(np.arange(4000.0, 6000.0, 0.3))
    assert Spectroscopy(wave_obs=wo, resample="conserving").resolve_conserving(wr) is True
    assert Spectroscopy(wave_obs=wo, resample="point").resolve_conserving(wr) is False


# ── 2b. the pixel integral is taken after the line-spread function ─────────────


@pytest.mark.parametrize(
    "pix_grid, mu, sigma_int, resolution, tol",
    [
        ("2A", 6000.0, 0.5, 6000.0 / (2.3548 * 1.0), 1e-3),
        ("2A", 6000.0, 0.5, 6000.0 / (2.3548 * 3.0), 1e-3),
        ("prism", 20000.0, 1.0, 100.0, 4e-3),
    ],
    ids=["2A-sigma_lsf-1A", "2A-sigma_lsf-3A", "prism-R100"],
)
def test_pixel_integral_follows_the_line_spread_function(pix_grid, mu, sigma_int, resolution, tol):
    """A pixel records the mean of the LSF-broadened light, not a broadened pixel mean.

    A narrow Gaussian line passes through a constant-R kernel (Gaussian in
    ln lambda, width sqrt(sigma_int^2 + sigma_lsf^2)); the expected pixel mean
    integrates that profile over the pixel edges. Broadening the pixel means
    instead is off by 3 % (2 Angstrom pixels) and 28 % (prism) of the peak; the
    remaining differences are the piecewise-constant kernel width of the
    variable-width convolution (``n_bins``) and its dependence on wavelength.
    """
    from tengri.observation.spectrum import project_spectrum

    wave = CASES[pix_grid][0]
    s_ln = np.hypot(sigma_int / mu, 1.0 / (2.3548 * resolution))
    flux = sigma_int * np.sqrt(2.0 * np.pi)
    half = 12.0 * s_ln * mu + 400.0
    wr = np.arange(mu - half, mu + half, min(sigma_int, s_ln * mu) / 20.0)
    sed = np.exp(-0.5 * ((wr - mu) / sigma_int) ** 2)

    def profile(lam):
        return (
            flux
            / (lam * s_ln * np.sqrt(2.0 * np.pi))
            * np.exp(-(np.log(lam / mu) ** 2) / (2 * s_ln**2))
        )

    e = _edges(wave)
    truth = np.array(
        [
            np.trapezoid(profile(g), g) / (b - a)
            for a, b in itertools.pairwise(e)
            for g in [np.linspace(a, b, 400)]
        ]
    )
    scale = float(lnu_to_fnu(1.0, DL, 0.0))
    got = (
        np.asarray(
            project_spectrum(
                jnp.asarray(sed),
                jnp.asarray(wr),
                jnp.asarray(wave),
                0.0,
                DL,
                resolution=resolution,
                sigma_lib_kms=0.0,
                n_bins=64,
                conserving=True,
            )
        )
        / scale
    )
    near = np.abs(wave - mu) < 5 * s_ln * mu + 2 * np.median(np.diff(wave))
    assert np.max(np.abs(got - truth)[near]) / truth.max() < tol


def test_lsf_scale_gradient_is_live_under_the_pixel_mean():
    """The pixel mean responds to ``lsf_scale`` and its gradient matches finite differences.

    R = 800 gives sigma_inst = 159 km/s, above the 70 km/s MILES library width,
    so the kernel width sqrt(sigma_inst^2 - sigma_lib^2) is nonzero; at an R
    where sigma_inst < sigma_lib the clamp makes the gradient identically zero.
    """
    from tengri.observation.spectrum import project_spectrum

    wr = np.arange(5900.0, 6100.0, 0.2)
    sed = jnp.asarray(np.exp(-0.5 * ((wr - 6000.0) / 1.5) ** 2) + 0.2)
    wave = jnp.asarray(np.arange(5950.0, 6050.0, 2.0))

    def peak(scale):
        flux = project_spectrum(
            sed,
            jnp.asarray(wr),
            wave,
            0.0,
            DL,
            resolution=800.0 / scale,
            sigma_lib_kms=70.0,
            conserving=True,
        )
        return jnp.max(flux) / lnu_to_fnu(1.0, DL, 0.0)

    g = float(jax.grad(peak)(1.0))
    fd = float((peak(1.0 + 1e-4) - peak(1.0 - 1e-4)) / 2e-4)
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-3)


# ── 3. every path reaches the same decision, and the same spectrum ─────────────


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("prsc_miles_chabrier_wNE")


def _prism_wave(lo=6000.0, hi=53000.0, resolution=100.0):
    w = [lo]
    while w[-1] < hi:
        w.append(w[-1] * (1.0 + 0.5 / resolution))
    return np.array(w)


CASES = {"2A": (np.arange(3800.0, 9200.0, 2.0), 2000.0), "prism": (_prism_wave(), 100.0)}


def _model(ssp, case, z, resample="auto"):
    wave, resolution = CASES[case]
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(
            photometry=Photometry.from_names(["des_g"]),
            spectroscopy=Spectroscopy(
                wave_obs=jnp.asarray(wave), resolution=resolution, resample=resample
            ),
        ),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
        },
        redshift=Fixed(z),
    )


@pytest.fixture
def decisions(monkeypatch):
    """Record every decision ``Spectroscopy.resolve_conserving`` returns."""
    log = []
    original = _Spectroscopy.resolve_conserving

    def recording(self, wave_rest_model, z_ref=0.0, *args, **kwargs):
        out = original(self, wave_rest_model, z_ref, *args, **kwargs)
        log.append(bool(out))
        return out

    monkeypatch.setattr(_Spectroscopy, "resolve_conserving", recording)
    return log


@pytest.mark.parametrize("z", [0.5, 2.0, 6.0])
@pytest.mark.parametrize("case", ["2A", "prism"])
def test_every_path_reaches_the_decision_of_the_rest_frame_rule(ssp, decisions, case, z):
    """The compiled kernel, ``predict_spectrum`` on a grid, ``observe_spectrum`` and
    ``Prediction.spectrum`` all resolve ``"auto"`` to the rule written out in the
    rest frame at the model's redshift.
    """
    m = _model(ssp, case, z)
    wave, _ = CASES[case]
    expected = _expected_decision(wave, np.asarray(m.wavelengths), z)
    p = dict(m.spec.sample(KEY))

    paths = {}
    decisions.clear()
    m.predict_observables(p)
    paths["Observation.predict"] = list(decisions)
    decisions.clear()
    m.predict_spectrum(p, wave_obs=jnp.asarray(wave))
    paths["predict_spectrum(wave_obs)"] = list(decisions)
    decisions.clear()
    m.predict(p).spectrum()
    paths["Prediction.spectrum"] = list(decisions)
    decisions.clear()
    result = SimpleNamespace(
        wavelength=jnp.asarray(m.wavelengths) * (1.0 + z), sed=jnp.asarray(m.predict(p).rest_sed())
    )
    m.observation.observe_spectrum(result, z, DL)
    paths["observe_spectrum"] = list(decisions)

    for name, got in paths.items():
        assert got and set(got) == {expected}, (
            f"{name} resolved {got}, rest-frame rule gives {expected}"
        )


@pytest.mark.parametrize("case", ["2A", "prism"])
def test_spectrum_paths_agree_when_exact(ssp, case):
    """The configured-grid kernel, the on-grid projector and ``Prediction.spectrum`` agree."""
    z = 0.5
    m = _model(ssp, case, z)
    wave, _ = CASES[case]
    p = dict(m.spec.sample(KEY))
    ref = np.asarray(m.predict_spectrum(p, wave_obs=jnp.asarray(wave)))
    # The paths pad the LSF from different static knowledge (#2832). The explicit grid
    # has a concrete sigma_v = 0 and pads by 256 px; the closure traces sigma_v and
    # lsf_scale, bounds each at 2000 km/s, and pads by 357 px. Both paddings cover the
    # kernel, and they differ by the sub-pixel ringing tail beyond 256 px, measured at
    # 1.1e-8 of peak, so agreement is held to 1e-7 rather than the 1e-10 of the rest.
    for other in (m.predict_spectrum(p), m.predict(p).spectrum()):
        np.testing.assert_allclose(np.asarray(other), ref, rtol=1e-7, atol=1e-7 * ref.max())


def test_auto_equals_conserving_on_coarse_pixels_and_differs_from_point(ssp):
    """On 2 Angstrom pixels at z = 0.5 ``"auto"`` is the pixel integral, not the point sample."""
    p = dict(_model(ssp, "2A", 0.5).spec.sample(KEY))
    a, c, pt = (
        np.asarray(_model(ssp, "2A", 0.5, r).predict_spectrum(p))
        for r in ("auto", "conserving", "point")
    )
    np.testing.assert_allclose(a, c, rtol=1e-12)
    assert np.max(np.abs(c - pt)) / c.max() > 1e-3


def test_sfh_spectrum_honors_resample(ssp):
    """``spectrum_from_sfh`` integrates 2 Angstrom pixels under auto, samples under point."""
    from tengri.analysis.simulate import sed_from_sfh, spectrum_from_sfh
    from tengri.cosmology import luminosity_distance

    z = 0.5
    wave = jnp.asarray(CASES["2A"][0])
    t, sfr = jnp.linspace(0.01, 13.7, 60), jnp.ones(60) * 5.0
    sed = sed_from_sfh(t, sfr, ssp)
    dl = luminosity_distance(z)
    scale = float(lnu_to_fnu(1.0, dl, z))
    wr = np.asarray(sed["wavelength"])
    s = np.asarray(sed["sed"])
    # Integral of the piecewise-linear model over each pixel, written independently.
    e = _edges(np.asarray(wave)) / (1.0 + z)
    dense = np.concatenate([np.linspace(a, b, 40) for a, b in itertools.pairwise(e)]).reshape(
        len(wave), 40
    )
    ref = np.trapezoid(np.interp(dense, wr, s), dense, axis=1) / np.diff(e) * scale
    auto = np.asarray(spectrum_from_sfh(t, sfr, ssp, wave, redshift=z)["flux"])
    point = np.asarray(spectrum_from_sfh(t, sfr, ssp, wave, redshift=z, resample="point")["flux"])
    np.testing.assert_allclose(auto, ref, rtol=2e-3, atol=1e-3 * ref.max())
    assert np.max(np.abs(point - ref)) / ref.max() > 5e-3


def test_spectrum_lut_refuses_a_pixel_integral(ssp):
    """The spectrum LUT samples at pixel centers; pixels wider than the model must be refused."""
    from tengri import SpectrumPrecomp

    wave, resolution = CASES["2A"]
    kwargs = dict(
        ssp_data=ssp,
        sfh={"type": "dpl", "all_params": FREE},
        redshift=Fixed(0.5),
        approx=SpectrumPrecomp(),
    )

    def obs(resample):
        return Observation(
            photometry=Photometry.from_names(["des_g"]),
            spectroscopy=Spectroscopy(
                wave_obs=jnp.asarray(wave), resolution=resolution, resample=resample
            ),
        )

    with pytest.raises(ValueError, match=r"approx=None"):
        SEDModel.build(observation=obs("auto"), **kwargs)
    SEDModel.build(observation=obs("point"), **kwargs)


def test_joint_photometry_lut_keeps_coarse_spectrum_exact(ssp):
    """A photometry LUT on a joint model does not promote a coarse-pixel spectrum onto the LUT."""
    from tengri import WavePrecomp

    wave, resolution = CASES["2A"]
    m = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(
            photometry=Photometry.from_names(["des_g"]),
            spectroscopy=Spectroscopy(wave_obs=jnp.asarray(wave), resolution=resolution),
        ),
        sfh={"type": "dpl", "all_params": FREE},
        redshift=Fixed(0.5),
        approx=WavePrecomp(),
    )
    assert m._approx.get("wave_precomp") and not m._approx.get("spectrum_precomp")


# ── 4. smoothness in redshift ──────────────────────────────────────────────────


def test_pixel_flux_gradient_in_redshift_has_no_steps():
    """The redshift gradient of a bin-integrated pixel is continuous across model nodes.

    The integral of a piecewise-linear model has a slope equal to the model value
    at the edge, which is continuous; a linear interpolation of node integrals
    gives a slope that jumps at every node. The scan moves the pixel edges across
    several 1 Angstrom nodes of a smooth spectrum and bounds the step between
    neighboring gradient values by 2 % of the gradient range.
    """
    wr = np.arange(5000.0, 6000.0, 1.0)
    sed = jnp.asarray(1.0 + 0.5 * np.sin(2 * np.pi * wr / 7.3))
    wave = jnp.asarray(np.array([5497.0, 5499.0, 5501.0, 5503.0, 5505.0]))
    wrj = jnp.asarray(wr)

    def flux(z):
        return compute_spectrum_conserving(sed, wrj, wave, z, DL)[2] / lnu_to_fnu(1.0, DL, 0.0)

    zs = jnp.asarray(np.linspace(0.0, 2 * 2.0 / 5500.0, 201))
    _, grad = jax.vmap(jax.value_and_grad(flux))(zs)
    grad = np.asarray(grad)
    eps = 1e-8
    fd = np.asarray((jax.vmap(flux)(zs + eps) - jax.vmap(flux)(zs - eps)) / (2 * eps))
    assert np.max(np.abs(grad - fd)) < 1e-4 * np.max(np.abs(grad))
    assert np.max(np.abs(np.diff(grad))) < 0.02 * (grad.max() - grad.min())
