# SPDX-License-Identifier: BSD-3-Clause
"""Censored limits are scored as limits on every scoring path (#2665, #2666, #2667).

An upper limit contributes ``ln Phi((F_lim - F_model) / sigma)`` and a lower
limit ``ln Phi((F_model - F_lim) / sigma)``. Three paths scored a limit as a
Gaussian detection at the limit value, or clamped the probability at 1e-30 so
the gradient vanished for a strongly violated limit:

* ``LineFluxData.log_likelihood`` / ``.chi2`` (#2665),
* the inlined loss branch for ``data_mask`` + spectroscopy/joint data, which
  never read ``line_flux_limit_mask`` (#2666),
* the geoVI / MGVI energies of ``jit_engine`` and ``native`` VI, which never
  read ``data_mask``; the NIFTy engines cannot represent censoring (#2667).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.noise import DETECTED, censored_neg_log_likelihood

pytestmark = pytest.mark.regression_bug

SIGMA = 0.1
LIMIT = 1.0
_KINDS = ("upper", "lower")


def _limit_data(kind: str) -> LineFluxData:
    """Two lines, the second a limit of the given kind; the first a detection."""
    flags = jnp.array([False, True])
    return LineFluxData(
        names=("Halpha", "Hbeta"),
        fluxes=jnp.array([LIMIT, LIMIT]),
        errors=jnp.array([SIGMA, SIGMA]),
        wavelengths=jnp.array([6564.61, 4862.68]),
        is_upper_limit=flags if kind == "upper" else None,
        is_lower_limit=flags if kind == "lower" else None,
    )


def _model_at_z(kind: str, z: float) -> jnp.ndarray:
    """Model fluxes putting the limit line at censored z-score ``z``."""
    shift = z * SIGMA
    limit_line = LIMIT - shift if kind == "upper" else LIMIT + shift
    return jnp.array([LIMIT, limit_line])


# ---------------------------------------------------------------- #2665


@pytest.mark.parametrize("kind", _KINDS)
def test_satisfied_limit_costs_nothing(kind):
    """A limit satisfied by 15 sigma adds ln Phi(15) = 0 to the detection term."""
    d = _limit_data(kind)
    ll = float(d.log_likelihood(_model_at_z(kind, 15.0)))
    detection = -0.5 * 0.0 - np.log(SIGMA) - 0.5 * np.log(2 * np.pi)
    assert ll == pytest.approx(detection, abs=1e-12)


@pytest.mark.parametrize("z", [-15.0, -30.0])
@pytest.mark.parametrize("kind", _KINDS)
def test_violated_limit_is_ln_phi_with_live_gradient(kind, z):
    """A limit violated by |z| sigma scores ln Phi(z) ~ -z^2/2 and keeps its gradient.

    The old probability clamp at 1e-30 floored the term near -69 and zeroed
    the gradient beyond z ~ -11.4, so the optimizer saw a flat plateau.
    ``d ln Phi / dz = phi(z) / Phi(z)`` ~ |z|, pulling the model back.
    """
    d = _limit_data(kind)
    model = _model_at_z(kind, z)
    detection = -np.log(SIGMA) - 0.5 * np.log(2 * np.pi)
    ll = float(d.log_likelihood(model))
    assert ll - detection == pytest.approx(stats.norm.logcdf(z), rel=1e-9)
    assert (ll - detection) / (-0.5 * z * z) == pytest.approx(1.0, abs=0.04)

    grad = float(jax.grad(d.log_likelihood)(model)[1])
    dlogphi_dz = float(np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z)))
    dz_dmodel = -1.0 / SIGMA if kind == "upper" else 1.0 / SIGMA
    assert grad == pytest.approx(dlogphi_dz * dz_dmodel, rel=1e-6)
    assert abs(grad) > 1.0


@pytest.mark.parametrize("kind", _KINDS)
def test_chi2_of_a_limit_is_twice_the_censored_energy(kind):
    """chi2 of a limit line is -2 ln Phi(z): the photometry path's 2 x energy, not dropped."""
    d = _limit_data(kind)
    z = -2.0
    model = _model_at_z(kind, z)
    energy = float(
        censored_neg_log_likelihood(
            d.fluxes[1:], d.errors[1:], model[1:], jnp.array([1 if kind == "upper" else -1])
        )
    )
    assert float(d.chi2(model)) == pytest.approx(2.0 * energy, rel=1e-12)
    assert 2.0 * energy == pytest.approx(-2.0 * stats.norm.logcdf(z), rel=1e-12)


def test_detection_only_keeps_full_gaussian_normalization():
    """Without limits, ln L keeps -ln sigma - 0.5 ln(2 pi) per line (evidence stays normalized)."""
    d = LineFluxData(
        names=("Halpha", "Hbeta"),
        fluxes=jnp.array([1.0, 2.0]),
        errors=jnp.array([0.1, 0.3]),
        wavelengths=jnp.array([6564.61, 4862.68]),
    )
    model = jnp.array([1.05, 1.7])
    r = (np.asarray(d.fluxes) - np.asarray(model)) / np.asarray(d.errors)
    expected = np.sum(-0.5 * r**2 - np.log(np.asarray(d.errors)) - 0.5 * np.log(2 * np.pi))
    same_expression = jnp.sum(
        -0.5 * ((d.fluxes - model) / d.errors) ** 2
        - jnp.log(d.errors)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    assert float(d.log_likelihood(model)) == float(same_expression)
    assert float(d.log_likelihood(model)) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- #2666 / #2667 models

FILTERS = ["sdss_g", "sdss_r", "sdss_i"]
WAVE_OBS = np.linspace(4000.0, 7000.0, 40)


@pytest.fixture(scope="module")
def joint_line_model():
    """A joint photometry + spectroscopy model with an H-alpha line-flux channel."""
    import tengri
    from tengri import (
        DEFAULT,
        Fixed,
        ForwardModel,
        Observation,
        Photometry,
        SEDModel,
        Spectroscopy,
        Uniform,
    )
    from tengri.observation.line_list import LineList

    lfd = LineFluxData(
        names=("Halpha",),
        wavelengths=jnp.asarray([6564.61]),
        fluxes=jnp.asarray([1.0e-17]),
        errors=jnp.asarray([1.0e-18]),
    )
    obs = Observation(
        photometry=Photometry.from_names(FILTERS),
        spectroscopy=Spectroscopy(wave_obs=WAVE_OBS, resolution=1000.0),
        lines=LineList.from_names(["Halpha"]),
        line_fluxes=lfd,
    )
    sed = SEDModel.build(
        ssp_data=tengri.load_ssp("fsps_prsc_miles_chabrier"),
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    return ForwardModel.build(sed=sed)


@pytest.mark.parametrize("kind", _KINDS)
def test_inlined_loss_branch_scores_line_limits_like_the_single_channel_path(
    joint_line_model, kind
):
    """The data_mask + joint branch scores a line limit with the same ln Phi term.

    Same model, same Halpha limit: the line-channel energy of the inlined
    branch equals ``-CensoredLikelihood.log_prob`` of the single-channel
    (photometry) path. Reading no ``line_flux_limit_mask`` scores the limit as
    a detection at the limit value and misses by many units.
    """
    from tengri import Data
    from tengri.inference.fitter import Fitter
    from tengri.inference.likelihoods import CensoredLikelihood
    from tengri.inference.loss_functions import _build_data_neg_log_likelihood_fn

    flux = np.array([1.0e-28, 1.2e-28, 1.4e-28])
    spec = np.full(WAVE_OBS.shape, 1.0e-28)
    # A limit far below the model so the censored and Gaussian scores differ.
    limit_value, limit_err = (1.0e-19, 1.0e-19) if kind == "upper" else (1.0e-14, 1.0e-16)
    data = Data(
        photometry=(flux, 0.05 * flux),
        spectrum=(spec, 0.05 * spec),
        lines={"Halpha": (limit_value, limit_err, kind)},
        censor=np.zeros(3, dtype=np.int32),
    )
    captured = {}
    import tengri.inference.fitter as fitmod

    real = fitmod.Fitter

    class _Spy(Fitter):
        def __init__(self, model, *a, **kw):
            super().__init__(model, *a, **kw)
            captured["fitter"] = self
            captured["model"] = model
            raise RuntimeError("stop-after-binding")

    fitmod.Fitter = _Spy
    try:
        joint_line_model.fit(data, method="map")
    except RuntimeError as exc:
        if "stop-after-binding" not in str(exc):
            raise
    finally:
        fitmod.Fitter = real
    fitter, model = captured["fitter"], captured["model"]

    args = fitter._build_data_args(model)
    assert "line_flux_limit_mask" in args
    params = {"sfh_delayed_log_total_mass": jnp.asarray(10.0)}
    full = params
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter)

    without = {k: v for k, v in args.items() if k != "line_flux_limit_mask"}
    e_with = float(neg_log_lik(full, args))
    e_without = float(neg_log_lik(full, without))

    model_lf = model.predict_line_fluxes(
        {k: v for k, v in full.items() if k in set(fitter.spec.free_params)},
        target_wavelengths=args["line_flux_waves"],
    )
    ref = CensoredLikelihood(
        obs=args["line_flux_obs"],
        err=args["line_flux_err"],
        mask=args["line_flux_limit_mask"],
        channel="line_fluxes",
    )
    e_ref = -float(ref.log_prob({"line_fluxes": model_lf}))
    chi2_half = 0.5 * float(
        jnp.sum(((args["line_flux_obs"] - model_lf) / args["line_flux_err"]) ** 2)
    )
    # e_without scores the line as a detection (0.5 chi2); the difference is the
    # line channel's censored energy minus that detection chi2.
    assert e_with - e_without == pytest.approx(e_ref - chi2_half, rel=1e-10, abs=1e-10)


# ---------------------------------------------------------------- #2667

LIM_BAND = 2
N_BANDS = 5


def _one_parameter_five_band_fitter(masked: bool):
    """One free parameter (log_total_mass), five 5% bands, band 2 an upper limit."""
    import tengri
    from tengri import (
        DEFAULT,
        Fitter,
        Fixed,
        Observation,
        Photometry,
        SEDModel,
        Uniform,
    )
    from tengri.observation.photometry import FilterCurve

    curves = []
    for i, c in enumerate([3800.0, 4800.0, 6200.0, 7600.0, 9000.0]):
        wv = np.linspace(c - 500, c + 500, 32)
        curves.append(FilterCurve(wave=wv, trans=np.ones_like(wv), name=f"b{i}"))
    obs = Observation(photometry=Photometry(filters=tuple(curves)))
    ssp = tengri.load_ssp()

    def build():
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )

    pred = np.asarray(build().predict_photometry({"sfh_dpl_log_total_mass": 10.5}))
    sig = 0.05 * pred
    data = pred.copy()
    data[LIM_BAND] = 3.0 * pred[LIM_BAND]  # a limit 3 x the model: satisfied at the truth
    mask = np.zeros(N_BANDS, dtype=int)
    mask[LIM_BAND] = 1
    return Fitter(
        build(), data=data, noise=sig, data_type="photometry", data_mask=mask if masked else None
    )


def _mean_log_mass(post) -> float:
    return float(np.mean(np.asarray(post.params["sfh_dpl_log_total_mass"])))


def test_vi_energy_with_a_satisfied_limit_matches_map_to_0p02_dex():
    """Native VI scores the upper limit as ln Phi, so it recovers what MAP recovers.

    Ignoring ``data_mask`` scores band 2 as a detection at 3 x the model and
    pulls log_total_mass to ~10.65 against the 10.50 of the censored fit.
    """
    key = jax.random.PRNGKey(1)
    map_mean = _mean_log_mass(_one_parameter_five_band_fitter(True).run("map", key=key))
    vi_mean = _mean_log_mass(
        _one_parameter_five_band_fitter(True).run(
            "native_vi_linear", key=key, n_iterations=20, allow_unvalidated=True
        )
    )
    assert abs(vi_mean - map_mean) < 0.02, f"native VI {vi_mean:.4f} vs map {map_mean:.4f}"


@pytest.mark.parametrize(
    "method", ["vi", "vi_nonlinear", "vi_nonlinear_fast", "vi_linear", "vi_linear_fast"]
)
def test_nifty_methods_refuse_a_censoring_mask(method):
    """NIFTy cannot represent a limit: raise, naming the methods that can."""
    fitter = _one_parameter_five_band_fitter(True)
    with pytest.raises(ValueError, match=r"(?s)NIFTy.*map.*mcmc_nuts"):
        fitter.run(method, key=jax.random.PRNGKey(1))


def test_nifty_methods_accept_an_all_detected_mask():
    """An all-zero mask censors nothing, so it does not trigger the refusal."""
    from tengri.inference.backends.vi.nifty import check_nifty_supports_data_mask

    check_nifty_supports_data_mask(jnp.zeros(5, dtype=jnp.int32))
    check_nifty_supports_data_mask(None)
    assert DETECTED == 0
