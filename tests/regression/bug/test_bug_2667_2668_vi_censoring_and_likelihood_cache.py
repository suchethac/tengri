# SPDX-License-Identifier: BSD-3-Clause
"""#2667 and #2668: the variational engines honor ``data_mask`` and own their data.

#2667. ``Fitter(..., data_mask=...)`` flags a photometric limit (``1`` upper,
``-1`` lower, the limit value in ``data``). ``map``, every ``mcmc_*`` sampler
and ``vi_fullrank`` score it with the Gaussian CDF; the NIFTy (``vi*``) and
native (``native_vi_*``) engines built their own ``0.5 chi^2`` from
``(data, noise)`` and scored a limit as a measurement at the limit value, with
no warning. They now score the censored energy of
:func:`tengri.observation.noise.censored_neg_log_likelihood`.

#2668. The NIFTy likelihood was cached on the model object with the data baked
in, so a second ``Fitter`` on the same model returned the first fit's
posterior. The cache now holds only the data-free physics.

Every closed form below is written with ``scipy.stats.norm``; none is read back
from the code under test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import tengri
from tengri import DEFAULT, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.config.exceptions import ParameterError
from tengri.inference._backend_registry import _BACKENDS
from tengri.inference._censoring import VI_METHODS_HONORING_DATA_MASK
from tengri.inference.mass_profile import PROFILE_MASS_BACKENDS
from tengri.observation import NoiseModel
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_NAME = "sfh_dpl_log_total_mass"
_TRUTH = 10.5
_LIM = 2  # the band that carries the limit
_CENTERS = (3800.0, 4800.0, 6200.0, 7600.0, 9000.0)
_KEY = jax.random.PRNGKey(1)


def _build_model(ssp, *, free_noise=False):
    """One free parameter (log total mass), five bands, optionally a free noise floor."""
    curves = []
    for i, c in enumerate(_CENTERS):
        wv = np.linspace(c - 500, c + 500, 32)
        curves.append(FilterCurve(wave=wv, trans=np.ones_like(wv), name=f"b{i}"))
    noise = NoiseModel(calibration_floor=Uniform(0.01, 0.2)) if free_noise else None
    obs = Observation(photometry=Photometry(filters=tuple(curves)), noise=noise)
    sfh = {"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)}
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=sfh,
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def truth(ssp):
    """Model photometry at the truth and 5 % noise (the shared mock)."""
    pred = np.asarray(_build_model(ssp).predict_photometry({_NAME: _TRUTH}))
    return pred, 0.05 * pred


def _fitter(
    ssp, truth, *, scale=1.0, mask_value=None, model=None, free_noise=False, all_bands=False
):
    """Fitter on the mock with band ``_LIM`` (or every band) replaced by ``scale`` x model.

    ``mask_value=None`` passes no ``data_mask``; otherwise band ``_LIM`` carries
    that flag (0 detected, 1 upper limit, -1 lower limit).
    """
    pred, sigma = truth
    data = pred * scale if all_bands else pred.copy()
    if not all_bands:
        data[_LIM] = pred[_LIM] * scale
    mask = None
    if mask_value is not None:
        mask = np.zeros(len(pred), dtype=int)
        mask[_LIM] = mask_value
    model = model if model is not None else _build_model(ssp, free_noise=free_noise)
    return Fitter(model, data=data, noise=sigma, data_type="photometry", data_mask=mask)


def _run_mean(fitter, method, **kw):
    """Posterior mean of the free parameter for a short, fixed-key run."""
    if method.startswith(("vi", "native_vi")):
        kw.setdefault("n_iterations", 20)
    if method.startswith("native_vi"):
        kw.update(allow_unvalidated=True, n_seeds=1)
    post = fitter.run(method, key=_KEY, verbose=False, **kw)
    return float(np.mean(np.asarray(post.params[_NAME])))


# ── closed form ──────────────────────────────────────────────────────────────


def _closed_form(data, sigma, pred, mask):
    """Per-band censored energy summed over bands, written with scipy only."""
    r = (data - pred) / sigma
    detected = 0.5 * r**2 + np.log(sigma)
    upper = -norm.logcdf((data - pred) / sigma)
    lower = -norm.logcdf((pred - data) / sigma)
    return float(np.sum(np.where(mask == 1, upper, np.where(mask == -1, lower, detected))))


def _pred_at(fitter, mass):
    return np.asarray(fitter.model.predict_photometry({_NAME: mass}), dtype=float)


def _latent(fitter, mass):
    return fitter.spec.get_distribution(_NAME).standardize(jnp.asarray(mass))


# ── the energy of every engine family, as a function of the latent ───────────


def _jit_engine_energy(fitter):
    """``build_jit_engine`` hamiltonian (native_vi_linear, geoVI/MGVI draws) minus the prior."""
    u0 = {_NAME: jnp.asarray(0.0)}
    engine = fitter._get_or_build_engine(u0)
    data_args = fitter._build_data_args(fitter.model)
    flatten = engine["flatten"]

    def energy(z):
        flat = flatten({_NAME: z})
        return engine["hamiltonian"](flat, data_args) - 0.5 * jnp.sum(flat**2)

    return energy


def _nifty_energy(fitter):
    """The NIFTy likelihood ``vi`` / ``vi_nonlinear_fast`` / ``vi_linear*`` minimize."""
    from tengri.inference.backends.vi.nifty import _build_nifty_likelihood

    likelihood = _build_nifty_likelihood(fitter)
    return lambda z: likelihood.energy({_NAME: z})


def _native_engine_energy(builder_name):
    """Energy of ``build_native_vi_{linear,nonlinear}_engine`` minus the prior."""

    def make(fitter):
        from tengri.inference.backends.vi import native
        from tengri.inference.jit_engine import get_or_build_signal_response

        engine = fitter._get_or_build_engine({_NAME: jnp.asarray(0.0)})
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        signal_response, _ = get_or_build_signal_response(fitter)
        hamiltonian = getattr(native, builder_name)(
            signal_response,
            jnp.asarray(fitter.data),
            jnp.asarray(fitter.noise),
            flatten,
            unflatten,
            mask=fitter.data_mask,
        )[2]

        def energy(z):
            flat = flatten({_NAME: z})
            return hamiltonian(flat) - 0.5 * jnp.sum(flat**2)

        return energy

    return make


_FAMILIES = {
    "jit_engine_hamiltonian": _jit_engine_energy,
    "nifty_likelihood": _nifty_energy,
    "native_linear_engine": _native_engine_energy("build_native_vi_linear_engine"),
    "native_nonlinear_engine": _native_engine_energy("build_native_vi_nonlinear_engine"),
}

# (mask flag on band 2, limit value as a multiple of the model at the truth)
_ENERGY_CASES = [
    pytest.param(0, 3.0, id="detected"),
    pytest.param(1, 3.0, id="upper_satisfied"),
    pytest.param(1, 0.3, id="upper_violated"),
    pytest.param(-1, 0.3, id="lower_satisfied"),
    pytest.param(-1, 3.0, id="lower_violated"),
]


@pytest.mark.parametrize("family", sorted(_FAMILIES))
@pytest.mark.parametrize("mass", [10.5, 10.9])
@pytest.mark.parametrize(("mask_value", "scale"), _ENERGY_CASES)
def test_vi_energy_equals_the_closed_form(ssp, truth, family, mass, mask_value, scale):
    """At a fixed latent position every VI energy equals the censored closed form.

    Tolerance 1e-10 relative: both sides are float64 evaluations of the same
    sum of ``0.5 r^2 + ln sigma`` / ``-ln Phi`` terms (scipy vs jax logcdf agree
    to ~1e-15), so 1e-10 leaves five orders of margin without admitting any
    wrong branch (the detection-only energy differs by O(1) to O(1e3)).
    """
    fitter = _fitter(ssp, truth, scale=scale, mask_value=mask_value)
    energy = _FAMILIES[family](fitter)
    got = float(energy(_latent(fitter, mass)))
    data = np.asarray(fitter.data, dtype=float)
    sigma = np.asarray(fitter.noise, dtype=float)
    mask = np.asarray(fitter.data_mask)
    want = _closed_form(data, sigma, _pred_at(fitter, mass), mask)
    assert got == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_without_a_mask_the_energy_is_half_chi_squared(ssp, truth, family):
    """No ``data_mask``: the energy is the detection ``0.5 chi^2`` of main, unchanged.

    Tolerance 1e-10 relative, as above. The detection energy carries no
    ``ln sigma`` term when no mask is given.
    """
    fitter = _fitter(ssp, truth, scale=3.0, mask_value=None)
    energy = _FAMILIES[family](fitter)
    got = float(energy(_latent(fitter, 10.9)))
    data = np.asarray(fitter.data, dtype=float)
    sigma = np.asarray(fitter.noise, dtype=float)
    want = 0.5 * float(np.sum(((data - _pred_at(fitter, 10.9)) / sigma) ** 2))
    assert got == pytest.approx(want, rel=1e-10)


# ── gradient of a violated limit ─────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_gradient_of_a_limit_violated_by_thirty_sigma(ssp, truth, family):
    """A lower limit 30 sigma above the model has a finite, nonzero, correct gradient.

    The limit is ``F = model + 30 sigma`` at the truth, so ``z = (m - F)/sigma =
    -30`` for ``-ln Phi(z)``: a clamped or log-of-Phi implementation underflows
    there. The gradient with respect to the latent is compared with a central
    finite difference of the closed form (step 1e-6, float64 truncation error
    ~1e-10 relative), tolerance 1e-6 relative.
    """
    pred, sigma = truth
    scale = (pred[_LIM] + 30.0 * sigma[_LIM]) / pred[_LIM]
    fitter = _fitter(ssp, truth, scale=scale, mask_value=-1)
    energy = _FAMILIES[family](fitter)
    z0 = _latent(fitter, _TRUTH)
    grad = float(jax.grad(energy)(z0))
    assert np.isfinite(grad)
    assert grad != 0.0

    data = np.asarray(fitter.data, dtype=float)
    mask = np.asarray(fitter.data_mask)
    h = 1e-6

    def closed(z):
        mass = float(fitter.spec.get_distribution(_NAME).unstandardize(jnp.asarray(z)))
        sigma_obs = np.asarray(fitter.noise, dtype=float)
        return _closed_form(data, sigma_obs, _pred_at(fitter, mass), mask)

    z0f = float(z0)
    fd = (closed(z0f + h) - closed(z0f - h)) / (2 * h)
    assert grad == pytest.approx(fd, rel=1e-6)


# ── recovery ─────────────────────────────────────────────────────────────────

#: Posterior means at ``limit = 3 x model`` on band 2 with NO ``data_mask``, measured
#: on main (``vi_*`` / ``native_vi_*``, key PRNGKey(1), 20 iterations). A detection fit
#: pulled to the limit value: truth 10.5 + log10 of the band-2 pull.
_MAIN_DETECTION_MEAN = {
    "vi_nonlinear_fast": 10.6455,
    "vi_linear_fast": 10.6456,
    "native_vi_linear": 10.6462,
    "native_vi_nonlinear": 10.6460,
}


@pytest.fixture(scope="module")
def map_means(ssp, truth):
    """MAP with the satisfied limit censored (10.5000) and scored as a detection (10.6461)."""
    return {
        "masked": _run_mean(_fitter(ssp, truth, scale=3.0, mask_value=1), "map"),
        "detection": _run_mean(_fitter(ssp, truth, scale=3.0), "map"),
    }


@pytest.mark.parametrize("method", sorted(_MAIN_DETECTION_MEAN))
def test_vi_with_a_satisfied_limit_matches_map_and_not_the_detection_fit(
    ssp, truth, map_means, method
):
    """A satisfied upper limit leaves the VI posterior at the censored MAP.

    Tolerance 0.005 dex against the censored MAP: the posterior standard
    deviation is 0.007 dex (measured), and VI and MAP agree to 6e-4 dex without
    a limit. The detection fit sits 0.146 dex away, so it is excluded by a
    margin of 0.1 dex (14 posterior standard deviations).
    """
    fitter = _fitter(ssp, truth, scale=3.0, mask_value=1)
    mean = _run_mean(fitter, method)
    assert map_means["masked"] == pytest.approx(_TRUTH, abs=5e-4)
    assert mean == pytest.approx(map_means["masked"], abs=5e-3)
    assert abs(mean - map_means["detection"]) > 0.1


@pytest.mark.parametrize("method", sorted(_MAIN_DETECTION_MEAN))
def test_vi_without_a_mask_is_unchanged_from_main(ssp, truth, method):
    """No ``data_mask``: the posterior mean is the value main returns.

    Tolerance 1e-3 dex: the fixed key makes the run deterministic on one
    machine; across JAX builds the value moves in the fourth decimal at most,
    far below the posterior standard deviation (0.007 dex) and the 0.146 dex
    shift this fix removes.
    """
    mean = _run_mean(_fitter(ssp, truth, scale=3.0), method)
    assert mean == pytest.approx(_MAIN_DETECTION_MEAN[method], abs=1e-3)


# ── refusal where the engine cannot score a limit ────────────────────────────


@pytest.mark.parametrize("method", sorted(VI_METHODS_HONORING_DATA_MASK))
def test_a_free_noise_model_with_a_limit_is_refused_by_name(ssp, truth, method):
    """``noise_frac_cal`` free plus a limit is refused, naming ``data_mask``."""
    fitter = _fitter(ssp, truth, scale=3.0, mask_value=1, free_noise=True)
    with pytest.raises(ParameterError, match="data_mask") as exc:
        _run_mean(fitter, method)
    assert "vi_fullrank" in str(exc.value)


def test_a_free_noise_model_with_an_all_detected_mask_is_not_refused(ssp, truth):
    """An all-zero ``data_mask`` flags no limit, so nothing is dropped and nothing is refused."""
    fitter = _fitter(ssp, truth, scale=1.0, mask_value=0, free_noise=True)
    assert np.isfinite(_run_mean(fitter, "vi_nonlinear_fast"))


# ── #2668: one model object, many Fitters ────────────────────────────────────

#: Tolerance between a fit on a reused model and the fit on a fresh model with the
#: same data and key: 1e-3 dex, set after measuring the run-to-run spread of the
#: same fixed-key fit (see the brief); well below the 0.60 dex separating the two
#: data sets and the 0.007 dex posterior standard deviation.
_REUSE_TOL = 1e-3


@pytest.mark.parametrize("free_noise", [False, True], ids=["fixed_noise", "free_noise"])
@pytest.mark.parametrize("method", ["vi_nonlinear_fast", "vi_linear"])
def test_two_fitters_on_one_model_each_fit_their_own_data(ssp, truth, method, free_noise):
    """Data 0.5x then 2x the model on ONE model object: each fit equals a fresh-model fit."""
    shared = _build_model(ssp, free_noise=free_noise)
    reused = [
        _run_mean(
            _fitter(ssp, truth, scale=s, model=shared, free_noise=free_noise, all_bands=True),
            method,
        )
        for s in (0.5, 2.0, 0.5)
    ]
    fresh = [
        _run_mean(_fitter(ssp, truth, scale=s, free_noise=free_noise, all_bands=True), method)
        for s in (0.5, 2.0)
    ]
    assert reused[0] == pytest.approx(fresh[0], abs=_REUSE_TOL)
    assert reused[1] == pytest.approx(fresh[1], abs=_REUSE_TOL)
    assert reused[2] == pytest.approx(fresh[0], abs=_REUSE_TOL)
    assert reused[1] - reused[0] > 0.5


# ── registry sweep: every backend honors or refuses ``data_mask`` ───────────

_LOSS_BACKENDS = frozenset(PROFILE_MASS_BACKENDS)
_VI_ENERGY_BACKENDS = frozenset(VI_METHODS_HONORING_DATA_MASK)
_MASK_CASES = [(0, 3.0), (1, 3.0), (1, 0.3), (-1, 0.3), (-1, 3.0)]


def _loss_seam_energy(fitter, z):
    """Data energy the shared loss gives (map, mcmc_*, laplace, pathfinder, nss)."""
    u = {_NAME: z}
    value = fitter._get_or_build_loss_fn()(u, fitter._build_data_args(fitter.model))
    return float(value) - 0.5 * float(z) ** 2


def _flat_logdensity_energy(fitter, z):
    """Data energy of the flat log-density (vi_fullrank, vi_meanfield, samplers)."""
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity

    logdensity, _, _init_flat, data_args = _get_flat_logdensity(fitter, {_NAME: z})
    return -float(logdensity(jnp.asarray([float(z)]), data_args)) - 0.5 * float(z) ** 2


@pytest.fixture(scope="module")
def loss_seam_cases(ssp, truth):
    """(closed form, loss energy, flat-logdensity energy) per mask case at mass 10.9."""
    rows = []
    for mask_value, scale in _MASK_CASES:
        fitter = _fitter(ssp, truth, scale=scale, mask_value=mask_value)
        z = _latent(fitter, 10.9)
        want = _closed_form(
            np.asarray(fitter.data, dtype=float),
            np.asarray(fitter.noise, dtype=float),
            _pred_at(fitter, 10.9),
            np.asarray(fitter.data_mask),
        )
        rows.append((want, _loss_seam_energy(fitter, z), _flat_logdensity_energy(fitter, z)))
    return rows


def test_every_registered_backend_has_a_declared_data_mask_family():
    """Registry == loss-reading backends | VI-energy backends; no backend is unclassified.

    A new backend fails here until it is added to ``PROFILE_MASS_BACKENDS`` (it
    reads the ``Fitter`` loss, which censors) or to
    ``VI_METHODS_HONORING_DATA_MASK`` (its own energy, closed-form tested above).
    """
    registered = set(_BACKENDS)
    assert not (_LOSS_BACKENDS & _VI_ENERGY_BACKENDS)
    unclassified = sorted(registered - _LOSS_BACKENDS - _VI_ENERGY_BACKENDS)
    stale = sorted((_LOSS_BACKENDS | _VI_ENERGY_BACKENDS) - registered)
    assert not unclassified, f"backend(s) with no data_mask classification: {unclassified}"
    assert not stale, f"data_mask classification names unregistered backend(s): {stale}"


@pytest.mark.parametrize("name", sorted(_BACKENDS))
def test_each_registered_backend_honors_or_refuses_data_mask(ssp, truth, loss_seam_cases, name):
    """Loss-reading backends: loss energies equal the closed form. VI backends: refuse by name.

    Loss family: the two seams a backend can read (``loss_fn`` and the flat
    log-density) equal the censored closed form for every mask value
    (1e-10 relative, float64). VI family: the censored energy is closed-form
    tested above for every engine; here the combination the engine cannot score,
    a limit with a free noise model, raises ``ParameterError`` naming ``data_mask``
    and the methods that honor it, before any compute.
    """
    if name in _VI_ENERGY_BACKENDS:
        fitter = _fitter(ssp, truth, scale=3.0, mask_value=1, free_noise=True)
        with pytest.raises(ParameterError, match="data_mask"):
            _run_mean(fitter, name)
        return
    assert name in _LOSS_BACKENDS
    for want, loss_value, logdensity_value in loss_seam_cases:
        assert loss_value == pytest.approx(want, rel=1e-10)
        assert logdensity_value == pytest.approx(want, rel=1e-10)


# ── #2668 sibling: the hmc_is evidence evaluation is cached on the model ─────


def test_hmc_is_evidence_on_a_reused_model_scores_its_own_data(ssp, truth):
    """``hmc_is`` on one model object: the second data set's log Z equals a fresh model's.

    The cached importance-sampling evaluation closed over the first Fitter's
    data, so a second data set was weighted against the wrong likelihood (log Z
    -1337.5 instead of -5.80 measured). Tolerance 1e-6 absolute: same key and
    data on a fresh model reproduce the evaluation exactly in float64.
    """
    kw = {"n_warmup": 100, "n_burnin": 20, "n_samples": 300, "n_is_draws": 4000}

    def log_z(model, scale):
        fitter = _fitter(ssp, truth, scale=scale, model=model, all_bands=True)
        return float(fitter.run("hmc_is", key=_KEY, verbose=False, **kw).log_evidence)

    shared = _build_model(ssp)
    log_z(shared, 0.5)
    reused = log_z(shared, 2.0)
    fresh = log_z(_build_model(ssp), 2.0)
    assert reused == pytest.approx(fresh, abs=1e-6)
