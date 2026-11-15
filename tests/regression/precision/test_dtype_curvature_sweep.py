# SPDX-License-Identifier: BSD-3-Clause
r"""Every curvature path works in float32 and float64, on every configuration class.

Owner ruling: every setting works with every mode. The float32 Laplace refusal
(``HESSIAN_BACKEND_SET`` in :mod:`tengri.inference.mass_profile`) claimed the
photometry Hessian is all-NaN in float32, but it was only checked on one narrow
fixture. This sweep measures the four curvature consumers on each configuration
class, in both dtypes:

- the Hessian of the fitter's flat log posterior at its MAP (the f64 MAP, cast);
- ``Fitter.run("laplace")`` (mean and covariance);
- the preconditioning metric (:func:`negative_hessian_metric`);
- the damped Newton polish (:func:`newton_polish`).

Assertions per cell: the Hessian is finite in both dtypes; the float32 Hessian
agrees with the float64 one to ``max|dH| / max|H| <= 1e-3`` at the same point;
the Laplace float32 mean is within 0.1 sigma of float64 and its covariance within
5 % relative. The Laplace comparison carries Monte Carlo noise: both runs draw
``n_samples=20000`` from the same key, so the sample-mean error is about
``sigma / sqrt(20000)`` (0.007 sigma) and the covariance error about 1 %. The 5 %
bar sits well above that.

Float32 is selected inside the test with ``jax.enable_x64(False)``, as in the other
precision tests; the module never sets ``JAX_ENABLE_X64``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, ForwardModel, SEDModel, generate_mock, recipes
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.inference.backends.map_dispatch import run_map
from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.newton import newton_polish
from tengri.inference.preconditioning import negative_hessian_metric
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.regression_bug

_SSP = "data/fsps_prsc_miles_chabrier.h5"
_BANDS_ONE = ["sdss_r"]
_BANDS_MULTI = ["sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]

#: Configuration classes. Each is the smallest fixture that exercises its feature.
CONFIGS = ("photometry_1band", "multiband", "censored", "presence", "nebular", "agn")
#: ``"auto"`` engages profiling where the model admits it and steps aside (with a logged
#: reason) where it does not: censored data and affine-in-mass AGN refuse an explicit True.
PROFILES = (False, "auto")
DTYPES = ("float64", "float32")

HESSIAN_REL_TOL = 1e-3
LAPLACE_MEAN_TOL_SIGMA = 0.1
LAPLACE_COV_TOL = 0.05
LAPLACE_SAMPLES = 20000
MAP_STEPS = 1000


def _recipe(config: str) -> dict:
    """Return the nested-dict recipe for one configuration class."""
    if config == "agn":
        return recipes.composable_agn()
    base = recipes.mock_recovery_minimal()
    if config == "nebular":
        base["neb"] = {"type": "cue", "all_params": Fixed(DEFAULT)}
    return base


def _bands(config: str) -> list[str]:
    multi = config in ("multiband", "censored", "presence")
    return _BANDS_MULTI if multi else _BANDS_ONE


def _build_model(config: str):
    """Build the SED model for ``config`` under the current x64 setting."""
    obs = Observation(photometry=Photometry.from_names(_bands(config)))
    return SEDModel.build(ssp_data=load_ssp_data(_SSP), observation=obs, **_recipe(config))


@functools.lru_cache(maxsize=None)
def _mock(config: str) -> dict:
    """Float64 mock data for ``config``, shared by both dtypes."""
    with jax.enable_x64(True):
        model = _build_model(config)
        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        mock = generate_mock(model, truth, key=key_mock, snr=30.0)
        flux = np.asarray(mock["flux_obs"], dtype=np.float64)
        noise = np.asarray(mock["noise"], dtype=np.float64)
    n = flux.shape[0]
    data_mask = None
    presence = None
    if config == "censored":
        # Band 0 becomes an upper limit (flag 1); its value is the limit.
        data_mask = np.zeros(n, dtype=np.int32)
        data_mask[0] = 1
    if config == "presence":
        presence = np.ones(n, dtype=np.float64)
        presence[0] = 0.0
    return {"flux": flux, "noise": noise, "data_mask": data_mask, "presence": presence}


def _fitter(config: str, profile: bool) -> Fitter:
    """Build the fitter for ``config`` under the current x64 setting."""
    model = _build_model(config)
    forward = ForwardModel.build(sed=model)
    mock = _mock(config)
    return Fitter(
        forward,
        data=jnp.asarray(mock["flux"]),
        noise=jnp.asarray(mock["noise"]),
        data_mask=None if mock["data_mask"] is None else jnp.asarray(mock["data_mask"]),
        presence=None if mock["presence"] is None else jnp.asarray(mock["presence"]),
        profile_mass=profile,
    )


def _attempt(fn):
    """Run one measurement; return ``("ok", value)`` or ``("error", message)``."""
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001 - the cell reports the failure, not a skip
        return ("error", f"{type(exc).__name__}: {exc}")


def _measure(config: str, profile: bool, dtype: str, ref: dict | None) -> dict:
    """Measure every curvature path for one (config, profile, dtype) cell.

    The float64 call computes the MAP and stores it; the float32 call reuses that
    MAP cast to float32, so both dtypes are evaluated at the same point.
    """
    x64 = dtype == "float64"
    with jax.enable_x64(x64):
        fitter = _fitter(config, profile)
        if ref is None:
            ctx = InferenceContext.from_target(fitter)
            res = run_map(ctx, key=jax.random.PRNGKey(1), n_steps=MAP_STEPS, verbose=False)
            mp_unb = ctx.unbounded_from_posterior(res)
        else:
            mp_unb = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float32), ref["mp_unb"])
        logd, _, x, data_args = _get_flat_logdensity(fitter, mp_unb)
        out = {
            "mp_unb": jax.tree.map(np.asarray, mp_unb),
            "dtype": str(jnp.result_type(x)),
            "profile_engaged": bool(getattr(fitter, "_profile_mass", False)),
        }
        out["hess"] = _attempt(lambda: np.asarray(jax.jit(jax.hessian(logd))(x, data_args)))
        out["metric"] = _attempt(
            lambda: np.asarray(negative_hessian_metric(logd, x, data_args))
        )
        out["newton"] = _attempt(
            lambda: np.asarray(
                newton_polish(x, logd, data_args, max_steps=5, tol=1e-3)[0]
            )
        )
        out["laplace"] = _attempt(lambda: _laplace_moments(fitter))
    return out


def _laplace_moments(fitter: Fitter) -> dict:
    """Mean and covariance of ``fitter.run("laplace")`` over sorted free names."""
    post = fitter.run("laplace", key=jax.random.PRNGKey(2), n_samples=LAPLACE_SAMPLES)
    names = sorted(post.samples)
    samples = np.stack([np.asarray(post.samples[n], dtype=np.float64) for n in names], 1)
    return {
        "names": names,
        "mean": samples.mean(0),
        "cov": np.cov(samples, rowvar=False),
    }


@functools.lru_cache(maxsize=None)
def _cell(config: str, profile: bool) -> dict:
    """Both dtypes for one (config, profile): ``{"float64": ..., "float32": ...}``."""
    m64 = _measure(config, profile, "float64", ref=None)
    m32 = _measure(config, profile, "float32", ref=m64)
    return {"float64": m64, "float32": m32}


def _cases():
    return [(c, p) for c in CONFIGS for p in PROFILES]


def _case_id(config: str, profile) -> str:
    return f"{config}-profile_{'off' if profile is False else 'auto'}"


_CASE_IDS = [_case_id(c, p) for c, p in _cases()]


def _value(cell: dict, dtype: str, path: str):
    status, value = cell[dtype][path]
    if status != "ok":
        pytest.fail(f"{dtype} {path} raised: {value}")
    return value


def _rel_max(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config,profile", _cases(), ids=_CASE_IDS)
def test_hessian_is_finite_in_both_dtypes(config, profile, dtype):
    """The flat log-posterior Hessian at the MAP is finite."""
    hess = _value(_cell(config, profile), dtype, "hess")
    assert np.all(np.isfinite(hess)), f"{dtype} Hessian has {np.sum(~np.isfinite(hess))} non-finite"


@pytest.mark.parametrize("config,profile", _cases(), ids=_CASE_IDS)
def test_hessian_float32_agrees_with_float64(config, profile):
    """Same point, same objective: float32 Hessian within 1e-3 of float64."""
    cell = _cell(config, profile)
    h32 = _value(cell, "float32", "hess")
    h64 = _value(cell, "float64", "hess")
    assert _rel_max(h32, h64) <= HESSIAN_REL_TOL


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config,profile", _cases(), ids=_CASE_IDS)
def test_preconditioning_metric_is_finite(config, profile, dtype):
    """The preconditioning metric at the MAP is finite and float32 tracks float64."""
    cell = _cell(config, profile)
    metric = _value(cell, dtype, "metric")
    assert np.all(np.isfinite(metric))
    if dtype == "float32":
        assert _rel_max(metric, _value(cell, "float64", "metric")) <= HESSIAN_REL_TOL


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config,profile", _cases(), ids=_CASE_IDS)
def test_newton_polish_is_finite(config, profile, dtype):
    """The damped Newton polish returns a finite point in both dtypes."""
    cell = _cell(config, profile)
    position = _value(cell, dtype, "newton")
    assert np.all(np.isfinite(position))


@pytest.mark.parametrize("config,profile", _cases(), ids=_CASE_IDS)
def test_profile_auto_resolves_identically_in_both_dtypes(config, profile):
    """``profile_mass="auto"`` engages (or steps aside) the same way in float32 and float64."""
    cell = _cell(config, profile)
    assert cell["float32"]["profile_engaged"] == cell["float64"]["profile_engaged"]


@pytest.mark.parametrize("config,profile", _cases(), ids=_CASE_IDS)
def test_laplace_float32_matches_float64(config, profile):
    """``fitter.run("laplace")``: float32 mean within 0.1 sigma, covariance within 5 %."""
    cell = _cell(config, profile)
    lap32 = _value(cell, "float32", "laplace")
    lap64 = _value(cell, "float64", "laplace")
    assert lap32["names"] == lap64["names"]
    sigma64 = np.sqrt(np.diag(lap64["cov"]))
    mean_gap = np.max(np.abs(lap32["mean"] - lap64["mean"]) / sigma64)
    assert mean_gap <= LAPLACE_MEAN_TOL_SIGMA, f"mean gap {mean_gap:.3f} sigma"
    cov_gap = _rel_max(lap32["cov"], lap64["cov"])
    assert cov_gap <= LAPLACE_COV_TOL, f"covariance relative gap {cov_gap:.3f}"
