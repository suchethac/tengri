"""Shared utilities for paper analysis scripts.

Provides standard mock generation, fitting pipelines, and plotting
defaults used across all figure-generation scripts.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri import (
    Fitter,
    Gaussian,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Posterior,
    Uniform,
    load_filter_set,
    load_ssp_data,
)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "analysis" / "figures"
FIG_DIR.mkdir(exist_ok=True)

SSP_FILE = DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
# Override via env var (e.g. for paper-figure output destination); defaults to local figures dir.
PAPER_FIG_DIR = Path(os.environ.get("TENGRI_PAPER_FIG_DIR", str(FIG_DIR)))

# ── PSD regimes ────────────────────────────────────────────────────
PSD_REGIMES = {
    "smooth": {"psd_sigma": 0.5, "psd_tau_myr": 200.0},
    "moderate": {"psd_sigma": 1.0, "psd_tau_myr": 50.0},
    "bursty": {"psd_sigma": 2.0, "psd_tau_myr": 20.0},
    "highly_bursty": {"psd_sigma": 3.0, "psd_tau_myr": 5.0},
}

# ── Default SFH params (star-forming galaxy) ──────────────────────
DEFAULT_SFH = dict(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.3, 2.0),
    sfh_dpl_tau_gyr=Uniform(1.0, 8.0),
    sfh_dpl_log_total_mass=Uniform(10.0, 11.5),
)

DEFAULT_SPS = dict(
    met_logzsol=Gaussian(-0.5, 0.3, lo=-2.0, hi=0.0),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.0),
    dust_slope=-0.7,
)


# ── Plotting defaults ─────────────────────────────────────────────
def setup_matplotlib():
    """Set publication-quality matplotlib defaults."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
        }
    )
    return plt


# ── Data loading (cached) ─────────────────────────────────────────
_SSP_CACHE = {}
_FILTER_CACHE = {}


def get_ssp():
    """Load SSP data (cached)."""
    if "ssp" not in _SSP_CACHE:
        _SSP_CACHE["ssp"] = load_ssp_data(str(SSP_FILE))
    return _SSP_CACHE["ssp"]


def get_filters(filter_names=None):
    """Load filter set (cached). Legacy — prefer get_observation()."""
    if filter_names is None:
        filter_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    key = tuple(filter_names)
    if key not in _FILTER_CACHE:
        _FILTER_CACHE[key] = load_filter_set(filter_names)
    return _FILTER_CACHE[key]


_OBS_CACHE = {}


def get_observation(filter_names=None):
    """Create an Observation with photometry (cached)."""
    if filter_names is None:
        filter_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    key = tuple(filter_names)
    if key not in _OBS_CACHE:
        _OBS_CACHE[key] = Observation(photometry=Photometry.from_names(list(filter_names)))
    return _OBS_CACHE[key]


# ── Model factory ─────────────────────────────────────────────────
def make_model(
    psd_regime: str,
    redshift: float = 0.1,
    stochastic: bool = True,
    n_grid: int = 128,
    free_psd: bool = False,
    filter_names: list | None = None,
) -> Model:
    """Create a Model for a given PSD regime.

    Parameters
    ----------
    psd_regime : str
        One of "smooth", "moderate", "bursty", "highly_bursty".
    redshift : float
        Galaxy redshift.
    stochastic : bool
        Whether to use the correlated-field SFH.
    n_grid : int
        Number of GP grid points.
    free_psd : bool
        If True, make psd_sigma and psd_tau_myr free parameters.
    filter_names : list, optional
        Filter names. Default: SDSS ugriz.
    """
    psd = PSD_REGIMES[psd_regime]

    spec_kwargs = dict(**DEFAULT_SFH, **DEFAULT_SPS)

    if free_psd:
        spec_kwargs["sfh_field_psd_sigma"] = Uniform(0.1, 4.0)
        spec_kwargs["sfh_field_psd_tau_myr"] = Uniform(1.0, 300.0)
    else:
        spec_kwargs["sfh_field_psd_sigma"] = psd["psd_sigma"]
        spec_kwargs["sfh_field_psd_tau_myr"] = psd["psd_tau_myr"]

    spec_kwargs["redshift"] = redshift
    spec_kwargs["stochastic"] = stochastic
    spec_kwargs["n_grid"] = n_grid

    spec = ParamSpec(**spec_kwargs)
    ssp = get_ssp()
    obs = get_observation(filter_names)

    return Model(spec, ssp, observation=obs)


# ── Mock generation ───────────────────────────────────────────────
@dataclass
class MockGalaxy:
    """A single mock galaxy with truth and observations."""

    true_params: dict
    flux_obs: jnp.ndarray
    noise: jnp.ndarray
    true_sfh: dict | None = None
    spec_obs: jnp.ndarray | None = None
    spec_noise: jnp.ndarray | None = None
    wave_spec: jnp.ndarray | None = None


def generate_mock_galaxy(
    model: Model, key, snr: float = 20.0, spec_snr: float = 15.0, wave_spec=None
) -> MockGalaxy:
    """Generate a single mock galaxy.

    Parameters
    ----------
    model : Model
        Configured model.
    key : PRNGKey
        Random key.
    snr : float
        Photometric S/N.
    spec_snr : float
        Spectroscopic S/N per pixel.
    wave_spec : array, optional
        If provided, also generate a mock spectrum.
    """
    key, param_key, noise_key, spec_key = jax.random.split(key, 4)
    true_params = model.spec.sample(param_key)
    mock = model.mock(true_params, snr=snr, key=noise_key)

    # SFH truth
    true_sfh = model.predict_sfh(true_params)

    result = MockGalaxy(
        true_params=true_params,
        flux_obs=mock.flux_obs,
        noise=mock.noise,
        true_sfh=true_sfh,
    )

    if wave_spec is not None:
        spec_true = model.predict_spectrum(true_params, wave_spec)
        spec_noise = spec_true / spec_snr
        spec_obs = spec_true + spec_noise * jax.random.normal(spec_key, shape=spec_true.shape)
        result.spec_obs = spec_obs
        result.spec_noise = spec_noise
        result.wave_spec = wave_spec

    return result


def generate_mock_population(
    model: Model, n_galaxies: int, key, snr: float = 20.0, **kwargs
) -> list[MockGalaxy]:
    """Generate a population of mock galaxies."""
    keys = jax.random.split(key, n_galaxies)
    return [generate_mock_galaxy(model, k, snr=snr, **kwargs) for k in keys]


# ── Fitting pipeline ──────────────────────────────────────────────
@dataclass
class FitResult:
    """Results from fitting a single mock galaxy."""

    posterior: Posterior
    true_params: dict
    true_sfh: dict
    method: str
    wall_time_s: float


def fit_galaxy(
    model: Model,
    galaxy: MockGalaxy,
    method: str = "mcmc_raytrace",
    data_type: str = "photometry",
    **kwargs,
) -> FitResult:
    """Fit a single mock galaxy.

    Parameters
    ----------
    model : Model
        The forward model.
    galaxy : MockGalaxy
        Mock data.
    method : str
        "map", "mcmc_raytrace", "mcmc_nuts", or "vi".
    data_type : str
        "photometry" or "spectroscopy".
    **kwargs
        Passed to fitter.run().
    """
    if data_type == "spectroscopy" and galaxy.spec_obs is not None:
        data = galaxy.spec_obs
        noise = galaxy.spec_noise
        # Rebuild model with spectroscopic observation
        from tengri import SpectroscopyConfig

        spec_obs = Observation(
            spectroscopy=SpectroscopyConfig(wave_obs=galaxy.wave_spec),
        )
        model = Model(model.spec, model.ssp_data, observation=spec_obs)
    else:
        data = galaxy.flux_obs
        noise = galaxy.noise

    fitter = Fitter(model, data, noise)

    # Default: MAP init → sampler
    t0 = time.time()
    key = kwargs.pop("key", jax.random.PRNGKey(42))

    if method in ("mcmc_raytrace", "mcmc_nuts", "vi", "vi_linear"):
        result_map = fitter.run("map", n_steps=1000, learning_rate=0.03, verbose=False, key=key)
        key = jax.random.fold_in(key, 1)
        posterior = fitter.run(method, init_from=result_map, key=key, verbose=False, **kwargs)
    else:
        posterior = fitter.run(method, key=key, verbose=False, **kwargs)

    wall_time = time.time() - t0

    return FitResult(
        posterior=posterior,
        true_params=galaxy.true_params,
        true_sfh=galaxy.true_sfh,
        method=method,
        wall_time_s=wall_time,
    )


# ── SFH residual metrics ─────────────────────────────────────────
def sfh_residuals(model: Model, fit_result: FitResult, t_max_gyr: float = 1.0) -> dict:
    """Compute SFH recovery metrics.

    Returns
    -------
    dict with keys:
        rmse_log_sfr: RMSE in log10(SFR) over full history
        rmse_log_sfr_recent: RMSE in log10(SFR) for t < t_max_gyr
        bias_log_sfr: mean bias in log10(SFR)
        coverage_68: fraction of truth within 68% CI
    """
    true_sfh = fit_result.true_sfh
    t_gyr = np.array(true_sfh["t_gyr"])
    sfr_true = np.array(true_sfh["sfr_full"])

    posterior = fit_result.posterior
    if posterior.samples is None:
        return {"rmse_log_sfr": np.nan}

    n_samples = min(posterior.diagnostics.get("n_samples", 50), 50)
    sfr_draws = []
    for i in range(n_samples):
        s_i = {k: posterior.samples[k][i] for k in posterior.samples}
        sfh_i = model.predict_sfh(s_i)
        sfr_draws.append(np.array(sfh_i["sfr_full"]))

    sfr_arr = np.array(sfr_draws)  # (n_samples, n_t)
    sfr_median = np.median(sfr_arr, axis=0)

    # Log-space metrics (avoid log(0))
    eps = 1e-10
    log_true = np.log10(np.maximum(sfr_true, eps))
    log_median = np.log10(np.maximum(sfr_median, eps))

    residual = log_median - log_true

    # Full history
    rmse_full = float(np.sqrt(np.mean(residual**2)))

    # Recent (t < t_max_gyr)
    mask_recent = t_gyr < t_max_gyr
    rmse_recent = (
        float(np.sqrt(np.mean(residual[mask_recent] ** 2))) if mask_recent.any() else np.nan
    )

    # Bias
    bias = float(np.mean(residual))

    # 68% coverage
    lo = np.percentile(np.log10(np.maximum(sfr_arr, eps)), 16, axis=0)
    hi = np.percentile(np.log10(np.maximum(sfr_arr, eps)), 84, axis=0)
    covered = (log_true >= lo) & (log_true <= hi)
    coverage = float(np.mean(covered))

    return {
        "rmse_log_sfr": rmse_full,
        "rmse_log_sfr_recent": rmse_recent,
        "bias_log_sfr": bias,
        "coverage_68": coverage,
    }
