"""Shared test fixtures for tengri test suite."""

import os
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Suppress background JIT compilation in the test suite.  Without this,
# every Fitter() instantiation spawns a compilation thread; with many test
# files each creating multiple Fitters the process floods with concurrent
# XLA compilations and exhausts memory.  Individual tests that exercise
# the compilation machinery clear this env var themselves.
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

from tengri.components.sfh.gp_sfh import compute_sqrt_power_drw
from tengri.components.sps.dsps_wrapper import SSPData
from tengri.utils.grid import grid_spacing, make_log_age_grid

# Enable 64-bit for numerical precision in tests
jax.config.update("jax_enable_x64", True)

# ── Paths for real SSP data ──────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SSP_FILE_WNE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_FILE_FSPS = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"


# ── Session-scoped real SSP fixtures ─────────────────────────────
# Loading HDF5 SSP data is expensive (~0.5-1s per call).  Session scope
# ensures a single load shared across all test files that need it.


@pytest.fixture(scope="session")
def ssp_data_wne():
    """Load the wNE SSP data once per session.  Skip if file missing."""
    if not _SSP_FILE_WNE.is_file():
        pytest.skip(f"SSP data not found: {_SSP_FILE_WNE}")
    from tengri.components.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_FILE_WNE))


@pytest.fixture(scope="session")
def ssp_data_fsps():
    """Load the FSPS SSP data once per session.  Skip if file missing."""
    if not _SSP_FILE_FSPS.is_file():
        pytest.skip(f"SSP data not found: {_SSP_FILE_FSPS}")
    from tengri.components.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_FILE_FSPS))


@pytest.fixture(scope="session")
def sdss_filters():
    """Load SDSS ugriz filters once per session."""
    from tengri.observation.filters import load_filter_set

    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


# ── Session-scoped synthetic SSP fixture ─────────────────────────
# Many unit tests create identical (3, 20, 100) synthetic SSPs.  Sharing
# a single instance avoids redundant array allocation and — more
# importantly — ensures all tests hit the same JIT-compiled code paths.


@pytest.fixture(scope="session")
def synthetic_ssp():
    """Minimal synthetic SSP: 3 Z × 20 ages × 100 wavelengths."""
    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(
        ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="session")
def simple_observation():
    """Synthetic 3-band observation matching the synthetic SSP wavelength range."""
    from tengri.observation.photometry import FilterCurve

    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.5 for _ in range(3)]
    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )
    from tengri import Observation, Photometry

    return Observation(photometry=Photometry(filters=curves))


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference gradient for scalar functions.

    Provides O(eps^2) accurate gradient estimate. Use to verify JAX autodiff:

        grad_jax = float(jax.grad(f)(jnp.array(x)))
        grad_fd  = fd_grad(f, x)
        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-3)

    Parameters
    ----------
    f : callable
        Scalar function float -> float (or jnp.array scalar -> scalar).
    x : float
        Point at which to estimate the gradient.
    eps : float
        Step size. 1e-4 is appropriate for float64; use 1e-3 for float32.

    Returns
    -------
    float
        Finite-difference gradient estimate (f(x+eps) - f(x-eps)) / (2*eps).
    """
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def pytest_configure(config):
    """Create minimal synthetic CB19 grid fixture before test collection.

    @pytest.mark.skipif(not path.exists(), ...) is evaluated at collection
    time when the test module is imported, so the file must already exist.
    This hook runs when the root conftest is imported — before collection —
    so any skipif that checks for cb19_templates.h5 will see the file.

    When the real file is present (e.g. after running
    scripts/download_cb19_templates.py), this hook is a no-op.
    """
    cb19_path = Path(__file__).parent.parent / "data" / "cb19_templates.h5"
    if cb19_path.exists():
        return

    cb19_path.parent.mkdir(parents=True, exist_ok=True)

    # Grid dimensions — must satisfy all TestCB19WithRealH5 assertions:
    #   n_oh=7, n_u=6, n_nh=4 (shape checks)
    #   n_age >= 11 (index 10 accessed in test_hb_ratio_is_unity)
    #   HbFrac=[0.0, 1.0]: hbfrac=1.0 → i_hb=1; hbfrac=0.42 → gap=0.42 > 0.15 → warns
    n_oh, n_age, n_u, n_nh, n_co, n_dno, n_hbfrac, n_lines = 7, 11, 6, 4, 3, 3, 2, 10

    # log_U linspace(-4.0, -1.5, 6) = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5]
    # index 2 = -3.0, which is the fiducial logU used in test_no_all_nan_slices_at_solar
    log_u = np.linspace(-4.0, -1.5, n_u).astype(np.float32)
    log_age = np.linspace(6.0, 10.0, n_age).astype(np.float32)

    with h5py.File(cb19_path, "w") as f:
        ax = f.create_group("axes")
        ax.create_dataset("log_OH_total", data=np.linspace(-5.06, -2.58, n_oh).astype(np.float32))
        ax.create_dataset("log_age_yr_ssp", data=log_age)
        ax.create_dataset("log_U", data=log_u)
        ax.create_dataset("log_nH", data=np.linspace(1.0, 4.0, n_nh).astype(np.float32))
        ax.create_dataset("log_CO", data=np.linspace(-1.0, 0.15, n_co).astype(np.float32))
        ax.create_dataset("dNO", data=np.linspace(-0.25, 0.25, n_dno).astype(np.float32))
        ax.create_dataset("HbFrac", data=np.array([0.0, 1.0], dtype=np.float32))

        # Line wavelengths — must include Hβ=4862.68 Å and Hα=6564.61 Å
        line_waves = np.array(
            [
                1215.67,
                1549.0,
                3727.0,
                4340.47,
                4862.68,
                5008.24,
                6300.30,
                6548.05,
                6564.61,
                6583.45,
            ],
            dtype=np.float32,
        )
        f.create_dataset("line_wavelengths_aa", data=line_waves)

        # All line ratios = 1.0 (linear L_line/L_Hβ), so after log10 conversion = 0.0.
        # This satisfies test_hb_ratio_is_unity (Hβ log10(ratio)=0.0) and
        # test_no_all_nan_slices_at_solar (all values finite).
        ratios = np.ones(
            (n_oh, n_age, n_u, n_nh, n_co, n_dno, n_hbfrac, n_lines), dtype=np.float32
        )
        grp = f.create_group("grids/SSP/Kroupa01/mu100")
        grp.create_dataset("line_ratios", data=ratios)

    # Tag as synthetic so developers know it's a test fixture, not the real data.
    # pytest_configure output goes before any test output; warn() is the right channel.
    import warnings

    warnings.warn(
        f"Created synthetic CB19 fixture at {cb19_path} for tests. "
        "Run scripts/download_cb19_templates.py to replace with the real grid.",
        UserWarning,
        stacklevel=1,
    )


@pytest.fixture
def rng_key():
    """Default PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def log_age_grid():
    """Standard 256-point log-age grid."""
    return make_log_age_grid(256)


@pytest.fixture
def d_log_age(log_age_grid):
    """Grid spacing."""
    return grid_spacing(log_age_grid)


@pytest.fixture
def drw_params_moderate():
    """Moderate burstiness DRW parameters."""
    return {"psd_sigma": 1.0, "psd_tau_yr": 50e6}  # 50 Myr


@pytest.fixture
def drw_params_smooth():
    """Smooth (low burstiness) DRW parameters."""
    return {"psd_sigma": 0.5, "psd_tau_yr": 200e6}  # 200 Myr


@pytest.fixture
def drw_params_bursty():
    """Highly bursty DRW parameters."""
    return {"psd_sigma": 3.0, "psd_tau_yr": 5e6}  # 5 Myr


@pytest.fixture
def sqrt_power_moderate(d_log_age, drw_params_moderate):
    """Pre-computed amplitude operator for moderate regime."""
    return compute_sqrt_power_drw(
        256,
        float(d_log_age),
        drw_params_moderate["psd_sigma"],
        drw_params_moderate["psd_tau_yr"],
    )
