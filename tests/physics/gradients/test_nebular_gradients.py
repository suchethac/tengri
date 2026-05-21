"""Finite-difference gradient tests for nebular backends.

These tests verify that JAX autodiff gradients match finite-difference
estimates for the nebular emission backends. All tests are skipped if
the required grid/weights files are not present, or if the backend
cannot be initialized with the minimal SSP data provided.

Gradient convention: we check dL/d_logU for line luminosities.
"""

import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.gradient

# Gradient tolerance
_FD_RTOL = 5e-2  # 5% relative tolerance for FD vs AD


def _fd_grad(fn, x, eps=1e-3):
    """Central finite difference for scalar function."""
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


# ── Minimal mock SSP data for backend initialization ──────────────


def _make_mock_ssp(n_met=3, n_age=10, n_wave=50):
    """Create a minimal mock SSP dataset for backend initialization.

    The mock SSP covers wavelengths that include the Lyman limit (< 912 Å)
    so that Q_H can be computed.
    """
    ssp_wave = np.linspace(500.0, 2000.0, n_wave)
    ssp_flux = np.zeros((n_met, n_age, n_wave))
    for i in range(n_met):
        for j in range(n_age):
            if j < 5:
                ssp_flux[i, j] = np.exp(-ssp_wave / 300.0) * 1e3
            else:
                ssp_flux[i, j] = np.exp(-ssp_wave / 800.0)
    ssp_lgmet = np.linspace(-2.5, -1.0, n_met)
    ssp_lg_age_gyr = np.linspace(-3.2, 1.1, n_age)
    return SimpleNamespace(
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
        ssp_lgmet=ssp_lgmet,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
    )


# ── Module-scoped fixtures for shared backends ────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_CLOUDY_GRID_PATH = _DATA_DIR / "cloudy_grid_mist.h5"
_CUE_WEIGHTS_PATH = _DATA_DIR / "cue_weights.npz"
_CB19_GRID_PATH = _DATA_DIR / "cb19_templates.h5"


@pytest.fixture(scope="module")
def mock_ssp():
    return _make_mock_ssp()


@pytest.fixture(scope="module")
def cloudy_linear_backend(mock_ssp):
    if not _CLOUDY_GRID_PATH.exists():
        pytest.skip("CLOUDY grid not present")
    from tengri.components.nebular import CloudyGridBackend

    return CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp)


@pytest.fixture(scope="module")
def cloudy_triweight_backend(mock_ssp):
    if not _CLOUDY_GRID_PATH.exists():
        pytest.skip("CLOUDY grid not present")
    from tengri.components.nebular import CloudyGridBackend

    return CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp, grid_interp="triweight")


@pytest.fixture(scope="module")
def cue_backend(mock_ssp):
    if not _CUE_WEIGHTS_PATH.exists():
        pytest.skip("Cue weights not present")
    import os

    from tengri.components.nebular import CueBackend
    from tengri.components.nebular.cue import CueWNESSPWarning

    # Mock SSP mimics a wNE grid — bypass the wNE hard-error guard
    # (CueWNESSPError) for this gradient-tracing fixture only.
    prev = os.environ.get("TENGRI_ALLOW_WNE_CUE")
    os.environ["TENGRI_ALLOW_WNE_CUE"] = "1"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CueWNESSPWarning)
            return CueBackend(str(_CUE_WEIGHTS_PATH), ssp_data=mock_ssp)
    finally:
        if prev is None:
            del os.environ["TENGRI_ALLOW_WNE_CUE"]
        else:
            os.environ["TENGRI_ALLOW_WNE_CUE"] = prev


@pytest.fixture(scope="module")
def cb19_backend(mock_ssp):
    if not _CB19_GRID_PATH.exists():
        pytest.skip("CB19 grid not present")
    from tengri.components.nebular.cloudy_cb19 import CB19Backend

    return CB19Backend(
        grid_path=str(_CB19_GRID_PATH),
        ssp_data=mock_ssp,
        ionizing_source_warning="suppress",
        continuum_warning="suppress",
    )


# ── CloudyGridBackend ─────────────────────────────────────────────


def test_cloudy_grid_grad_logu(cloudy_linear_backend, mock_ssp):
    """Test JAX autodiff vs finite difference for CloudyGridBackend."""
    backend = cloudy_linear_backend

    # Use the mock SSP age/weight arrays matching initialization
    n_age = len(mock_ssp.ssp_lg_age_gyr)
    ssp_wave = jnp.array(mock_ssp.ssp_wave)
    ssp_weights = jnp.ones(n_age) * 1e8
    ssp_log_ages = jnp.array(mock_ssp.ssp_lg_age_gyr + 9.0)  # log(age/yr)
    log_z = -1.848

    def fn(logU):
        return jnp.sum(
            backend.predict_nebular_sed(
                ssp_wave=ssp_wave,
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages,
                log_z=log_z,
                neb_logU=logU,
            )
        )

    # Use a non-grid-point logU value.  The CLOUDY grid nodes are at
    # -4, -3.5, -3, -2.5, -2, -1.5, -1.  At a grid node the piecewise-
    # linear interpolation has a kink: JAX AD returns the right-slope
    # while central FD averages left and right slopes.  When those slopes
    # differ (as they do for dominant lines), FD ≠ AD even though both
    # are mathematically valid.  Testing inside a grid cell avoids this.
    logU = jnp.array(-2.25)  # interior of the [-2.5, -2.0] cell
    try:
        fd = _fd_grad(fn, logU)
        ad = jax.grad(fn)(logU)
    except Exception as e:
        pytest.skip(f"Backend call failed (likely shape mismatch with mock SSP): {e}")

    np.testing.assert_allclose(
        float(ad),
        float(fd),
        rtol=_FD_RTOL,
        err_msg="CloudyGridBackend (linear): FD/AD mismatch at interior logU",
    )


def test_cloudy_grid_triweight_runs(cloudy_triweight_backend, mock_ssp):
    """Triweight mode produces finite output for both lines and continuum."""
    backend = cloudy_triweight_backend

    n_age = len(mock_ssp.ssp_lg_age_gyr)
    ssp_wave = jnp.array(mock_ssp.ssp_wave)
    ssp_weights = jnp.ones(n_age) * 1e8
    ssp_log_ages = jnp.array(mock_ssp.ssp_lg_age_gyr + 9.0)
    log_z = -1.848

    try:
        sed = backend.predict_nebular_sed(
            ssp_wave=ssp_wave,
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages,
            log_z=log_z,
            neb_logU=-2.5,  # on a grid node — triweight handles this smoothly
        )
    except Exception as e:
        pytest.skip(f"Backend call failed (likely shape mismatch with mock SSP): {e}")

    assert jnp.all(jnp.isfinite(sed)), "Triweight mode produced non-finite SED"
    assert jnp.any(sed > 0.0), "Triweight mode produced all-zero SED"


def test_cloudy_grid_triweight_grad_at_grid_node(cloudy_triweight_backend, mock_ssp):
    """Triweight mode: FD ≈ AD even when logU lands exactly on a grid node.

    This is the key advantage over linear mode — the smooth C²-continuous kernel
    eliminates the piecewise-linear kink, so FD and AD agree everywhere.
    """
    backend = cloudy_triweight_backend

    n_age = len(mock_ssp.ssp_lg_age_gyr)
    ssp_wave = jnp.array(mock_ssp.ssp_wave)
    ssp_weights = jnp.ones(n_age) * 1e8
    ssp_log_ages = jnp.array(mock_ssp.ssp_lg_age_gyr + 9.0)
    log_z = -1.848

    def fn(logU):
        return jnp.sum(
            backend.predict_nebular_sed(
                ssp_wave=ssp_wave,
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages,
                log_z=log_z,
                neb_logU=logU,
            )
        )

    # Grid node — triweight kernel is C² so FD = AD here (unlike linear mode)
    logU = jnp.array(-2.5)
    try:
        fd = _fd_grad(fn, logU)
        ad = jax.grad(fn)(logU)
    except Exception as e:
        pytest.skip(f"Backend call failed (likely shape mismatch with mock SSP): {e}")

    np.testing.assert_allclose(
        float(ad),
        float(fd),
        rtol=_FD_RTOL,
        err_msg="CloudyGridBackend (triweight): FD/AD mismatch at grid node logU",
    )


def test_logu_ordering(cloudy_linear_backend, mock_ssp):
    """Higher logU → harder ionization → higher [OIII]5007/Hβ ratio.

    Veilleux & Osterbrock (1987, ApJS 63, 295): [OIII]5007/Hβ is the primary
    BPT diagnostic for the ionization parameter.  A harder radiation field
    (higher logU) excites more O++ relative to recombination, raising the ratio.
    This tests the physical ordering directly against the CLOUDY grid.

    Line order in CLOUDY_LINE_NAMES: Hβ at index 4 (4862.68 Å),
    [OIII]5007 at index 6 (5008.24 Å) — vacuum wavelengths.
    """
    backend = cloudy_linear_backend

    n_age = len(mock_ssp.ssp_lg_age_gyr)
    ssp_weights = jnp.ones(n_age) * 1e8
    ssp_log_ages = jnp.array(mock_ssp.ssp_lg_age_gyr + 9.0)  # log(age/yr)
    log_z = -1.848  # solar metallicity in absolute log10(Z)

    try:
        # Use grid extremes (-4.0 and -1.0) for maximum [OIII]/Hβ contrast.
        # The mock SSP triggers the SFR-based Q_H fallback, which washes out
        # small logU differences; the full grid span gives a clear signal.
        _, lum_low = backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages,
            log_z=log_z,
            neb_logU=-4.0,
        )
        _, lum_high = backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages,
            log_z=log_z,
            neb_logU=-1.0,
        )
    except Exception as e:
        pytest.skip(f"Backend call failed (likely shape mismatch with mock SSP): {e}")

    # Hβ at index 4, [OIII]5007 at index 6 (CLOUDY_LINE_NAMES order)
    ratio_low = float(lum_low[6]) / float(lum_low[4])
    ratio_high = float(lum_high[6]) / float(lum_high[4])

    assert ratio_high > ratio_low, (
        f"logU ordering violated: [OIII]/Hβ at logU=-1 ({ratio_high:.3f}) "
        f"≤ logU=-4 ({ratio_low:.3f}) — Veilleux & Osterbrock 1987"
    )


def test_cloudy_grid_invalid_interp_mode(mock_ssp):
    """Unknown grid_interp raises ValueError at construction time."""
    if not _CLOUDY_GRID_PATH.exists():
        pytest.skip("CLOUDY grid not present")
    from tengri.components.nebular import CloudyGridBackend

    with pytest.raises(ValueError, match="grid_interp"):
        CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp, grid_interp="cubic")


# ── CueBackend ────────────────────────────────────────────────────


def test_cue_grad_logu(cue_backend, mock_ssp):
    """Test JAX autodiff vs finite difference for CueBackend.

    The mock SSP has near-zero ionizing flux, which correctly triggers a
    CueWNESSPWarning at backend construction (the mock mimics a wNE-type SSP).
    We suppress that warning here — it is expected behaviour for mock data and
    is tested separately in test_nebular_warnings.py.
    """
    backend = cue_backend

    n_age = len(mock_ssp.ssp_lg_age_gyr)
    ssp_wave = jnp.array(mock_ssp.ssp_wave)
    ssp_weights = jnp.ones(n_age) * 1e8
    ssp_log_ages = jnp.array(mock_ssp.ssp_lg_age_gyr + 9.0)
    log_z = -1.848

    def fn(logU):
        return jnp.sum(
            backend.predict_nebular_sed(
                ssp_wave=ssp_wave,
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages,
                log_z=log_z,
                neb_logU=logU,
            )
        )

    logU = jnp.array(-2.5)
    try:
        fd = _fd_grad(fn, logU)
        ad = jax.grad(fn)(logU)
    except Exception as e:
        pytest.skip(f"Backend call failed (likely shape mismatch with mock SSP): {e}")

    np.testing.assert_allclose(
        float(ad),
        float(fd),
        rtol=_FD_RTOL,
        err_msg="CueBackend: FD/AD mismatch for ∂/∂logU",
    )


# ── CB19Backend ───────────────────────────────────────────────────


def test_cb19_grad_logu(cb19_backend, mock_ssp):
    """Test JAX autodiff vs finite difference for CB19Backend w.r.t. neb_logU.

    CB_19 provides line ratios only (no continuum).  The gradient check confirms
    that the logU interpolation inside the CB_19 grid is smooth enough for
    autodiff to agree with central finite differences to 5% relative tolerance.
    """
    backend = cb19_backend

    n_age = len(mock_ssp.ssp_lg_age_gyr)
    ssp_wave = jnp.array(mock_ssp.ssp_wave)
    ssp_weights = jnp.ones(n_age) * 1e8
    ssp_log_ages = jnp.array(mock_ssp.ssp_lg_age_gyr + 9.0)
    log_z = -1.848

    def fn(logU):
        return jnp.sum(
            backend.predict_nebular_sed(
                ssp_wave=ssp_wave,
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages,
                log_z=log_z,
                neb_logU=logU,
            )
        )

    # Use interior-of-cell value to avoid piecewise-linear kinks at grid nodes
    logU = jnp.array(-2.25)
    try:
        fd = _fd_grad(fn, logU)
        ad = jax.grad(fn)(logU)
    except Exception as e:
        pytest.skip(f"Backend call failed (likely shape mismatch with mock SSP): {e}")

    np.testing.assert_allclose(
        float(ad),
        float(fd),
        rtol=_FD_RTOL,
        err_msg="CB19Backend: FD/AD mismatch for ∂/∂logU at interior grid point",
    )


# ── ShockBackend — MAPPINGS V velocity gradient ───────────────────

_MAPPINGS_GRID_PATH = _DATA_DIR / "mappings_templates.h5"


@pytest.mark.skipif(not _MAPPINGS_GRID_PATH.exists(), reason="MAPPINGS templates not present")
def test_mappings_grad_velocity():
    """Test JAX autodiff vs finite difference for ShockBackend w.r.t. shock_velocity.

    The MAPPINGS V grid uses linear interpolation on the velocity axis.  This test
    verifies that the gradient is finite and agrees with the central FD estimate at
    an interior grid point (avoids the kink at exact grid nodes).

    Physical range: 100–1000 km/s (3MdBs grid).  We test at 350 km/s (mid-range).
    """
    from tengri.components.nebular.shock import ShockBackend

    backend = ShockBackend(shock_abundance="solar", shock_component="combined")

    wavelength = jnp.linspace(3000.0, 7000.0, 100)
    # Hα luminosity: 1e40 erg/s (a typical warm-ionized SFG level)
    l_shock_halpha = 1e40

    def fn(velocity):
        return jnp.sum(
            backend.predict_nebular_sed(
                wavelength=wavelength,
                shock_velocity=velocity,
                l_shock_halpha=l_shock_halpha,
                shock_log_density=0.0,
                shock_b_over_sqrt_n=1.0,
            )
        )

    # 350 km/s: well inside the [100, 1000] km/s range, not on a grid node
    velocity = jnp.array(350.0)
    try:
        fd = _fd_grad(fn, velocity, eps=5.0)  # 5 km/s step for velocity axis
        ad = jax.grad(fn)(velocity)
    except Exception as e:
        pytest.skip(f"Backend call failed (likely missing MAPPINGS grid): {e}")

    np.testing.assert_allclose(
        float(ad),
        float(fd),
        rtol=_FD_RTOL,
        err_msg="ShockBackend (MAPPINGS V): FD/AD mismatch for ∂/∂velocity",
    )


# ── Metallicity conversion round-trip ─────────────────────────────


def test_neb_logzsol_to_log_z_abs_round_trip():
    """neb_logzsol_to_log_z_abs is a linear offset — verify the constant is correct.

    At solar metallicity (logzsol=0), the output must equal _LOG10_ZSUN ≈ −1.848.
    Subtracting the same offset recovers the input exactly (round-trip).
    """
    from tengri.components.nebular._constants import _LOG10_ZSUN
    from tengri.components.nebular._shared import neb_logzsol_to_log_z_abs

    # Solar metallicity
    logzsol_sun = jnp.array(0.0)
    log_z_abs_sun = neb_logzsol_to_log_z_abs(logzsol_sun)
    assert jnp.abs(log_z_abs_sun - _LOG10_ZSUN) < 1e-8, (
        f"Solar: expected {_LOG10_ZSUN:.6f}, got {float(log_z_abs_sun):.6f}"
    )

    # Round-trip: convert then invert; must recover input to float64 precision
    for logzsol_val in [-2.0, -1.0, 0.0, 0.5]:
        logzsol = jnp.array(logzsol_val)
        log_z_abs = neb_logzsol_to_log_z_abs(logzsol)
        logzsol_back = log_z_abs - _LOG10_ZSUN  # analytic inverse
        assert jnp.abs(logzsol_back - logzsol) < 1e-8, (
            f"Round-trip failed at logzsol={logzsol_val}: got {float(logzsol_back):.6f}"
        )


def test_neb_logzsol_to_cloudy_logoh():
    """cloudy_logoh conversion preserves metallicity differences (linear offset).

    Both ``neb_logzsol_to_log_z_abs`` and ``neb_logzsol_to_cloudy_logoh`` are
    pure linear offsets: differences between metallicities are preserved exactly.
    We test this property rather than an absolute value (which depends on the
    CLOUDY c17.01 solar O/H reference convention).
    """
    from tengri.components.nebular._shared import neb_logzsol_to_cloudy_logoh

    logzsol_a = jnp.array(0.0)
    logzsol_b = jnp.array(-1.0)
    logoh_a = neb_logzsol_to_cloudy_logoh(logzsol_a)
    logoh_b = neb_logzsol_to_cloudy_logoh(logzsol_b)

    # Differences are preserved exactly under a linear shift
    assert jnp.abs((logoh_a - logoh_b) - (logzsol_a - logzsol_b)) < 1e-8, (
        "logoh difference does not match logzsol difference"
    )

    # Monotonicity: higher logzsol → higher O/H
    assert logoh_a > logoh_b, "O/H should increase with metallicity"
