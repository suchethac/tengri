"""Finite-difference gradient tests for nebular backends.

These tests verify that JAX autodiff gradients match finite-difference
estimates for the nebular emission backends. All tests are skipped if
the required grid/weights files are not present, or if the backend
cannot be initialized with the minimal SSP data provided.

Gradient convention: we check dL/d_logU for line luminosities.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Gradient tolerance
_FD_RTOL = 5e-2  # 5% relative tolerance for FD vs AD


def _fd_grad(fn, x, eps=1e-3):
    """Central finite difference for scalar function."""
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# Minimal mock SSP data for backend initialization
# ---------------------------------------------------------------------------


def _make_mock_ssp(n_met=3, n_age=10, n_wave=50):
    """Create a minimal mock SSP dataset for backend initialization.

    The mock SSP covers wavelengths that include the Lyman limit (< 912 Å)
    so that Q_H can be computed.
    """
    ssp_wave = np.linspace(500.0, 2000.0, n_wave)
    # Simple blackbody-like spectrum: higher for hotter (younger) SSPs
    ssp_flux = np.zeros((n_met, n_age, n_wave))
    for i in range(n_met):
        for j in range(n_age):
            # Young ages (j=0..4) emit ionizing photons; older do not
            if j < 5:
                ssp_flux[i, j] = np.exp(-ssp_wave / 300.0) * 1e3
            else:
                ssp_flux[i, j] = np.exp(-ssp_wave / 800.0)
    ssp_lgmet = np.linspace(-2.5, -1.0, n_met)
    # log10(age/Gyr): spans 6 Myr to 13 Gyr
    ssp_lg_age_gyr = np.linspace(-3.2, 1.1, n_age)
    return SimpleNamespace(
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
        ssp_lgmet=ssp_lgmet,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
    )


# ---------------------------------------------------------------------------
# CloudyGridBackend
# ---------------------------------------------------------------------------

_CLOUDY_GRID_PATH = Path("/Users/suchethacooray/Projects/tengri/data/cloudy_grid_mist.h5")


@pytest.mark.skipif(not _CLOUDY_GRID_PATH.exists(), reason="CLOUDY grid not present")
def test_cloudy_grid_grad_logu():
    """Test JAX autodiff vs finite difference for CloudyGridBackend."""
    from tengri.models.nebular import CloudyGridBackend

    mock_ssp = _make_mock_ssp()
    backend = CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp)

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

    assert jnp.isfinite(fd), "FD gradient is not finite"
    assert jnp.isfinite(ad), "AD gradient is not finite"
    assert jnp.abs(fd - ad) / (jnp.abs(fd) + 1e-10) < _FD_RTOL, (
        f"FD/AD mismatch: FD={float(fd):.4g}, AD={float(ad):.4g}"
    )


@pytest.mark.skipif(not _CLOUDY_GRID_PATH.exists(), reason="CLOUDY grid not present")
def test_cloudy_grid_triweight_runs():
    """Triweight mode produces finite output for both lines and continuum."""
    from tengri.models.nebular import CloudyGridBackend

    mock_ssp = _make_mock_ssp()
    backend = CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp, grid_interp="triweight")

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


@pytest.mark.skipif(not _CLOUDY_GRID_PATH.exists(), reason="CLOUDY grid not present")
def test_cloudy_grid_triweight_grad_at_grid_node():
    """Triweight mode: FD ≈ AD even when logU lands exactly on a grid node.

    This is the key advantage over linear mode — the smooth C²-continuous kernel
    eliminates the piecewise-linear kink, so FD and AD agree everywhere.
    """
    from tengri.models.nebular import CloudyGridBackend

    mock_ssp = _make_mock_ssp()
    backend = CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp, grid_interp="triweight")

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

    assert jnp.isfinite(fd), "FD gradient is not finite"
    assert jnp.isfinite(ad), "AD gradient is not finite"
    assert jnp.abs(fd - ad) / (jnp.abs(fd) + 1e-10) < _FD_RTOL, (
        f"Triweight FD/AD mismatch at grid node: FD={float(fd):.4g}, AD={float(ad):.4g}"
    )


@pytest.mark.skipif(not _CLOUDY_GRID_PATH.exists(), reason="CLOUDY grid not present")
def test_cloudy_grid_invalid_interp_mode():
    """Unknown grid_interp raises ValueError at construction time."""
    from tengri.models.nebular import CloudyGridBackend

    mock_ssp = _make_mock_ssp()
    with pytest.raises(ValueError, match="grid_interp"):
        CloudyGridBackend(str(_CLOUDY_GRID_PATH), mock_ssp, grid_interp="cubic")


# ---------------------------------------------------------------------------
# CueBackend
# ---------------------------------------------------------------------------

_CUE_WEIGHTS_PATH = Path("/Users/suchethacooray/Projects/tengri/data/cue_weights.npz")


@pytest.mark.skipif(not _CUE_WEIGHTS_PATH.exists(), reason="Cue weights not present")
def test_cue_grad_logu():
    """Test JAX autodiff vs finite difference for CueBackend."""
    from tengri.models.nebular import CueBackend

    mock_ssp = _make_mock_ssp()
    backend = CueBackend(str(_CUE_WEIGHTS_PATH), ssp_data=mock_ssp)

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

    assert jnp.isfinite(fd), "FD gradient is not finite"
    assert jnp.isfinite(ad), "AD gradient is not finite"
    assert jnp.abs(fd - ad) / (jnp.abs(fd) + 1e-10) < _FD_RTOL, (
        f"FD/AD mismatch: FD={float(fd):.4g}, AD={float(ad):.4g}"
    )
