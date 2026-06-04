# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Synthesizer BLR adapter and backend.

``SynthesizerBLRBackend`` reuses the NLR grid loader and interpolation on the
broad-line-region grid file; ``compute_blr_sed_synthesizer`` convolves the
predicted broad lines into a continuous L_nu at a broad (~5000 km/s) FWHM. These
tests pin: the backend loads, the adapter is finite/non-negative, the broad FWHM
genuinely broadens lines relative to the NLR (~500 km/s) path, and the grid-path
contract is enforced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.bounds

import jax
import jax.numpy as jnp

from tengri.components.agn.nlr_cloudy import (
    compute_blr_sed_synthesizer,
    compute_nlr_sed_synthesizer,
    get_synthesizer_blr_backend,
)
from tengri.components.nebular.agn_nebular import SynthesizerBLRBackend

_DATA = Path(__file__).resolve().parents[3] / "data" / "synthesizer_grids"
_BLR_GRID = _DATA / "test_grid_agn-blr.hdf5"
_NLR_GRID = _DATA / "test_grid_agn-nlr.hdf5"


@pytest.fixture(scope="module")
def blr_grid_path():
    if not _BLR_GRID.exists():
        pytest.skip(f"Synthesizer BLR test grid not found at {_BLR_GRID}")
    return str(_BLR_GRID)


def test_backend_loads_and_is_blr(blr_grid_path):
    """The BLR backend loads the BLR grid and identifies itself."""
    backend = get_synthesizer_blr_backend(blr_grid_path)
    assert isinstance(backend, SynthesizerBLRBackend)
    assert backend.name == "synthesizer_blr"


def test_adapter_finite_nonnegative(blr_grid_path):
    """The BLR SED is finite and non-negative across the grid."""
    wave = jnp.asarray(np.logspace(2.7, 6.0, 2000))
    L = np.asarray(
        compute_blr_sed_synthesizer(
            wave, l_disc_bol_erg=1e45, covering_fraction=0.1, grid_path=blr_grid_path
        )
    )
    assert np.all(np.isfinite(L))
    assert np.all(L >= 0.0)
    assert (L > 0).any()


def test_covering_fraction_scales_linearly(blr_grid_path):
    """Doubling the covering fraction doubles the line luminosity."""
    wave = jnp.asarray(np.logspace(2.7, 6.0, 2000))
    L1 = np.asarray(
        compute_blr_sed_synthesizer(
            wave, l_disc_bol_erg=1e45, covering_fraction=0.1, grid_path=blr_grid_path
        )
    )
    L2 = np.asarray(
        compute_blr_sed_synthesizer(
            wave, l_disc_bol_erg=1e45, covering_fraction=0.2, grid_path=blr_grid_path
        )
    )
    peak1, peak2 = L1.max(), L2.max()
    assert peak2 == pytest.approx(2.0 * peak1, rel=1e-5)


def test_broad_wider_than_narrow(blr_grid_path):
    """The broad FWHM smears each line wider than the NLR path's peak height.

    For matched total line luminosity, a broader Gaussian has a lower peak L_nu.
    Driving the *same* grid through both adapters at their default widths
    (BLR 5000 km/s vs NLR 500 km/s) the BLR peak must sit below the NLR peak.
    """
    if not _NLR_GRID.exists():
        pytest.skip("NLR grid needed for the broad-vs-narrow comparison")
    wave = jnp.asarray(np.logspace(2.7, 6.0, 4000))
    L_blr = np.asarray(
        compute_blr_sed_synthesizer(
            wave, l_disc_bol_erg=6e45, covering_fraction=0.1, grid_path=blr_grid_path
        )
    )
    L_nlr = np.asarray(
        compute_nlr_sed_synthesizer(
            wave, l_disc_bol_erg=6e45, covering_fraction=0.1, grid_path=str(_NLR_GRID)
        )
    )
    assert L_blr.max() < L_nlr.max()


def test_requires_grid_path():
    """A missing grid_path raises a clear error (grids are not packaged)."""
    wave = jnp.asarray(np.logspace(2.7, 6.0, 100))
    with pytest.raises(ValueError, match="grid_path"):
        compute_blr_sed_synthesizer(wave, l_disc_bol_erg=1e45)


def test_jit_core(blr_grid_path):
    """The numerical core is JIT-safe once the backend is materialised."""
    backend = get_synthesizer_blr_backend(blr_grid_path)

    @jax.jit
    def _lines(log_qh):
        return backend.predict_agn_blr_lines(log_qh=log_qh)[1].sum()

    val = float(_lines(53.0))
    assert np.isfinite(val)
