# SPDX-License-Identifier: BSD-3-Clause
"""Tests for nebular backend warning and error emission.

These tests verify that:
1. Each backend emits the correct warning class at instantiation
2. The 'raise' mode raises the correct exception type
3. The 'suppress' mode is silent
4. has_continuum attributes are correct
"""

import contextlib
import warnings
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_CB19_GRID_PATH = Path(__file__).resolve().parents[2] / "data" / "cb19_templates.h5"

from tengri.components.nebular import (
    BakedInBackend,
    BakedInNebularWarning,
    CB19Backend,
    CB19IonizingSpectrumWarning,
    CB19NoContinuumWarning,
    IonizingSpectrumInconsistencyError,
    IonizingSpectrumInconsistencyWarning,
    MappingsPhotoAGNBackend,
    MappingsPhotoStellarBackend,
    NebularBackend,
)

# ── BakedInBackend ────────────────────────────────────────────────


def test_baked_in_warns_by_default():
    """BakedInBackend warns with BakedInNebularWarning by default."""
    with pytest.warns(BakedInNebularWarning):
        BakedInBackend()


def test_baked_in_raises_when_requested():
    """BakedInBackend raises ValueError in 'raise' mode."""
    with pytest.raises(ValueError, match="fixed logU"):
        BakedInBackend(ionizing_source_warning="raise")


def test_baked_in_suppress_is_silent():
    """BakedInBackend is silent in 'suppress' mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BakedInBackend(ionizing_source_warning="suppress")  # Must not raise


def test_baked_in_has_continuum_true():
    """BakedInBackend.has_continuum is True (already in SSP)."""
    b = BakedInBackend(ionizing_source_warning="suppress")
    assert b.has_continuum is True


# ── CB19Backend ───────────────────────────────────────────────────


@pytest.mark.skipif(not _CB19_GRID_PATH.exists(), reason="CB19 grid file not present")
def test_cb19_warns_by_default():
    """CB19Backend warns with CB19IonizingSpectrumWarning by default."""
    with pytest.warns(CB19IonizingSpectrumWarning):
        CB19Backend(ionizing_source_warning="warn", continuum_warning="suppress")


def test_cb19_continuum_warning():
    """CB19Backend warns about continuum absence."""
    with pytest.warns(CB19NoContinuumWarning), contextlib.suppress(FileNotFoundError):
        CB19Backend(ionizing_source_warning="suppress", continuum_warning="warn")


def test_cb19_has_continuum_false():
    """CB19Backend.has_continuum is False (lines only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b = CB19Backend(ionizing_source_warning="suppress", continuum_warning="suppress")
        except FileNotFoundError:
            pytest.skip("CB19 grid file not present")
    assert b.has_continuum is False


# ── CB19 degenerate-grid guard (#924) ─────────────────────────────


def _write_flat_cb19_grid(src: Path, dst: Path, value: float = 1.0) -> None:
    """Copy a CB19 grid file but flatten its line_ratios to a single constant.

    Reproduces the shipped-placeholder failure mode of #924, where every entry
    of ``grids/SSP/Kroupa01/mu100/line_ratios`` is identical.
    """
    import shutil

    import h5py
    import numpy as np

    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        ds = f["grids/SSP/Kroupa01/mu100/line_ratios"]
        ds[...] = np.full(ds.shape, value, dtype=ds.dtype)


@pytest.mark.skipif(not _CB19_GRID_PATH.exists(), reason="CB19 grid file not present")
def test_cb19_load_warns_on_degenerate_grid(tmp_path):
    """load_cb19_grid warns loudly when every line ratio is identical (#924).

    A flat grid gives all 10 emission lines the same luminosity (Halpha/Hbeta =
    1.0 instead of the Case B 2.87) — plausible-looking, silently wrong line
    physics. The backend must surface this at load time, not trust it.
    """
    from tengri.components.nebular.cloudy_cb19 import (
        CB19DegenerateGridWarning,
        load_cb19_grid,
    )

    degenerate = tmp_path / "cb19_flat.h5"
    _write_flat_cb19_grid(_CB19_GRID_PATH, degenerate)
    with pytest.warns(CB19DegenerateGridWarning, match="DEGENERATE"):
        load_cb19_grid(degenerate)


@pytest.mark.skipif(not _CB19_GRID_PATH.exists(), reason="CB19 grid file not present")
def test_cb19_load_silent_on_real_grid():
    """The shipped (non-degenerate) CB19 grid loads without the degenerate warning."""
    from tengri.components.nebular.cloudy_cb19 import (
        CB19DegenerateGridWarning,
        load_cb19_grid,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", CB19DegenerateGridWarning)
        load_cb19_grid(_CB19_GRID_PATH)  # must not raise the degenerate-grid warning


# ── MappingsPhotoStellarBackend ───────────────────────────────────


def test_mappings_stellar_raises_by_default():
    """MappingsPhotoStellarBackend raises IonizingSpectrumInconsistencyError
    with default ionizing_source_warning='raise'."""
    with pytest.raises(IonizingSpectrumInconsistencyError):
        MappingsPhotoStellarBackend("nonexistent.h5", ionizing_source_warning="raise")


def test_mappings_stellar_warn_mode():
    """MappingsPhotoStellarBackend warns in 'warn' mode."""
    with (
        pytest.warns(IonizingSpectrumInconsistencyWarning),
        contextlib.suppress(FileNotFoundError, OSError),
    ):
        MappingsPhotoStellarBackend("nonexistent.h5", ionizing_source_warning="warn")


def test_mappings_stellar_suppress_mode():
    """MappingsPhotoStellarBackend is silent in 'suppress' mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", IonizingSpectrumInconsistencyWarning)
        with contextlib.suppress(FileNotFoundError, OSError):
            MappingsPhotoStellarBackend("nonexistent.h5", ionizing_source_warning="suppress")


# ── MappingsPhotoAGNBackend ───────────────────────────────────────


def test_mappings_agn_warns_by_default():
    """MappingsPhotoAGNBackend warns with default ionizing_source_warning='warn'."""
    with (
        pytest.warns(IonizingSpectrumInconsistencyWarning),
        contextlib.suppress(FileNotFoundError, OSError),
    ):
        MappingsPhotoAGNBackend("nonexistent.h5")


def test_mappings_agn_suppress_is_silent():
    """MappingsPhotoAGNBackend is silent in 'suppress' mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", IonizingSpectrumInconsistencyWarning)
        with contextlib.suppress(FileNotFoundError, OSError):
            MappingsPhotoAGNBackend("nonexistent.h5", ionizing_source_warning="suppress")


# ── has_continuum correctness ─────────────────────────────────────


def test_has_continuum_table():
    """Verify has_continuum for all backends that can be instantiated without
    grid files."""
    b_baked = BakedInBackend(ionizing_source_warning="suppress")
    assert b_baked.has_continuum is True, "BakedIn should have continuum (in SSP)"

    # CueBackend — skip if weights file missing
    from pathlib import Path

    from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH

    if Path(_DEFAULT_CUE_WEIGHTS_PATH).exists():
        from tengri.components.nebular import CueBackend

        c = CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))
        assert c.has_continuum is True, "Cue should have continuum"


# ── Protocol compliance ───────────────────────────────────────────


def test_baked_in_satisfies_protocol():
    """BakedInBackend satisfies NebularBackend Protocol."""
    b = BakedInBackend(ionizing_source_warning="suppress")
    assert isinstance(b, NebularBackend), "BakedInBackend must satisfy NebularBackend Protocol"
