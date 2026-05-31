# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate the Slone & Netzer (2012) disc module against AGNfitter-rX.

The SN12 alpha-disc templates were ported from AGNfitter-rX's ``SN12.pickle``
into tengri's HDF5 grid by ``scripts/build_slone_netzer_grid.py``. This test
reads both the original pickle (via the build script's safe unpickler) and
tengri's runtime grid, and verifies the templates match after regridding to a
common wavelength axis.

The Eddington axis follows AGNfitter-rX's own labelling: SED column ``j`` is
``logEddra-values[j]`` (the first 12 of the 259-entry list). See the build
script for the full provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chex
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_GRID_PATH = _DATA_DIR / "slone_netzer_disc_grid.h5"
_AGNFITTER_PICKLE = Path("/tmp/AGNfitter-rX/models/BBB/SN12.pickle")
_N_EDD = 12

if not _GRID_PATH.is_file():
    pytest.skip(
        "Slone & Netzer grid not found at " + str(_GRID_PATH),
        allow_module_level=True,
    )

if not _AGNFITTER_PICKLE.is_file():
    pytest.skip(
        "AGNfitter SN12.pickle not found at "
        + str(_AGNFITTER_PICKLE)
        + " (clone with: git clone --branch AGNfitter-rX_v0.1 "
        + "https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX)",
        allow_module_level=True,
    )


def _safe_load_sn12(pickle_path: Path) -> dict:
    """Safely unpickle SN12.pickle using the build script's allow-list."""
    from build_slone_netzer_grid import _preflight_opcode_scan, _RestrictedUnpickler

    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        obj = _RestrictedUnpickler(fh, encoding="latin1").load()
    if not isinstance(obj, dict):
        raise TypeError(f"SN12 pickle root is {type(obj).__name__}, expected dict.")
    return obj


@pytest.fixture(scope="module")
def sn12_grid():
    """Load the tengri Slone & Netzer grid (numpy arrays)."""
    with h5py.File(str(_GRID_PATH), "r") as f:
        g = f["slone_netzer"]
        return {
            "log_mbh": np.asarray(g["log_mbh"][:], dtype=np.float64),
            "log_edd": np.asarray(g["log_edd"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def agnfitter_sn12():
    """Load AGNfitter-rX's SN12.pickle (safely)."""
    return _safe_load_sn12(_AGNFITTER_PICKLE)


class TestSloneNetzerPort:
    """The ported grid reproduces AGNfitter-rX's SN12 templates."""

    def test_axes_match(self, sn12_grid, agnfitter_sn12):
        """M_BH and Eddington axes match AGNfitter-rX's labelling."""
        np.testing.assert_allclose(
            sn12_grid["log_mbh"],
            np.asarray(agnfitter_sn12["logBHmass-values"], dtype=np.float64).ravel(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            sn12_grid["log_edd"],
            np.asarray(agnfitter_sn12["logEddra-values"], dtype=np.float64).ravel()[:_N_EDD],
            rtol=1e-12,
        )

    @pytest.mark.parametrize("mbh_idx,edd_idx", [(0, 0), (4, 6), (8, 11)])
    def test_template_matches(self, sn12_grid, agnfitter_sn12, mbh_idx, edd_idx):
        """tengri's stored template matches AGNfitter's regridded SED to <1%."""
        c_aa_s = 2.99792458e18
        freq = np.asarray(agnfitter_sn12["frequency"], dtype=np.float64).ravel()
        sed = np.asarray(agnfitter_sn12["SED"], dtype=np.float64)[:, mbh_idx, edd_idx]
        wave = c_aa_s / freq
        order = np.argsort(wave)
        agn_regridded = np.interp(
            sn12_grid["wavelength"], wave[order], sed[order], left=0.0, right=0.0
        )
        np.testing.assert_allclose(
            sn12_grid["template"][mbh_idx, edd_idx],
            agn_regridded,
            rtol=0.01,
            atol=1e-30,
            err_msg=f"Port diverges at (mbh={mbh_idx}, edd={edd_idx})",
        )


class TestSloneNetzerRuntime:
    """The runtime closure behaves correctly under JAX."""

    @pytest.fixture(scope="class")
    def runtime(self):
        from tengri.components.agn.slone_netzer import create_slone_netzer_from_grid

        return create_slone_netzer_from_grid(str(_GRID_PATH))

    def test_evaluates_finite(self, runtime):
        wavelength = np.geomspace(500.0, 5e4, 256)
        sed = runtime(wavelength, agn_log_lbol=11.0, agn_log_mbh=8.6, agn_log_ledd=-2.0)
        chex.assert_tree_all_finite(sed)
        assert sed.shape == wavelength.shape

    def test_luminosity_scaling(self, runtime):
        wavelength = np.geomspace(500.0, 5e4, 256)
        lo = runtime(wavelength, agn_log_lbol=10.0, agn_log_mbh=8.6, agn_log_ledd=-2.0)
        hi = runtime(wavelength, agn_log_lbol=10.301, agn_log_mbh=8.6, agn_log_ledd=-2.0)
        assert np.median(hi / (lo + 1e-30)) > 1.5

    def test_gradient_flows(self, runtime):
        def loss(log_mbh):
            wavelength = np.geomspace(500.0, 5e4, 64)
            return jnp.sum(
                runtime(wavelength, agn_log_lbol=11.0, agn_log_mbh=log_mbh, agn_log_ledd=-2.0)
            )

        grad = jax.grad(loss)(8.6)
        assert np.isfinite(grad)
        assert abs(grad) > 0
