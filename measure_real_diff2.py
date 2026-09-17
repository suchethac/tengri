#!/usr/bin/env python
"""Measure real float64 SED differences using predict_nebular_sed."""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import tempfile
import warnings

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings('ignore')

from tengri.components.nebular.cloudy_cb19 import CB19Backend, _frac_idx
from tests._cb19_grid import write_synthetic_cb19_grid
from tengri import load_ssp_data

with tempfile.TemporaryDirectory() as tmpdir:
    grid_path = write_synthetic_cb19_grid(Path(tmpdir) / "cb19_templates.h5")
    ssp_path = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    ssp_data = load_ssp_data(ssp_path)

    # Three NON-grid-aligned parameter points
    test_points = [
        (-3.123456789, 0.0123456),
        (-2.71828182, -0.0654321),
        (-2.5, 0.0987654),
    ]

    # Define old _frac_idx with float32 cast
    def _frac_idx_old(val, grid):
        val_clipped = jnp.clip(val, grid[0], grid[-1])
        n = grid.shape[0]
        idx = jnp.searchsorted(grid, val_clipped, side="right") - 1
        idx = jnp.clip(idx, 0, n - 2)
        dx = grid[idx + 1] - grid[idx]
        frac = jnp.where(dx > 0, (val_clipped - grid[idx]) / dx, 0.0)
        return (idx + frac).astype(jnp.float32)  # OLD: cast to float32

    print("Measuring float64 SED differences (predict_nebular_sed):")
    print()

    max_rel_diff = 0.0
    for i, (neb_logU, neb_logZ_gas) in enumerate(test_points):
        # Create fresh backends for each test
        backend_new = CB19Backend(grid_path=grid_path, ssp_data=ssp_data)

        # Common parameters
        wave = np.linspace(1000, 1e5, 500)
        p = {
            "neb_logU": np.float64(neb_logU),
            "neb_logZ_gas": np.float64(neb_logZ_gas),
            "neb_fesc": np.float64(0.0),
            "neb_fesc_lya": np.float64(0.0),
            "neb_fdust": np.float64(0.0),
            "neb_log_nH": np.float64(2.0),
            "neb_co": np.float64(-0.4),
            "neb_dno": np.float64(0.0),
            "neb_hbfrac": np.float64(0.5),
        }

        # Measure with NEW code
        sed_new = backend_new.predict_nebular_sed(p, np.float64(0.0), wave)

        # Swap in old _frac_idx
        import tengri.components.nebular.cloudy_cb19 as cb19_module
        old_frac_idx_ref = cb19_module._frac_idx
        cb19_module._frac_idx = _frac_idx_old

        # Create fresh backend with old _frac_idx active
        backend_old = CB19Backend(grid_path=grid_path, ssp_data=ssp_data)
        sed_old = backend_old.predict_nebular_sed(p, np.float64(0.0), wave)

        # Restore
        cb19_module._frac_idx = old_frac_idx_ref

        # Compute differences
        sed_new = np.asarray(sed_new, dtype=np.float64)
        sed_old = np.asarray(sed_old, dtype=np.float64)

        # Only consider non-zero elements
        nonzero_mask = np.abs(sed_old) > 1e-25
        if np.any(nonzero_mask):
            rel_diff = np.max(np.abs((sed_new[nonzero_mask] - sed_old[nonzero_mask]) / sed_old[nonzero_mask]))
        else:
            rel_diff = 0.0

        max_rel_diff = max(max_rel_diff, rel_diff)

        print(f"Point {i+1} (logU={neb_logU}, logZ_gas={neb_logZ_gas}):")
        print(f"  max|new-old|/|old| = {rel_diff:.3e}")

    print()
    print(f"Overall max relative difference: {max_rel_diff:.3e}")
