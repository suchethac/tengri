#!/usr/bin/env python
"""Measure real float64 SED differences between old (float32-rounded) and new coordinate paths."""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import tempfile

jax.config.update("jax_enable_x64", True)

from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tests._cb19_grid import write_synthetic_cb19_grid
from tengri import load_ssp_data

with tempfile.TemporaryDirectory() as tmpdir:
    grid_path = write_synthetic_cb19_grid(Path(tmpdir) / "cb19_templates.h5")
    ssp_path = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    ssp_data = load_ssp_data(ssp_path)
    backend = CB19Backend(grid_path=grid_path, ssp_data=ssp_data)

    # Three NON-grid-aligned parameter points
    test_points = [
        {"neb_logU": -3.123456789, "neb_logZ_gas": 0.0123456},
        {"neb_logU": -2.71828182, "neb_logZ_gas": -0.0654321},
        {"neb_logU": -2.5, "neb_logZ_gas": 0.0987654},
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

    print("Measuring float64 SED differences (non-grid-aligned points):")
    print()

    max_rel_diff = 0.0
    for i, params in enumerate(test_points):
        ssp_log_ages_yr = jnp.array([np.log10(100e6)])
        ssp_weights = jnp.array([1.0])

        # Measure with NEW code (current)
        _, lum_new = backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=np.float64(-0.5),
            neb_logU=np.float64(params["neb_logU"]),
            neb_logZ_gas=np.float64(params["neb_logZ_gas"]),
            neb_fesc=np.float64(0.0),
            neb_fesc_lya=np.float64(0.0),
            neb_fdust=np.float64(0.0),
            neb_log_nH=np.float64(2.0),
            neb_co=np.float64(-0.4),
            neb_dno=np.float64(0.0),
            neb_hbfrac=np.float64(0.5),
        )

        # Monkeypatch OLD code in
        import tengri.components.nebular.cloudy_cb19 as cb19_module
        old_frac_idx = cb19_module._frac_idx
        cb19_module._frac_idx = _frac_idx_old

        _, lum_old = backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=np.float64(-0.5),
            neb_logU=np.float64(params["neb_logU"]),
            neb_logZ_gas=np.float64(params["neb_logZ_gas"]),
            neb_fesc=np.float64(0.0),
            neb_fesc_lya=np.float64(0.0),
            neb_fdust=np.float64(0.0),
            neb_log_nH=np.float64(2.0),
            neb_co=np.float64(-0.4),
            neb_dno=np.float64(0.0),
            neb_hbfrac=np.float64(0.5),
        )

        cb19_module._frac_idx = old_frac_idx

        # Compute differences
        lum_new = np.asarray(lum_new, dtype=np.float64)
        lum_old = np.asarray(lum_old, dtype=np.float64)

        rel_diff = np.max(np.abs((lum_new - lum_old) / np.maximum(np.abs(lum_old), 1e-20)))
        max_rel_diff = max(max_rel_diff, rel_diff)

        print(f"Point {i+1} (logU={params['neb_logU']}, logZ_gas={params['neb_logZ_gas']}):")
        print(f"  max|new-old|/|old| = {rel_diff:.3e}")

    print()
    print(f"Overall max relative difference: {max_rel_diff:.3e}")
