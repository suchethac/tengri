#!/usr/bin/env python
"""Measure gradient differences between old (float32-rounded) and new paths."""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import tempfile

# Force x64 on
jax.config.update("jax_enable_x64", True)

from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tests._cb19_grid import write_synthetic_cb19_grid
from tengri import load_ssp_data

# Create backends
with tempfile.TemporaryDirectory() as tmpdir:
    grid_path = write_synthetic_cb19_grid(Path(tmpdir) / "cb19_templates.h5")

    # Load SSP data from the data directory
    ssp_path = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    ssp_data = load_ssp_data(ssp_path)
    backend = CB19Backend(grid_path=grid_path, ssp_data=ssp_data)

    print("Measuring gradient differences (w.r.t. neb_co) between old and new paths:")
    print()

    ssp_log_ages_yr = jnp.array([np.log10(100e6)])
    ssp_weights = jnp.array([1.0])
    log_z = np.float64(-0.5)

    def objective_new(neb_co_val):
        _, lum = backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=np.float64(-2.5),
            neb_logZ_gas=np.float64(-0.5),
            neb_fesc=np.float64(0.0),
            neb_fesc_lya=np.float64(0.0),
            neb_fdust=np.float64(0.0),
            neb_log_nH=np.float64(2.0),
            neb_co=neb_co_val,
            neb_dno=np.float64(0.0),
            neb_hbfrac=np.float64(0.5),
        )
        return jnp.sum(lum)

    # Compute gradient with new code (current)
    grad_fn_new = jax.grad(objective_new)
    grad_new = float(grad_fn_new(np.float64(-0.4)))

    # Compute gradient with old code
    import tengri.components.nebular.cloudy_cb19 as cb19_module
    old_frac_idx = cb19_module._frac_idx
    cb19_module._frac_idx = lambda val, grid: (
        (
            jnp.clip(jnp.searchsorted(grid, jnp.clip(val, grid[0], grid[-1]), side="right") - 1, 0, grid.shape[0] - 2)
            + jnp.where(
                (grid[jnp.clip(jnp.searchsorted(grid, jnp.clip(val, grid[0], grid[-1]), side="right") - 1, 0, grid.shape[0] - 2) + 1]
                 - grid[jnp.clip(jnp.searchsorted(grid, jnp.clip(val, grid[0], grid[-1]), side="right") - 1, 0, grid.shape[0] - 2)])
                > 0,
                (jnp.clip(val, grid[0], grid[-1])
                 - grid[jnp.clip(jnp.searchsorted(grid, jnp.clip(val, grid[0], grid[-1]), side="right") - 1, 0, grid.shape[0] - 2)])
                / (grid[jnp.clip(jnp.searchsorted(grid, jnp.clip(val, grid[0], grid[-1]), side="right") - 1, 0, grid.shape[0] - 2) + 1]
                   - grid[jnp.clip(jnp.searchsorted(grid, jnp.clip(val, grid[0], grid[-1]), side="right") - 1, 0, grid.shape[0] - 2)]),
                0.0
            )
        ).astype(jnp.float32)
    )

    def objective_old(neb_co_val):
        _, lum = backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=np.float64(-2.5),
            neb_logZ_gas=np.float64(-0.5),
            neb_fesc=np.float64(0.0),
            neb_fesc_lya=np.float64(0.0),
            neb_fdust=np.float64(0.0),
            neb_log_nH=np.float64(2.0),
            neb_co=neb_co_val,
            neb_dno=np.float64(0.0),
            neb_hbfrac=np.float64(0.5),
        )
        return jnp.sum(lum)

    grad_fn_old = jax.grad(objective_old)
    grad_old = float(grad_fn_old(np.float64(-0.4)))

    cb19_module._frac_idx = old_frac_idx

    print(f"Gradient (new, float64 coords):  {grad_new:.6e}")
    print(f"Gradient (old, float32 coords):  {grad_old:.6e}")
    print(f"Difference:                      {abs(grad_new - grad_old):.6e}")

    if abs(grad_old) > 1e-20:
        rel_diff = abs(grad_new - grad_old) / abs(grad_old)
        print(f"Relative difference:             {rel_diff:.6e}")
    else:
        print(f"Old gradient is essentially zero (underflow occurred)")

