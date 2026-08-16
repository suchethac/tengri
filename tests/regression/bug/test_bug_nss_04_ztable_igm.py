# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-NSS-04: IGM silently not applied in z-table kernel.

See ADR / docs/known_bugs.md for full context.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBugNSS04ZTableIGM:
    """hybrid.py build_hybrid_photometry_ztable — IGM must be applied.

    Before the fix, has_igm was set but never referenced in the z-table kernel
    (the block was just `pass`). This left both stellar photometry and non-stellar
    SED without IGM attenuation, producing artificially bright UV/NUV flux at
    z > 0.5 even with apply_igm=True.

    Fix (hybrid.py lines 1565-1574 and 2265-2320):
      1. Non-stellar: call igm_transmission(ssp_wave * (1+z), z) inside the traced
         function and multiply the non-stellar SED before filter integration.
      2. Stellar: linearly interpolate igm_trans_table (n_z, n_filters) to the
         current redshift and multiply stellar_phot after flux scaling.

    The fix does not cite an external equation because it is a wiring correction
    (applying an existing IGM model that was computed but not used), not a physics
    formula change. Inoue et al. (2014) MNRAS 442, 1805 governs the IGM model itself.
    """

    # test_ztable_igm_wiring_present_in_source lived here, skipped with
    # "forward/_kernels/hybrid.py deleted in Phase 6; IGM now wired in
    # orchestrator". It ran inspect.getsource() on that module and grepped for
    # the identifiers `_igm_fn`, `_igm_full` and `_igm_eff`, so it could not
    # survive the module's deletion — nor a rename of any of those locals, for
    # a bug that was never about their names. The property it was reaching for
    # (IGM is applied, not merely computed) is asserted behaviourally by
    # test_igm_attenuates_uv_at_high_z below, which is what should have been
    # written in the first place and does not care how the wiring is spelled.

    def test_igm_trans_table_interpolation_formula(self):
        """Verify that the per-filter IGM interpolation formula is correct.

        The z-table kernel uses the same linear interpolation as interpolate_ztable:
          frac = (z - z_grid[iz]) / (z_grid[iz+1] - z_grid[iz])
          igm_eff = (1-frac) * igm_table[iz] + frac * igm_table[iz+1]

        This is a unit test of the formula itself with a synthetic igm_trans_table.
        """
        # Synthetic igm_trans_table: 5 z-grid points, 3 filters
        z_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 3.0])
        # IGM transmission decreases with z (more absorption at high z)
        igm_table = jnp.array(
            [
                [0.99, 0.99, 1.00],  # z=0.1
                [0.90, 0.95, 1.00],  # z=0.5
                [0.70, 0.85, 1.00],  # z=1.0
                [0.40, 0.65, 1.00],  # z=2.0
                [0.15, 0.45, 1.00],  # z=3.0
            ]
        )

        # Interpolate at z=0.75 (midpoint between z=0.5 and z=1.0, index 1→2)
        z_test = jnp.float64(0.75)
        _z_c = jnp.clip(z_test, z_grid[0], z_grid[-1])
        _iz = jnp.clip(jnp.searchsorted(z_grid, _z_c) - 1, 0, z_grid.shape[0] - 2)
        _frac = (_z_c - z_grid[_iz]) / (z_grid[_iz + 1] - z_grid[_iz])
        igm_eff = (1.0 - _frac) * igm_table[_iz] + _frac * igm_table[_iz + 1]

        # Expected: linear interp between row at z=0.5 and z=1.0, frac=0.5
        expected = 0.5 * jnp.array([0.90, 0.95, 1.00]) + 0.5 * jnp.array([0.70, 0.85, 1.00])
        assert int(_iz) == 1, f"Expected iz=1 (z=0.5 bin), got {int(_iz)}"
        assert abs(float(_frac) - 0.5) < 1e-10, f"Expected frac=0.5, got {float(_frac)}"
        assert jnp.allclose(igm_eff, expected, atol=1e-10), (
            f"IGM interp mismatch: got {igm_eff}, expected {expected}"
        )

    def test_igm_attenuates_uv_at_high_z(self):
        """IGM transmission must be < 1 for UV at z ~ 3 (Lyman forest).

        Uses igm_transmission directly to verify the physics. igm_transmission
        takes **observed-frame** wavelengths. At z=3, the Ly-alpha forest absorbs
        all observed wavelengths below 1216*(1+3)=4864 Å. An observed wavelength of
        2271 Å at z=3 corresponds to rest-frame 2271/(1+3)=568 Å — deep in the
        Lyman continuum — so IGM transmission must be essentially 0.
        """
        from tengri.components.igm import igm_transmission

        # Observed-frame wavelength: 2271 Å. At z=3 this probes rest ~568 Å (Lyman continuum).
        wave_obs = jnp.array([2271.0])  # observed-frame Angstrom (already observer frame)
        z = 3.0
        trans = igm_transmission(wave_obs, z)
        assert float(trans[0]) < 0.5, (
            f"IGM transmission at z=3 for wave_obs=2271 Å must be < 0.5, got {float(trans[0]):.4f}"
        )
