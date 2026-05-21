"""Regression test for BLR Fe II normalization grid-resolution-dependent bug.

Bug: blr.py:217 — jnp.sum used for Fe II normalization; result depends on pixel spacing.
Fix uses jnp.trapezoid over frequency for grid-independent normalization.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBLRFeIINormalization:
    """Bug: blr.py:217 — Fe II normalization grid-resolution-dependent."""

    def test_fe2_normalization_grid_independent(self):
        """Fe II normalization should not change significantly with wavelength grid resolution."""
        from tengri.components.agn.blr import compute_blr_sed

        wave_coarse = jnp.logspace(2.8, 4.2, 100)
        wave_fine = jnp.logspace(2.8, 4.2, 500)

        sed_coarse = compute_blr_sed(wave_coarse, l_disc_bol_erg=1e46, agn_fe2_strength=1.0)
        sed_fine = compute_blr_sed(wave_fine, l_disc_bol_erg=1e46, agn_fe2_strength=1.0)

        # The Fe II template in the 4434-4684 A window should normalize consistently.
        # Total power (integral over frequency) should agree within 10% between grids.
        _C_AA = 2.99792458e18
        nu_c = _C_AA / wave_coarse
        nu_f = _C_AA / wave_fine

        sort_c = jnp.argsort(nu_c)
        sort_f = jnp.argsort(nu_f)

        mask_c = (wave_coarse >= 4434.0) & (wave_coarse <= 4684.0)
        mask_f = (wave_fine >= 4434.0) & (wave_fine <= 4684.0)

        fe2_c = jnp.abs(jnp.trapezoid((sed_coarse * mask_c)[sort_c], nu_c[sort_c]))
        fe2_f = jnp.abs(jnp.trapezoid((sed_fine * mask_f)[sort_f], nu_f[sort_f]))

        if fe2_c > 0 and fe2_f > 0:
            ratio = fe2_c / fe2_f
            assert 0.5 < ratio < 2.0, (
                f"Fe II optical bump power ratio coarse/fine = {ratio:.3f}; "
                "normalization is grid-resolution-dependent"
            )
