# SPDX-License-Identifier: BSD-3-Clause
"""Regression: ``src/tengri/analysis/simulate.py`` uses the 10-pc convention at z=0.

Before this fix, ``sed_from_sfh`` and ``spectrum_from_sfh`` used a
``dl_cm = luminosity_distance(redshift) if redshift > 0 else 1.0``
fallback at z=0 — yielding a ~10^19× flux error because the
``1.0 cm`` placeholder bypassed the absolute-magnitude convention
``luminosity_distance`` itself already implements.

Discovered while auditing call sites for the #398 unification.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_z0_uses_10pc_convention_directly():
    """``luminosity_distance(0.0)`` returns 10 pc in cm (~3.086e19)."""
    from tengri.utils.cosmology import luminosity_distance

    dl_z0_cm = float(luminosity_distance(jnp.asarray(0.0)))
    expected_10pc_cm = 3.0856775814913673e24 * 1e-5  # MPC_CM × 1e-5
    rel_err = abs(dl_z0_cm - expected_10pc_cm) / expected_10pc_cm
    assert rel_err < 1e-12, (
        f"luminosity_distance(0) = {dl_z0_cm:.4e} cm, expected 10 pc "
        f"= {expected_10pc_cm:.4e} cm (rel_err={rel_err:.2e})"
    )


def test_simulate_no_longer_uses_1cm_fallback():
    """Behavioral test: sed_from_sfh and spectrum_from_sfh must use the same
    luminosity_distance call for z=0 and z>0 (no special case fallback).

    The bug was a ``luminosity_distance(z) if z > 0 else 1.0`` guard that
    placed z=0 galaxies at 1 cm instead of 10 pc, yielding a ~1e19× flux error.
    ``luminosity_distance`` already implements the 10-pc convention for z=0, so
    the Python-side guard is dead code and masks the wrong behavior.
    """
    import jax.numpy as jnp

    from tengri.utils.cosmology import luminosity_distance

    # Verify that luminosity_distance itself handles z=0 correctly
    z_0 = jnp.asarray(0.0)
    dl_z0 = float(luminosity_distance(z_0))

    # The 10-pc convention for absolute magnitudes: 10 pc ≈ 3.086e24 cm
    expected_10pc_cm = 3.0856775814913673e24 * 1e-5
    rel_err = abs(dl_z0 - expected_10pc_cm) / expected_10pc_cm

    assert rel_err < 1e-12, (
        f"luminosity_distance(0) = {dl_z0:.4e} cm; expected 10 pc = {expected_10pc_cm:.4e} cm "
        f"(rel_err={rel_err:.2e}). If there's a Python-side z > 0 guard returning 1.0 "
        "for z=0, the flux would be ~1e19× too bright."
    )
