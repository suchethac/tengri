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
    """Read the source of ``simulate.py`` and assert the broken
    ``else 1.0`` fallback is gone. Static check rather than a
    physical round-trip — the round-trip test would need SSP
    fixtures and the entire forward model to set up."""
    import inspect

    from tengri.analysis import simulate

    src = inspect.getsource(simulate)
    assert "luminosity_distance(redshift) if redshift > 0 else 1.0" not in src, (
        "Reintroduced the 10^19× z=0 flux bug. ``luminosity_distance`` "
        "already handles z=0 via the 10-pc absolute-magnitude convention; "
        "the Python-side guard is dead code."
    )
