# SPDX-License-Identifier: BSD-3-Clause
"""compute_mass_remaining_fraction reachable on tengri.* — closes #447."""

import jax.numpy as jnp
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract


def test_top_level_export():
    assert hasattr(tengri, "compute_mass_remaining_fraction")
    assert "compute_mass_remaining_fraction" in tengri.__all__


def test_sps_namespace_export():
    assert hasattr(tengri.sps, "compute_mass_remaining_fraction")


def test_returns_n_age_curve_in_unit_interval():
    """Per-age surviving-mass fraction must be a (n_age,) curve in (0, 1]."""
    age_gyr = jnp.array([0.001, 0.01, 0.1, 1.0, 5.0, 10.0])
    frac = tengri.compute_mass_remaining_fraction(age_gyr, imf="chabrier")
    frac_np = np.asarray(frac)
    assert frac.shape == (6,)
    # All entries in (0, 1].
    assert (frac_np > 0).all()
    assert (frac_np <= 1.0 + 1e-6).all()
    # Monotonically decreasing — older populations have more mass return.
    diffs = np.diff(frac_np)
    assert (diffs <= 1e-6).all(), f"surviving fraction should be monotonic, got {diffs}"


def test_unknown_imf_raises():
    age = jnp.array([1.0])
    with pytest.raises(ValueError, match="Unknown IMF"):
        tengri.compute_mass_remaining_fraction(age, imf="bogus")
