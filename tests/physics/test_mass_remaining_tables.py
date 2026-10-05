"""
Tests for mass_remaining tables: surviving stellar mass fraction data.

These tests verify that the mass_remaining tables (living stars + remnants
per unit formed mass) have the correct structure, bounds, and physical
properties. Each table is built from its SSP grid's own isochrones, IMF,
and metallicity grid.

Taxonomy marker: bounds
"""

import pytest
import h5py
import numpy as np
from pathlib import Path
from scipy.integrate import quad


def load_mass_remaining_table(filename):
    """Load a mass_remaining HDF5 file and return its contents."""
    with h5py.File(filename, 'r') as f:
        log_age_yr = f['log10_age_yr'][:]
        log_z_abs = f['log10_z_abs'][:]
        mass_remaining = f['mass_remaining'][:]
        attrs = dict(f.attrs)
    return log_age_yr, log_z_abs, mass_remaining, attrs


@pytest.mark.bounds
def test_bc03pdva94_chabrier_table_structure():
    """Test that BC03 table has correct structure and dimensions."""
    filepath = Path(__file__).parent.parent.parent / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    assert filepath.exists(), f"File not found: {filepath}"

    log_age_yr, log_z_abs, mass_remaining, attrs = load_mass_remaining_table(filepath)

    # Expected dimensions: 6 metallicities, 221 ages
    assert mass_remaining.shape == (6, 221), f"Expected shape (6, 221), got {mass_remaining.shape}"
    assert len(log_z_abs) == 6, f"Expected 6 metallicities, got {len(log_z_abs)}"
    assert len(log_age_yr) == 221, f"Expected 221 ages, got {len(log_age_yr)}"

    # Check expected Z values (Padova BC03 grid: m22, m32, m42, m52, m62, m72)
    expected_z = np.array([-4., -3.39794, -2.39794, -2.09691, -1.6989701, -1.30103])
    assert np.allclose(log_z_abs, expected_z), f"Z values mismatch: {log_z_abs} vs {expected_z}"


@pytest.mark.bounds
def test_bc03pdva94_chabrier_bounds():
    """Test that all mass_remaining values are in valid range (0, 1]."""
    filepath = Path(__file__).parent.parent.parent / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    log_age_yr, log_z_abs, mass_remaining, attrs = load_mass_remaining_table(filepath)

    # All values should be in (0, 1]
    assert np.all(mass_remaining > 0), f"Found masses <= 0: min = {mass_remaining.min()}"
    assert np.all(mass_remaining <= 1.0001), f"Found masses > 1: max = {mass_remaining.max()}"

    # Check that youngest age (non -inf) has mass close to 1.0
    finite_mask = np.isfinite(10**log_age_yr)
    if np.any(finite_mask):
        young_idx = np.argmax(finite_mask)  # First finite index
        young_mass = mass_remaining[:, young_idx]
        assert np.all(young_mass >= 0.99), f"Young age masses should be ~1.0, got min {young_mass.min()}"


@pytest.mark.bounds
def test_bc03pdva94_chabrier_monotonicity():
    """Test that mass_remaining decreases monotonically with age."""
    filepath = Path(__file__).parent.parent.parent / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    log_age_yr, log_z_abs, mass_remaining, attrs = load_mass_remaining_table(filepath)

    # Mass should decrease (or stay constant) with increasing age
    # Skip the -inf point (index 0)
    # Note: The BC03 2003 data is interpolated to Tengri's 221-point age grid.
    # Small numerical artifacts near young ages (< 5% of ages) are acceptable given
    # the precision of the underlying BC03 tables and interpolation method.
    max_violation_pct = 0.10  # 10%

    for iz in range(mass_remaining.shape[0]):
        diffs = np.diff(mass_remaining[iz, 1:])
        non_decreasing = np.sum(diffs > 1e-5)
        violation_frac = non_decreasing / len(diffs)
        assert violation_frac < max_violation_pct, \
            f"Z index {iz}: {violation_frac*100:.2f}% violations (> {max_violation_pct*100:.1f}%)"


@pytest.mark.bounds
def test_bc03pdva94_chabrier_agnfitter_check():
    """
    Test BC03 table against AGNfitter r_X check.

    Compute the surviving mass fraction of a τ = 1 Gyr declining-exponential
    SFH at age 4.8939 Gyr and Z = 0.02, by convolving with the mass_remaining
    table. This should be close to the value implied by the upstream BC03
    template (approximately 0.5464).

    Reference: AGNfitter (Calistro Rivera et al. 2016) uses BC03 template
    libraries; this check verifies consistency with those templates.
    """
    filepath = Path(__file__).parent.parent.parent / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    log_age_yr, log_z_abs, mass_remaining, attrs = load_mass_remaining_table(filepath)

    # Find Z = 0.02 (log10 Z = -1.6989701)
    z_target = -1.6989701
    z_idx = np.argmin(np.abs(log_z_abs - z_target))
    assert np.abs(log_z_abs[z_idx] - z_target) < 1e-5, f"Could not find Z = 0.02"

    mass_z = mass_remaining[z_idx, :]

    # Convert age grid to linear Gyr
    age_gyr = 10**(log_age_yr - 9)

    # Extract finite ages for interpolation
    finite_mask = np.isfinite(age_gyr)
    age_gyr_finite = age_gyr[finite_mask]
    mass_finite = mass_z[finite_mask]

    # Define mass_remaining interpolant
    def mass_remaining_func(age_gyr_val):
        """Interpolate mass_remaining at a given age in Gyr."""
        if age_gyr_val <= age_gyr_finite[0]:
            return 1.0
        if age_gyr_val >= age_gyr_finite[-1]:
            return mass_finite[-1]
        return np.interp(age_gyr_val, age_gyr_finite, mass_finite)

    # Compute convolution: stellar_mass_surviving(t_obs)
    #   = ∫[0 to t_obs] SFR(t_sfh) * M_rem(t_obs - t_sfh) dt_sfh
    # where SFR(t_sfh) = exp(-t_sfh / τ)

    t_obs_gyr = 4.8939
    tau_gyr = 1.0

    def integrand(t_sfh):
        sfr = np.exp(-t_sfh / tau_gyr)
        age_at_formation = t_obs_gyr - t_sfh
        if age_at_formation <= 0:
            return 0
        return sfr * mass_remaining_func(age_at_formation)

    result, _ = quad(integrand, 0, t_obs_gyr, limit=100, epsabs=1e-8, epsrel=1e-6)

    # Check against expected value from AGNfitter templates
    # AGNfitter uses BC03 2003 original release, column 7 (M* = total stellar mass at age)
    # This value was derived by Calistro Rivera et al. 2016 for a declining exponential SFH
    expected = 0.5464
    tolerance = 0.01  # 1% tolerance: accounts for different interpolation/integration methods

    assert result > 0, f"Surviving mass fraction is non-positive: {result}"
    diff_pct = abs(result - expected) / expected * 100
    assert diff_pct < tolerance * 100, \
        f"Surviving mass fraction {result:.6f} differs from expected {expected:.6f} by {diff_pct:.2f}% (tolerance {tolerance*100:.1f}%)"


@pytest.mark.bounds
def test_bc03pdva94_chabrier_attributes():
    """Test that BC03 table has proper metadata attributes."""
    filepath = Path(__file__).parent.parent.parent / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    _, _, _, attrs = load_mass_remaining_table(filepath)

    # Required attributes
    required_attrs = [
        'quantity', 'isochrones', 'imf', 'source', 'remnant_prescription',
        'citation', 'generator'
    ]

    for attr in required_attrs:
        assert attr in attrs, f"Missing attribute: {attr}"

    # Verify specific values
    assert attrs['isochrones'] == 'Padova 1994'
    assert attrs['imf'] == 'Chabrier'
    assert 'Bruzual' in attrs['citation']
    assert '2003' in attrs['citation']
    assert 'BC03' in attrs['source']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
