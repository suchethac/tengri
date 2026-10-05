# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the tabulated Schreiber et al. (2018) dust library (``schreiber2016``).

The model evaluates the dust-continuum and PAH templates CIGALE ships under the
module name ``schreiber2016`` (Schreiber et al. 2018, A&A 609, A30). The PAH
band centroids checked here are the published aromatic complexes (3.3, 6.2, 7.7
and 11.3 micron; Smith et al. 2007, ApJ 656, 770), not values read off the model.
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.dust.emission import DUST_EMISSION_MODELS, cmb_corrected_temperature
from tests._bounds import assert_non_negative

_C_AA = 2.99792458e18  # speed of light [Angstrom/s]

#: (lower, upper, published centroid) of the PAH complexes, micron.
_PAH_COMPLEXES_UM = ((3.0, 3.8, 3.3), (5.5, 7.0, 6.2), (7.0, 8.5, 7.7), (10.5, 12.0, 11.3))


def schreiber2016(*args, **kwargs):
    """The registry model (resolved lazily, so a missing file fails the test that uses it)."""
    return DUST_EMISSION_MODELS["schreiber2016"](*args, **kwargs)


class TestSchreiber2016:
    """Test suite for the ``schreiber2016`` dust library."""

    @pytest.fixture
    def wavelength_grid(self):
        """IR grid inside the library's tabulated range (1 micron to 3 mm)."""
        return jnp.geomspace(1.0e4, 3.0e7, 800)

    @pytest.fixture
    def l_absorbed(self):
        """Test absorbed luminosity."""
        return 1.0  # Lsun

    @pytest.mark.parametrize("f_pah", [0.0, 0.2, 1.0])
    def test_energy_conservation(self, wavelength_grid, l_absorbed, f_pah):
        """The frequency integral of L_nu equals L_absorbed for any PAH fraction."""
        l_nu = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=f_pah)
        nu = _C_AA / wavelength_grid
        np.testing.assert_allclose(-jnp.trapezoid(l_nu, nu), l_absorbed, rtol=0.01)

    @pytest.mark.parametrize(("lo_um", "hi_um", "centroid_um"), _PAH_COMPLEXES_UM)
    def test_pah_template_peaks_at_the_published_complexes(
        self, wavelength_grid, l_absorbed, lo_um, hi_um, centroid_um
    ):
        """The pure-PAH template peaks within 1.5% of each published aromatic complex."""
        l_nu = np.asarray(schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=1.0))
        wave_um = np.asarray(wavelength_grid) / 1.0e4
        window = (wave_um >= lo_um) & (wave_um <= hi_um)
        peak_um = wave_um[window][np.argmax((l_nu * _C_AA / wave_um)[window])]
        assert abs(peak_um / centroid_um - 1.0) < 0.015, (peak_um, centroid_um)

    def test_temperature_dependence(self, wavelength_grid, l_absorbed):
        """Higher T_dust shifts the continuum peak blueward."""
        cold = schreiber2016(wavelength_grid, l_absorbed, dust_T=20.0, dust_f_pah=0.0)
        warm = schreiber2016(wavelength_grid, l_absorbed, dust_T=40.0, dust_f_pah=0.0)
        assert wavelength_grid[jnp.argmax(warm)] < wavelength_grid[jnp.argmax(cold)]

    def test_f_pah_clipped_to_range(self, wavelength_grid, l_absorbed):
        """f_pah outside [0, 1] is clipped."""
        clipped = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=1.5)
        one = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=1.0)
        np.testing.assert_allclose(clipped, one, rtol=1e-12)

    def test_positive_luminosity(self, wavelength_grid, l_absorbed):
        """Output is always non-negative."""
        l_nu = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=0.5)
        assert_non_negative(l_nu, name="l_nu")

    def test_mixed_model_lies_between_the_pure_components(self, wavelength_grid, l_absorbed):
        """The renormalized per-kg mixture is a convex combination of the two pure SEDs."""
        pure_continuum = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=0.0)
        pure_pah = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=1.0)
        mixed = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=0.5)
        lo = jnp.minimum(pure_continuum, pure_pah)
        hi = jnp.maximum(pure_continuum, pure_pah)
        assert bool(jnp.all(mixed >= lo * (1.0 - 1e-9)))
        assert bool(jnp.all(mixed <= hi * (1.0 + 1e-9)))

    def test_zero_absorption_gives_zero_output(self, wavelength_grid):
        """Zero absorbed luminosity gives zero output."""
        l_nu = schreiber2016(wavelength_grid, 0.0, dust_T=30.0, dust_f_pah=0.5)
        assert jnp.allclose(l_nu, 0.0)

    def test_output_shape(self, wavelength_grid, l_absorbed):
        """Output shape matches the input wavelength grid."""
        l_nu = schreiber2016(wavelength_grid, l_absorbed, dust_T=30.0, dust_f_pah=0.5)
        chex.assert_equal_shape([l_nu, wavelength_grid])

    def test_cmb_temperature_correction_raises_the_effective_temperature(self):
        """The CMB heating utility raises the effective temperature at high redshift."""
        t_z0 = cmb_corrected_temperature(30.0, 0.0, 1.5)
        t_z3 = cmb_corrected_temperature(30.0, 3.0, 1.5)
        assert t_z3 > t_z0
