# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate IGM transmission against bagpipes (Inoue+2014).

Both tengri and bagpipes implement the same Inoue et al. (2014)
prescription using the same coefficient tables (Table 2). The
implementations differ only in:

- tengri: pure JAX, analytical piecewise power laws, takes observed-frame wavs
- bagpipes: numpy, same piecewise power laws, takes rest-frame wavs

With the coefficient tables matched to the paper the two agree to <1e-3
away from the sharp Lyman-series edges, where each code samples the same
step at slightly offset grid positions (a resolution artifact, not a
physics disagreement). The below-Ly-alpha coverage here is deliberate:
an earlier version asserted agreement only *above* Ly-alpha and thereby
missed a DLA-coefficient transcription bug that over-absorbed the z >= 2
Lyman continuum (fixed; see
tests/components/igm/test_inoue14_dla_coefficients.py).
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

bagpipes_igm = pytest.importorskip(
    "bagpipes.models.making.igm_inoue2014",
    reason="bagpipes not installed",
)

from tengri.components.igm import igm_transmission


class TestIGMTransmissionCrossval:
    """Compare tengri IGM vs bagpipes Inoue+2014."""

    @pytest.mark.parametrize("z_source", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_transmission_matches_bagpipes(self, rest_wavelengths, z_source):
        """IGM transmission should agree within ~10% across redshifts.

        tengri takes observed-frame wavelengths; bagpipes takes
        rest-frame. We convert: wave_obs = rest_wavs * (1 + z).
        """
        rest_wavs = rest_wavelengths
        wave_obs = rest_wavs * (1.0 + z_source)

        # tengri (observed-frame input)
        trans_tengri = np.asarray(igm_transmission(jnp.array(wave_obs), z_source))

        # bagpipes (rest-frame input)
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z_source)

        # Both should be in [0, 1]
        assert np.all(trans_tengri >= 0.0) and np.all(trans_tengri <= 1.0 + 1e-10)
        assert np.all(trans_bagpipes >= 0.0) and np.all(trans_bagpipes <= 1.0 + 1e-10)

        # Focus on wavelengths above Ly-alpha (where the lower-bound
        # difference between implementations has minimal effect)
        above_lya = rest_wavs > 1216.0
        if np.sum(above_lya) < 3:
            pytest.skip(f"Too few points above Ly-alpha at z={z_source}")

        np.testing.assert_allclose(
            trans_tengri[above_lya],
            trans_bagpipes[above_lya],
            atol=1e-4,
            err_msg=f"IGM transmission mismatch above Ly-alpha at z={z_source}",
        )

    @pytest.mark.parametrize("z_source", [3.0, 4.0, 5.0])
    def test_lyman_series_continuum_matches_bagpipes(self, z_source):
        """Below Ly-alpha the DLA + LAF terms must match bagpipes too.

        This is the region the DLA coefficients govern. A systematic
        coefficient error shifts the whole curve, so we assert a small
        *median* |Δ| (robust to the isolated grid-sampling spikes at the
        sharp Lyman-series edges). The pre-fix DLA transcription produced a
        median |Δ| ~0.02-0.09 here; the correct coefficients give <1e-3.
        """
        rest_wavs = np.linspace(820.0, 1210.0, 2000)
        wave_obs = rest_wavs * (1.0 + z_source)
        trans_tengri = np.asarray(igm_transmission(jnp.array(wave_obs), z_source))
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z_source)

        median_abs = float(np.median(np.abs(trans_tengri - trans_bagpipes)))
        assert median_abs < 2e-3, (
            f"Below-Ly-alpha IGM disagrees with bagpipes at z={z_source}: "
            f"median |Δ| = {median_abs:.3e} (DLA/LAF coefficient regression?)"
        )

    def test_full_transmission_above_lya(self, rest_wavelengths):
        """Both codes should give T~1 for wavelengths well above Ly-alpha."""
        rest_wavs = np.linspace(1300, 2000, 50)
        z = 2.0
        wave_obs = rest_wavs * (1.0 + z)

        trans_tengri = np.asarray(igm_transmission(jnp.array(wave_obs), z))
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z)

        np.testing.assert_allclose(trans_tengri, 1.0, atol=1e-6)
        np.testing.assert_allclose(trans_bagpipes, 1.0, atol=1e-6)

    def test_zero_redshift_bagpipes_no_absorption(self, rest_wavelengths):
        """At z=0, bagpipes correctly gives no IGM absorption."""
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavelengths, 0.0)
        np.testing.assert_allclose(trans_bagpipes, 1.0, atol=1e-10)

    def test_zero_redshift_tengri_no_absorption(self, rest_wavelengths):
        """At z=0, tengri should give exactly no IGM absorption."""
        wave_obs = rest_wavelengths  # obs = rest at z=0
        trans_tengri = np.asarray(igm_transmission(jnp.array(wave_obs), 0.0))

        np.testing.assert_allclose(trans_tengri, 1.0, atol=1e-10)

    def test_high_redshift_strong_absorption(self):
        """At z=6, UV below Ly-alpha should be heavily absorbed in both."""
        rest_wavs = np.array([900.0, 1000.0, 1100.0])
        z = 6.0
        wave_obs = rest_wavs * (1.0 + z)

        trans_tengri = np.asarray(igm_transmission(jnp.array(wave_obs), z))
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z)

        # Both should show strong absorption (T < 0.5) below Ly-alpha
        assert np.all(trans_tengri < 0.5), "tengri: weak absorption at z=6"
        assert np.all(trans_bagpipes < 0.5), "bagpipes: weak absorption at z=6"

    @pytest.mark.parametrize("z_source", [1.0, 3.0, 5.0])
    def test_monotonic_with_redshift(self, z_source):
        """Higher redshift should produce more absorption (lower T)."""
        rest_wavs = np.array([1000.0, 1100.0, 1200.0])

        z_low = max(0.5, z_source - 1.0)
        z_high = z_source

        wave_obs_low = rest_wavs * (1.0 + z_low)
        wave_obs_high = rest_wavs * (1.0 + z_high)

        trans_low = np.asarray(igm_transmission(jnp.array(wave_obs_low), z_low))
        trans_high = np.asarray(igm_transmission(jnp.array(wave_obs_high), z_high))

        # Higher z should have less transmission (more absorption)
        assert np.mean(trans_high) <= np.mean(trans_low), (
            f"Higher z should have more absorption: "
            f"T(z={z_low})={np.mean(trans_low):.3f}, T(z={z_high})={np.mean(trans_high):.3f}"
        )
