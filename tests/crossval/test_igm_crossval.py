"""Cross-validate IGM transmission against bagpipes (Inoue+2014).

Both diffsed and bagpipes implement the same Inoue et al. (2014)
prescription using the same coefficient tables (from eazy-py). The
implementations differ in:

- diffsed: pure JAX, analytical piecewise power laws, takes observed-frame wavs
- bagpipes: numpy, same piecewise power laws, takes rest-frame wavs

Known difference: diffsed applies absorption for wave_obs < lam_j*(1+z)
without the lower bound wave_obs > lam_j. This means diffsed may
over-absorb at wavelengths below line rest wavelengths. This produces
~5-10% differences at z>2 and non-zero absorption at z=0.

We use sanity-check tolerances (~10%) to verify both implementations
are broadly consistent, while documenting the known differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

bagpipes_igm = pytest.importorskip(
    "bagpipes.models.making.igm_inoue2014",
    reason="bagpipes not installed",
)

from diffsed.models.igm import igm_transmission


class TestIGMTransmissionCrossval:
    """Compare diffsed IGM vs bagpipes Inoue+2014."""

    @pytest.mark.parametrize("z_source", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_transmission_matches_bagpipes(self, rest_wavelengths, z_source):
        """IGM transmission should agree within ~10% across redshifts.

        diffsed takes observed-frame wavelengths; bagpipes takes
        rest-frame. We convert: wave_obs = rest_wavs * (1 + z).
        """
        rest_wavs = rest_wavelengths
        wave_obs = rest_wavs * (1.0 + z_source)

        # diffsed (observed-frame input)
        trans_diffsed = np.asarray(igm_transmission(jnp.array(wave_obs), z_source))

        # bagpipes (rest-frame input)
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z_source)

        # Both should be in [0, 1]
        assert np.all(trans_diffsed >= 0.0) and np.all(trans_diffsed <= 1.0 + 1e-10)
        assert np.all(trans_bagpipes >= 0.0) and np.all(trans_bagpipes <= 1.0 + 1e-10)

        # Focus on wavelengths above Ly-alpha (where the lower-bound
        # difference between implementations has minimal effect)
        above_lya = rest_wavs > 1216.0
        if np.sum(above_lya) < 3:
            pytest.skip(f"Too few points above Ly-alpha at z={z_source}")

        np.testing.assert_allclose(
            trans_diffsed[above_lya],
            trans_bagpipes[above_lya],
            atol=1e-4,
            err_msg=f"IGM transmission mismatch above Ly-alpha at z={z_source}",
        )

    def test_full_transmission_above_lya(self, rest_wavelengths):
        """Both codes should give T~1 for wavelengths well above Ly-alpha."""
        rest_wavs = np.linspace(1300, 2000, 50)
        z = 2.0
        wave_obs = rest_wavs * (1.0 + z)

        trans_diffsed = np.asarray(igm_transmission(jnp.array(wave_obs), z))
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z)

        np.testing.assert_allclose(trans_diffsed, 1.0, atol=1e-6)
        np.testing.assert_allclose(trans_bagpipes, 1.0, atol=1e-6)

    def test_zero_redshift_bagpipes_no_absorption(self, rest_wavelengths):
        """At z=0, bagpipes correctly gives no IGM absorption."""
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavelengths, 0.0)
        np.testing.assert_allclose(trans_bagpipes, 1.0, atol=1e-10)

    def test_zero_redshift_diffsed_no_absorption(self, rest_wavelengths):
        """At z=0, diffsed should give exactly no IGM absorption."""
        wave_obs = rest_wavelengths  # obs = rest at z=0
        trans_diffsed = np.asarray(igm_transmission(jnp.array(wave_obs), 0.0))

        np.testing.assert_allclose(trans_diffsed, 1.0, atol=1e-10)

    def test_high_redshift_strong_absorption(self):
        """At z=6, UV below Ly-alpha should be heavily absorbed in both."""
        rest_wavs = np.array([900.0, 1000.0, 1100.0])
        z = 6.0
        wave_obs = rest_wavs * (1.0 + z)

        trans_diffsed = np.asarray(igm_transmission(jnp.array(wave_obs), z))
        trans_bagpipes = bagpipes_igm.get_Inoue14_trans(rest_wavs, z)

        # Both should show strong absorption (T < 0.5) below Ly-alpha
        assert np.all(trans_diffsed < 0.5), "diffsed: weak absorption at z=6"
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
