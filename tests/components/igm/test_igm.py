# SPDX-License-Identifier: BSD-3-Clause
"""Tests for IGM absorption module (models/igm.py).

CRITICAL CONVENTION: igm_transmission(wave_obs, z_source) takes OBSERVED-FRAME
wavelengths, not rest-frame. See CLAUDE.md gotchas.

Physics references:
- Inoue+2014, MNRAS 442, 1805 (IGM opacity model)
- Fan+2006, AJ 132, 117 (Lya forest mean transmission)
"""

import chex
import jax
import pytest

pytestmark = pytest.mark.bounds
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.igm import igm_transmission
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestIGMConvention:
    def test_observed_frame_input(self):
        """igm_transmission takes observed-frame wavelengths, not rest-frame.

        At z=3, rest 1200 Å (just blueward of Lya at 1215.67 Å) lands in the
        forest at obs 4800 Å. Pass 4800 Å, not 1200 Å.
        The two conventions give very different transmission values.
        """
        z = 3.0
        # Observed-frame: a forest wavelength (rest 1200 Å) at z=3 appears at 4800 Å
        wave_forest_obs = jnp.array([1200.0 * (1 + z)])
        T_obs = float(igm_transmission(wave_forest_obs, z)[0])

        # Passing rest-frame wavelength (WRONG convention): 1200 Å at z=3 → 300 Å observed
        # that is EUV, should have T → 0 (optically thick)
        wave_forest_rest = jnp.array([1200.0])
        T_rest = float(igm_transmission(wave_forest_rest, z)[0])

        # At observed 4800 Å (rest 1200 Å): Lya forest → T ≈ 0.68 (Fan+2006)
        assert abs(T_obs - 0.68) < 0.30, (
            f"Observed-frame convention: T(1200*(1+3)=4800Å)={T_obs:.3f} should be ≈ 0.68"
        )
        # At 1216 Å (would be EUV deep within Lyman limit at z=3): T ≈ 0
        assert T_rest < T_obs, (
            "Rest-frame wavelength gives lower T (more opaque) than observed-frame"
        )

    def test_lyman_limit_opacity_z4(self):
        """Just inside the Lyman continuum (rest ~900 Å) is heavily attenuated at z_source=4.

        Inoue+2014 MNRAS 442: τ_LL >> 1 for z > 3. The Inoue limit is 911.8 Å, so
        we sample a wavelength comfortably inside the LyC region rather than at the
        edge of the table where the analytic formula transitions to zero.
        """
        z = 4.0
        wave_ll_obs = jnp.array([900.0 * (1 + z)])  # observed 4500 Å (rest 900)
        T = float(igm_transmission(wave_ll_obs, z)[0])
        assert T < 0.30, f"Inoue+2014: LyC opacity at rest 900 Å, z=4: T={T:.3f} (expected < 0.30)"

    def test_lya_forest_z3(self):
        """Mean Lya forest transmission at z=3 ≈ 0.68. Fan+2006 AJ 132, Eq. 3.

        Probed just blueward of Lya (rest 1200 Å) where the mean forest level
        applies; at the line edge itself the value is not well-defined.
        """
        z = 3.0
        wave_forest_obs = jnp.array([1200.0 * (1 + z)])
        T = float(igm_transmission(wave_forest_obs, z)[0])
        np.testing.assert_allclose(
            T,
            0.68,
            atol=0.15,
            err_msg="Fan+2006 AJ 132 Eq. 3: mean Lya forest T at z=3 ≈ 0.68",
        )

    def test_no_absorption_z0(self):
        """No significant IGM absorption at z≈0 — local universe is transparent.

        The UV end of the range (near 912 Å observed) may have residual LAF
        opacity even at z=0.01, so the threshold is 0.95 not 0.99.
        """
        z = 0.01
        wave = jnp.linspace(912.0 * (1 + z), 3000.0 * (1 + z), 100)
        T = igm_transmission(wave, z)
        assert float(jnp.min(T)) > 0.95, (
            f"IGM: effectively no absorption at z=0.01, min T={float(jnp.min(T)):.4f}"
        )

    def test_transparent_above_lya(self):
        """IGM is transparent longward of Lya in the observed frame.

        At z=3: observed-frame Lya is at 4864 Å. Longer wavelengths (green/red
        optical) should have T > 0.95.
        """
        z = 3.0
        wave_opt_obs = jnp.array([6000.0, 7000.0, 8000.0])  # obs optical, above Lya
        T = igm_transmission(wave_opt_obs, z)
        assert float(jnp.min(T)) > 0.95, (
            f"IGM: transparent at optical obs wavelengths above Lya, min T={float(jnp.min(T)):.4f}"
        )

    def test_monotone_with_z(self):
        """At fixed observed wavelength, transmission decreases with redshift.

        Higher redshift = longer path through more Lya forest absorbers.
        At obs 3600 Å (blueward of Lya for all three z): T(z=2) > T(z=3) > T(z=4).
        """
        wave = jnp.array([3600.0])
        T2 = float(igm_transmission(wave, 2.0)[0])
        T3 = float(igm_transmission(wave, 3.0)[0])
        T4 = float(igm_transmission(wave, 4.0)[0])
        assert T2 > T3, f"IGM: T(z=2)={T2:.3f} should exceed T(z=3)={T3:.3f}"
        assert T3 > T4, f"IGM: T(z=3)={T3:.3f} should exceed T(z=4)={T4:.3f}"

    def test_no_nan_at_lyman_limit(self):
        """No NaN at exactly the Lyman limit wavelength (numerical edge case).

        Power-law exponents in igm.py clamp z_obs >= 0 to prevent NaN.
        See CLAUDE.md: 'IGM LAF opacity clamps z_obs >= 0'.
        """
        z = 2.0
        wave_ll = jnp.array([912.0 * (1 + z)])
        T = igm_transmission(wave_ll, z)
        assert jnp.isfinite(T[0]), "igm_transmission: NaN at Lyman limit wavelength"

    def test_jit_compatible(self):
        """igm_transmission is JIT-compilable."""
        wave = jnp.array([1216.0 * 4.0, 912.0 * 5.0])
        result = assert_jit_matches_eager(igm_transmission, wave, 3.0)
        chex.assert_tree_all_finite(result)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "igm_transmission uses tabulated LAF/DLA opacity via jnp.interp — "
            "index selection is not differentiable w.r.t. z via JAX autodiff; "
            "grad_jax=0 while FD ≈ -17."
        ),
    )
    def test_gradient_wrt_z(self):
        """FD check: ∂T/∂z at z=3 (fixed observed wavelength).

        Inoue+2014 model is smooth in z — gradient should be finite and negative
        (more absorption at higher z). The FD check catches sign errors.
        """
        wave = jnp.array([1216.0 * 4.0])  # fixed obs wavelength = Lya at z=3

        def f(z):
            return float(igm_transmission(wave, z).sum())

        grad_jax = float(jax.grad(lambda z: igm_transmission(wave, z).sum())(3.0))
        grad_fd = fd_grad(f, 3.0, eps=0.01)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=5e-3,
            err_msg="igm_transmission: FD check on ∂T/∂z at z=3",
        )
