# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for bug #2739: nthcomp coordinates in input dtype.

Bug: Kubota-Done warm Comptonization casts interpolation coordinates to float32,
causing 1e-6 SED differences between JIT and eager evaluation, and AD/FD gradient
errors of 1.03e-4 to 2.2e-4 (above 1e-4 bar).

Fix: Keep coordinates and weights in input dtype (float64 under x64, float32 in
pure-float32 mode); template tables stay float32 but promote before arithmetic.
"""

import jax
import jax.numpy as jnp
import pytest

from tengri.components.agn._nthcomp import load_nthcomp_table, nthcomp_lnu_interp


@pytest.fixture
def nthcomp_table():
    """Load nthcomp templates, skipping if unavailable."""
    table = load_nthcomp_table()
    if table is None:
        pytest.skip("nthcomp templates not available")
    return table


class TestNthcompKubotaDoneJitEager:
    """kubota_done JIT vs eager precision (#2739)."""

    @pytest.mark.regression_bug
    def test_kubota_done_jit_eager_at_mbh_spin_points(self):
        """kubota_done disc L_nu at log M_BH 7-8 and spin 0/0.7 agrees to 1e-9.

        The defect shows through the 3-zone disc, not just nthcomp_lnu_interp
        alone: when kubota_done evaluates the warm and hot zones, the templated
        nthcomp shape gets multiplied by ring luminosities (~1e66 erg/s), so a
        float32-truncated coordinate shifts the boundary between template cells
        by 1e-6 of the shape, moving L_nu by 1e-6 of the flux. Measurement points
        match the brief: log M_BH in [7.0, 7.5, 8.0], spin in [0.0, 0.7].
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        wavelength = jnp.logspace(0, 8, 150)
        points = [
            {"log_mbh": 7.0, "a_spin": 0.0},
            {"log_mbh": 7.0, "a_spin": 0.7},
            {"log_mbh": 7.5, "a_spin": 0.0},
            {"log_mbh": 7.5, "a_spin": 0.7},
            {"log_mbh": 8.0, "a_spin": 0.0},
            {"log_mbh": 8.0, "a_spin": 0.7},
        ]

        jitted_disc = jax.jit(kubota_done_full_agn)

        for pt in points:
            # Bolometric luminosity set relative to M_BH
            log_lbol = pt["log_mbh"] + 1.0

            eager = kubota_done_full_agn(
                wavelength,
                agn_log_lbol=log_lbol,
                agn_log_mbh=pt["log_mbh"],
                agn_a_spin=pt["a_spin"],
            )

            traced = jitted_disc(
                wavelength,
                agn_log_lbol=log_lbol,
                agn_log_mbh=pt["log_mbh"],
                agn_a_spin=pt["a_spin"],
            )

            peak = jnp.max(eager)
            mask = eager > 1e-6 * peak
            if jnp.any(mask):
                rel_err = jnp.abs(traced[mask] - eager[mask]) / (
                    jnp.abs(eager[mask]) + 1e-30
                )
                max_err = jnp.max(rel_err)
                assert max_err < 1e-9, (
                    f"JIT/eager kubota_done differ by {max_err:.3e} "
                    f"at M_BH 10^{pt['log_mbh']}, a={pt['a_spin']}"
                )

    @pytest.mark.regression_bug
    def test_nthcomp_coordinates_in_input_dtype(self, nthcomp_table):
        """nthcomp coordinates preserve input precision (float64/float32).

        Direct test: nthcomp_lnu_interp at fixed gamma/kTe/kTbb must agree
        between JIT and eager to 1e-9, confirming coordinates stay in input dtype.
        """
        nu = jnp.logspace(10, 20, 80)
        gammas = [1.5, 2.0, 2.5]
        kTes = [0.05, 0.13, 0.3]
        kTbbs = [1e-3, 5e-3, 1e-2]

        jitted = jax.jit(
            lambda g, t, b: nthcomp_lnu_interp(nu, g, t, b, _template=nthcomp_table)
        )

        for gamma in gammas:
            for kte in kTes:
                for ktbb in kTbbs:
                    eager = nthcomp_lnu_interp(
                        nu, gamma, kte, ktbb, _template=nthcomp_table
                    )
                    traced = jitted(gamma, kte, ktbb)

                    peak = jnp.max(eager)
                    mask = eager > 1e-6 * peak
                    if jnp.any(mask):
                        rel_err = jnp.abs(traced[mask] - eager[mask]) / (
                            jnp.abs(eager[mask]) + 1e-30
                        )
                        max_err = jnp.max(rel_err)
                        assert max_err < 1e-9, (
                            f"nthcomp JIT/eager differ by {max_err:.3e} "
                            f"at gamma={gamma}, kTe={kte}, kTbb={ktbb}"
                        )

    @pytest.mark.regression_bug
    def test_float32_mode_output_finite(self, nthcomp_table):
        """Float32 mode output is finite and stays float32.

        With JAX_ENABLE_X64=0, coordinates are float32, output stays float32,
        and all values remain finite (no ceiling overflow).
        """
        nu = jnp.logspace(10, 20, 40)
        nu_f32 = nu.astype(jnp.float32)
        gamma = jnp.asarray(2.0, dtype=jnp.float32)
        kte = jnp.asarray(0.13, dtype=jnp.float32)
        ktbb = jnp.asarray(5e-3, dtype=jnp.float32)

        with jax.enable_x64(False):
            result = nthcomp_lnu_interp(
                nu_f32, gamma, kte, ktbb, _template=nthcomp_table
            )
            assert result.dtype == jnp.float32
            assert jnp.all(jnp.isfinite(result))
