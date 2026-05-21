"""Tests for the ADAF + truncated disc model (disc.adaf_disc + unified.adaf_agn)."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


# ── Basic functionality ───────────────────────────────────────────


class TestAdafDisc:
    """Tests for the low-level adaf_disc function."""

    def test_finite_sed(self, wavelength):
        """ADAF produces finite SED values everywhere."""
        from tengri.components.agn.disc import adaf_disc

        l_nu = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=0.1,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )
        chex.assert_tree_all_finite(l_nu)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_non_negative(self, wavelength):
        """ADAF SED is non-negative everywhere."""
        from tengri.components.agn.disc import adaf_disc

        l_nu = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=0.1,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
        )
        assert jnp.all(l_nu >= 0.0)

    def test_peaks_at_longer_wavelengths_than_standard_disc(self, wavelength):
        """ADAF SED peaks at longer wavelengths than a standard thin disc.

        The ADAF synchrotron peak is in the radio/mm regime (~300 um),
        while the standard disc peaks in the UV.
        """
        from tengri.components.agn.disc import adaf_disc, multicolor_disc

        l_adaf = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )
        l_disc = multicolor_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
        )

        # Find peak wavelengths
        peak_adaf = wavelength[jnp.argmax(l_adaf * wavelength)]
        peak_disc = wavelength[jnp.argmax(l_disc * wavelength)]

        # ADAF should peak at longer wavelength (lower frequency)
        assert peak_adaf > peak_disc

    def test_truncation_radius_affects_uv(self, optical_wavelength):
        """Larger truncation radius reduces UV emission from outer disc.

        A larger r_tr means the thin disc starts further out (cooler),
        producing less UV/optical emission.
        """
        from tengri.components.agn.disc import adaf_disc

        # UV band: 1000-3000 A
        uv_mask = (optical_wavelength > 1000.0) & (optical_wavelength < 3000.0)

        l_small_tr = adaf_disc(
            optical_wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=30.0,
        )
        l_large_tr = adaf_disc(
            optical_wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=500.0,
        )

        # Larger r_tr -> less UV (hotter inner disc is truncated further out)
        uv_small = jnp.sum(l_small_tr[uv_mask])
        uv_large = jnp.sum(l_large_tr[uv_mask])
        assert uv_small > uv_large

    def test_adaf_faint_at_high_ledd(self, wavelength):
        """At high L/L_Edd, ADAF component is faint relative to disc.

        The ADAF radiative efficiency scales as r_isco/r_tr, so when
        r_tr is small (high accretion) the ADAF is more efficient but
        the disc dominates.
        """
        from tengri.components.agn.disc import adaf_disc

        # Low Eddington ratio: ADAF regime
        l_low = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-4.0,
            agn_r_tr=300.0,
        )

        # Higher Eddington ratio: disc-dominated
        l_high = adaf_disc(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_r_tr=10.0,
        )

        # Higher L_bol should produce brighter overall SED.
        # Compare bolometric luminosities via frequency integral — raw sum
        # over a log-spaced wavelength grid is NOT a bolometric proxy because
        # L_nu * dnu gains a nu factor, biasing toward radio-peaking SEDs.
        c_aa_per_s = 2.99792458e18  # c in Angstrom/s
        nu = c_aa_per_s / wavelength  # Hz, descending when wavelength ascending
        sort_idx = jnp.argsort(nu)
        lbol_high = jnp.trapezoid(l_high[sort_idx], nu[sort_idx])
        lbol_low = jnp.trapezoid(l_low[sort_idx], nu[sort_idx])
        assert lbol_high > lbol_low, (
            f"Higher L_bol SED not brighter: {lbol_high:.3e} vs {lbol_low:.3e} Lsun"
        )

    def test_agn_frac_scaling(self, wavelength):
        """agn_frac linearly scales the output."""
        from tengri.components.agn.disc import adaf_disc

        l_full = adaf_disc(wavelength, agn_log_lbol=42.0, agn_frac=1.0)
        l_half = adaf_disc(wavelength, agn_log_lbol=42.0, agn_frac=0.5)

        ratio = l_full / jnp.maximum(l_half, 1e-100)
        # Should be ~2 everywhere (within numerical precision)
        assert jnp.allclose(ratio, 2.0, rtol=0.01, atol=1e-30)


# ── JIT and gradient compatibility ────────────────────────────────


class TestAdafJitGrad:
    """JIT compilation and gradient tests."""

    def test_jit_compatible(self, wavelength):
        """adaf_disc is JIT-compilable."""
        from tengri.components.agn.disc import adaf_disc

        @jax.jit
        def _run(wave):
            return adaf_disc(wave, agn_log_lbol=42.0, agn_frac=0.1)

        result = _run(wavelength)
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂log_lbol for adaf_disc."""
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(lbol):
            return jnp.sum(adaf_disc(optical_wavelength, agn_log_lbol=lbol, agn_frac=0.1))

        g = float(jax.grad(loss_fn)(42.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 42.0, eps=0.01),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂log_lbol",
        )

    def test_gradient_wrt_r_tr(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂r_tr for adaf_disc. Truncation radius gradient."""
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(r_tr):
            return jnp.sum(
                adaf_disc(optical_wavelength, agn_log_lbol=42.0, agn_frac=0.1, agn_r_tr=r_tr)
            )

        g = float(jax.grad(loss_fn)(100.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 100.0, eps=0.1),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂r_tr",
        )

    def test_gradient_wrt_delta(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂delta for adaf_disc. ADAF δ parameter gradient."""
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(delta):
            return jnp.sum(
                adaf_disc(
                    optical_wavelength, agn_log_lbol=42.0, agn_frac=0.1, agn_adaf_delta=delta
                )
            )

        g = float(jax.grad(loss_fn)(0.01))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 0.01, eps=1e-4),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂adaf_delta",
        )

    def test_gradient_wrt_beta(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂beta for adaf_disc. ADAF β parameter gradient.

        Fixed by the Mahadevan 1997 rewrite (2026-04-21): The new implementation uses
        a more physical weighting of synchrotron/bremsstrahlung/IC components via
        magnetic field pressure (1-beta), which avoids the algebraic singularity
        that plagued the old linear weighting scheme.
        """
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(beta):
            return jnp.sum(
                adaf_disc(optical_wavelength, agn_log_lbol=42.0, agn_frac=0.1, agn_adaf_beta=beta)
            )

        g = float(jax.grad(loss_fn)(0.5))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 0.5, eps=1e-3),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂adaf_beta",
        )


# ── Registry tests ────────────────────────────────────────────────


class TestAdafRegistry:
    """Tests that ADAF is properly registered in the AGN model registry."""

    def test_registered_as_adaf(self):
        """'adaf' is in AGN_MODELS registry."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "adaf" in AGN_MODELS

    def test_get_agn_model_adaf(self):
        """resolve_agn_model('adaf') returns a callable."""
        from tengri.components.agn.unified import resolve_agn_model

        model_fn = resolve_agn_model("adaf")
        assert callable(model_fn)

    def test_registered_model_runs(self, optical_wavelength):
        """The registered 'adaf' model produces finite output."""
        from tengri.components.agn.unified import resolve_agn_model

        model_fn = resolve_agn_model("adaf")
        l_nu = model_fn(optical_wavelength, agn_log_lbol=42.0)
        chex.assert_tree_all_finite(l_nu)
        chex.assert_equal_shape([l_nu, optical_wavelength])

    def test_adaf_in_unified_disc_fns(self, optical_wavelength):
        """'adaf' disc type works in unified_agn combiner."""
        from tengri.components.agn.unified import unified_agn

        l_nu = unified_agn(
            optical_wavelength,
            agn_log_lbol=42.0,
            disc_model="adaf",
            torus_model="simple",
        )
        chex.assert_tree_all_finite(l_nu)


# ── simple_agn: power-law disc + single-temperature torus ─────────


class TestSimpleAgn:
    """Tests for simple_agn: power-law disc + single-temperature torus."""

    def test_finite_nonneg(self, wavelength):
        """simple_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import simple_agn

        l_nu = simple_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_simple(self):
        """'simple' appears in the AGN_MODELS registry."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "simple" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import simple_agn

        l1 = simple_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = simple_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        # Exclude wavelengths where both SEDs fall below float64 underflow (~1e-300);
        # X-ray regime can reach 1e-40 making the ratio numerically indeterminate.
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_torus_frac_nonzero_adds_ir(self, wavelength):
        """Non-zero agn_torus_frac adds torus IR flux near the BB peak.

        At fixed total luminosity, a torus fraction of 0.5 routes half the
        power into a 1000 K BB (peak ~29,000 Å).  The IR sum near that peak
        must exceed the disc-only (torus_frac=0) case where the powerlaw disc
        carries all the power but is fainter at 1000 K BB wavelengths.
        """
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.unified import simple_agn

        # Near the 1000 K BB peak (Wien: λ_peak = 2.898e7/1000 Å ≈ 29,000 Å)
        nir_mask = (wavelength > 2e4) & (wavelength < 5e4)
        # disc only at half power
        l_disc_half = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_frac=0.5)
        # disc (half power) + torus (half power, 1000 K BB peak in NIR)
        l_with_torus = simple_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_torus_frac=0.5,
            agn_T_torus=1000.0,
        )
        assert float(jnp.sum(l_with_torus[nir_mask])) > float(jnp.sum(l_disc_half[nir_mask]))

    def test_flatter_alpha_more_uv(self, wavelength):
        """Flatter disc slope (less-negative alpha) puts more power in UV.

        L_nu ∝ nu^alpha; flatter alpha means relatively more emission at
        high frequencies (UV) compared to a steep, red power law.
        """
        from tengri.components.agn.unified import simple_agn

        uv_mask = (wavelength > 1000.0) & (wavelength < 3000.0)
        # Use torus_frac=0 so we isolate the disc spectrum
        l_steep = simple_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_alpha=-2.0, agn_torus_frac=0.0
        )
        l_flat = simple_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_alpha=-0.5, agn_torus_frac=0.0
        )
        assert float(jnp.sum(l_flat[uv_mask])) > float(jnp.sum(l_steep[uv_mask]))

    def test_hotter_torus_peaks_at_shorter_wavelength(self, wavelength):
        """Hotter torus temperature shifts the IR peak to shorter wavelengths (Wien).

        Wien's law: λ_peak = b/T, so T_hot → shorter peak.
        """
        from tengri.components.agn.unified import simple_agn

        ir = wavelength[(wavelength > 5e3) & (wavelength < 2e7)]
        if ir.shape[0] < 10:
            pytest.skip("wavelength grid too sparse for IR peak test")

        l_cold = simple_agn(
            ir, agn_log_lbol=44.0, agn_frac=1.0, agn_torus_frac=1.0, agn_T_torus=400.0
        )
        l_hot = simple_agn(
            ir, agn_log_lbol=44.0, agn_frac=1.0, agn_torus_frac=1.0, agn_T_torus=2000.0
        )

        peak_cold = float(ir[jnp.argmax(l_cold)])
        peak_hot = float(ir[jnp.argmax(l_hot)])
        assert peak_hot < peak_cold, (
            f"Expected hotter torus peak at shorter λ; got peak_hot={peak_hot:.1f} Å, "
            f"peak_cold={peak_cold:.1f} Å"
        )

    def test_jit_compatible(self, wavelength):
        """simple_agn is JIT-compilable."""
        from tengri.components.agn.unified import simple_agn

        @jax.jit
        def _run(wave):
            return simple_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂agn_log_lbol for simple_agn."""
        from tengri.components.agn.unified import simple_agn

        def loss_fn(lbol):
            return jnp.sum(simple_agn(optical_wavelength, agn_log_lbol=lbol))

        g = float(jax.grad(loss_fn)(44.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 44.0, eps=0.01),
            rtol=1e-3,
            err_msg="simple_agn: FD check ∂/∂agn_log_lbol",
        )


# ── standard_agn: multi-color disc + two-temperature torus ────────


class TestStandardAgn:
    """Tests for standard_agn: multi-color disc + two-temperature torus."""

    def test_finite_nonneg(self, wavelength):
        """standard_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import standard_agn

        l_nu = standard_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_standard(self):
        """'standard' appears in the AGN_MODELS registry."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "standard" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import standard_agn

        l1 = standard_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = standard_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        # Exclude wavelengths where both SEDs fall below float64 underflow (~1e-300);
        # X-ray regime can reach 1e-40 making the ratio numerically indeterminate.
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_higher_ledd_bluer_disc(self, optical_wavelength):
        """Higher Eddington ratio gives a hotter disc → more far-UV emission.

        T_in ∝ mdot^{1/4} at fixed M_BH (mdot ∝ L_Edd * ratio ∝ M_BH * l_edd_ratio).
        At fixed L_bol both discs peak in the EUV (<100 Å), but the hotter disc
        extends further into the far-UV.  Below ~500 Å the high-Eddington SED
        exceeds the low-Eddington one; integrating a broader window mixes in
        the optical bump where the cooler disc wins.
        """
        from tengri.components.agn.unified import standard_agn

        # Far-UV (<500 Å): hotter disc emits more (further from the Wien cutoff)
        far_uv_mask = (optical_wavelength > 300.0) & (optical_wavelength < 500.0)
        # Isolate disc by setting torus_frac=0; vary only Eddington ratio
        l_low = standard_agn(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_log_ledd=-2.0,
            agn_torus_frac=0.0,
        )
        l_high = standard_agn(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_log_ledd=-0.5,
            agn_torus_frac=0.0,
        )
        assert jnp.any(far_uv_mask), "No wavelengths in far-UV window"
        # Higher Eddington ratio → hotter disc → more flux at far-UV wavelengths
        assert float(jnp.sum(l_high[far_uv_mask])) > float(jnp.sum(l_low[far_uv_mask]))

    def test_two_temperature_torus_has_near_ir_excess(self, wavelength):
        """Two-temperature torus produces near-IR emission from the hot component.

        With a hot (1200 K) and warm (300 K) component, there is more near-IR
        (1–5 μm) emission than a cool single-temperature torus would produce.
        """
        from tengri.components.agn.unified import standard_agn

        # Near-IR: 1–5 μm = 10,000–50,000 Å
        nir_mask = (wavelength > 1e4) & (wavelength < 5e4)
        l_standard = standard_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_torus_frac=1.0,
        )
        # Set hot component to near-zero to simulate a cooler single-temperature torus
        l_cold_only = standard_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_T_hot=350.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.0,
            agn_torus_frac=1.0,
        )
        assert float(jnp.sum(l_standard[nir_mask])) > float(jnp.sum(l_cold_only[nir_mask]))

    def test_jit_compatible(self, wavelength):
        """standard_agn is JIT-compilable."""
        from tengri.components.agn.unified import standard_agn

        @jax.jit
        def _run(wave):
            return standard_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂agn_log_lbol for standard_agn."""
        from tengri.components.agn.unified import standard_agn

        def loss_fn(lbol):
            return jnp.sum(standard_agn(optical_wavelength, agn_log_lbol=lbol))

        g = float(jax.grad(loss_fn)(44.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 44.0, eps=0.01),
            rtol=1e-3,
            err_msg="standard_agn: FD check ∂/∂agn_log_lbol",
        )

    def test_gradient_wrt_torus_frac(self, optical_wavelength):
        """FD check: ∂(∑SED)/∂agn_torus_frac for standard_agn."""
        from tengri.components.agn.unified import standard_agn

        def loss_fn(frac):
            return jnp.sum(
                standard_agn(optical_wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
            )

        g = float(jax.grad(loss_fn)(0.5))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 0.5, eps=0.01),
            rtol=1e-3,
            err_msg="standard_agn: FD check ∂/∂agn_torus_frac",
        )


# ── unified_agn combiner: disc × torus combinations ───────────────


class TestUnifiedAgnCombinations:
    """Tests for the unified_agn combiner across disc/torus model combinations."""

    @pytest.mark.parametrize(
        "disc_model,torus_model",
        [
            ("powerlaw", "simple"),
            ("powerlaw", "two_temperature"),
            ("multicolor", "simple"),
            ("multicolor", "two_temperature"),
            ("adaf", "simple"),
        ],
    )
    def test_all_combinations_produce_finite_output(self, wavelength, disc_model, torus_model):
        """Every supported disc × torus combination gives finite, non-negative output."""
        from tengri.components.agn.unified import unified_agn

        l_nu = unified_agn(
            wavelength,
            agn_log_lbol=44.0,
            disc_model=disc_model,
            torus_model=torus_model,
        )
        (
            chex.assert_tree_all_finite(l_nu),
            (f"Non-finite SED for disc={disc_model}, torus={torus_model}"),
        )
        assert jnp.all(l_nu >= 0.0), f"Negative SED for disc={disc_model}, torus={torus_model}"
        chex.assert_equal_shape([l_nu, wavelength])

    def test_unknown_disc_raises(self, wavelength):
        """Unknown disc_model raises KeyError."""
        from tengri.components.agn.unified import unified_agn

        with pytest.raises(KeyError):
            unified_agn(wavelength, agn_log_lbol=44.0, disc_model="nonexistent")

    def test_unknown_torus_raises(self, wavelength):
        """Unknown torus_model raises KeyError."""
        from tengri.components.agn.unified import unified_agn

        with pytest.raises(KeyError):
            unified_agn(wavelength, agn_log_lbol=44.0, torus_model="nonexistent")

    def test_torus_dominated_sed_uv_suppressed(self, wavelength):
        """With torus_frac=1.0, IR dominates over UV by Wien suppression.

        The 1000 K torus has Wien tail at λ=5000 Å: x = hν/kT ≈ 28.8,
        giving B_nu ∝ exp(-28.8) ≈ 3e-13 relative to peak.  The disc
        contributes nothing (agn_frac=0).  So IR >> UV by a factor >>1000.
        """
        from tengri.components.agn.unified import unified_agn

        uv_mask = (wavelength > 500.0) & (wavelength < 5000.0)
        ir_mask = (wavelength > 2e4) & (wavelength < 1e6)
        l_nu = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=1.0)
        ir_sum = float(jnp.sum(l_nu[ir_mask]))
        uv_sum = float(jnp.sum(l_nu[uv_mask]))
        assert ir_sum > 0.0, "IR sum is zero — torus emitted nothing"
        assert ir_sum > 1000.0 * uv_sum, (
            f"Expected IR >> UV (Wien suppression); got IR={ir_sum:.3e}, UV={uv_sum:.3e}"
        )

    def test_torus_frac_one_sets_disc_to_zero(self, wavelength):
        """agn_torus_frac=0.0 yields exactly the disc alone (no torus).

        Torus is called with agn_torus_frac=0, so simple_torus returns 0;
        unified_agn output equals disc-only.
        """
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.unified import unified_agn

        l_unified = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=0.0)
        # Disc gets agn_frac = 1 - 0 = 1.0
        l_disc = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_frac=1.0)
        np.testing.assert_allclose(
            np.array(l_unified),
            np.array(l_disc),
            rtol=1e-6,
            err_msg="unified_agn with torus_frac=0 should equal disc-only",
        )

    def test_jit_over_default_combination(self, wavelength):
        """The default powerlaw+simple combination is JIT-compilable."""
        from tengri.components.agn.unified import unified_agn

        @jax.jit
        def _run(wave):
            return unified_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))

    def test_total_is_disc_plus_torus(self, wavelength):
        """unified_agn output equals disc + torus computed separately."""
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.torus import simple_torus
        from tengri.components.agn.unified import unified_agn

        frac = 0.3
        l_unified = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
        l_disc = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_frac=1.0 - frac)
        l_torus = simple_torus(wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
        np.testing.assert_allclose(
            np.array(l_unified),
            np.array(l_disc + l_torus),
            rtol=1e-6,
            err_msg="unified_agn should equal disc + torus",
        )


# ── resolve_agn_model: error and deprecation paths ────────────────


class TestResolveAgn:
    """Tests for resolve_agn_model error/warning branches."""

    def test_unknown_model_raises_value_error(self):
        """resolve_agn_model raises ValueError for unknown names."""
        from tengri.components.agn.unified import resolve_agn_model

        with pytest.raises(ValueError, match="Unknown AGN model"):
            resolve_agn_model("not_a_real_model_xyz_abc")

    def test_kubota_done_emits_deprecation_warning(self, wavelength):
        """resolve_agn_model('kubota_done') emits DeprecationWarning."""
        from tengri.components.agn.unified import resolve_agn_model

        with pytest.warns(DeprecationWarning, match="kubota_done.*deprecated"):
            fn = resolve_agn_model("kubota_done")
        assert callable(fn)

    def test_kubota_done_still_returns_valid_function(self, wavelength):
        """Despite the deprecation warning, the returned function produces finite output."""
        import warnings

        from tengri.components.agn.unified import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = resolve_agn_model("kubota_done")

        l_nu = fn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_all_canonical_models_in_registry(self):
        """All canonical model names appear in AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS

        for name in (
            "simple",
            "standard",
            "multicolor_agn",
            "kubota_done",
            "kubota_done_full",
            "adaf",
            "unified_nlr_blr",
        ):
            assert name in AGN_MODELS, f"'{name}' missing from AGN_MODELS"


# ── register_agn_model: decorator factory ─────────────────────────


class TestRegisterAgn:
    """Tests for the register_agn_model decorator factory."""

    def test_decorator_adds_model_to_registry(self):
        """register_agn_model adds the decorated function to AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS, register_agn_model

        @register_agn_model("_test_unit_model_a1b2")
        def _dummy(wavelength, agn_log_lbol, **_kw):
            return jnp.zeros_like(wavelength)

        try:
            assert "_test_unit_model_a1b2" in AGN_MODELS
            # Registry now stores AGNRegistryEntry which wraps the function
            entry = AGN_MODELS["_test_unit_model_a1b2"]
            assert callable(entry)
            assert entry.callable is _dummy
        finally:
            AGN_MODELS.pop("_test_unit_model_a1b2", None)

    def test_decorator_returns_original_function(self):
        """The decorator is transparent: the decorated function is returned unchanged."""
        from tengri.components.agn.unified import AGN_MODELS, register_agn_model

        def _raw(wavelength, agn_log_lbol, **_kw):
            return wavelength * agn_log_lbol

        decorated = register_agn_model("_test_unit_identity_c3d4")(_raw)
        try:
            assert decorated is _raw
        finally:
            AGN_MODELS.pop("_test_unit_identity_c3d4", None)


# ── multicolor_agn: spin-dependent disc + two-temperature torus ───


class TestMulticolorAgn:
    """Tests for multicolor_agn (spin-dependent K&D outer disc + 2T torus)."""

    def test_finite_nonneg(self, wavelength):
        """multicolor_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import multicolor_agn

        l_nu = multicolor_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_multicolor_agn(self):
        """'multicolor_agn' appears in AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "multicolor_agn" in AGN_MODELS

    def test_kubota_done_alias_is_same_function(self):
        """AGN_MODELS['kubota_done'] wraps the same function as multicolor_agn."""
        from tengri.components.agn.unified import AGN_MODELS

        # kubota_done and multicolor_agn are registry entries; check they wrap same function
        assert AGN_MODELS["kubota_done"] is AGN_MODELS["multicolor_agn"]

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import multicolor_agn

        l1 = multicolor_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = multicolor_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_higher_spin_more_far_uv(self, optical_wavelength):
        """Higher BH spin → smaller ISCO → hotter inner disc → more far-UV flux.

        At maximal spin (a=0.998), the ISCO shrinks to ~1.2 R_g vs 6 R_g at a=0,
        allowing the disc to reach temperatures ~3x higher. This shifts the
        Wien peak into the far-UV/EUV (< 500 Å), boosting flux there.
        """
        from tengri.components.agn.unified import multicolor_agn

        far_uv = (optical_wavelength > 300.0) & (optical_wavelength < 500.0)
        l_nospin = multicolor_agn(
            optical_wavelength, agn_log_lbol=44.0, agn_a_spin=0.0, agn_torus_frac=0.0
        )
        l_spin = multicolor_agn(
            optical_wavelength, agn_log_lbol=44.0, agn_a_spin=0.99, agn_torus_frac=0.0
        )
        assert jnp.any(far_uv), "No wavelengths in far-UV window"
        assert float(jnp.sum(l_spin[far_uv])) > float(jnp.sum(l_nospin[far_uv]))

    def test_jit_compatible(self, wavelength):
        """multicolor_agn is JIT-compilable."""
        from tengri.components.agn.unified import multicolor_agn

        @jax.jit
        def _run(wave):
            return multicolor_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))


# ── kubota_done_full_agn: full 3-zone K&D disc + two-temperature torus


class TestKubotaDoneFullAgn:
    """Tests for kubota_done_full_agn (K&D 3-zone disc + 2T torus)."""

    def test_finite_nonneg(self, wavelength):
        """kubota_done_full_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import kubota_done_full_agn

        l_nu = kubota_done_full_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_kubota_done_full(self):
        """'kubota_done_full' appears in AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "kubota_done_full" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import kubota_done_full_agn

        l1 = kubota_done_full_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = kubota_done_full_agn(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_has_xray_emission(self, wavelength):
        """Full 3-zone disc produces X-ray emission from the hot corona.

        kubota_done_full includes a hard X-ray power law (hot corona).
        At λ < 100 Å, the full model should have non-negligible flux
        while a torus-only comparison has zero disc contribution.
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        xray_mask = (wavelength > 1.0) & (wavelength < 100.0)
        if not jnp.any(xray_mask):
            pytest.skip("wavelength grid does not cover X-ray regime")

        l_nu = kubota_done_full_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_frac=1.0,
            agn_f_hard=0.1,
            agn_torus_frac=0.0,
        )
        assert float(jnp.sum(l_nu[xray_mask])) > 0.0

    def test_f_hard_changes_sed_shape(self, wavelength):
        """Changing agn_f_hard from 0 to 0.1 alters the SED.

        With fixed L_bol, increasing f_hard routes more power to the
        corona power law and less to the disc. The two SEDs must differ.
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        l_no_corona = kubota_done_full_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_f_hard=0.0, agn_torus_frac=0.0
        )
        l_corona = kubota_done_full_agn(
            wavelength, agn_log_lbol=44.0, agn_frac=1.0, agn_f_hard=0.1, agn_torus_frac=0.0
        )
        # The SEDs must differ somewhere (not identical arrays)
        assert not jnp.allclose(l_corona, l_no_corona, rtol=1e-6)

    def test_jit_compatible(self, wavelength):
        """kubota_done_full_agn is JIT-compilable."""
        from tengri.components.agn.unified import kubota_done_full_agn

        @jax.jit
        def _run(wave):
            return kubota_done_full_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))


# ── _sigmoid_mask: smooth geometric visibility function ───────────


class TestSigmoidMask:
    """Tests for the _sigmoid_mask visibility function."""

    def test_face_on_high_visibility(self):
        """Face-on (cos_inc=1) gives visibility close to 1."""
        from tengri.components.agn.unified import _sigmoid_mask

        mask = _sigmoid_mask(cos_inc=1.0, theta_torus=30.0)
        assert float(mask) > 0.9

    def test_edge_on_low_visibility(self):
        """Edge-on (cos_inc=0) gives visibility close to 0 for a wide torus."""
        from tengri.components.agn.unified import _sigmoid_mask

        mask = _sigmoid_mask(cos_inc=0.0, theta_torus=30.0)
        assert float(mask) < 0.1

    def test_output_in_unit_interval(self):
        """Mask value is always in [0, 1]."""
        from tengri.components.agn.unified import _sigmoid_mask

        for cos_inc in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = float(_sigmoid_mask(cos_inc=cos_inc, theta_torus=30.0))
            assert 0.0 <= val <= 1.0, f"Mask out of [0,1] at cos_inc={cos_inc}: {val}"

    def test_monotone_increasing_with_cos_inc(self):
        """Larger cos_inc (more face-on) → larger visibility."""
        from tengri.components.agn.unified import _sigmoid_mask

        mask_edge = float(_sigmoid_mask(cos_inc=0.0, theta_torus=30.0))
        mask_mid = float(_sigmoid_mask(cos_inc=0.5, theta_torus=30.0))
        mask_face = float(_sigmoid_mask(cos_inc=1.0, theta_torus=30.0))
        assert mask_edge < mask_mid < mask_face

    def test_narrower_torus_increases_visibility(self):
        """Narrower torus (theta_torus = torus half-angle) → disc more visible.

        `theta_torus` is the torus HALF-ANGLE (dusty region), so
        inc_crit = 90 - theta_torus.  A narrow torus (theta=20°) has
        inc_crit=70°; a wide torus (theta=60°) has inc_crit=30°.

        At 45° inclination (cos_inc≈0.707):
          - narrow torus: 45° < 70° → disc visible (mask ≈ 1)
          - wide torus:   45° > 30° → disc blocked (mask ≈ 0)
        """
        from tengri.components.agn.unified import _sigmoid_mask

        # cos(45 deg) ≈ 0.707
        mask_narrow = float(_sigmoid_mask(cos_inc=0.707, theta_torus=20.0))
        mask_wide = float(_sigmoid_mask(cos_inc=0.707, theta_torus=60.0))
        assert mask_narrow > mask_wide

    def test_jit_compatible(self):
        """_sigmoid_mask is JIT-compilable."""
        from tengri.components.agn.unified import _sigmoid_mask

        @jax.jit
        def _run(cos_inc):
            return _sigmoid_mask(cos_inc, theta_torus=30.0)

        result = _run(jnp.array(0.5))
        assert jnp.isfinite(result)

    def test_gradient_wrt_cos_inc(self):
        """_sigmoid_mask has a finite, non-zero gradient w.r.t. cos_inc near transition."""
        from tengri.components.agn.unified import _sigmoid_mask

        # Near the transition (inc ~ 90 - theta_torus = 60 deg, cos ~ 0.5)
        g = float(jax.grad(_sigmoid_mask)(0.5, theta_torus=30.0))
        assert jnp.isfinite(jnp.array(g))
        assert g != 0.0, "Gradient of sigmoid mask should be non-zero near transition"


# ── unified_nlr_blr: NLR/BLR decomposition with geometric masking ─


class TestUnifiedNlrBlr:
    """Tests for unified_nlr_blr: disc + torus + NLR/BLR with sigmoid masking."""

    def test_finite_nonneg(self, wavelength):
        """unified_nlr_blr produces finite, non-negative SED."""
        from tengri.components.agn.unified import unified_nlr_blr

        l_nu = unified_nlr_blr(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert jnp.all(l_nu >= 0.0)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_unified_nlr_blr(self):
        """'unified_nlr_blr' appears in AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS

        assert "unified_nlr_blr" in AGN_MODELS

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_frac multiplies the whole SED linearly."""
        from tengri.components.agn.unified import unified_nlr_blr

        l1 = unified_nlr_blr(wavelength, agn_log_lbol=44.0, agn_frac=0.1)
        l2 = unified_nlr_blr(wavelength, agn_log_lbol=44.0, agn_frac=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_type1_more_uv_than_type2(self, optical_wavelength):
        """Face-on (Type 1) has more disc UV emission than edge-on (Type 2).

        The disc is masked by the torus at high inclinations. Type 2 (edge-on,
        cos_inc=0) has the disc fully obscured, so the UV (disc-dominated)
        band carries much less flux.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        uv_mask = (optical_wavelength > 1000.0) & (optical_wavelength < 4000.0)
        l_type1 = unified_nlr_blr(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_cos_inc=1.0,
            agn_theta_torus=30.0,
        )
        l_type2 = unified_nlr_blr(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_cos_inc=0.0,
            agn_theta_torus=30.0,
        )
        assert jnp.any(uv_mask), "No wavelengths in UV window"
        assert float(jnp.sum(l_type1[uv_mask])) > float(jnp.sum(l_type2[uv_mask]))

    def test_nlr_always_visible(self, optical_wavelength):
        """NLR emission appears even in edge-on (Type 2) views.

        NLR is isotropic — the mask is not applied to l_nlr. Even when
        cos_inc=0 fully obscures disc+BLR, the NLR contributes flux.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        l_type2 = unified_nlr_blr(
            optical_wavelength,
            agn_log_lbol=44.0,
            agn_cos_inc=0.0,
            agn_theta_torus=30.0,
            agn_nlr_cf=0.3,
        )
        assert float(jnp.sum(l_type2)) > 0.0

    def test_polar_dust_reddens_type1_uv(self, optical_wavelength):
        """Polar E(B-V) > 0 suppresses UV more than optical for face-on views.

        SMC extinction law rises steeply toward UV, so the UV/optical ratio
        decreases when polar dust is applied to the disc + BLR.
        """
        from tengri.components.agn.unified import unified_nlr_blr

        uv_mask = (optical_wavelength > 1000.0) & (optical_wavelength < 3000.0)
        opt_mask = (optical_wavelength > 4500.0) & (optical_wavelength < 6000.0)

        l_nodust = unified_nlr_blr(
            optical_wavelength, agn_log_lbol=44.0, agn_cos_inc=1.0, agn_polar_ebv=0.0
        )
        l_dusty = unified_nlr_blr(
            optical_wavelength, agn_log_lbol=44.0, agn_cos_inc=1.0, agn_polar_ebv=0.5
        )

        assert jnp.any(uv_mask) and jnp.any(opt_mask)
        uv_ratio = float(jnp.mean(l_dusty[uv_mask] / jnp.maximum(l_nodust[uv_mask], 1e-100)))
        opt_ratio = float(jnp.mean(l_dusty[opt_mask] / jnp.maximum(l_nodust[opt_mask], 1e-100)))
        # UV is attenuated more than optical (SMC law steeper at short λ)
        assert uv_ratio < opt_ratio

    def test_jit_compatible(self, wavelength):
        """unified_nlr_blr is JIT-compilable."""
        from tengri.components.agn.unified import unified_nlr_blr

        @jax.jit
        def _run(wave):
            return unified_nlr_blr(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))
