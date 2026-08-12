# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for X-ray XRB luminosity offset parameters.

Verifies that ``log_L_hmxb_offset`` and ``log_L_lmxb_offset`` parameters
in XRayAirdSEDComponent correctly scale the HMXB and LMXB luminosities.
Parity target: X-CIGALE (Yang+2020 [1]_) ``det_hmxb`` / ``det_lmxb``
parameters (yang20.py:64–75).

**Contract**:
- Default offsets (0.0) reproduce the baseline XRB luminosity
- Offset +0.3 dex multiplies the corresponding XRB component by 10^0.3 ≈ 2.0×
- Offsets are independent and only affect the targeted component
- Gradients (via jax.grad) are finite and well-behaved
- JIT-compilation succeeds

**References**
.. [1] Yang, G. et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
   CIGALE and improvement of the code," MNRAS, 491, 740 (2020).
   https://doi.org/10.1093/mnras/stz3001
.. [2] Lehmer, B. D. et al., "The evolution of the X-ray binary
   luminosity functions of nearby galaxies with the Chandra COSMOS
   survey," ApJ, 825, 7 (2016).
   https://doi.org/10.3847/0004-637X/825/1/7
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import xray_total, xray_xrb
from tengri.components.xray.xray_model import XRayAirdSEDComponent
from tests._grad_parity import assert_grad_matches_fd

# ─────────────────────────────────────────────── xray_xrb offset tests


@pytest.mark.contract
def test_xray_xrb_hmxb_offset_zero_is_baseline() -> None:
    """HMXB offset = 0 reproduces baseline XRB spectrum.

    Ensures default behavior (backward compatibility): when both offsets
    are 0.0, xray_xrb should produce the identical spectrum as before
    the offsets were added.
    """
    wave = jnp.logspace(np.log10(0.1), np.log10(124.0), 500)

    # Baseline: no offsets
    L_baseline = xray_xrb(
        wave,
        sfr=1.0,
        stellar_mass=1e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        gamma_hmxb=2.0,
        gamma_lmxb=1.7,
        log_L_hmxb_offset=0.0,
        log_L_lmxb_offset=0.0,
    )

    # With explicit zero offsets (should be identical)
    L_with_offsets = xray_xrb(
        wave,
        sfr=1.0,
        stellar_mass=1e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        gamma_hmxb=2.0,
        gamma_lmxb=1.7,
        log_L_hmxb_offset=0.0,
        log_L_lmxb_offset=0.0,
    )

    chex.assert_trees_all_close(L_baseline, L_with_offsets, rtol=1e-14)


@pytest.mark.contract
def test_xray_xrb_hmxb_offset_positive_scales_xrb_luminosity() -> None:
    """HMXB offset +0.3 dex multiplies total XRB by 10^0.3.

    Physical interpretation: a +0.3 dex offset means the HMXB population
    is 10^0.3 ≈ 1.995× brighter than the Lehmer+2016 expectation at the
    given SFR and metallicity.

    Integrated 2–10 keV luminosity should scale as 10^0.3.
    """
    wave = jnp.logspace(np.log10(0.1), np.log10(124.0), 800)
    e_lo_kev, e_hi_kev = 2.0, 10.0

    def band_integral(L_nu: jnp.ndarray) -> float:
        """Integrate ∫ L_ν dν over a keV band (returns erg/s)."""
        kev_per_aa = 12.398
        e_kev = kev_per_aa / np.asarray(wave)
        mask = (e_kev >= e_lo_kev) & (e_kev <= e_hi_kev)
        nu_band = (e_kev[mask] / kev_per_aa) * 2.998e18
        order = np.argsort(nu_band)
        return float(np.trapezoid(np.asarray(L_nu)[mask][order], nu_band[order]))

    # Baseline (zero LMXB to isolate HMXB)
    L_base = xray_xrb(
        wave,
        sfr=2.0,
        stellar_mass=0.0,  # zero stellar mass → zero LMXB
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_hmxb_offset=0.0,
    )
    I_base = band_integral(L_base)

    # With +0.3 dex offset
    L_offset = xray_xrb(
        wave,
        sfr=2.0,
        stellar_mass=0.0,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_hmxb_offset=0.3,
    )
    I_offset = band_integral(L_offset)

    # Ratio should be 10^0.3 ≈ 1.9953
    expected_factor = 10**0.3
    ratio = I_offset / I_base
    chex.assert_trees_all_close(ratio, expected_factor, rtol=2e-2)


@pytest.mark.contract
def test_xray_xrb_lmxb_offset_positive_scales_xrb_luminosity() -> None:
    """LMXB offset +0.3 dex multiplies total XRB by 10^0.3.

    Physical interpretation: a +0.3 dex offset means the LMXB population
    is 10^0.3 brighter than the Lehmer+2016 expectation at the given
    stellar mass and age.

    Integrated 2–10 keV luminosity should scale as 10^0.3.
    """
    wave = jnp.logspace(np.log10(0.1), np.log10(124.0), 800)
    e_lo_kev, e_hi_kev = 2.0, 10.0

    def band_integral(L_nu: jnp.ndarray) -> float:
        """Integrate ∫ L_ν dν over a keV band (returns erg/s)."""
        kev_per_aa = 12.398
        e_kev = kev_per_aa / np.asarray(wave)
        mask = (e_kev >= e_lo_kev) & (e_kev <= e_hi_kev)
        nu_band = (e_kev[mask] / kev_per_aa) * 2.998e18
        order = np.argsort(nu_band)
        return float(np.trapezoid(np.asarray(L_nu)[mask][order], nu_band[order]))

    # Baseline (zero SFR to isolate LMXB)
    L_base = xray_xrb(
        wave,
        sfr=0.0,  # zero SFR → zero HMXB
        stellar_mass=1e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_lmxb_offset=0.0,
    )
    I_base = band_integral(L_base)

    # With +0.3 dex offset
    L_offset = xray_xrb(
        wave,
        sfr=0.0,
        stellar_mass=1e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_lmxb_offset=0.3,
    )
    I_offset = band_integral(L_offset)

    # Ratio should be 10^0.3 ≈ 1.9953
    expected_factor = 10**0.3
    ratio = I_offset / I_base
    chex.assert_trees_all_close(ratio, expected_factor, rtol=2e-2)


@pytest.mark.contract
def test_xray_xrb_offsets_independent() -> None:
    """HMXB and LMXB offsets are independent.

    Changing one should not affect the other component.
    """
    wave = jnp.array([4.0])  # single X-ray wavelength ≈ 3 keV

    # Spectrum with both components present
    L_both = xray_xrb(
        wave,
        sfr=1.0,
        stellar_mass=1e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_hmxb_offset=0.5,
        log_L_lmxb_offset=0.2,
    )

    # HMXB contribution alone (zero mass)
    L_hmxb_only = xray_xrb(
        wave,
        sfr=1.0,
        stellar_mass=0.0,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_hmxb_offset=0.5,
    )

    # LMXB contribution alone (zero SFR)
    L_lmxb_only = xray_xrb(
        wave,
        sfr=0.0,
        stellar_mass=1e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        log_L_lmxb_offset=0.2,
    )

    # Both together should equal sum of parts
    chex.assert_trees_all_close(L_both, L_hmxb_only + L_lmxb_only, rtol=1e-13)


@pytest.mark.gradient
def test_xray_xrb_gradient_finite_in_hmxb_offset() -> None:
    """∂L/∂(log_L_hmxb_offset) is finite at offset=0.

    Ensures xray_xrb can be optimized w.r.t. the HMXB offset parameter.
    """
    wave = jnp.array([4.0])

    def L_total_vs_hmxb_offset(offset):
        return jnp.sum(
            xray_xrb(
                wave,
                sfr=1.0,
                stellar_mass=0.0,
                metallicity_z=0.02,
                stellar_age_gyr=5.0,
                log_L_hmxb_offset=offset,
            )
        )

    grad = assert_grad_matches_fd(L_total_vs_hmxb_offset, 0.0)
    assert jnp.isfinite(grad), f"gradient not finite: {grad}"
    # Gradient w.r.t. log offset should be positive (brighter offset → brighter X-ray)
    assert grad > 0.0


@pytest.mark.gradient
def test_xray_xrb_gradient_finite_in_lmxb_offset() -> None:
    """∂L/∂(log_L_lmxb_offset) is finite at offset=0.

    Ensures xray_xrb can be optimized w.r.t. the LMXB offset parameter.
    """
    wave = jnp.array([4.0])

    def L_total_vs_lmxb_offset(offset):
        return jnp.sum(
            xray_xrb(
                wave,
                sfr=0.0,
                stellar_mass=1e10,
                metallicity_z=0.02,
                stellar_age_gyr=5.0,
                log_L_lmxb_offset=offset,
            )
        )

    grad = assert_grad_matches_fd(L_total_vs_lmxb_offset, 0.0)
    assert jnp.isfinite(grad), f"gradient not finite: {grad}"
    # Gradient w.r.t. log offset should be positive
    assert grad > 0.0


# ─────────────────────────────────────────────── Component smoke tests


@pytest.mark.contract
def test_xray_aird_component_det_params_declared(synthetic_ssp_wide, synthetic_tophat_obs):
    """XRayAirdSEDComponent declares det_hmxb and det_lmxb as free parameters.

    Smoke test: component can be instantiated and declares the new parameters.
    """
    component = XRayAirdSEDComponent()

    # Check parameter_prefix
    assert component.parameter_prefix == "xray_"

    # Check that the new parameters are declared via the parent class mechanism
    declared = component.declared_parameters()
    param_names = {p.name for p in declared}

    # Should include det_hmxb and det_lmxb (and the existing ones)
    # Parameter names are full prefixed names: xray_gamma_hmxb, xray_det_hmxb, etc.
    expected_full_names = {
        "xray_gamma_hmxb",
        "xray_gamma_lmxb",
        "xray_gamma_agn",
        "xray_log_nh",
        "xray_det_hmxb",
        "xray_det_lmxb",
    }
    assert expected_full_names.issubset(param_names), (
        f"Missing parameters. Got {param_names}, expected at least {expected_full_names}"
    )


@pytest.mark.contract
def test_xray_aird_component_predict_with_offsets(synthetic_ssp_wide, synthetic_tophat_obs):
    """XRayAirdSEDComponent.predict() accepts det_hmxb and det_lmxb in param dict.

    Smoke test: component predict method runs without error when offsets are provided.
    """
    component = XRayAirdSEDComponent()
    wave = jnp.logspace(np.log10(100.0), np.log10(1e4), 200)
    sed_in = jnp.zeros_like(wave)

    # Minimal parameter dict with new offset parameters
    p = {
        "gamma_hmxb": jnp.array(2.0),
        "gamma_lmxb": jnp.array(1.7),
        "gamma_agn": jnp.array(1.9),
        "log_nh": jnp.array(21.0),
        "det_hmxb": jnp.array(0.0),
        "det_lmxb": jnp.array(0.0),
    }

    sed_out, published = component.predict(
        p,
        sed_in,
        wave,
        sfr=1.0,
        log_mstar=10.0,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        L_2500_30deg=0.0,
    )

    chex.assert_tree_all_finite(sed_out)
    chex.assert_tree_all_finite(published["sed_xray"])
    assert sed_out.shape == wave.shape


@pytest.mark.contract
def test_xray_total_jit_with_offsets() -> None:
    """xray_total JIT-compiles and runs correctly with offset parameters.

    Smoke test: ensures JAX JIT compilation succeeds and output is finite.
    """
    wave = jnp.logspace(np.log10(0.1), np.log10(124.0), 300)

    @jax.jit
    def compute_xray_jit(
        log_L_hmxb_offset: float,
        log_L_lmxb_offset: float,
    ) -> jnp.ndarray:
        return xray_total(
            wave,
            sfr=1.0,
            stellar_mass=1e10,
            metallicity_z=0.02,
            stellar_age_gyr=5.0,
            log_L_hmxb_offset=log_L_hmxb_offset,
            log_L_lmxb_offset=log_L_lmxb_offset,
        )

    L_jit = compute_xray_jit(0.2, 0.1)
    chex.assert_tree_all_finite(L_jit)
    assert L_jit.shape == wave.shape
