#!/usr/bin/env python
"""Measure LUT FLOP contribution using direct HLO cost analysis.

Creates synthetic SSP data directly to avoid fixture issues.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Photometry, Observation
import tengri.components.stellar.sps_loader as sps_loader


def create_minimal_model(lut_enabled=True):
    """Create a minimal config II model for measurement."""
    # Create synthetic SSP data with minimal coverage
    wave_ssp = jnp.logspace(2.5, 4.5, 100)  # 300 AA to 30,000 AA
    age_gyr = jnp.array([0.1, 0.3, 1.0, 3.0, 10.0, 13.8])
    met_abs = jnp.array([-2.5, -2.0, -1.5, -1.0, 0.0])

    sps_data = sps_loader.SSPData(
        wave=wave_ssp,
        l_sun=jnp.ones((len(age_gyr), len(met_abs), len(wave_ssp))),  # Flat luminosity
        age_gyr=age_gyr,
        met_abs=met_abs,
        sources={"all": "synthetic"}
    )

    # Create minimal observation
    wave_obs = jnp.logspace(2.8, 4.2, 20)  # Subset of SED
    obs = Observation(
        wave=wave_obs,
        specs=None,
        photos=[
            Photometry(
                filt_name="tophat",
                flux_nu_Fnu=None,
                wave_eff=jnp.array([1000.0, 2000.0, 5000.0, 10000.0]),
                response=None,
                tophat_width=jnp.array([100.0, 200.0, 500.0, 1000.0])
            )
        ],
        redshift=0.1
    )

    # Temporarily disable LUT if needed
    if not lut_enabled:
        import tengri.forward.sed_model as sm
        original_ok = sm.SEDModel._EB_ATTEN_FREE_OK
        sm.SEDModel._EB_ATTEN_FREE_OK = frozenset({
            "dust_tau_bc",
            "dust_tau_diff",
            "dust_eta_balance",
            "dust_log_L_ir",
        })

    try:
        model = SEDModel.build(
            ssp_data=ssp_data,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "tau_v": Uniform(0.0, 3.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )
    finally:
        if not lut_enabled:
            sm.SEDModel._EB_ATTEN_FREE_OK = original_ok

    return model


def measure_gradient_flops(model, seed=0):
    """Measure gradient FLOP count from compiled HLO."""
    params = model.spec.sample(jax.random.PRNGKey(seed))

    def loss_fn(p):
        return jnp.sum(model.predict_photometry(p))

    grad_fn = jax.grad(loss_fn)

    # Compile and get cost analysis
    compiled = grad_fn.lower(params).compile()
    cost = compiled.cost_analysis()

    # Extract FLOPs
    if isinstance(cost, dict):
        if 'flops' in cost:
            return int(cost['flops'])
        elif 'total_flops' in cost:
            return int(cost['total_flops'])

    return None


def main():
    print("=" * 70)
    print("LUT FLOP CONTRIBUTION MEASUREMENT (Direct HLO Analysis)")
    print("=" * 70)
    print()

    print("Building model WITH LUT enabled...")
    try:
        model_with_lut = create_minimal_model(lut_enabled=True)
        lut = model_with_lut._energy_balance_lut_cache
        print(f"  LUT exists: {lut is not None}")
        if lut:
            print(f"  tau_bc_grid: shape {lut.tau_bc_grid.shape}, values {lut.tau_bc_grid}")
            print(f"  tau_diff_grid: shape {lut.tau_diff_grid.shape}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print()
    print("Building model WITHOUT LUT (disabled via allowlist)...")
    try:
        model_without_lut = create_minimal_model(lut_enabled=False)
        lut = model_without_lut._energy_balance_lut_cache
        print(f"  LUT exists: {lut is not None}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print()
    print("Measuring gradient FLOPs via compiled HLO cost_analysis()...")
    print()

    print("WITH LUT enabled:")
    try:
        flops_with = measure_gradient_flops(model_with_lut)
        if flops_with is not None:
            print(f"  Gradient FLOPs: {flops_with:,}")
        else:
            print(f"  Could not extract FLOP count")
    except Exception as e:
        print(f"  ERROR: {e}")
        flops_with = None

    print()
    print("WITHOUT LUT (allowlist disabled):")
    try:
        flops_without = measure_gradient_flops(model_without_lut)
        if flops_without is not None:
            print(f"  Gradient FLOPs: {flops_without:,}")
        else:
            print(f"  Could not extract FLOP count")
    except Exception as e:
        print(f"  ERROR: {e}")
        flops_without = None

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if flops_with is not None and flops_without is not None:
        ratio = flops_without / flops_with if flops_with > 0 else float('inf')
        savings = ((flops_without - flops_with) / flops_without * 100) if flops_without > 0 else 0

        print(f"WITH LUT:    {flops_with:,} FLOPs")
        print(f"WITHOUT LUT: {flops_without:,} FLOPs")
        print(f"Ratio (without/with): {ratio:.4f}x")
        print(f"LUT savings: {savings:+.1f}%")
        print()

        if abs(ratio - 1.0) < 0.02:
            print("CONCLUSION: LUT provides <2% isolated FLOP difference on photometry path.")
            print("Band response optimization (#2487) is the dominant savings.")
        elif ratio > 1.0:
            print(f"CONCLUSION: LUT saves ~{(ratio-1)*100:.1f}% of FLOPs.")
        else:
            print(f"CONCLUSION: LUT increases FLOPs by ~{(1-ratio)*100:.1f}% (overhead).")
    else:
        print("Could not measure FLOPs. cost_analysis() may not be available.")

    print()


if __name__ == "__main__":
    main()
