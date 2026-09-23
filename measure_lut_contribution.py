#!/usr/bin/env python
"""Measure the energy-balance LUT's isolated FLOP contribution on single-screen dust.

Methodology: Build single-screen models, one with LUT enabled (dust_tau_v in
_EB_ATTEN_FREE_OK) and one without. Both have band response enabled (from PR #2487).
Measure gradient FLOPs for predict_photometry on config II using compiled HLO cost.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import WavePrecomp


def build_model_with_lut_state(lut_enabled):
    """Build config II single-screen model with/without LUT enabled."""
    # Import fixture factories
    from tests.conftest import synthetic_ssp_wide, synthetic_tophat_obs

    ssp = synthetic_ssp_wide()
    obs = synthetic_tophat_obs()

    # If LUT is disabled, we need to monkeypatch _EB_ATTEN_FREE_OK temporarily
    if not lut_enabled:
        import tengri.forward.sed_model as sm
        original_ok = sm.SEDModel._EB_ATTEN_FREE_OK
        sm.SEDModel._EB_ATTEN_FREE_OK = frozenset(
            {
                "dust_tau_bc",
                "dust_tau_diff",
                "dust_eta_balance",
                "dust_log_L_ir",
            }
        )

    model = SEDModel.build(
        ssp_data=ssp,
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

    if not lut_enabled:
        sm.SEDModel._EB_ATTEN_FREE_OK = original_ok

    return model


def measure_flops(model, seed=0):
    """Measure gradient FLOPs using jax.grad and cost_analysis."""
    params = model.spec.sample(jax.random.PRNGKey(seed))

    def loss_fn(p):
        return jnp.sum(model.predict_photometry(p))

    grad_fn = jax.grad(loss_fn)

    # Compile the gradient
    compiled = grad_fn.lower(params).compile()

    # Get FLOP cost from cost analysis
    cost = compiled.cost_analysis()
    if isinstance(cost, dict) and 'flops' in cost:
        return cost['flops']

    # If cost analysis doesn't have flops, try to extract from the compiled module
    try:
        module_str = str(compiled.as_compiled_module())
        if 'flop_count' in module_str:
            # Try to extract FLOP count
            import re
            match = re.search(r'flop_count["\']?\s*[:=]\s*(\d+)', module_str)
            if match:
                return int(match.group(1))
    except:
        pass

    return None


def main():
    print("=" * 70)
    print("LUT FLOP CONTRIBUTION MEASUREMENT")
    print("=" * 70)
    print()

    print("Building model WITH LUT enabled (dust_tau_v in _EB_ATTEN_FREE_OK)...")
    model_with_lut = build_model_with_lut_state(lut_enabled=True)

    lut = model_with_lut._energy_balance_lut_cache
    print(f"  LUT cache exists: {lut is not None}")
    if lut is not None:
        print(f"  tau_bc_grid shape: {lut.tau_bc_grid.shape}, value: {lut.tau_bc_grid[0]}")
        print(f"  tau_diff_grid shape: {lut.tau_diff_grid.shape}")
        print(f"  tau_diff range: [{lut.tau_diff_grid.min()}, {lut.tau_diff_grid.max()}]")
    print()

    print("Building model WITHOUT LUT enabled (LUT disabled via allowlist)...")
    model_without_lut = build_model_with_lut_state(lut_enabled=False)

    lut = model_without_lut._energy_balance_lut_cache
    print(f"  LUT cache exists: {lut is not None}")
    print()

    print("Measuring gradient FLOPs...")
    print()

    print("WITH LUT:")
    flops_with = measure_flops(model_with_lut)
    print(f"  FLOPs: {flops_with}")
    print()

    print("WITHOUT LUT:")
    flops_without = measure_flops(model_without_lut)
    print(f"  FLOPs: {flops_without}")
    print()

    if flops_with is not None and flops_without is not None:
        ratio = flops_without / flops_with if flops_with > 0 else 0
        savings_pct = (1 - flops_with / flops_without) * 100 if flops_without > 0 else 0
        print(f"FLOP RATIO (without / with): {ratio:.3f}x")
        print(f"LUT savings: {savings_pct:.1f}%")
        print()
        if abs(ratio - 1.0) < 0.01:
            print("RESULT: LUT provides negligible isolated FLOP savings on photometry path")
        else:
            print(f"RESULT: LUT provides ~{ratio:.1f}x speedup (or slowdown if <1)")
    else:
        print("Could not measure FLOPs using cost_analysis()")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
