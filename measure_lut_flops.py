#!/usr/bin/env python
"""Measure FLOP reduction from energy-balance LUT on single-component dust."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TENGRI_DISABLE_PRECOMP_CACHE"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform, Observation
from tengri.observation import Photometry
from tengri.forward.sed_model import WavePrecomp

# Setup observation
F = [
    "hst_f606w", "hst_f160w", "irac_45", "Spitzer_MIPS_24mu",
    "Spitzer_MIPS_70mu", "Herschel_Pacs_green", "Herschel_SPIRE_PSW",
    "Herschel_SPIRE_PMW"
]
obs = Observation(photometry=Photometry.from_names(F))
ssp = tengri.load_ssp("fsps_prsc_c3k_a_chabrier")

def build_model():
    """Build single-component dust attenuation model."""
    return SEDModel.build(
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

def measure_flops(model, params):
    """Measure FLOPs of photometry gradient."""
    cost_analysis = jax.jit(
        jax.grad(lambda q: jnp.sum(model.predict_photometry(q)))
    ).lower(params).compile().cost_analysis()
    if isinstance(cost_analysis, list):
        cost_analysis = cost_analysis[0]
    return float(cost_analysis["flops"])

# Build and measure with LUT enabled
print("Building model with LUT enabled...")
model_on = build_model()
params = dict(model_on.spec.sample(jax.random.PRNGKey(0)))
params["dust_tau_v"] = np.float64(1.37)
flops_on = measure_flops(model_on, params)
print(f"  With LUT:    {flops_on:,} FLOPs")

# Build and measure with LUT disabled
print("Building model with LUT disabled...")
original = SEDModel._EB_ATTEN_FREE_OK
SEDModel._EB_ATTEN_FREE_OK = frozenset(original - {"dust_tau_v"})
model_off = build_model()
SEDModel._EB_ATTEN_FREE_OK = original
flops_off = measure_flops(model_off, params)
print(f"  Without LUT: {flops_off:,} FLOPs")

# Report results
ratio = flops_off / flops_on
print(f"\n{'='*60}")
print(f"Speedup ratio: {ratio:.1f}x")
print(f"  Baseline (no LUT):  {flops_off:,} FLOPs")
print(f"  With LUT:           {flops_on:,} FLOPs")
print(f"  Reduction:          {flops_off - flops_on:,} FLOPs")

if flops_on < flops_off * 0.9:  # >10% improvement
    print(f"\n✓ SUCCESS: LUT is being consumed and providing measurable speedup")
else:
    print(f"\n✗ FAILURE: LUT not reducing FLOPs significantly")
print(f"{'='*60}")
