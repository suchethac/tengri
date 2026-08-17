# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1303: population fit with a free noise parameter.

Per-galaxy ``noise_frac_cal`` is ``(N_gal,)``; the data and noise it scales
are ``(N_gal, n_data)``. Before the fix the variable-noise closures inside
``build_jit_engine`` broadcast the two directly and crashed with
``Incompatible shapes for broadcasting: shapes=[(N_gal,), (N_gal, n_data)]``
before the fit started. The fix gives ``f_cal`` a trailing axis, guarded on
``noise.ndim == 2``.

This test runs the issue's actual surface end-to-end —
``Fitter(ForwardModel.build(population=...))`` with ``noise_frac_cal`` free
in the spec, the exact construction from the #1303 report — so it fails if
the guard is absent or mis-gated. It does NOT re-implement the reshape
logic.

Kill requirement (mutation-validated): changing ``noise.ndim == 2`` to
``noise.ndim == 99`` at the guard sites in
``src/tengri/inference/jit_engine.py`` must turn this test red with the
original broadcasting ``ValueError``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.slow

_N_GAL = 4


def test_population_fit_with_free_noise_frac_cal_runs(synthetic_ssp, simple_observation):
    """The #1303 crash scenario completes and returns a finite result.

    Pre-fix this raised ``ValueError: Incompatible shapes for broadcasting:
    shapes=[(4,), (4, 3)]`` from the variable-noise likelihood.

    N_GAL MUST DIFFER from the number of bands (4 vs 3 here): with
    N_gal == n_bands the unguarded ``(N_gal,) * (N_gal, n_bands)``
    broadcast SUCCEEDS silently but misaligned (per-galaxy values applied
    per-band), and this test would pass with the guard absent.
    """
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(2.0),
        sfh_dpl_log_total_mass=Uniform(9.0, 11.0),
        redshift=Fixed(0.05),
        # The one ingredient that matters for #1303: a free noise parameter
        # switches build_jit_engine onto the variable-noise branch.
        noise_frac_cal=Uniform(0.01, 0.2),
    )
    template = SEDModel(spec, synthetic_ssp, observation=simple_observation)

    # Non-vacuity preconditions: the spec actually frees the noise parameter,
    # and the fit is genuinely batched (N_gal, n_data). Without these the test
    # could pass while exercising the scalar path the bug never touched.
    assert "noise_frac_cal" in template.spec.free_params

    pop = PopulationSEDModel(
        sed=template,
        galaxies=[
            {"flux_obs": jnp.ones(3) * 1e-18, "noise": jnp.ones(3) * 1e-19} for _ in range(_N_GAL)
        ],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    fitter = Fitter(forward)
    assert fitter.data.shape == (_N_GAL, 3)
    assert fitter.noise.shape == (_N_GAL, 3)

    # Build the REAL engine and evaluate its variable-noise hamiltonian at a
    # real position — the exact closure that crashed. (The fixed-noise
    # hamiltonian sibling in build_jit_engine carries no f_cal and needs no
    # guard, so this is the one seam #1303 lives on.)
    dummy_pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    engine = fitter._build_jit_engine(dummy_pos)
    xi = engine["flatten"](dummy_pos)

    # PopulationSpecView.n_latent must agree with the engine's real flat
    # dimension — the value the #1408 auto-pick sites compare (a view without
    # n_latent breaks every hierarchical "auto"/"mcmc" dispatch).
    assert fitter.spec.n_latent == engine["d_total"], (
        f"spec.n_latent ({fitter.spec.n_latent}) != engine d_total ({engine['d_total']})"
    )

    value = engine["hamiltonian"](xi, fitter._data_args)
    assert np.isfinite(np.asarray(value)), f"hamiltonian is non-finite: {value}"

    grad = jax.grad(lambda x: engine["hamiltonian"](x, fitter._data_args))(xi)
    assert np.all(np.isfinite(np.asarray(grad))), "hamiltonian gradient is non-finite"
