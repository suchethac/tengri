# SPDX-License-Identifier: BSD-3-Clause
"""#1999: dense-mass HMC on the SpectrumPrecomp arm samples live.

The frozen arm from the 2026-08-24 evidence matrix (the notebook-06
construction: D=6 spectroscopy, SpectrumPrecomp, HMC_VALIDATED settings,
dense mass matrix) died with ~100% post-burnin divergence -- window
adaptation returned a step size above the stability limit of its own
returned metric, and the #2110 guard raised ``DeadFitError``. The
post-adaptation stability probe (``_stabilize_dense_mass_step``) backs
the step off until proposals survive, so the same fit must now sample
live -- and with the probe disabled, the freeze must come back, which is
what keeps this fix load-bearing.

Needs the FSPS MILES grid; skips when the file is absent.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SSP_PATH = Path(__file__).resolve().parents[2] / "data" / "fsps_prsc_miles_chabrier.h5"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _SSP_PATH.exists(), reason="FSPS MILES SSP grid not present"),
]

HMC_VALIDATED = dict(
    method="mcmc_hmc",
    n_warmup=1000,
    n_samples=600,
    n_leapfrog_steps=20,
    dense_mass_matrix=True,
    target_accept_rate=0.9,
)


def _build_forward_and_data(noise_seed):
    import tengri
    from tengri import (
        DEFAULT,
        FREE,
        Fixed,
        ForwardModel,
        Observation,
        SEDModel,
        Spectroscopy,
        SpectrumPrecomp,
        Uniform,
        builders,
    )

    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    wave_obs = jnp.linspace(3800.0, 9200.0, 260)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=2000))
    sed_model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=SpectrumPrecomp(),
        sfh=builders.sfh.tsnorm(
            all_params=Fixed(DEFAULT), log_total_mass=FREE, peak_lbt_gyr=FREE, width_gyr=FREE
        ),
        dust_attenuation=builders.dust.two_component(
            all_params=Fixed(DEFAULT),
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )
    forward = ForwardModel.build(sed=sed_model)

    truth = sed_model.spec.sample(jax.random.PRNGKey(0))
    truth = {
        **truth,
        "met_logzsol": jnp.array(-0.30),
        "dust_tau_bc": jnp.array(0.30),
        "dust_tau_diff": jnp.array(0.25),
        "sfh_tsnorm_log_total_mass": jnp.array(10.5),
    }
    truth_full = {
        **sed_model.spec.get_fixed_values(),
        **{k: float(v) for k, v in truth.items()},
    }
    p_spec = np.asarray(sed_model.predict_spectrum(truth_full, wave_obs=wave_obs))
    noise = p_spec / 30.0
    flux = p_spec + np.random.default_rng(noise_seed).normal(size=p_spec.shape) * noise
    return sed_model, forward, flux, noise


def test_dense_mass_spectrum_precomp_fit_is_live():
    """The previously frozen arm samples live under the stability probe."""
    from tengri import Data

    sed_model, forward, flux, noise = _build_forward_and_data(noise_seed=0)
    posterior = forward.fit(
        Data(spectrum=(flux, noise)),
        key=jax.random.PRNGKey(1),
        precondition=False,
        **HMC_VALIDATED,
    )
    n_div = int(posterior.diagnostics.get("n_divergent", -1))
    uniq = [len(np.unique(np.asarray(posterior.samples[p]))) for p in sed_model.spec.free_params]
    assert n_div <= 30, f"expected a live chain, got {n_div} divergent draws"
    assert min(uniq) > 400, f"chain barely moves: unique/param={uniq}"
    assert "dense_mass_step_backoffs" in posterior.diagnostics
    # Divergence count and unique-draw fraction alone are uninformative on
    # this codebase: a 2026-09-05 campaign measured cells at R-hat 2.97 with
    # zero divergences and unique fractions ~1.0. Pair all three.
    rhat_max = max(posterior.rhat().values())
    assert rhat_max < 1.2, f"between-chain disagreement survived: max R-hat {rhat_max:.3f}"


def test_probe_is_load_bearing(monkeypatch):
    """A probe that reports instability halves the step the fit samples with.

    Guards against the probe being silently disconnected from the dense-mass
    path. Until 2026-09-12 this test disabled the probe and asserted that the
    #1999 freeze returned; it no longer does on this fixture -- the L-BFGS
    MAP seed (#2311) starts the adaptation where the unprobed step is already
    stable, with or without mass profiling (0 divergences either way) -- so a
    natural freeze is not a signal the test can rely on. The wiring is: the
    probe's verdict on the adapted step is what decides the step size the
    sampler runs at. Force the verdict (unstable once, then stable) and read
    the decision off the fit: exactly one backoff, a step half the adapted
    one, and the chain still live. Distinct noise seed so the adaptation cache
    from the live test cannot serve a step to this fit.
    """
    from tengri import Data
    from tengri.inference.backends.mcmc import _shared

    verdicts = iter([1.0, 0.0])  # first probe: every trajectory diverged; second: none
    seen_steps: list[float] = []

    def _forced_probe(kernel, state, ld, da, step, imm, max_doublings):
        seen_steps.append(float(step))
        return jnp.asarray(next(verdicts))

    monkeypatch.setattr(_shared, "_probe_nuts_stability_jit", _forced_probe)

    sed_model, forward, flux, noise = _build_forward_and_data(noise_seed=1)
    posterior = forward.fit(
        Data(spectrum=(flux, noise)),
        key=jax.random.PRNGKey(1),
        precondition=False,
        **HMC_VALIDATED,
    )
    assert posterior.diagnostics["dense_mass_step_backoffs"] == 1, (
        "the probe reported instability once, yet the fit records no backoff: "
        f"probe disconnected (diagnostics={posterior.diagnostics})"
    )
    assert len(seen_steps) == 2 and seen_steps[1] == pytest.approx(seen_steps[0] / 2), (
        f"the second probe should see half the adapted step, saw {seen_steps}"
    )
    assert posterior.diagnostics["step_size"] == pytest.approx(seen_steps[1]), (
        "the sampler did not run at the step the probe settled on"
    )
    n_div = int(posterior.diagnostics.get("n_divergent", -1))
    uniq = [len(np.unique(np.asarray(posterior.samples[p]))) for p in sed_model.spec.free_params]
    assert n_div <= 30 and min(uniq) > 400, (
        f"halving the step should leave the chain live: {n_div} divergent, unique/param={uniq}"
    )
