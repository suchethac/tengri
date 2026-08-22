# SPDX-License-Identifier: BSD-3-Clause
"""#1177: the MCLMC backends must actually sample under blackjax >= 1.6.

blackjax made two independent breaking changes between 1.3 and 1.6:

1. ``mclmc_find_L_and_step_size`` / ``adjusted_mclmc_find_L_and_step_size``
   gained a required ``logdensity_fn``; in the adjusted variant it was
   inserted as the *second positional* parameter, so any positional call
   silently binds the wrong argument.
2. The kernel returned by ``build_kernel`` changed contract. It is no longer
   a closure over the log-density: it is called per step as
   ``kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size)``
   (and the adjusted kernel additionally unpacks ``integration_steps_params``
   as a *tuple*).

Only the first was fixed initially, which is why the backends still raised
``TypeError: kernel() missing 2 required positional arguments: 'L' and
'step_size'`` on a real 1.6 install.

These tests drive tengri's own ``run_mclmc`` / ``run_adjusted_mclmc`` — not
blackjax directly — because a test that re-implements the call it is meant to
guard cannot fail when the production path breaks. Neutering the kernel
construction in ``backends/mcmc/mclmc.py`` or the per-step call in
``backends/mcmc/_shared.py`` makes them fail.

The runners are invoked directly rather than through ``fit(method=...)``:
``mcmc_mclmc`` is registered ``tier="broken"`` for unrelated mixing reasons
(R-hat ~ 1.7), so the dispatcher refuses it by design.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _blackjax_has_16_kernel_contract() -> bool:
    """True when the installed blackjax uses the >= 1.6 per-step kernel contract."""
    try:
        import blackjax
    except ImportError:
        return False
    try:
        kernel = blackjax.mcmc.mclmc.build_kernel(
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )
    except TypeError:
        return False  # pre-1.6 build_kernel requires logdensity_fn
    return "logdensity_fn" in inspect.signature(kernel).parameters


requires_blackjax_16 = pytest.mark.skipif(
    not _blackjax_has_16_kernel_contract(),
    reason="#1177: MCLMC backends require blackjax >= 1.6 (per-step kernel contract)",
)


@pytest.fixture(scope="module")
def tiny_fitter():
    """A 2-free-parameter photometric target — the smallest honest MCLMC problem."""
    from tengri import FREE, Fixed, ForwardModel, Observation, Photometry, SEDModel
    from tengri.components.stellar.sps.dsps_wrapper import SSPData
    from tengri.inference.fitter import Fitter
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(3000.0, 10000.0, 60)
    ages = jnp.linspace(-1.0, 1.14, 12)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    flux_grid = jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5
    ssp = SSPData(ssp_wave=wave, ssp_flux=flux_grid, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
        for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
    )
    obs = Observation(photometry=Photometry(filters=curves))
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl"},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
        redshift=Fixed(0.05),
    )
    truth = {"dust_tau_bc": 0.3, "dust_tau_diff": 0.2}
    data = jnp.asarray(np.asarray(sed.predict_photometry(truth)))
    noise = jnp.asarray(0.05 * np.abs(np.asarray(data)))
    forward = ForwardModel.build(sed=sed, observation=obs)
    return Fitter(forward, data=data, noise=noise)


@requires_blackjax_16
def test_run_mclmc_samples_end_to_end(tiny_fitter):
    """``run_mclmc`` must complete adaptation AND sampling, not just build a kernel."""
    from tengri.inference.backends.mcmc.mclmc import run_mclmc

    posterior = run_mclmc(
        tiny_fitter, key=jax.random.PRNGKey(0), n_warmup=40, n_samples=60, verbose=False
    )
    draws = np.asarray(posterior.samples["dust_tau_bc"])
    assert draws.shape == (60,), f"expected 60 draws, got {draws.shape}"
    assert np.all(np.isfinite(draws)), "MCLMC produced non-finite samples"


@requires_blackjax_16
def test_run_adjusted_mclmc_samples_end_to_end(tiny_fitter):
    """The adjusted variant exercises the second changed seam.

    Its adaptation takes ``logdensity_fn`` as positional #2, and its kernel
    unpacks ``integration_steps_params`` as a tuple — a bare scalar raises
    ``TypeError: iteration over a 0-d array``.
    """
    from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc

    posterior = run_adjusted_mclmc(
        tiny_fitter, key=jax.random.PRNGKey(1), n_warmup=40, n_samples=60, verbose=False
    )
    draws = np.asarray(posterior.samples["dust_tau_bc"])
    assert draws.shape == (60,), f"expected 60 draws, got {draws.shape}"
    assert np.all(np.isfinite(draws)), "adjusted MCLMC produced non-finite samples"
