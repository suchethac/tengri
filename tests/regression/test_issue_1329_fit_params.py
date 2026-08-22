# SPDX-License-Identifier: BSD-3-Clause
"""#1329 — fit(params=...) per-fit override must reach the FORWARD PASS.

The bug this guards is a *silent relabel*: an override applied only to the
output param dict (``_to_physical``) while the loss the optimizer minimizes keeps
running at the model's fixed value. Such a fix passes an echo-only check
(``result.params["redshift"]`` reflects the override) yet changes nothing about
the fit. The real proof, asserted below, is that the FREE-parameter MAP diverges
between two redshift overrides — because the same observed photometry maps to
different physical parameters at different redshift.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, ForwardModel, Observation, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve


def build_test_model(use_waveprecomp=False):
    """A small hermetic model (synthetic SSP, no data files)."""
    from tengri import FREE
    from tengri.forward.sed_model import WavePrecomp

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
        redshift=Fixed(0.5),
    )
    truth = {"dust_tau_bc": 0.3, "dust_tau_diff": 0.2}
    data = jnp.asarray(np.asarray(sed.predict_photometry(truth)))
    noise = jnp.asarray(0.05 * np.abs(np.asarray(data)))

    forward = ForwardModel.build(sed=sed, observation=obs)
    if use_waveprecomp:
        forward = forward.with_approx(WavePrecomp(catalog_z_range=(0.01, 2.0), n_z=30))
    return forward, data, noise, sed


class TestFitParamsOverride:
    """params=... per-fit override reaching the forward pass (#1329)."""

    pytestmark = pytest.mark.regression_bug

    def test_params_override_reaches_forward_pass(self):
        """The override changes the FIT, not just the echoed output.

        This is the load-bearing assertion. Under a silent relabel the two
        free-param MAPs would be identical; a real override makes them diverge.
        """
        forward, data, noise, sed = build_test_model(use_waveprecomp=True)
        free = list(sed.spec.free_params)

        r1 = forward.fit(data, noise, method="map", params={"redshift": 0.1}, n_steps=100)
        r2 = forward.fit(data, noise, method="map", params={"redshift": 1.5}, n_steps=100)

        # Echoed redshift reflects the override (necessary, NOT sufficient — an
        # output-only relabel also passes this).
        assert abs(float(r1.params["redshift"]) - 0.1) < 1e-5
        assert abs(float(r2.params["redshift"]) - 1.5) < 1e-5

        # The forward pass actually used the override: the free-param MAP diverges.
        # (Under the silent-relabel bug this max delta is exactly 0.)
        max_free_delta = max(abs(float(r1.params[p]) - float(r2.params[p])) for p in free)
        assert max_free_delta > 1e-3, (
            f"free-param MAP identical across redshift overrides (max Δ={max_free_delta:.2e}) "
            "— the override did not reach the forward pass (silent relabel)."
        )

    def test_params_naming_free_param_raises(self):
        """params key naming a FREE parameter raises ValueError."""
        forward, data, noise, sed = build_test_model()

        free_params = sed.spec.free_params
        assert free_params, "test model must have at least one free parameter"
        free_name = free_params[0]
        with pytest.raises(ValueError, match="free"):
            forward.fit(data, noise, method="map", params={free_name: 0.5})

    def test_params_naming_nonexistent_param_raises(self):
        """params key naming a nonexistent parameter raises ValueError."""
        forward, data, noise, _ = build_test_model()

        with pytest.raises(ValueError, match="not a valid parameter"):
            forward.fit(data, noise, method="map", params={"nonexistent_param": 0.5})

    def test_model_unmutated_after_params_fit(self):
        """The model object is not mutated by a params= fit."""
        forward, data, noise, sed = build_test_model()

        original_fixed = dict(sed.spec.get_fixed_values())
        forward.fit(data, noise, method="map", params={"redshift": 0.1})
        current_fixed = dict(sed.spec.get_fixed_values())
        assert original_fixed == current_fixed

    def test_two_overrides_do_not_collide_in_the_loss_cache(self):
        """Distinct overrides get distinct baked loss functions (#1329 cache key).

        The loss closure bakes ``fitter._fixed_values`` and is cached on the model
        by ``_engine_cache_key()``. If the override were not part of that key, the
        second fit would silently reuse the first fit's baked redshift. Fitting in
        the OTHER order and comparing to the first run's MAP catches that collision.
        """
        forward, data, noise, sed = build_test_model(use_waveprecomp=True)
        free = list(sed.spec.free_params)

        # Fit z=1.5 first this time; if the cache collided on the previous test's
        # ordering, the MAP for a given override would depend on fit order.
        a = forward.fit(data, noise, method="map", params={"redshift": 1.5}, n_steps=100)
        b = forward.fit(data, noise, method="map", params={"redshift": 0.1}, n_steps=100)
        max_delta = max(abs(float(a.params[p]) - float(b.params[p])) for p in free)
        assert max_delta > 1e-3, (
            "the two overrides produced identical MAPs — likely a loss-cache "
            "collision (override missing from _engine_cache_key)."
        )


class TestUnknownKwargValidation:
    """Half B (fit-level unknown-kwarg validation) is a deferred follow-up.

    #1329 Half A (the params= override) is implemented and load-bearing above.
    Half B — validating unknown kwargs against the union of the resolved backend
    runner + Fitter.run + fit signatures — is intentionally NOT built here: a
    naive version would reject legitimate backend options (n_warmup, n_samples,
    forward_chunk_size, ...). These tests pin only what is TRUE today, so the
    deferral is honest and a future Half B cannot silently break real kwargs.
    """

    pytestmark = pytest.mark.regression_bug

    def test_unknown_kwarg_rejected_by_backend_today(self):
        """A garbage kwarg is rejected today by the backend runner, incidentally.

        This is NOT #1329 validation (deferred) — it documents that an unknown
        kwarg does not pass silently through the MAP backend. If a future change
        makes this pass silently, Half B should be prioritized.
        """
        forward, data, noise, _ = build_test_model()
        with pytest.raises((TypeError, ValueError)):
            forward.fit(data, noise, method="map", notakwarg=1)

    def test_real_backend_kwargs_still_accepted(self):
        """Real MAP kwargs are not rejected (guards a future over-eager Half B)."""
        forward, data, noise, _ = build_test_model()
        for kwargs in ({"n_steps": 50}, {"learning_rate": 0.01}, {"verbose": False}):
            result = forward.fit(data, noise, method="map", **kwargs)
            assert result is not None
