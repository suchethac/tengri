# SPDX-License-Identifier: BSD-3-Clause
r"""Bug #2378: float32 Hessian refusal names the dtype in both routes.

## Mechanism

The float32 refusal on ``HESSIAN_BACKEND_SET`` backends claims to cover two Hessian
routes: ``jax.hessian`` (laplace) and preconditioning's ``negative_hessian_metric``
(optional metric whitening for NUTS). Only laplace is guarded. Preconditioning under
float32 fails with a misleading diagnostic that names "a diverged MAP" as the cause
when the cause is the dtype. The SED model's photometry Hessian is all-NaN in float32
(a JAX forward-over-reverse seam), not specific to profiling.

## Reviewer scenario

The disable-then-raise pattern at ``mass_profile.py:719-720`` mutates a ``Fitter``
unconditionally but then raises. A caller catching that ``ValueError`` and reusing the
Fitter (the warm-start MAP→NUTS pattern) gets a Fitter in an inconsistent state: the
profiled loss function caches are keyed on the original free-parameter set, which no
longer matches ``fitter._free_names``.

## Assertions

(a) Under float32, ``fitter.run("laplace")`` or direct resolver call raises ``ValueError``
    whose message contains "float32", and afterwards ``fitter._profile_mass`` and the
    free-parameter count are unchanged from before the call (invariant: reuse-after-failure
    is safe).

(b) Under float32, preconditioning's non-finite-metric error message names "float32"
    as a candidate cause.

(c) Under float64, the preconditioning message does not mention float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import ForwardModel, SEDModel, generate_mock, recipes
from tengri.inference.fitter import Fitter
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.observation import Observation, Photometry


class TestFloat32HessianRefusal:
    """Float32 refusal names the dtype in both laplace and preconditioning routes."""

    @pytest.fixture
    def ssp_bare(self):
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")

    @pytest.fixture
    def minimal_model(self, ssp_bare):
        """Minimal stellar model for fast Fitter construction."""
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        return SEDModel.build(
            ssp_data=ssp_bare,
            observation=obs,
            **recipes.mock_recovery_minimal(),
        )

    @pytest.fixture
    def fitter_f32(self, minimal_model):
        """Fitter under float32, with profiling enabled."""
        forward = ForwardModel.build(sed=minimal_model)
        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = minimal_model.spec.sample(key_truth)
        mock = generate_mock(minimal_model, truth, key=key_mock, snr=30.0)
        flux = jnp.asarray(mock["flux_obs"])
        noise = jnp.asarray(mock["noise"])
        return Fitter(
            forward,
            data=flux,
            noise=noise,
            profile_mass="auto",
        )

    def test_laplace_under_float32_raises_with_dtype_in_message(self, fitter_f32):
        """Laplace refusal under float32 names the dtype in its error."""
        with jax.enable_x64(False):
            assert jnp.result_type(float) == jnp.float32
            # Record state before the call
            profile_mass_before = fitter_f32._profile_mass
            free_names_before = tuple(fitter_f32._free_names)

            with pytest.raises(ValueError) as excinfo:
                fitter_f32.run("laplace")

            msg = str(excinfo.value)
            assert "float32" in msg.lower(), (
                f"float32 refusal message should name the dtype: {msg}"
            )

            # Invariant: reuse-after-failure is safe
            assert fitter_f32._profile_mass == profile_mass_before, (
                "disable-then-raise mutated _profile_mass even though it raises"
            )
            assert tuple(fitter_f32._free_names) == free_names_before, (
                "free-parameter count changed after exception (reuse-after-failure breaks)"
            )

    def test_preconditioning_under_float32_names_dtype(self):
        """Preconditioning non-finite-metric error names float32 as a candidate cause."""

        def quartic(xi):
            return -0.5 * jnp.sum(xi**4)

        # NaN at one coordinate triggers non-finite metric
        bad_init = jnp.asarray([0.0, jnp.nan, 0.0, 0.0])

        with jax.enable_x64(False):
            assert jnp.result_type(float) == jnp.float32
            with pytest.raises(ValueError) as excinfo:
                prepare_preconditioning(quartic, bad_init, 0.0, precondition=True)

            msg = str(excinfo.value)
            assert "float32" in msg.lower(), (
                f"preconditioning error under float32 should name the dtype: {msg}"
            )

    def test_preconditioning_under_float64_does_not_name_dtype(self):
        """Preconditioning non-finite-metric error under float64 should not name float32."""

        def quartic(xi):
            return -0.5 * jnp.sum(xi**4)

        # NaN at one coordinate triggers non-finite metric
        bad_init = jnp.asarray([0.0, jnp.nan, 0.0, 0.0])

        with jax.enable_x64(True):
            assert jnp.result_type(float) == jnp.float64
            with pytest.raises(ValueError) as excinfo:
                prepare_preconditioning(quartic, bad_init, 0.0, precondition=True)

            msg = str(excinfo.value)
            # Under float64 the NaN is a true divergence, not a dtype artifact
            assert "float32" not in msg.lower(), (
                f"preconditioning error under float64 should not blame float32: {msg}"
            )
