# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2306: bare CB19Backend gradients underflow to zero.

Bug: ``CB19Backend`` without ``ssp_data`` has ``_log_qh_scale=0.0`` (line 835),
so ``_lum_scale ≈ 1.25e-46``. When interpolation coordinates are cast to float32
by ``_frac_idx`` (line 639), the cotangent flowing back through the float32
coordinate underflows to exactly 0.0 in the backward pass. All grid-axis
parameter gradients (``neb_co``, ``neb_hbfrac``, etc.) are therefore silently
exact zero, while forward values move correctly.

The hazard is **not reachable via ``SEDModel.build``** (always supplies
``ssp_data``), only direct-backend use in tests and downstream code. CI guards
for gradient assertions police test *declarations*, not this runtime path.

Guard: a bare-backend regression test asserting finite **and nonzero** gradients
for grid-axis parameters at the float64 precision level, using the (#2100/#2178)
assertion convention.

Mutation checks:
1. re-add `.astype(jnp.float32)` to `_frac_idx`'s return: gradient tests fail
   'Gradient is exactly zero'; forward-value test passes (2026-09-17).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.cloudy_cb19 import CB19Backend


@pytest.mark.regression_bug
class TestBareBackendGradients:
    """Gradients of bare CB19Backend must be finite and nonzero."""

    @pytest.fixture
    def bare_backend(self, tmp_path):
        """A CB19Backend without ssp_data — _lum_scale ≈ 1.25e-46."""
        from tests._cb19_grid import write_synthetic_cb19_grid

        grid_path = write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5")
        # Bare backend: no ssp_data means no Q_H precomputation
        return CB19Backend(grid_path=grid_path, ssp_data=None)

    def test_neb_co_gradient_finite_and_nonzero(self, bare_backend):
        """Gradient of predicted nebular SED w.r.t. neb_co is finite and nonzero.

        This assertion must pass in float64 precision (jax.enable_x64 is forced
        on in tests/conftest.py). The cotangent underflow happens in float32,
        not in the backward pass itself, so float64 is the correctness gate.
        Under pure float32 (jax_enable_x64=False), the coordinate dtype remains
        float32 either way (behavior unchanged).
        """
        import jax

        from tengri.components.nebular.cloudy_cb19 import _frac_idx

        # Verify coordinate dtype behavior under float32 is unchanged
        with jax.enable_x64(False):
            grid_f32 = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
            coord_f32 = _frac_idx(1.5, grid_f32)
            assert coord_f32.dtype == jnp.float32, (
                f"Coordinate should remain float32 under x64=False, got {coord_f32.dtype}"
            )

        # Enable x64 to force float64 computation
        with jax.enable_x64(True):
            # Create a synthetic SFH: single age bin at 100 Myr
            ssp_log_ages_yr = jnp.array([np.log10(100e6)])  # 100 Myr
            ssp_weights = jnp.array([1.0])  # unit weight
            log_z = np.float64(-0.5)  # solar metallicity

            # Create a scalar objective: sum of nebular line luminosities
            def objective(neb_co_val: float) -> jnp.ndarray:
                # predict_nebular_line_luminosities signature expects these kwargs
                _, lum = bare_backend.predict_nebular_line_luminosities(
                    ssp_weights=ssp_weights,
                    ssp_log_ages_yr=ssp_log_ages_yr,
                    log_z=log_z,
                    neb_logU=np.float64(-2.5),
                    neb_logZ_gas=np.float64(-0.5),
                    neb_fesc=np.float64(0.0),
                    neb_fesc_lya=np.float64(0.0),
                    neb_fdust=np.float64(0.0),
                    neb_log_nH=np.float64(2.0),
                    neb_co=neb_co_val,
                    neb_dno=np.float64(0.0),
                    neb_hbfrac=np.float64(0.5),
                )
                # Return sum as a scalar to differentiate
                return jnp.sum(lum)

            # Compute gradient w.r.t. neb_co at the mid-grid point
            grad_fn = jax.grad(objective)
            grad_value = float(grad_fn(np.float64(-0.4)))

            # Assert finite and nonzero (the #2100/#2178 convention)
            assert np.isfinite(grad_value), (
                f"Gradient is not finite: {grad_value}; "
                "float32 underflow in coordinate cast?"
            )
            assert grad_value != 0.0, (
                f"Gradient is exactly zero: {grad_value}; "
                "cotangent underflow in float32 coordinate."
            )

    def test_neb_hbfrac_gradient_finite_and_nonzero(self, bare_backend):
        """Gradient of predicted nebular SED w.r.t. neb_hbfrac is finite and nonzero."""
        import jax

        with jax.enable_x64(True):
            ssp_log_ages_yr = jnp.array([np.log10(100e6)])  # 100 Myr
            ssp_weights = jnp.array([1.0])
            log_z = np.float64(-0.5)

            def objective(neb_hbfrac_val: float) -> jnp.ndarray:
                _, lum = bare_backend.predict_nebular_line_luminosities(
                    ssp_weights=ssp_weights,
                    ssp_log_ages_yr=ssp_log_ages_yr,
                    log_z=log_z,
                    neb_logU=np.float64(-2.5),
                    neb_logZ_gas=np.float64(-0.5),
                    neb_fesc=np.float64(0.0),
                    neb_fesc_lya=np.float64(0.0),
                    neb_fdust=np.float64(0.0),
                    neb_log_nH=np.float64(2.0),
                    neb_co=np.float64(-0.4),
                    neb_dno=np.float64(0.0),
                    neb_hbfrac=neb_hbfrac_val,
                )
                return jnp.sum(lum)

            grad_fn = jax.grad(objective)
            grad_value = float(grad_fn(np.float64(0.5)))

            assert np.isfinite(grad_value), (
                f"Gradient is not finite: {grad_value}; "
                "float32 underflow in coordinate cast?"
            )
            assert grad_value != 0.0, (
                f"Gradient is exactly zero: {grad_value}; "
                "cotangent underflow in float32 coordinate."
            )

    def test_forward_value_changes_with_neb_co(self, bare_backend):
        """Control: forward value changes when neb_co changes (sanity check)."""
        ssp_log_ages_yr = jnp.array([np.log10(100e6)])  # 100 Myr
        ssp_weights = jnp.array([1.0])
        log_z = np.float64(-0.5)

        _, lum_low = bare_backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=np.float64(-2.5),
            neb_logZ_gas=np.float64(-0.5),
            neb_fesc=np.float64(0.0),
            neb_fesc_lya=np.float64(0.0),
            neb_fdust=np.float64(0.0),
            neb_log_nH=np.float64(2.0),
            neb_co=np.float64(-0.5),
            neb_dno=np.float64(0.0),
            neb_hbfrac=np.float64(0.5),
        )
        val_low = np.asarray(jnp.sum(lum_low), dtype=np.float64)

        _, lum_high = bare_backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=np.float64(-2.5),
            neb_logZ_gas=np.float64(-0.5),
            neb_fesc=np.float64(0.0),
            neb_fesc_lya=np.float64(0.0),
            neb_fdust=np.float64(0.0),
            neb_log_nH=np.float64(2.0),
            neb_co=np.float64(-0.3),
            neb_dno=np.float64(0.0),
            neb_hbfrac=np.float64(0.5),
        )
        val_high = np.asarray(jnp.sum(lum_high), dtype=np.float64)

        assert val_low != val_high, (
            f"Forward value does not change when neb_co changes; "
            f"expected {val_low} != {val_high}"
        )

