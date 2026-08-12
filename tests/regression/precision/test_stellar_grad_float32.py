# SPDX-License-Identifier: BSD-3-Clause
r"""The stellar SED must be *differentiable* in pure float32, not just finite (#1206).

A finite forward value is not enough for a float32 fit — the fit needs
``grad``. The CSP mass-scaling ``lnu_age = total_mass * ssp_flux * L_sun`` keeps
its forward value in range by ordering the multiplies (``total_mass * ssp_flux``
~1e-5 lands before ``L_sun``), but autodiff's local Jacobian for that product is
``total_mass * L_sun`` ~ 3.8e43, which overflows float32 (3.4e38) to ``inf`` as a
standalone intermediate under XLA's *fused* reverse pass — even though the true
gradient is in range (the unfused ``jax_debug_nans`` path is finite).

The fix folds ``L_sun`` into the params-independent SSP operand *inside* the
einsum, so the only Jacobians autodiff forms are ``total_mass`` (~1e10) and the
erg-scaled SSP (~3.8e18), both representable. This test pins the reverse pass,
which a forward-only finiteness check (``test_dust_ir_float32``) cannot see.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

#: Brings the synthetic fixture's per-Msun flux down to a real grid's regime,
#: so ~1e43 mass scales are exercised (see test_dust_ir_float32).
_SSP_FLUX_SCALE = 1.0e-17


def _physical_ssp(ssp):
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * _SSP_FLUX_SCALE,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _model(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Uniform(-1.0, 0.2), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": FREE,
            "*": FIXED,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.7),
            "*": FIXED,
        },
        redshift=Fixed(0.1),
    )


@pytest.mark.parametrize("param", ["sfh_delayed_log_total_mass", "met_logzsol"])
def test_stellar_sed_gradient_is_finite_in_pure_float32(synthetic_ssp_wide, param):
    """d(sum sed_intrinsic)/d(param) must be finite under fused float32 execution."""
    ssp = _physical_ssp(synthetic_ssp_wide)
    model = _model(ssp)
    base = {"sfh_delayed_log_total_mass": 10.0, "met_logzsol": -0.3}

    def sed_sum(x):
        p = dict(base)
        p[param] = x
        return jnp.sum(model.predict_state(p).sed_intrinsic)

    # float64 reference: the true gradient, and proof the objective is smooth here.
    with jax.enable_x64(True):
        g64 = float(np.asarray(jax.grad(sed_sum)(base[param])))
    assert np.isfinite(g64) and g64 != 0.0, "setup: float64 gradient is not a finite nonzero"

    with jax.enable_x64(False):
        g32 = np.asarray(jax.grad(sed_sum)(np.float32(base[param])))
    assert g32.dtype == jnp.float32, "precondition: genuinely float32"
    assert np.isfinite(g32), (
        f"d(sed)/d({param}) is non-finite in fused float32 — the mass-scale "
        "backward materialized total_mass*L_sun (~3.8e43, inf in float32); L_sun "
        "must be folded into the SSP operand inside the einsum"
    )
    # Same gradient to float32 precision (the fold is algebraically identical).
    assert abs(g32.astype(np.float64) / g64 - 1.0) < 1e-3, (
        f"float32 gradient {float(g32):.4e} departs from float64 {g64:.4e}"
    )
