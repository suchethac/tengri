# SPDX-License-Identifier: BSD-3-Clause
r"""``compute_fisher_matrix`` in pure float32 — two defects, one hiding the other (#1542).

The public FIM returned NaN for every pure-float32 call, from two independent
causes::

    jac = jax.jacobian(predict_from_flat)(flat)  # jacobian IS jacrev
    noise_inv = 1.0 / noise**2  # sigma ~ 5e-32  ->  inf
    weighted_jac = jac * noise_inv[:, None]  # 0 * inf = NaN

1. ``jax.jacobian`` is reverse mode, and in float32 the reverse-mode Jacobian of
   a raw flux is **exactly all zeros** (#1388/#1415). ``jacfwd`` is alive on the
   identical model.
2. ``1.0 / noise**2`` overflows. ``(1/sigma)**2`` does not help — it overflows
   too. The restructure to ``(J/sigma)^T (J/sigma)`` is required.

**The zero is more dangerous than the NaN, and this is the point of the file.**
Removing only the ``inf`` yields a finite, entirely-zero Fisher matrix — the
zero Jacobian had been hiding behind the NaN. NaN and ``inf`` propagate loudly;
a zero FIM inverts to *infinite confidence*, and a zero gradient stops an
optimizer at its start point looking converged.

So ``assert jnp.isfinite(x).all()`` is not a passing grade here, and every test
below that checks finiteness also checks non-zero. That asymmetry — loud
failures are cheap, silent plausible ones are expensive — is why the two are
asserted together rather than separately.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.analysis.diagnostics.fisher import compute_fisher_matrix, fisher_parameter_errors

pytestmark = pytest.mark.regression_bug

_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}
_NAMES = sorted(_PARAMS)


def _model(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
    )


def _physical(ssp):
    """Bring the synthetic per-Msun flux down to a real grid's regime."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * 1.0e-17,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _fim(ssp, *, x64):
    with jax.enable_x64(x64):
        model = _model(_physical(ssp))
        pred = model.predict_photometry(_PARAMS)
        noise = jnp.abs(pred) * 0.05
        fim, names = compute_fisher_matrix(model, _PARAMS, noise, param_names=_NAMES)
        return np.asarray(fim, dtype=np.float64), names


def test_setup_the_noise_reciprocal_really_overflows_float32(synthetic_ssp_wide):
    """Guard the guard: if 1/sigma**2 fitted in float32 this file proves nothing."""
    with jax.enable_x64(False):
        model = _model(_physical(synthetic_ssp_wide))
        noise = jnp.abs(model.predict_photometry(_PARAMS)) * 0.05
        reciprocal = 1.0 / np.asarray(noise, dtype=np.float64) ** 2
    assert reciprocal.min() > 3.4e38, (
        f"1/sigma**2 min is {reciprocal.min():.3e}, inside the float32 window — this "
        "fixture no longer exercises the overflow"
    )


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "pure_float32"])
def test_fisher_matrix_is_finite_and_not_all_zero(synthetic_ssp_wide, x64):
    """Both halves, together. Finiteness alone would pass on a zero Jacobian."""
    fim, _ = _fim(synthetic_ssp_wide, x64=x64)

    assert np.isfinite(fim).all(), (
        f"FIM has non-finite entries in {'float64' if x64 else 'pure float32'}. In float32 "
        "this is 1/noise**2 overflowing to inf and multiplying a zero Jacobian (#1542)"
    )
    assert not np.all(fim == 0.0), (
        "the FIM is entirely zero. This is the failure that hides behind the NaN: a zero "
        "Fisher matrix inverts to INFINITE confidence, so it is worse than the crash it "
        "replaced. It means the Jacobian came back all zeros — jax.jacobian is jacrev, "
        "which is dead in float32 on raw fluxes (#1388/#1415); use jacfwd"
    )
    assert np.all(np.diag(fim) > 0.0), (
        f"FIM diagonal is not strictly positive: {np.diag(fim)}. Every parameter that "
        "moves the prediction must carry information"
    )


def test_float32_matches_float64(synthetic_ssp_wide):
    """The restructure must not merely be finite — it must be the same matrix."""
    fim64, _ = _fim(synthetic_ssp_wide, x64=True)
    fim32, _ = _fim(synthetic_ssp_wide, x64=False)

    rel = np.abs(fim32 - fim64) / np.maximum(np.abs(fim64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 FIM differs from float64 by {rel.max():.3e} relative. Finite but wrong "
        "is the failure mode a bare isfinite check cannot see"
    )


def test_parameter_errors_are_usable_in_float32(synthetic_ssp_wide):
    """The consumer, end to end — a zero FIM would surface here as zero or inf errors."""
    fim32, _ = _fim(synthetic_ssp_wide, x64=False)
    fim64, _ = _fim(synthetic_ssp_wide, x64=True)
    err32 = np.asarray(fisher_parameter_errors(jnp.asarray(fim32)), dtype=np.float64)
    err64 = np.asarray(fisher_parameter_errors(jnp.asarray(fim64)), dtype=np.float64)

    assert np.isfinite(err32).all() and (err32 > 0).all(), (
        f"1-sigma errors from the float32 FIM are {err32} — a zero FIM inverts to "
        "infinite confidence, i.e. zero error bars, which reads as a spectacular fit"
    )
    rel = np.abs(err32 - err64) / err64
    assert rel.max() < 1e-2, f"float32 error bars differ from float64 by {rel.max():.3e}"


def test_reverse_mode_really_is_dead_here(synthetic_ssp_wide):
    """Pins the defect, so the fix cannot outlive the reason for it.

    If ``jacrev`` ever becomes viable in float32, this fails and whoever changed
    it is told they can simplify ``compute_fisher_matrix`` back. Without it the
    ``jacfwd`` choice would look like an arbitrary preference.
    """
    with jax.enable_x64(False):
        model = _model(_physical(synthetic_ssp_wide))
        flat = jnp.array([_PARAMS[n] for n in _NAMES])

        def predict(v):
            return model.predict_photometry({k: v[i] for i, k in enumerate(_NAMES)})

        jac_rev = jax.jacrev(predict)(flat)
        jac_fwd = jax.jacfwd(predict)(flat)

    assert np.all(np.asarray(jac_rev) == 0.0), (
        "the reverse-mode Jacobian is no longer identically zero in float32. If #1388/"
        "#1415 was fixed, compute_fisher_matrix can go back to jax.jacobian — but "
        "re-measure before doing it"
    )
    assert not np.all(np.asarray(jac_fwd) == 0.0), (
        "setup: forward mode is also zero here, so this configuration cannot demonstrate "
        "the asymmetry — pick parameters that move the photometry"
    )
