# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 gradients must survive the bolometric peak factorizations (#1436).

Nine reductions in ``src/`` factor an integrand by its own peak to keep float32
intermediates in range, then multiply the peak back — directly, or as a
``log10(peak)`` term. ``(x/p) * p`` is ``x`` for *any* ``p``, so the peak's
derivative contributions cancel analytically. Autodiff has to cancel them
**numerically**, through two separate paths. Float64 cancels; float32 does not, and
what survives is an uncancelled term.

Only ``apply_log10_scale`` held its peak under ``stop_gradient`` (#1415). The rest did
not, and the two seams that carry the largest scales — the dust energy balance
(~+44.5 dex) and the AGN bolometric renormalization (~+34.6 dex) — were measurably
wrong:

===================  ==========================  =========================
model                float32 grad vs float64     after ``stop_gradient``
===================  ==========================  =========================
stellar + dust        7.3e-05                     7.3e-05  (no such seam)
+ dust IR             **2.96e-01**                7.7e-04
+ AGN                 **3.00e-01**                1.1e-03
===================  ==========================  =========================

A ~30% error, finite and plausible, in the gradient a pure-float32 fit descends. It
would have converged confidently to the wrong answer, and none of the existing guards
could see it: float64 is correct, the float32 *forward* pass is correct to ~1e-6, the
values are never NaN, and the one float32 gradient guard covered stellar + dust — the
single configuration with no large positive scale seam.

Float64 is untouched: ``stop_gradient`` is a no-op forward, and the removed derivative
terms sum to zero. Measured bit-identical for all three models here, against the
pre-fix tree at full precision.

This module covers the **positive**-scale seams. The negative-scale one (the flux
projection, ~-58 dex) is
``test_float32_gradient_accuracy.py``, where the failure mode is underflow to zero
rather than a wrong finite value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.context import InferenceContext

pytestmark = pytest.mark.regression_bug

_BASE = dict(
    sfh={
        "type": "delayed",
        "all_params": FIXED,
        "log_total_mass": Uniform(9.0, 11.0),
        "tau_gyr": 1.0,
        "age_gyr": 5.0,
    },
    redshift=Fixed(0.1),
)

#: Each adds one large-positive-scale seam to the same stellar backbone, so a failure
#: names the seam rather than "float32 is bad".
_SEAM_MODELS = {
    # Dust IR re-emission normalizes its template to L_ir (~1e43), so the energy
    # balance peak-factors the absorbed bolometric integral: forward/energy_balance.py
    # and utils/sed_quantities.py.
    "dust_ir": dict(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        },
        dust_emission={"type": "dale2014", "all_params": FIXED},
    ),
    # The CIGALE-joint AGN renormalization forms trapz(L_disc) ~ L_bol (~1e44), peak
    # factored in components/agn/disc.py.
    "agn": dict(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Fixed(10.5),  # #2069: pinned to break flat direction
            "fracAGN": 0.1,
        },
    ),
    # Every seam at once — dust IR, Cue (Q_H ~1e56), AGN, radio, X-ray, shock — which
    # is what a science model actually looks like, and the case where a defect at one
    # seam could compound with, or cancel against, another. Measured 1.30e-03.
    #
    # Kept because coverage here has to be enumerated **by seam**, not by picking a
    # representative model: #1436 hid for as long as it did precisely because the one
    # model under test (stellar + dust) was the only one with no large positive scale
    # seam, so it measured 7.3e-05 and passed while dust IR and AGN were 30% wrong.
    "panchromatic": dict(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        },
        dust_emission={"type": "dale2014_cigale", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED},
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Fixed(10.5),  # #2069: pinned to break flat direction
            "fracAGN": 0.1,
        },
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        xray={"type": "simple"},
        shock={"frac": 0.1},
    ),
}


@pytest.fixture(scope="module")
def obs():
    # herschel_250 is load-bearing: without a far-IR band the dust IR component
    # contributes almost nothing to the likelihood and the defect is invisible.
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1", "herschel_250"])
    )


def _build(ssp, obs, groups):
    return SEDModel.build(ssp_data=ssp, observation=obs, **_BASE, **groups)


def _nlp_gradient(ssp, obs, groups, flux, noise, *, x64, dtype):
    with jax.enable_x64(x64):
        model = _build(ssp, obs, groups)
        ctx = InferenceContext.from_target(
            Fitter(model, jnp.asarray(flux, dtype=dtype), jnp.asarray(noise, dtype=dtype))
        )
        data_args = ctx.data_args
        names = sorted(ctx.initial_params(jax.random.PRNGKey(1)))
        point = {k: jnp.asarray(0.0, dtype=dtype) for k in names}
        grad = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, data_args))(point)
        return names, np.array([float(np.asarray(grad[k])) for k in names])


@pytest.mark.parametrize("seam", sorted(_SEAM_MODELS))
def test_float32_likelihood_gradient_survives_the_peak_factorization(ssp_bare, obs, seam):
    """The float32 gradient must track float64 once a big-scale seam is in the model.

    Compared against **float64 autodiff** rather than float32 finite differences,
    deliberately: at this point the small ``agn_log_lbol`` gradient (~-0.24) makes
    same-precision finite differences noisy at the several-percent level, which is
    larger than the tolerance worth enforcing. Float64 autodiff is the trustworthy
    reference here — it agrees with float64 finite differences to 2e-04, and the
    float32 *forward* model agrees with float64 to ~1e-6, so a disagreement of this
    size can only come from the reverse pass.
    """
    groups = _SEAM_MODELS[seam]

    # One mock, float64, so both precisions fit identical data.
    with jax.enable_x64(True):
        model = _build(ssp_bare, obs, groups)
        truth = {
            n: float(model.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in model.spec.free_params
        }
        mock = model.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
        flux = np.asarray(mock.flux_obs, dtype=np.float64)
        noise = np.asarray(mock.noise, dtype=np.float64)
    del model, mock

    names, g64 = _nlp_gradient(ssp_bare, obs, groups, flux, noise, x64=True, dtype=jnp.float64)
    _, g32 = _nlp_gradient(ssp_bare, obs, groups, flux, noise, x64=False, dtype=jnp.float32)

    assert np.all(np.isfinite(g32)), f"float32 gradient is non-finite for the {seam} seam: {g32}"
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    # 1e-2 sits two orders below the 0.30 defect and an order above the 1.1e-3 the fix
    # achieves, so it is neither brittle nor able to pass while the bug is present.
    assert rel.max() < 1e-2, (
        f"float32 likelihood gradient disagrees with float64 by {rel.max():.2e} for the "
        f"{seam} seam (names={names}, f32={g32}, f64={g64}). A ~0.3 relative error means "
        "a bolometric peak factorization lost its stop_gradient (#1436) — the peak became "
        "differentiable again and its two autodiff paths no longer cancel in float32."
    )
