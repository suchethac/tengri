# SPDX-License-Identifier: BSD-3-Clause
r"""The radio SED must be finite in pure float32 (#1206).

Radio emission is driven by two luminosities that overflow float32 (max
3.4e38):

* the SF synchrotron and free-free terms scale with ``L_ir`` (~1e43 erg/s),
* the AGN jet term with the radio-loudness B-band reference derived from
  ``L_agn_bol`` (~1e46 erg/s).

The linear ``L_ir`` / ``L_agn_bol`` arrive as ``inf`` in float32, and every
radio kernel then divides them by a large constant (``L_ir / (3.75e12 · 10^{q}``
for the FIR–radio correlation, ``L_ir / L_{IR,⊙}`` for free-free) — but
``inf / finite = inf``, so the whole radio SED is ``inf`` no matter how the
divide is ordered. The fix threads the float32-safe log companions
``log_L_ir`` / ``log_L_agn_bol`` into the kernels, which form the *representable*
~1e28 erg/s/Hz radio luminosity directly with ``pow10(log_L_ir − log10(const))``
so the ~1e43 value never materializes. The log path activates only in float32;
float64 keeps the exact linear divide (bit-identical).

The FIR–radio relation is non-linear in ``L_ir`` (Bell 2003 synchrotron
suppression, ``L ∝ L_ir^{1.3}``), so a reference-luminosity + linear-rescale
trick is *not* valid here — the log substitution forms the true quotient and is
exact to float32 precision. The AGN jet is exercised with a **power-law** disc
so the composable-AGN ``L_4400_intrinsic`` it reads is itself exact under the
float32 factoring (the multicolor disc's L_bol-dependent shape is a separate,
documented float32 limitation — see ``test_agn_lbol_shape_dependence``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}


def _model(ssp, radio):
    """A stellar + dust(+IR) + power-law-disc AGN + radio model. ``radio`` is the group."""
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": 0.5,
            "tau_bc": 0.0,
        },
        dust_emission={"type": "dale2014_cigale", "all_params": FIXED},
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "powerlaw", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Uniform(9.0, 13.0),
            "fracAGN": 0.1,
        },
        radio=radio,
        redshift=Fixed(0.1),
    )


def _sed_radio(ssp, radio, x64):
    with jax.enable_x64(x64):
        model = _model(ssp, radio)
        dtype = jnp.float64 if x64 else jnp.float32
        p = {k: jnp.asarray(v, dtype=dtype) for k, v in _TRUTH.items()}
        st = model.predict_state(p)
        return np.asarray(st.derived["sed_radio"]), st.derived["sed_radio"].dtype


@pytest.mark.parametrize(
    "radio",
    [
        pytest.param({"type": "condon92"}, id="sf+ff+agn_powerlaw"),
        pytest.param({"sf": {"type": "bell2003"}, "agn": {"type": "none"}}, id="sf+ff_only"),
    ],
)
def test_sed_radio_finite_and_matches_f64_in_float32(ssp_bare, radio):
    """``sed_radio`` is finite in pure float32 and matches float64 to float32 eps."""
    ref, _ = _sed_radio(ssp_bare, radio, x64=True)
    f32, dt = _sed_radio(ssp_bare, radio, x64=False)

    assert dt == jnp.float32, "precondition: the forward grid is genuinely float32"
    assert np.all(np.isfinite(f32)), (
        "sed_radio is non-finite in pure float32 — the SF/free-free L_ir divide "
        "or the AGN jet L_agn_bol fallback still materializes the out-of-range "
        "linear luminosity"
    )
    # Compare only where the radio SED is non-negligible: the long-wavelength
    # radio regime. The optical/IR end is ~1e-160 in float64 and flushes to zero
    # in float32 — a meaningless ratio there, not a radio-model error.
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < 1e-3, (
        f"sed_radio float32 vs float64 max rel = {rel.max():.2e} on the live radio "
        "band — larger than float32 rounding, so a kernel is not forming the "
        "radio luminosity from the log companion"
    )


def test_neg_log_posterior_gradient_finite_with_radio_in_float32(ssp_bare):
    """A fit through a radio model must have a finite float32 gradient.

    The radio SED feeds the photometry likelihood (WISE bands sit on the radio
    Rayleigh–Jeans tail only faintly, but the projection sums the full SED), so a
    reverse-pass ``inf`` in a radio kernel would poison ``grad(nlp)`` even with a
    finite forward. Uses the SF-only radio group to isolate the ``L_ir`` path.
    """
    from tengri import Fitter
    from tengri.inference.context import InferenceContext

    radio = {"sf": {"type": "bell2003"}, "agn": {"type": "none"}}
    with jax.enable_x64(True):
        model = _model(ssp_bare, radio)
        mock = model.mock(_TRUTH, snr=30.0, key=jax.random.PRNGKey(0))
        flux = np.asarray(mock.flux_obs, dtype=np.float64)
        noise = np.asarray(mock.noise, dtype=np.float64)

    with jax.enable_x64(False):
        model = _model(ssp_bare, radio)
        ctx = InferenceContext.from_target(Fitter(model, jnp.asarray(flux), jnp.asarray(noise)))
        da = ctx.data_args
        p = ctx.initial_params(jax.random.PRNGKey(3))
        g = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, da))(p)
        leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(g)]
    assert leaves and leaves[0].dtype == jnp.float32, "precondition: genuinely float32"
    assert all(np.all(np.isfinite(v)) for v in leaves), (
        "grad(nlp) is non-finite in pure float32 with radio — a radio kernel's "
        "reverse pass overflows"
    )
