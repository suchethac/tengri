# SPDX-License-Identifier: BSD-3-Clause
r"""The multicolor accretion disc must be exact in pure float32 (#1206).

The Shakura-Sunyaev ``multicolor_disc`` forms several cgs intermediates that
overflow float32 (max 3.4e38) at a realistic AGN luminosity even though every
*result* is representable:

* ``L_bol = 10^log_lbol · L_sun`` ~1e44 erg/s, ``L_Edd`` ~1e46, the ``t_in**4``
  accretion numerator ~1e58 — yet ``mdot`` ~1e24 g/s and ``t_in`` ~1e5 K are in
  range;
* the EUV-tail and renormalization bolometric integrals ``∫L_ν dν`` ~1e43 erg/s
  — yet the ratio that sets the tail amplitude / renorm scale is O(1e-4)–O(1).

The float32 path forms each in log10 and materializes only the representable
result (``pow10``), peak-factoring the energy integrals. This is float32-gated:
float64 keeps the exact linear arithmetic (bit-identical).

Because the disc *shape* (temperature) depends on L_bol, the AGN component
additionally evaluates the disc SHAPE at the true L_bol while normalizing its
MAGNITUDE to a low reference (``agn_log_lbol_shape``), so the whole
composable-AGN runner — which works in the overflowing ~1e40 ``L_lambda`` space —
stays in range and the true magnitude is re-applied downstream. The net effect:
the multicolor disc, its intrinsic 2500/4400 Å luminosities, and the full AGN SED
are **exact** in pure float32, not merely finite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(200.0, 1.0e7, 3000)


def _disc(dtype):
    from tengri.components.agn.disc import multicolor_disc

    w = jnp.asarray(_WAVE, dtype=dtype)
    return np.asarray(
        multicolor_disc(
            w,
            agn_log_lbol=jnp.asarray(11.0, dtype=dtype),
            agn_frac=jnp.asarray(1.0, dtype=dtype),
            agn_log_mbh=jnp.asarray(8.0, dtype=dtype),
            agn_a_spin=jnp.asarray(0.0, dtype=dtype),
        )
    )


def test_multicolor_disc_finite_and_matches_f64_in_float32():
    """``multicolor_disc`` L_ν is finite and float64-accurate in pure float32."""
    with jax.enable_x64(True):
        ref = _disc(jnp.float64)
    with jax.enable_x64(False):
        f32 = _disc(jnp.float32)
    assert f32.dtype == jnp.float32
    assert np.all(np.isfinite(f32)), (
        "multicolor_disc is non-finite in pure float32 — a cgs intermediate "
        "(L_bol ~1e44, L_Edd ~1e46, t_in**4 ~1e58, or the ~1e43 bolometric "
        "integral) overflowed instead of being formed in log space"
    )
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < 1e-3, (
        f"multicolor_disc float32 vs float64 max rel = {rel.max():.2e} — larger "
        "than float32 rounding, so the log-space temperature or renorm diverges"
    )


def _kubota(dtype):
    from tengri.components.agn.disc import kubota_done_disc

    w = jnp.asarray(_WAVE, dtype=dtype)
    return np.asarray(
        kubota_done_disc(
            w,
            agn_log_lbol=jnp.asarray(11.0, dtype=dtype),
            agn_frac=jnp.asarray(1.0, dtype=dtype),
            agn_log_mbh=jnp.asarray(8.0, dtype=dtype),
            agn_a_spin=jnp.asarray(0.0, dtype=dtype),
        )
    )


def test_kubota_done_disc_finite_and_matches_f64_in_float32():
    """The Kubota & Done three-zone disc is finite and float64-accurate in float32.

    Beyond the L_bol / L_Edd / t_in overflows shared with the multicolor disc, the
    three-zone model adds several ~1e42–1e44 erg/s intermediates that overflow
    float32: the bisection's ``l0``, the seed-photon and zone bolometric
    integrals, and the warm-zone ring bolometric ``p_plain * ring_area`` (~1e42,
    though the ring L_nu ~1e27 is representable). The float32 path works these in
    L_sun units / reordered so no out-of-range intermediate forms.
    """
    with jax.enable_x64(True):
        ref = _kubota(jnp.float64)
    with jax.enable_x64(False):
        f32 = _kubota(jnp.float32)
    assert f32.dtype == jnp.float32
    assert np.all(np.isfinite(f32)), (
        "kubota_done_disc is non-finite in pure float32 — a three-zone cgs "
        "intermediate (l0, seed/zone bolometric integral, or the warm-ring "
        "p_plain*area ~1e42) overflowed instead of being formed in L_sun units"
    )
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < 1e-3, (
        f"kubota_done_disc float32 vs float64 max rel = {rel.max():.2e} — larger "
        "than float32 rounding, so a zone's L_sun bookkeeping diverges"
    )


def _adaf(dtype):
    from tengri.components.agn.adaf import adaf_spectrum

    w = jnp.asarray(_WAVE, dtype=dtype)
    return np.asarray(
        adaf_spectrum(
            w,
            agn_log_lbol=jnp.asarray(11.0, dtype=dtype),
            agn_frac=jnp.asarray(1.0, dtype=dtype),
            agn_log_mbh=jnp.asarray(8.0, dtype=dtype),
        )
    )


def test_adaf_spectrum_finite_and_matches_f64_in_float32():
    """The Mahadevan ADAF spectrum is finite and float64-accurate in float32.

    ADAF works in dimensionless units (``m``, ``mdot``), so its only float32
    hazards are the ~1e44 erg/s ``l_bol_erg`` and the ~3e46 ``coeff`` in the mdot
    inversion, and the ~1e43 erg/s spectral integral in the renormalization — all
    worked in L_sun so only the ratios (mdot ~1e-2, the normalized shape) form.
    """
    with jax.enable_x64(True):
        ref = _adaf(jnp.float64)
    with jax.enable_x64(False):
        f32 = _adaf(jnp.float32)
    assert f32.dtype == jnp.float32
    assert np.all(np.isfinite(f32)), (
        "adaf_spectrum is non-finite in pure float32 — the mdot-inversion coeff "
        "(~3e46) / l_bol_erg (~1e44) or the ~1e43 renorm integral overflowed"
    )
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < 1e-3, (
        f"adaf_spectrum float32 vs float64 max rel = {rel.max():.2e} — larger than "
        "float32 rounding, so the L_sun mdot/renorm bookkeeping diverges"
    )


def _composable_intrinsics(ssp, dtype):
    """Return ``(L_4400, L_2500, sed_agn_peak)`` for a multicolor-disc AGN."""
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
    model = SEDModel.build(
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
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Uniform(9.0, 13.0),
            "fracAGN": 0.1,
        },
        redshift=Fixed(0.1),
    )
    p = {
        k: jnp.asarray(v, dtype=dtype)
        for k, v in {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}.items()
    }
    d = model.predict_state(p).derived
    return (
        float(np.asarray(d["L_4400_intrinsic"])),
        float(np.asarray(d["L_2500_intrinsic"])),
        float(np.abs(np.asarray(d["sed_agn"])).max()),
    )


def test_composable_multicolor_disc_agn_exact_in_float32(ssp_bare):
    """The composable multicolor-disc AGN matches float64 in pure float32.

    Before the shape/normalization split, the float32 factoring evaluated a cold
    reference disc: ``L_4400_intrinsic`` came out ~0 (underflow of the ~1e-171
    factored value), corrupting the radio-loudness reference and X-ray alpha_ox.
    Now the SHAPE is taken at the true L_bol, so every intrinsic luminosity and
    the AGN SED agree across precisions.
    """
    with jax.enable_x64(True):
        l4400_64, l2500_64, sed_64 = _composable_intrinsics(ssp_bare, jnp.float64)
    with jax.enable_x64(False):
        l4400_32, l2500_32, sed_32 = _composable_intrinsics(ssp_bare, jnp.float32)

    assert l4400_64 > 1.0e10, "precondition: float64 L_4400 is physical"
    for name, v64, v32 in [
        ("L_4400_intrinsic", l4400_64, l4400_32),
        ("L_2500_intrinsic", l2500_64, l2500_32),
        ("sed_agn_peak", sed_64, sed_32),
    ]:
        rel = abs(v32 - v64) / abs(v64)
        assert rel < 1e-3, (
            f"{name}: float32 {v32:.4e} vs float64 {v64:.4e} (rel {rel:.2e}) — the "
            "multicolor disc's true-L_bol shape is not reaching the float32 path"
        )


def test_multicolor_disc_agn_fit_gradient_finite_in_float32(ssp_bare):
    """A fit through a multicolor-disc AGN has a finite float32 gradient.

    The headline of the disc-internals fix: with a shape-changing physical disc
    in the model, ``grad(neg_log_posterior)`` is finite in pure float32, so
    gradient-based inference (MAP/NUTS) runs end-to-end. A reverse-pass overflow
    in the disc temperature or renorm would poison it even with a finite forward.
    """
    from tengri import Fitter
    from tengri.inference.context import InferenceContext

    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w3", "wise_w4"]))

    def _model():
        return SEDModel.build(
            ssp_data=ssp_bare,
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
            dust_emission={"type": "dale2014", "all_params": FIXED},
            agn={
                "type": "composable",
                "all_params": FIXED,
                "disc": {"type": "multicolor", "all_params": FIXED},
                "torus": {"type": "skirtor", "all_params": FIXED},
                "norm": "cigale_joint",
                "log_lbol": Uniform(9.0, 12.0),
                "fracAGN": 0.1,
            },
            redshift=Fixed(0.1),
        )

    truth = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5, "agn_log_lbol": 11.0}
    with jax.enable_x64(True):
        mock = _model().mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
        flux = np.asarray(mock.flux_obs, dtype=np.float64)
        noise = np.asarray(mock.noise, dtype=np.float64)

    with jax.enable_x64(False):
        ctx = InferenceContext.from_target(Fitter(_model(), jnp.asarray(flux), jnp.asarray(noise)))
        da = ctx.data_args
        for i in range(3):
            p = ctx.initial_params(jax.random.fold_in(jax.random.PRNGKey(0), i))
            g = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, da))(p)
            leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(g)]
            assert leaves and leaves[0].dtype == jnp.float32, "precondition: genuinely float32"
            assert all(np.all(np.isfinite(v)) for v in leaves), (
                f"grad(nlp) non-finite at draw {i} in pure float32 — a multicolor-disc "
                "reverse-pass overflow"
            )
