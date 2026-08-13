"""
Component Orchestrator End-to-End
==================================

TODO[examples-sweep]: This script uses low-level component orchestration
(build_components, run_components) which is experimental Phase II-2.6 API
intended for infrastructure use, not recommended for user-facing examples.

Recommended replacement: Use SEDModel.build() with recipe-based composition.
See plot_joint_fit.py and plot_radio_xray.py for the public-API path.

The orchestrator layer may change; forward-compatible SED building goes
through the SEDModel.build() nested-dict grammar and recipes.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

try:
    from tengri.forward import build_components, chain_summary, run_components
    from tengri.protocols.component import PipelineState

    components = build_components(
        ssp_data=ssp,
        sfh_model="tsnorm",
        metallicity_model="ramp",
        nebular_backend="baked_in",
        # `multicolor_agn` is deprecated: it was always a composable chain in
        # disguise (disc=multicolor + torus=silva04). Spell it out.
        agn_model="composable",
        agn_disc_block="multicolor",
        agn_torus_block="silva04",
        agn_norm="conserving",
        dust_law_bc="calzetti",
        dust_emission_model="modified_blackbody",
        use_radio=True,
        use_xray=True,
        use_igm=True,
    )
    print("chain:", chain_summary(components))

    state0 = PipelineState(
        wave=ssp.ssp_wave,
        sed_observed=jnp.ones(len(ssp.ssp_wave)),
    )
    params = {
        "sfh_tsnorm_log_total_mass": jnp.asarray(1.0),
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        "met_logzsol_0": jnp.asarray(-1.0),
        "met_logzsol_final": jnp.asarray(0.0),
        "agn_log_lbol": jnp.asarray(11.0),
        "agn_lum_ratio": jnp.asarray(0.1),
        "dust_tau_bc": jnp.asarray(1.0),
        "dust_tau_diff": jnp.asarray(0.3),
        "dust_slope": jnp.asarray(-0.7),
        "dust_T": jnp.asarray(35.0),
        "dust_beta_ir": jnp.asarray(1.6),
        "dust_epsilon_mbb": jnp.asarray(1.0),
        "radio_q_ir": jnp.asarray(2.64),
        "radio_alpha_sf": jnp.asarray(0.8),
        "radio_loudness": jnp.asarray(0.0),
        "radio_alpha_agn": jnp.asarray(0.7),
        "radio_T_e": jnp.asarray(1e4),
        "radio_alpha_ff": jnp.asarray(-0.1),
        "xray_gamma_hmxb": jnp.asarray(2.0),
        "xray_gamma_lmxb": jnp.asarray(1.6),
        "xray_gamma_agn": jnp.asarray(1.8),
        "xray_E_cut": jnp.asarray(300.0),
        "xray_delta_alpha_ox": jnp.asarray(-1.4),
        # log10 N_H [cm^-2] photoelectric absorption (#768); 20 = unobscured.
        "xray_log_nh": jnp.asarray(20.0),
        # Lehmer+2016 XRB luminosity offsets [dex], read since #1706. 0.0 is the
        # declared default: the term scales by 10**offset, so this is no offset.
        # Every xray_* parameter is indexed directly by the component, so this
        # hand-rolled dict -- the point of the low-level orchestrator API -- must
        # supply each one; a missing key is a KeyError, not a silent default.
        "xray_det_hmxb": jnp.asarray(0.0),
        "xray_det_lmxb": jnp.asarray(0.0),
        "redshift": jnp.asarray(0.0),
    }

    pipeline = jax.jit(lambda p: run_components(components, state0, p))
    state = pipeline(params)

    _nu = 2.998e18 / ssp.ssp_wave
    _L_bol = float(jnp.abs(jnp.trapezoid(state.sed_intrinsic, _nu)))
    _logM = float(state.derived["log_mstar"])
    print(f"L_bol (stellar)        = {_L_bol:.3g} erg/s")
    print(f"log_mstar              = {_logM:.3f}  ({10**_logM:.3g} Msun)")
    print(f"L_ir (dust)            = {float(state.derived['L_ir']):.3g} erg/s")
    print(f"L_agn_bol              = {float(state.derived['L_agn_bol']):.3g} erg/s")

    wave = ssp.ssp_wave
    sed = state.sed_intrinsic
    mask = (wave > 100) & (wave < 1e7) & (sed > 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(wave[mask], wave[mask] * sed[mask], label="total")
    if "sed_dust_attenuated" in state.derived:
        ax.loglog(
            wave[mask],
            wave[mask] * jnp.maximum(state.derived["sed_dust_attenuated"][mask], 1e-30),
            ":",
            label="stellar (post-dust)",
        )
    if "sed_dust_ir" in state.derived:
        ax.loglog(
            wave[mask],
            wave[mask] * jnp.maximum(state.derived["sed_dust_ir"][mask], 1e-30),
            "--",
            label="dust IR",
        )
    if "sed_agn" in state.derived:
        ax.loglog(
            wave[mask],
            wave[mask] * jnp.maximum(state.derived["sed_agn"][mask], 1e-30),
            "-.",
            label="AGN",
        )
    ax.set_xlabel("rest-frame wavelength [Å]")
    ax.set_ylabel("λ × L_λ  [erg/s]")
    ax.legend()
    ax.set_ylim(1e30, 1e45)
    fig.tight_layout()
    plt.savefig("plot_orchestrator_demo.png", dpi=150, bbox_inches="tight")

except ImportError as e:
    print(f"Orchestrator API not available: {e}")
    print("Use SEDModel.build() instead (see examples/quickstart and examples/advanced).")
