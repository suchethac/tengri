"""
Component orchestrator end-to-end
==================================

Build a full multiwavelength SED — stellar + nebular + AGN + dust +
radio + X-ray + IGM — without going through :class:`tengri.SEDModel`,
using :func:`tengri.forward.build_components` and
:func:`tengri.forward.run_components` (Phase II-2.6 public API).

Every physics block is **swappable** by changing a config string:

- Dust attenuation law: ``"power_law"``, ``"calzetti"``, ``"smc"``,
  ``"cardelli"``, …
- IR emission template: ``"modified_blackbody"``, ``"casey2012"``,
  ``"dale2014"``, ``"draine_li2007"``, ``"draine_li2014"``.
- AGN model: ``"simple"``, ``"standard"``, ``"kubota_done_full"``, …
- Nebular backend: ``"baked_in"``, ``"cloudy_grid"``, ``"cue"``, ``"shock"``.

The orchestrator chain JIT-compiles end-to-end with bit-exact match
to the eager path (rtol=1e-12). Cross-component data flows through
``state.derived``: dust publishes ``L_ir`` for radio + X-ray, AGN
publishes ``L_agn_bol`` for X-ray, stellar publishes ``log_mstar``
+ ``lnu_age`` + 9 more keys.
"""

# %%
# Build the pipeline
# ------------------

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.core.component import PipelineState
from tengri.forward import build_components, chain_summary, run_components


def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

components = build_components(
    ssp_data=ssp,
    sfh_model="tsnorm",
    metallicity_model="ramp",
    nebular_backend="baked_in",
    agn_model="simple",
    dust_law_bc="calzetti",
    dust_emission_model="modified_blackbody",
    use_radio=True,
    use_xray=True,
    use_igm=True,
)
print("chain:", chain_summary(components))

# %%
# Run a forward pass
# ------------------

state0 = PipelineState(
    wave=ssp.ssp_wave,
    sed_observed=jnp.ones(len(ssp.ssp_wave)),  # IGM transmits this
)
params = {
    # tsnorm SFH (peak 10 Msun/yr at 2 Gyr lookback, 1 Gyr width)
    "sfh_tsnorm_log_peak_sfr": jnp.asarray(1.0),
    "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
    "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
    "sfh_tsnorm_skew": jnp.asarray(0.0),
    "sfh_tsnorm_trunc": jnp.asarray(3.0),
    # Ramp metallicity: -1 dex (oldest) → solar (today)
    "met_logzsol_0": jnp.asarray(-1.0),
    "met_logzsol_final": jnp.asarray(0.0),
    # AGN (10^11 Lsun bolometric, 10% scaling)
    "agn_log_lbol": jnp.asarray(11.0),
    "agn_frac": jnp.asarray(0.1),
    # Dust two-component
    "dust_tau_bc": jnp.asarray(1.0),
    "dust_tau_diff": jnp.asarray(0.3),
    "dust_slope": jnp.asarray(-0.7),
    "dust_T": jnp.asarray(35.0),
    "dust_beta_ir": jnp.asarray(1.6),
    # Multiwavelength defaults
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
    "xray_alpha_ox": jnp.asarray(-1.4),
    "redshift": jnp.asarray(0.0),
}

# JIT-compiled
pipeline = jax.jit(lambda p: run_components(components, state0, p))
state = pipeline(params)

# %%
# Inspect the cross-component publications
# ----------------------------------------

_nu = 2.998e18 / ssp.ssp_wave
_L_bol = float(jnp.abs(jnp.trapezoid(state.sed_intrinsic, _nu)))
_logM = float(state.derived["log_mstar"])
print(f"L_bol (stellar)        = {_L_bol:.3g} erg/s")
print(f"log_mstar              = {_logM:.3f}  ({10**_logM:.3g} Msun)")
print(f"L_ir (dust)            = {float(state.derived['L_ir']):.3g} erg/s")
print(f"L_agn_bol              = {float(state.derived['L_agn_bol']):.3g} erg/s")
if 'sed_radio' in state.derived:
    print(f"L_radio peak           = {float(state.derived['sed_radio'].max()):.3g} erg/s/Hz")
if 'sed_xray' in state.derived:
    print(f"L_xray peak            = {float(state.derived['sed_xray'].max()):.3g} erg/s/Hz")
if 'sfr_10myr' in state.derived:
    print(f"sfr_10myr              = {float(state.derived['sfr_10myr']):.3f} Msun/yr")
if 'nion' in state.derived:
    print(f"nion                   = {float(state.derived['nion']):.3g} photons/s")

# %%
# Plot the SED
# ------------

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
ax.set_title("Component-orchestrator end-to-end")
ax.legend()
ax.set_ylim(1e30, 1e45)
fig.tight_layout()
plt.savefig("plot_orchestrator_demo.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Swap the dust law and re-run — composability in action
# ------------------------------------------------------

components_smc = build_components(
    ssp_data=ssp,
    sfh_model="tsnorm",
    metallicity_model="ramp",
    nebular_backend="baked_in",
    agn_model="simple",
    dust_law_bc="smc",  # ← only change
    dust_emission_model="modified_blackbody",
    use_radio=True,
    use_xray=True,
    use_igm=True,
)
state_smc = jax.jit(lambda p: run_components(components_smc, state0, p))(params)
print(f"\ncalzetti L_ir = {float(state.derived['L_ir']):.3g}")
print(f"smc      L_ir = {float(state_smc.derived['L_ir']):.3g}")
print("(swapping the law changes the absorbed-luminosity integral; energy balance still exact)")
