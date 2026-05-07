# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SED anatomy: kitchen-sink panchromatic decomposition
#
# **What you'll learn:**
# - How every emission channel that ``tengri`` models — stellar, nebular,
#   dust attenuation, dust IR re-emission, AGN (disc + torus), X-ray, radio —
#   contributes to a galaxy's panchromatic SED from 0.1 Å (hard X-ray) to
#   3×10¹¹ Å (radio).
# - That *isolating* each component is one ``Parameters`` flag away — the
#   same model construction grammar used to fit data.
# - That the orchestrator's published wavelength grid is the canonical
#   substrate: ``state.sed_intrinsic`` is the running total at chain end,
#   ``state.derived["sed_<name>"]`` keys are the per-component bookkeeping.
#
# **Prerequisites:** [``00_quickstart.py``](00_quickstart.py),
# [``01_why_jax.py``](01_why_jax.py).
# **Next:** [``03_discovering_the_menu.py``](03_discovering_the_menu.py)
# to learn the ``list_*().filter().names()`` discovery API.
# **Runtime budget:** ≤ 60 s on CPU.

# %%
import os
import sys
import warnings

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
except NameError:
    _nb_dir = os.getcwd()
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))

_src = os.path.join(_repo_root, "src")
if os.path.isdir(os.path.join(_src, "tengri")):
    sys.path.insert(0, _src)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _nb_dir)

# nbconvert kernel cwd is notebooks/, switch to repo root so data/ resolves.
if os.path.isdir(os.path.join(_repo_root, "data")):
    os.chdir(_repo_root)

import jax
import jax.numpy as jnp
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tengri")

from _plot_style import COLORS, setup_style

setup_style()

import tengri as tg
from tengri import Fixed, Parameters, SEDModel, load_ssp_data

print(f"tengri {tg.__version__}")

FIGDIR = os.path.join("notebooks", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## Part 1: Build the kitchen-sink model
#
# Every component flag ``Parameters`` accepts is set here. With
# ``predict_via_orchestrator``, each component's contribution is published
# as a key in ``state.derived``, so we can isolate them at plot time
# without re-running the model.
#
# We need a **bare-stellar SSP** because Cue's nebular emulator infers
# the ionizing field from the SSP spectrum directly — wNE SSPs (with
# baked-in nebular) leave that field at log10(Q_H) ≈ 0 and Cue would
# refuse to construct (see ``CueWNESSPError`` and notebook 08).

# %%
ssp_data = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
print(
    f"SSP grid: {ssp_data.ssp_flux.shape[0]} Z × "
    f"{ssp_data.ssp_flux.shape[1]} ages × {ssp_data.ssp_flux.shape[-1]} λ"
)

# Star-forming, moderately dusty, modest AGN, z = 0.5
spec_full = Parameters(
    # Stellar population — truncated normal SFH peaking 0.3 Gyr ago,
    # 1 Gyr wide. Choosing a star-forming SFH here so Cue's nebular
    # continuum + line predictions are non-trivially populated; a
    # passive SFH (e.g. ``dexp`` with peak in the distant past)
    # produces ~zero recent Q_H and Cue floors at 1e-100 internally.
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(0.7),  # 5 Msun/yr peak
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.3),  # peaked 300 Myr ago
    sfh_tsnorm_width_gyr=Fixed(1.0),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(13.8),
    met_logzsol=Fixed(-0.3),
    # Two-component dust attenuation (birth-cloud + diffuse)
    dust_tau_bc=Fixed(0.8),
    dust_tau_diff=Fixed(0.4),
    dust_slope=Fixed(-0.7),
    # Dust IR re-emission via energy balance (Dale et al. 2014)
    dust_emission="dale2014",
    dust_T=Fixed(35.0),
    dust_qpah=Fixed(2.5),
    # Nebular continuum + lines via Cue (neural CLOUDY emulator)
    nebular_cue=True,
    neb_logU=Fixed(-2.5),
    neb_logZ_gas=Fixed(-0.3),
    # AGN — qsogen disc + torus.  ``agn_log_lbol`` is log10(L_bol/L_sun)
    # at the API surface (CLAUDE.md), not log10(erg/s).  10^11 L_sun ≈
    # 4×10^44 erg/s — a moderate Seyfert-class AGN.
    agn_model="qsogen",
    agn_log_lbol=Fixed(11.0),
    # ``agn_frac`` defaults to 0 in the registry (a multiplicative on/off
    # switch on top of ``agn_log_lbol``). Set to 1 so the AGN we just
    # configured actually contributes.
    agn_frac=Fixed(1.0),
    agn_torus_frac=Fixed(0.4),
    agn_cos_inc=Fixed(0.7),
    # X-ray (host XRBs + AGN power-law)
    xray=True,
    # Radio (FIR-radio correlation + AGN)
    radio=True,
    radio_q_ir=Fixed(2.64),
    # IGM attenuation (Madau 1995)
    apply_igm=True,
    redshift=Fixed(0.5),
)

t0 = __import__("time").perf_counter()
model_full = SEDModel(spec_full, ssp_data, observation=None)
t_build = __import__("time").perf_counter() - t0
print(f"  ⏱  SEDModel construction          {t_build:.2f} s")

# Single forward pass — everything below reads from this state.
params = spec_full.sample(jax.random.PRNGKey(0))
t0 = __import__("time").perf_counter()
state = model_full.predict_via_orchestrator(params)
t_pred = __import__("time").perf_counter() - t0
print(f"  ⏱  predict_via_orchestrator cold  {t_pred:.2f} s")

# Second pass with τ → 0 to recover the un-attenuated stellar+nebular
# reference. Same model, different params — no special-casing needed.
params_no_dust = {**params, "dust_tau_bc": jnp.array(0.0), "dust_tau_diff": jnp.array(0.0)}
state_no_dust = model_full.predict_via_orchestrator(params_no_dust)

# %% [markdown]
# ## Part 2: Inspect what's in the state
#
# Each component's published quantities live in ``state.derived``. The
# top-level ``state.sed_intrinsic`` is the running sum threaded through
# the chain — by the end of the orchestrator it equals the panchromatic
# total.

# %%
print(f"\nstate.wave shape: {state.wave.shape}")
print(f"state.wave range: {float(state.wave.min()):.2g} to {float(state.wave.max()):.2g} Å")
print("\nPer-component peaks [erg/s/Hz]:")
for k in (
    "sed_dust_attenuated",  # stellar + nebular, after dust attenuation
    "sed_nebular",  # nebular continuum + lines
    "sed_dust_ir",  # thermal IR re-emission
    "sed_agn",  # AGN disc + torus
    "sed_xray",  # X-ray (XRBs + AGN)
    "sed_radio",  # radio synchrotron + AGN
):
    v = state.derived.get(k)
    if v is not None and hasattr(v, "max"):
        peak = float(np.asarray(v).max())
        print(f"  {k:<30s} {peak:>12.3e}")

# %% [markdown]
# ## Part 3: The kitchen-sink figure
#
# Plot every component on a single panchromatic ν L_ν panel. Wavelength
# bands are shaded so the reader can see at a glance which channel
# dominates each regime. The y-axis is ν L_ν in solar luminosities, the
# physically natural unit for "how much energy is leaving the galaxy at
# this wavelength."

# %%
_C_AA_S = 2.99792458e18  # speed of light [Å/s]
_LSUN_ERG = 3.828e33  # IAU 2015 solar luminosity [erg/s]

wave_aa = np.asarray(state.wave)
wave_um = wave_aa * 1e-4
nu_hz = _C_AA_S / wave_aa


def to_nulnu_lsun(lnu_erg):
    """L_nu [erg/s/Hz] → ν L_ν [L_sun]."""
    arr = np.asarray(lnu_erg)
    return arr * nu_hz / _LSUN_ERG


# Components: (key, label, color, linestyle, linewidth, z-order)
_d = state.derived
_d_nodust = state_no_dust.derived
components = [
    (
        # Pre-attenuation stellar+nebular reference (dashed)
        to_nulnu_lsun(_d_nodust["sed_dust_attenuated"]),
        "Stars + nebular (no dust)",
        "#999999", "--", 1.4, 2,
    ),
    (
        to_nulnu_lsun(_d["sed_dust_attenuated"]),
        "Stars + nebular (attenuated)",
        "#3a86ff", "-", 2.0, 4,
    ),
    (
        to_nulnu_lsun(_d.get("sed_nebular", np.zeros_like(wave_aa))),
        "Nebular only",
        "#06d6a0", "-", 1.6, 3,
    ),
    (
        to_nulnu_lsun(_d.get("sed_dust_ir", np.zeros_like(wave_aa))),
        "Dust IR re-emission",
        "#f77f00", "-", 2.0, 4,
    ),
    (
        to_nulnu_lsun(_d.get("sed_agn", np.zeros_like(wave_aa))),
        "AGN (disc + torus)",
        "#9d4edd", "-", 2.0, 4,
    ),
    (
        to_nulnu_lsun(_d.get("sed_xray", np.zeros_like(wave_aa))),
        "X-ray (XRBs + AGN)",
        "#e63946", "-", 1.8, 4,
    ),
    (
        to_nulnu_lsun(_d.get("sed_radio", np.zeros_like(wave_aa))),
        "Radio (synchrotron)",
        "#588157", "-", 1.8, 4,
    ),
    (
        to_nulnu_lsun(state.sed_intrinsic),
        "Total",
        "#1a1a1a", "-", 2.5, 5,
    ),
]

# Y-range: pick from total, three decades down from peak.
total = to_nulnu_lsun(state.sed_intrinsic)
y_peak = float(np.nanmax(total[total > 0]))
y_top = y_peak * 5
y_bot = y_peak * 1e-5

fig, ax = plt.subplots(1, 1, figsize=(13, 7))

# Wavelength bands (rest-frame interpretation; with z=0.5 these are blueshifted).
bands = [
    (1e-5, 1e-3, "X-ray", "#e8d0f5"),
    (1e-3, 0.0091, "EUV", "#f5e6d3"),
    (0.0091, 0.4, "UV", "#fff0a0"),
    (0.4, 0.7, "Optical", "#e0f5e0"),
    (0.7, 30.0, "NIR / MIR", "#fde8d0"),
    (30.0, 1e3, "FIR / sub-mm", "#fcc890"),
    (1e3, 1e7, "Radio", "#dce8f5"),
]
for lo, hi, lbl, col in bands:
    ax.axvspan(lo, hi, color=col, alpha=0.35, zorder=0)
    ax.text(
        np.sqrt(lo * hi),
        y_top * 0.7,
        lbl,
        ha="center", va="top", fontsize=10,
        color="#444444", style="italic", zorder=1,
    )

# Plot each component, masking values that fall below the panel floor
# (otherwise log autoscale would squash everything — see notebook rules §2).
floor = y_bot * 0.5
for y, lbl, col, ls, lw, z in components:
    mask = y > floor
    if mask.any():
        ax.plot(wave_um[mask], y[mask], ls=ls, lw=lw, color=col, label=lbl, zorder=z)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1e7)
ax.set_ylim(y_bot, y_top)
ax.set_xlabel(r"Rest wavelength [$\mu$m]", fontsize=12)
ax.set_ylabel(r"$\nu L_{\nu}$ [$L_\odot$]", fontsize=12)
ax.set_title(
    f"Panchromatic SED at z={float(params['redshift']):.2f}: every component on one panel",
    fontsize=12,
)
ax.legend(loc="lower center", frameon=False, fontsize=9, ncol=4)
ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_sed_anatomy_hero.png"), dpi=200, bbox_inches="tight")
plt.close()
print("✓ Saved 02_sed_anatomy_hero.png")

# %% [markdown]
# ## Part 4: Component isolation — same panel, one component at a time
#
# Useful for explaining what each component contributes in isolation.
# Same wavelength grid as above, identical normalization. The bottom
# row shows the IGM transmission separately because it multiplies
# (rather than adds to) the SED.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
panels = [
    ("Stars + nebular", to_nulnu_lsun(_d["sed_dust_attenuated"]), "#3a86ff"),
    ("Nebular continuum + lines", to_nulnu_lsun(_d.get("sed_nebular", np.zeros_like(wave_aa))), "#06d6a0"),
    ("Dust IR re-emission", to_nulnu_lsun(_d.get("sed_dust_ir", np.zeros_like(wave_aa))), "#f77f00"),
    ("AGN (disc + torus)", to_nulnu_lsun(_d.get("sed_agn", np.zeros_like(wave_aa))), "#9d4edd"),
    ("X-ray (XRBs + AGN)", to_nulnu_lsun(_d.get("sed_xray", np.zeros_like(wave_aa))), "#e63946"),
    ("Radio (synchrotron)", to_nulnu_lsun(_d.get("sed_radio", np.zeros_like(wave_aa))), "#588157"),
]
for ax_p, (label, y, col) in zip(axes.flat, panels):
    mask = y > floor
    if mask.any():
        ax_p.plot(wave_um[mask], y[mask], color=col, lw=2.0)
        # Faint reference: total
        ax_p.plot(wave_um, total, color="#cccccc", lw=0.8, ls="-", zorder=0)
    ax_p.set_xscale("log")
    ax_p.set_yscale("log")
    ax_p.set_xlim(1e-5, 1e7)
    ax_p.set_ylim(y_bot, y_top)
    ax_p.set_title(label, fontsize=11)
    ax_p.grid(True, alpha=0.25, which="both")
for ax_p in axes[-1, :]:
    ax_p.set_xlabel(r"Rest $\lambda$ [$\mu$m]")
for ax_p in axes[:, 0]:
    ax_p.set_ylabel(r"$\nu L_{\nu}$ [$L_\odot$]")

fig.suptitle("Each component on its own (faint grey = full panchromatic total)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_component_isolation.png"), dpi=200, bbox_inches="tight")
plt.close()
print("✓ Saved 02_component_isolation.png")

# %% [markdown]
# ## Part 5: IGM transmission and dust attenuation as multiplicative factors
#
# IGM (Lyman-alpha forest) at z=0.5 is mild — most of the action lives at
# z ≳ 2. Re-running with z=3 makes it dramatic. Dust attenuation
# transmission is recovered by ``sed_dust_attenuated /
# sed_dust_attenuated_no_dust``.

# %%
# IGM at multiple redshifts
fig, (ax_igm, ax_dust) = plt.subplots(1, 2, figsize=(14, 5))

for z, color in [(0.5, "#3a86ff"), (1.0, "#06d6a0"), (2.0, "#f77f00"), (3.0, "#e63946")]:
    p = {**params, "redshift": jnp.array(z)}
    s = model_full.predict_via_orchestrator(p)
    igm_t = np.asarray(s.derived["igm_transmission"])
    # Plot in observed-frame wavelength so the Lyman alpha forest is visible
    wave_obs = np.asarray(s.wave) * (1.0 + z) * 1e-4  # μm
    ax_igm.plot(wave_obs, igm_t, color=color, lw=1.8, label=f"z = {z:.1f}")

ax_igm.set_xscale("log")
ax_igm.set_xlim(0.05, 0.4)
ax_igm.set_ylim(-0.05, 1.15)
ax_igm.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
ax_igm.set_xlabel(r"Observed $\lambda$ [$\mu$m]")
ax_igm.set_ylabel("IGM transmission")
ax_igm.set_title("IGM (Madau 1995) — Lyman-α forest", fontsize=11)
ax_igm.legend(loc="lower right", frameon=False)
ax_igm.grid(True, alpha=0.25)

# Dust attenuation curve (transmission vs wavelength)
intrinsic = np.asarray(_d_nodust["sed_dust_attenuated"])
attenuated = np.asarray(_d["sed_dust_attenuated"])
mask_optical = (wave_um >= 0.1) & (wave_um <= 5.0) & (intrinsic > 0)
trans = attenuated[mask_optical] / intrinsic[mask_optical]
ax_dust.plot(wave_um[mask_optical], trans, color="#9d4edd", lw=2.0)
ax_dust.set_xscale("log")
ax_dust.set_xlim(0.1, 5.0)
ax_dust.set_ylim(-0.05, 1.05)
ax_dust.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
ax_dust.set_xlabel(r"Rest $\lambda$ [$\mu$m]")
ax_dust.set_ylabel(r"$L_\nu^{\rm attenuated} / L_\nu^{\rm intrinsic}$")
ax_dust.set_title(r"Dust attenuation transmission ($\tau_{\rm bc}=0.8,\ \tau_{\rm diff}=0.4$)", fontsize=11)
ax_dust.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_redshift_igm.png"), dpi=200, bbox_inches="tight")
plt.close()
print("✓ Saved 02_redshift_igm.png")

# %% [markdown]
# ## Part 6: Layer-by-layer build
#
# Same model, parameters peeled back one at a time. Lets the reader see
# what each component *adds* (or removes) from the cumulative SED.

# %%
fig, ax = plt.subplots(1, 1, figsize=(13, 6))

stages = [
    # Strip dust IR + AGN + radio + xray to leave just the stars
    (
        {"dust_tau_bc": 0.0, "dust_tau_diff": 0.0,
         "agn_log_lbol": -10.0, "dust_emission": None},
        "(1) Stars + nebular only",
        "#3a86ff",
    ),
    # Add dust attenuation
    (
        {"agn_log_lbol": -10.0, "dust_emission": None},
        "(2) + Dust attenuation",
        "#06d6a0",
    ),
    # Add dust IR re-emission
    (
        {"agn_log_lbol": -10.0},
        "(3) + Dust IR re-emission",
        "#f77f00",
    ),
    # Full kitchen sink
    (
        {},
        "(4) + AGN + X-ray + radio (full)",
        "#1a1a1a",
    ),
]

# Stage 1 needs a model rebuild because dust_emission=None changes structure.
# For simplicity, run the parameter-only stages from the existing model.
# (Stages 2-4 only flip parameters; stage 1 is computed separately.)
spec_no_emit = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(0.7),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.3),
    sfh_tsnorm_width_gyr=Fixed(1.0),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(13.8),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.0), dust_tau_diff=Fixed(0.0), dust_slope=Fixed(-0.7),
    nebular_cue=True,
    neb_logU=Fixed(-2.5), neb_logZ_gas=Fixed(-0.3),
    redshift=Fixed(0.5),
)
model_stars = SEDModel(spec_no_emit, ssp_data, observation=None)
state_stars = model_stars.predict_via_orchestrator(spec_no_emit.sample(jax.random.PRNGKey(0)))

# Stars-only chain uses a different (shorter) wave grid; convert here
# rather than via the kitchen-sink-grid helper.
_wave_stars = np.asarray(state_stars.wave)
_nu_stars = _C_AA_S / _wave_stars
_nulnu_stars = np.asarray(state_stars.sed_intrinsic) * _nu_stars / _LSUN_ERG
ax.plot(
    _wave_stars * 1e-4,
    _nulnu_stars,
    color="#3a86ff", lw=2.0, label="(1) Stars + nebular only", zorder=4,
)

# Stages 2-4: parameter overrides on the kitchen-sink model.
for params_override, label, color in [
    ({"dust_tau_bc": 0.8, "dust_tau_diff": 0.4,
      "agn_frac": 0.0}, "(2) + Dust attenuation only", "#06d6a0"),
    ({"agn_frac": 0.0}, "(3) + Dust IR re-emission", "#f77f00"),
    ({}, "(4) + AGN + X-ray + radio", "#1a1a1a"),
]:
    p = {**params, **{k: jnp.array(v) for k, v in params_override.items()}}
    s = model_full.predict_via_orchestrator(p)
    ax.plot(
        wave_um,
        to_nulnu_lsun(s.sed_intrinsic),
        color=color, lw=2.0, label=label, zorder=5,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1e7)
ax.set_ylim(y_bot, y_top)
ax.set_xlabel(r"Rest $\lambda$ [$\mu$m]")
ax.set_ylabel(r"$\nu L_{\nu}$ [$L_\odot$]")
ax.set_title("Building up the SED layer by layer", fontsize=12)
ax.legend(loc="lower center", frameon=False, fontsize=10, ncol=2)
ax.grid(True, alpha=0.25, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_layer_anatomy.png"), dpi=200, bbox_inches="tight")
plt.close()
print("✓ Saved 02_layer_anatomy.png")

# %% [markdown]
# ## Part 7: Units convention reference
#
# tengri carries SED quantities as **L_ν in erg/s/Hz** through the chain.
# The same number gets reinterpreted as **f_ν in erg/s/cm²/Hz** when
# divided by ``4π d_L²``, or as an **AB magnitude** via standard
# zero-point. Here's all three sides at once for the kitchen-sink galaxy.

# %%
# At z=0.5, dL ≈ 2895 Mpc = 8.94e27 cm. ``state`` for z=0.5 is already in
# rest-frame L_nu; project to observed-frame f_nu using the helper.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

lnu = np.asarray(state.sed_intrinsic)
ax = axes[0]
mask_pos = lnu > 0
ax.plot(wave_um[mask_pos], lnu[mask_pos], color="#3a86ff", lw=1.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.05, 1000.0)
ax.set_xlabel(r"Rest $\lambda$ [$\mu$m]"); ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("(a) Rest-frame luminosity density")
ax.grid(True, alpha=0.25, which="both")

ax = axes[1]
fnu_ujy = np.asarray(tg.units.fnu_to_ujy(np.asarray(state.sed_intrinsic) / (4.0 * np.pi * (8.94e27) ** 2)))
m = fnu_ujy > 1e-4
ax.plot(wave_um[m] * 1.5, fnu_ujy[m], color="#06d6a0", lw=1.8)  # observed wave = rest * (1+z)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.05, 1000.0)
ax.set_xlabel(r"Observed $\lambda$ [$\mu$m]"); ax.set_ylabel(r"$f_\nu$ [$\mu$Jy]")
ax.set_title("(b) Observed flux density")
ax.grid(True, alpha=0.25, which="both")

ax = axes[2]
m_ab = -2.5 * np.log10(np.maximum(fnu_ujy * 1e-6, 1e-30) / 3631.0)
m = (fnu_ujy > 1e-4) & (m_ab < 35)
ax.plot(wave_um[m] * 1.5, m_ab[m], color="#f77f00", lw=1.8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(0.05, 1000.0)
ax.set_xlabel(r"Observed $\lambda$ [$\mu$m]"); ax.set_ylabel(r"$m_{\rm AB}$ [mag]")
ax.set_title("(c) AB magnitude")
ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_units_convention.png"), dpi=200, bbox_inches="tight")
plt.close()
print("✓ Saved 02_units_convention.png")

# %% [markdown]
# ## Recap
#
# | Component | ``state.derived`` key | Wavelength regime |
# |---|---|---|
# | Stellar continuum | (in ``sed_dust_attenuated`` after dust) | UV → NIR |
# | Nebular continuum + lines | ``sed_nebular`` | UV → optical |
# | Dust attenuation | (multiplicative on ``sed_dust_attenuated``) | UV → near-IR |
# | Dust IR re-emission | ``sed_dust_ir`` | mid-IR → FIR |
# | AGN (disc + torus) | ``sed_agn`` | UV → MIR |
# | X-ray (XRBs + AGN) | ``sed_xray`` | hard X-ray |
# | Radio (synchrotron) | ``sed_radio`` | cm → m |
# | IGM transmission | ``igm_transmission`` | < Lyα (high z) |
#
# One ``predict_via_orchestrator`` call gives you all of them. To toggle
# any component off, drop its ``Parameters`` flag (e.g. ``radio=False``)
# or zero its parameter (``agn_log_lbol=Fixed(-10.0)`` makes AGN
# negligible). The chain remains JIT-compatible; this is the substrate
# that the inference notebooks (05–08) use to compute likelihoods.

# %%
tg.cite(model_full)
print("\n✓ Notebook 02 (SED anatomy) complete. Next: 03_discovering_the_menu.py")
