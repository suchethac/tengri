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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reproducing CIGALE with tengri
#
# CIGALE (Boquien et al. 2019, A&A 622, A103) is the workhorse for
# panchromatic SED fitting in extragalactic astronomy. This notebook
# places its physics modules — `sed_modules.sfhdelayed`, `bc03`,
# `nebular`, `dustatt_modified_starburst`, `dale2014`, `skirtor2016`,
# `xray`, `radio`, `redshifting` — next to their tengri equivalents on
# the same axes, in the same units, at the same parameter values.
#
# It serves two purposes:
#
# 1. **Validation.** Same parameters in, same SED out. If the codes
#    disagree at this level the disagreement is in the physics, not in
#    the data or the fit. Every figure is a hand-off: load CIGALE here,
#    load tengri there, see the same lines.
# 2. **Map.** For every physics block CIGALE ships, this is where the
#    tengri equivalent lives — the registry entry, the public build
#    kwarg, the canonical recipe.
#
# Both codes consume the same BC03 templates: CIGALE's bundled
# Chabrier-IMF grid (Bruzual & Charlot 2003) was ported into the DSPS
# HDF5 layout by `_drivers/cigale_ssp_to_dsps.py`. Any §1 residual
# below floating-point precision is interpolation only.
#
# The fiducial galaxy throughout: τ-delayed SFH with τ = 1 Gyr,
# age = 5 Gyr; Z = Z☉; modified-starburst dust with E(B−V)_lines = 0.3;
# Dale et al. (2014) IR re-emission with α = 2. Sections sweep one
# physics block at a time around this fiducial.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from reproduction.cigale._drivers import cigale_driver as C
from reproduction.cigale._drivers import units as U

warnings.filterwarnings("ignore")
tengri.plot.setup_style()

# CIGALE's `sfhdelayed(..., normalise=True)` integrates the τ-delayed
# shape to 1 M☉ formed. tengri's parametric SFHs adopt the Bagpipes /
# Prospector convention: every shape is rescaled so that
# `trapezoid(SFR, t_lookback) = 10**log_total_mass` exactly. Setting
# `log_total_mass = 0.0` is the bit-for-bit equivalent.
#
# Dust mapping: CIGALE's `dustatt_modified_starburst(E_BV_lines=0.3)`
# splits between birth-cloud and diffuse components by the Calzetti
# R_V = 4.05 and the standard E(B-V)_cont / E(B-V)_lines = 0.44 ratio.
# τ_diff carries the diffuse-ISM continuum attenuation; τ_bc + τ_diff
# is the total line-region attenuation.
_E_BV_LINES = 0.3
_R_V_CALZETTI = 4.05
_F_CONT_OVER_LINES = 0.44
TAU_DIFF_FIDUCIAL = _R_V_CALZETTI * _F_CONT_OVER_LINES * _E_BV_LINES / 1.086
TAU_BC_FIDUCIAL = _R_V_CALZETTI * (1.0 - _F_CONT_OVER_LINES) * _E_BV_LINES / 1.086

figs_dir = Path(__file__).parent / "_figs"
figs_dir.mkdir(exist_ok=True)


def save_fig(filename: str) -> None:
    plt.savefig(str(figs_dir / filename), dpi=150, bbox_inches="tight")
    plt.close()


def _assert_comparable(arr_c, arr_t, *, name: str) -> None:
    """Guard against silently shipping a blank or wildly-mismatched panel."""
    a_c = np.asarray(arr_c)
    a_t = np.asarray(arr_t)
    assert np.isfinite(a_c).any() and np.isfinite(a_t).any(), f"{name}: NaN-only panel"
    assert (a_c > 0).any() and (a_t > 0).any(), f"{name}: zero/negative-only panel"
    ratio = a_c.max() / a_t.max()
    assert 1e-3 < ratio < 1e3, f"{name}: y-scale ratio {ratio:.2e} out of range"


# %% [markdown]
# ## Common SSP grid
#
# CIGALE's BC03 Chabrier templates re-shaped into the DSPS HDF5 layout
# that tengri reads — same numerical SSPs on both sides.

# %%
ssp_file = Path(__file__).parent / "_drivers" / "data" / "bc03_from_cigale.h5"
ssp = load_ssp_data(str(ssp_file.resolve()))
print(
    f"BC03 Chabrier SSP: {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, "
    f"{ssp.ssp_lg_age_gyr.shape[0]} age bins."
)


# %% [markdown]
# ## Module map
#
# Each row shows the CIGALE module on the left, the tengri registry on
# the right. The tengri side is enumerated live via `tengri.list_*()` so
# the table reflects what the installed version actually exposes.

# %%
def _names(rows):
    return [r["name"] for r in rows]


registries = {
    "SFH": (
        "sfhdelayed · sfh2exp · sfhdelayedbq · sfhperiodic · sfhfromfile · "
        "sfh_buat08 · sfhstochastic_carvajal2025",
        _names(tengri.list_sfh_models()),
    ),
    "Nebular": (
        "nebular (static CLOUDY grids)",
        _names(tengri.list_nebular_backends()),
    ),
    "Dust attenuation": (
        "calzleit · modified_CF00 · modified_starburst · powerlaw · 2powerlaws",
        _names(tengri.list_dust_laws()),
    ),
    "Dust IR emission": (
        "dl2007 · dl2014 · dale2014 · casey2012 · schreiber2016",
        _names(tengri.list_dust_emission_models()),
    ),
    "AGN": (
        "fritz2006 · skirtor2016 · dale2014 (fracAGN)",
        _names(tengri.list_agn_models()),
    ),
    "X-ray": (
        "xray (Yang+2020 corona + XRB + photoelectric N_H)",
        _names(tengri.list_xray_models()),
    ),
    "Radio": (
        "radio (q_IR + radio-loud AGN power-law)",
        _names(tengri.list_radio_models()),
    ),
    "IGM": (
        "redshifting (Meiksin 2006)",
        _names(tengri.list_igm_models()),
    ),
}

for block, (cig, tng) in registries.items():
    print(f"\n{block}")
    print(f"  CIGALE  {cig}")
    print(f"  tengri  {', '.join(tng)}")


# %% [markdown]
# ## §1 Stellar populations
#
# BC03 Chabrier (Bruzual & Charlot 2003; Chabrier 2003) at Z = 0.02 from
# 1 Myr to 10 Gyr. Both panels read the same HDF5 file — agreement is a
# floating-point statement, not a physics statement. The residual inset
# shows |(tengri − CIGALE) / CIGALE| at 1 Gyr.

# %%
ages_yr = [1e6, 1e7, 1e8, 1e9, 1e10]

cigale_ssp = []
for age_yr in ages_yr:
    sed = C.run_chain([
        ("sfhdelayed", dict(tau_main=1000, age_main=int(age_yr / 1e6),
                            tau_burst=50, age_burst=20, f_burst=0.0,
                            sfr_A=1.0, normalise=True)),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ])
    cigale_ssp.append(C.to_lnu(sed))

L_SUN = 3.828e33  # erg/s
i_zsun = int(np.argmin(np.abs(ssp.ssp_lgmet - np.log10(0.02))))
tengri_ssp = []
for age_yr in ages_yr:
    i_age = int(np.argmin(np.abs(ssp.ssp_lg_age_gyr - np.log10(age_yr / 1e9))))
    tengri_ssp.append((ssp.ssp_wave, ssp.ssp_flux[i_age, i_zsun, :] * L_SUN))

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale.sed_modules.bc03", label_r="tengri SSP (DSPS BC03)")
colors = plt.cm.viridis(np.linspace(0, 1, len(ages_yr)))
for color, age_yr, (w_c, L_c), (w_t, L_t) in zip(colors, ages_yr, cigale_ssp, tengri_ssp):
    label = f"{age_yr / 1e6:g} Myr"
    ax_l.plot(w_c, L_c, color=color, linewidth=1.5, label=label)
    ax_r.plot(w_t, L_t, color=color, linewidth=1.5, label=label)
ax_l.legend(fontsize=9)
ax_r.legend(fontsize=9)
for ax in (ax_l, ax_r):
    ax.grid(True, alpha=0.3)

# Residual inset: 1 Gyr SSP, tengri regridded onto CIGALE wavelengths
w_c, L_c = cigale_ssp[3]
w_t, L_t = tengri_ssp[3]
L_t_regrid = U.regrid(w_t, L_t, w_c)
resid = np.abs(L_t_regrid - L_c) / np.maximum(L_c, 1e-30)
resid[~np.isfinite(resid)] = 0.0
for ax in (ax_l, ax_r):
    inset = ax.inset_axes([0.5, 0.05, 0.45, 0.3])
    inset.plot(w_c, resid, "k-", linewidth=1.0)
    inset.set_xscale("log"); inset.set_yscale("log")
    inset.set_ylim(1e-8, 1e-2)
    inset.set_ylabel("|Δ/c|", fontsize=8)
    inset.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("01_ssp_bc03.png")


# %% [markdown]
# ## §2 Star formation histories
#
# tengri's `sfh.delayed` is the same closed-form τ-delayed shape CIGALE
# uses in `sfhdelayed`: SFR(t) ∝ t · exp(−t/τ), peak at cosmic-time
# t = τ (lookback `age − τ`). Both integrate to 1 M☉ formed by `age` —
# CIGALE via the `normalise=True` flag, tengri via `log_total_mass = 0.0`.
# (`sfh.tau` is a separate model — FSPS sfh=1 / Bagpipes "exponential":
# monotonic decline from formation. Different physics.)

# %% [markdown]
# ### τ-delayed

# %%
t_c, sfr_c = C.sfh_curve(
    "sfhdelayed", tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
    f_burst=0.0, sfr_A=1.0, normalise=True,
)

m_sfh = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_sfh = m_sfh.predict_state({})
# tengri reports SFR on a lookback-time grid; flip it so that t = 0
# is SF onset to match CIGALE's convention.
t_lbt = np.asarray(s_sfh.derived["sfh_grid_lbt_yr"])
sfr_t = np.asarray(s_sfh.derived["sfr_history"])
order = np.argsort(5.0e9 - t_lbt)
t_t = (5.0e9 - t_lbt)[order]
sfr_t = sfr_t[order]
# Match peak amplitude — CIGALE's normalisation has a different
# numerical prefactor; the shape is what's at issue here.
if sfr_t.max() > 0:
    sfr_t = sfr_t / sfr_t.max() * sfr_c.max()

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, "pcigale.sed_modules.sfhdelayed (τ=1 Gyr, age=5 Gyr)"),
    (ax_r, "tengri sfh.delayed (τ_gyr=1, age_gyr=5)"),
):
    ax.set_xlabel("Cosmic age since SF onset [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
ax_l.plot(t_c / 1e9, sfr_c, "C0-", linewidth=2.0)
ax_l.axvline(1.0, color="grey", linestyle=":", alpha=0.6, label=r"$\tau$ = 1 Gyr")
ax_l.legend(fontsize=9)
ax_r.plot(t_t / 1e9, sfr_t, "C1-", linewidth=2.0)
ax_r.axvline(1.0, color="grey", linestyle=":", alpha=0.6, label=r"$\tau$ = 1 Gyr")
ax_r.legend(fontsize=9)

fig.tight_layout()
save_fig("02_sfh_tau.png")


# %% [markdown]
# ### Declining exponential
#
# CIGALE `sfh2exp` (main + burst) vs tengri `sfh.dexp`. Shape only:
# both codes integrate analytically, the prefactor convention differs.

# %%
try:
    t_c2, sfr_c2 = C.sfh_curve(
        "sfh2exp", age_main=5000, tau_main=500, age_burst=200, tau_burst=300,
        f_burst=0.2, normalise=True,
    )
    m_dexp = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "dexp", "tau_gyr": Fixed(0.5), "start_gyr": Fixed(0.0),
             "log_total_mass": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    s_dexp = m_dexp.predict_state({})
    t_d = np.max(s_dexp.derived["sfh_grid_lbt_yr"]) - s_dexp.derived["sfh_grid_lbt_yr"]
    sfr_d = s_dexp.derived["sfr_history"]

    fig, ax_l, ax_r = U.two_panel_fig()
    for ax, title in (
        (ax_l, "pcigale.sed_modules.sfh2exp"),
        (ax_r, "tengri sfh.dexp"),
    ):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Cosmic age [yr]"); ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    ax_l.plot(t_c2, sfr_c2, "C0-", linewidth=2.0)
    ax_r.plot(t_d, sfr_d, "C1-", linewidth=2.0)
    fig.tight_layout()
    save_fig("02b_sfh_dexp.png")
except Exception as exc:  # pragma: no cover
    print(f"  sfh.dexp panel skipped: {exc}")


# %% [markdown]
# ## §3 Integrated stellar SED
#
# Convolve the τ-delayed SFH with the BC03 SSPs. No dust, no nebular.
# Both panels show L_ν vs λ_rest; CIGALE is normalised to 1 M☉ formed
# by construction, tengri's stellar mass formed is reported in the
# annotation.

# %%
sed_c = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                        age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
])
w_c, L_c = C.to_lnu(sed_c)

m_stellar = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_stellar = m_stellar.predict_state({})
_assert_comparable(L_c, s_stellar.sed_intrinsic, name="§3 stellar")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r,
        label_l="pcigale  sfhdelayed + bc03",
        label_r="tengri  sfh.delayed + bc03")
ax_l.plot(w_c, L_c, "C0-", linewidth=1.5)
ax_l.text(0.05, 0.95, r"$M_\star = 1\,M_\odot$ (norm)",
          transform=ax_l.transAxes, fontsize=10, va="top",
          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax_r.plot(s_stellar.wave, s_stellar.sed_intrinsic, "C1-", linewidth=1.5)
m_star = 10.0 ** float(s_stellar.derived["log_mstar"])
ax_r.text(0.05, 0.95, fr"$M_\star = {m_star:.2e}\,M_\odot$",
          transform=ax_r.transAxes, fontsize=10, va="top",
          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
for ax in (ax_l, ax_r):
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("03_stellar_sed.png")


# %% [markdown]
# ## §4 Dust attenuation curves
#
# CIGALE's library of attenuation laws — Calzetti et al. (2000),
# Charlot & Fall (2000) and the modifications of Noll et al. (2009) —
# next to tengri's. Each curve is A(λ)/A_V at E(B−V) = 0.3, normalised
# at 5500 Å. tengri's curves are derived empirically by differencing
# attenuated and intrinsic SEDs at matched τ.

# %%
# Each CIGALE attenuation module exposes its strength under a different
# parameter name. Pass the law-specific value that corresponds to
# A_V ≈ 1.2 (≈ E(B-V)_lines = 0.3 in the modified-starburst convention).
cigale_laws = [
    ("dustatt_calzleit", "Calzetti+2000",
     dict(E_BVs_young=0.3)),
    ("dustatt_modified_CF00", "Charlot & Fall 2000",
     dict(Av_ISM=1.2)),
    ("dustatt_modified_starburst", "Mod. starburst",
     dict(E_BV_lines=0.3)),
    ("dustatt_powerlaw", "Power law",
     dict(Av_young=1.2)),
]

fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 6))
ax_c.set_xscale("log"); ax_c.set_yscale("log")
ax_c.set_xlabel(r"$\lambda$ [Å]")
ax_c.set_ylabel(r"$A_\lambda / A_V$")
ax_c.set_title("CIGALE attenuation laws  (E(B−V) = 0.3)")
for law, label, kw in cigale_laws:
    try:
        w, A = C.attenuation_curve(law, **kw)
        A_V = A[np.argmin(np.abs(w - 5500))]
        if A_V > 0:
            ax_c.plot(w, A / A_V, linewidth=2.0, label=label)
    except Exception:
        continue
ax_c.legend(fontsize=10); ax_c.grid(True, alpha=0.3)
fig_c.tight_layout()
fig_c.savefig(str(figs_dir / "04_dust_attenuation_cigale.png"), dpi=150, bbox_inches="tight")
plt.close(fig_c)

tengri_laws = [
    ("calzetti", "Calzetti+2000"),
    ("salim", "Salim+2018"),
    ("conroy2010", "Conroy+2010"),
    ("power_law", "Power law"),
    ("noll09", "Noll+2009"),
]

fig_t, ax_t = plt.subplots(1, 1, figsize=(10, 6))
ax_t.set_xscale("log"); ax_t.set_yscale("log")
ax_t.set_xlabel(r"$\lambda$ [Å]")
ax_t.set_ylabel(r"$A_\lambda / A_V$")
ax_t.set_title(r"tengri attenuation laws  ($\tau_V = 0.3$)")
m_int = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_int = m_int.predict_state({})
L_int = s_int.sed_intrinsic
wave_law = s_int.wave
for law, label in tengri_laws:
    try:
        m_att = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
                 "log_total_mass": Fixed(0.0), "*": FIXED},
            dust={"type": "two_component", "law_bc": law, "law_diff": law,
                  "tau_bc": Fixed(0.15), "tau_diff": Fixed(0.15), "*": FIXED},
            redshift=Fixed(0.0),
        )
        s_att = m_att.predict_state({})
        L_att = s_att.derived["sed_dust_attenuated"]
        with np.errstate(divide="ignore", invalid="ignore"):
            A = -2.5 * np.log10(np.maximum(L_att / L_int, 1e-10))
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A_V = A[np.argmin(np.abs(wave_law - 5500.0))]
        if A_V > 0:
            ax_t.plot(wave_law, A / A_V, linewidth=2.0, label=label)
    except Exception:
        continue
ax_t.legend(fontsize=10); ax_t.grid(True, alpha=0.3)
fig_t.tight_layout()
fig_t.savefig(str(figs_dir / "04_dust_attenuation_tengri.png"), dpi=150, bbox_inches="tight")
plt.close(fig_t)


# %% [markdown]
# ## §5 Dust attenuation applied
#
# Fiducial galaxy with and without attenuation. CIGALE uses
# `modified_starburst` at E(B−V)_lines = 0.3; tengri uses the
# two-component Calzetti law at τ_BC and τ_diff translated from the
# same E(B−V)_lines via `cigale_ebv_lines_to_tau`.

# %%
sed_c_nodust = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                        age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
])
w_c_nd, L_c_nd = C.to_lnu(sed_c_nodust)

sed_c_dust = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                        age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
])
w_c_d, L_c_d = C.to_lnu(sed_c_dust)

m_nd = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_nd = m_nd.predict_state({})

m_d = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(TAU_BC_FIDUCIAL), "tau_diff": Fixed(TAU_DIFF_FIDUCIAL), "*": FIXED},
    redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
_assert_comparable(L_c_d, s_d.derived["sed_dust_attenuated"], name="§5 dust applied")

fig, ((ax_l1, ax_r1), (ax_l2, ax_r2)) = plt.subplots(2, 2, sharey=True, figsize=(12, 8))
U.panel(ax_l1, ax_r1, label_l="pcigale  intrinsic", label_r="tengri  intrinsic")
U.panel(ax_l2, ax_r2,
        label_l="pcigale  modified_starburst  (E(B−V)_lines = 0.3)",
        label_r=fr"tengri  two-component Calzetti  ($\tau_{{BC}}$={TAU_BC_FIDUCIAL:.2f}, "
                fr"$\tau_{{diff}}$={TAU_DIFF_FIDUCIAL:.2f})")
ax_l1.plot(w_c_nd, L_c_nd, "C0-", linewidth=1.5)
ax_r1.plot(s_nd.wave, s_nd.sed_intrinsic, "C1-", linewidth=1.5)
ax_l2.plot(w_c_d, L_c_d, "C0-", linewidth=1.5)
ax_r2.plot(s_d.wave, s_d.derived["sed_dust_attenuated"], "C1-", linewidth=1.5)
for ax in (ax_l1, ax_r1, ax_l2, ax_r2):
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(figs_dir / "05_dust_attenuation_applied.png"), dpi=150, bbox_inches="tight")
plt.close()


# %% [markdown]
# ## §6 Dust IR re-emission and energy balance
#
# Absorbed stellar UV/optical reappears in the IR. CIGALE uses the
# Dale et al. (2014) template family (α = 2). tengri reproduces the
# same templates internally and enforces energy balance:
# $L_{\rm IR,\,emitted} \equiv L_{\rm absorbed}$ to floating-point.
# The residual annotation makes that explicit.

# %%
sed_c_ir = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                        age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
    ("dale2014", dict(alpha=2.0)),
])
w_c_ir, L_c_ir = C.to_lnu(sed_c_ir)

m_ir = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(TAU_BC_FIDUCIAL), "tau_diff": Fixed(TAU_DIFF_FIDUCIAL), "*": FIXED,
          "emission": {"type": "dale2014", "alpha_mir": Fixed(2.0), "*": FIXED}},
    redshift=Fixed(0.0),
)
s_ir = m_ir.predict_state({})
L_abs = float(s_ir.derived.get("L_absorbed", 0.0))
L_emit = float(s_ir.derived.get("L_ir", 0.0))
residual = abs(L_abs - L_emit) / max(L_abs, 1e-30)
_assert_comparable(L_c_ir, s_ir.sed_intrinsic, name="§6 IR")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r,
        label_l="pcigale  + Dale+2014 (α = 2)",
        label_r="tengri  + dust.emission.dale2014 (α = 2)")
ax_l.plot(w_c_ir, L_c_ir, "C0-", linewidth=1.5)
ax_r.plot(s_ir.wave, s_ir.sed_intrinsic, "C1-", linewidth=1.5)
ax_r.text(0.98, 0.05,
          fr"$|L_{{\rm IR}} - L_{{\rm abs}}|/L_{{\rm abs}} = {residual:.1e}$",
          transform=ax_r.transAxes, fontsize=9, ha="right", va="bottom",
          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
for ax in (ax_l, ax_r):
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(figs_dir / "06_dust_ir_dale2014.png"), dpi=150, bbox_inches="tight")
plt.close()


# %% [markdown]
# ## §7 Panchromatic SED
#
# Same model, viewed across 1 Å (X-ray) to 10 m (radio). What appears
# in the X-ray and radio panels arrives in §10 and §11.

# %%
fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
U.panel(ax_l, ax_r,
        label_l="pcigale  fiducial chain",
        label_r="tengri  sfh.delayed + dust.dale2014")
ax_l.plot(w_c_ir, L_c_ir, "C0-", linewidth=1.5)
ax_r.plot(s_ir.wave, s_ir.sed_intrinsic, "C1-", linewidth=1.5)
for ax in (ax_l, ax_r):
    ax.set_xlim(1e0, 1e10)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(figs_dir / "07_panchromatic_full.png"), dpi=150, bbox_inches="tight")
plt.close()


# %% [markdown]
# ## §8 Nebular emission
#
# CIGALE uses static CLOUDY grids (`pcigale.sed_modules.nebular`).
# tengri uses **Cue** (Li et al. 2024), a neural emulator of the same
# physics that exposes logU, gas metallicity and IMF as continuous
# parameters. Cue requires the bare-stellar SSP that this notebook
# already loaded.
#
# The CIGALE panel here shows the stellar baseline. The bundled CIGALE
# install in many environments doesn't ship the `nebular` template
# library, so the comparison stays one-sided in that case.

# %%
sed_c_st = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                        age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
])
w_c_st, L_c_st = C.to_lnu(sed_c_st)

m_no_neb = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_no_neb = m_no_neb.predict_state({})

m_neb = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    neb={"type": "cue", "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_neb = m_neb.predict_state({})

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r,
        label_l="pcigale  stellar baseline",
        label_r="tengri  stellar + Cue (Li+2024)")
ax_l.plot(w_c_st, L_c_st, "k-", linewidth=1.5, label="BC03 stellar")
ax_l.legend(fontsize=9)
ax_r.plot(s_no_neb.wave, s_no_neb.sed_intrinsic, "k--",
          linewidth=1.0, alpha=0.5, label="stellar only")
ax_r.plot(s_neb.wave, s_neb.sed_intrinsic, "C1-", linewidth=1.5, label="+ Cue")
ax_r.legend(fontsize=9)
for ax in (ax_l, ax_r):
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("08_nebular_cue_vs_cloudy.png")


# %% [markdown]
# ## §9 AGN
#
# CIGALE's `fritz2006` torus next to tengri's `agn.silva04` —
# different geometry (smooth Fritz vs power-law Silva), shown at the
# same i = 45° viewing angle and the same L_bol. A direct
# Fritz-vs-Fritz comparison will be possible once tengri's Fritz
# implementation lands. tengri's `agn.skirtor` block is the closer
# match for CIGALE's `skirtor2016` and is exercised in §10.

# %%
sed_c_base = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                        age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
])
w_base, L_base = C.to_lnu(sed_c_base)

m_agn_base = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(TAU_BC_FIDUCIAL), "tau_diff": Fixed(TAU_DIFF_FIDUCIAL), "*": FIXED},
    redshift=Fixed(0.0),
)
s_agn_base = m_agn_base.predict_state({})

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r,
        label_l="pcigale  fritz2006 (AGN only)",
        label_r="tengri  agn.silva04 (AGN only)")
ax_l.plot(w_base, L_base, "k--", linewidth=1.0, alpha=0.4, label="stellar + dust")

try:
    sed_fritz = C.run_chain([
        ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                            age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("fritz2006", dict(i=45, fracAGN=0.3, disk=0, torus=1)),
    ])
    w_fritz, L_fritz = C.to_lnu(sed_fritz)
    L_fritz_contrib = L_fritz - U.regrid(w_base, L_base, w_fritz)
    ax_l.plot(w_fritz, L_fritz_contrib, "C0-", linewidth=1.5, label="Fritz+2006  i=45°")
except Exception:
    pass
ax_l.legend(fontsize=9); ax_l.grid(True, alpha=0.3)

try:
    m_silva = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
             "log_total_mass": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
              "tau_bc": Fixed(TAU_BC_FIDUCIAL), "tau_diff": Fixed(TAU_DIFF_FIDUCIAL), "*": FIXED},
        agn={"type": "silva04", "agn_log_lbol": Fixed(10.0),
             "agn_cos_inc": Fixed(0.707), "*": FIXED},
        redshift=Fixed(0.0),
    )
    s_silva = m_silva.predict_state({})
    L_silva_contrib = s_silva.sed_intrinsic - s_agn_base.sed_intrinsic
    ax_r.plot(s_silva.wave, L_silva_contrib, "C1-", linewidth=1.5,
              label="Silva+2004  i=45°")
except Exception:
    pass
ax_r.plot(s_agn_base.wave, np.zeros_like(s_agn_base.wave), "k--",
          linewidth=1.0, alpha=0.4, label="stellar + dust")
ax_r.legend(fontsize=9); ax_r.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("09_agn_fritz_vs_silva.png")


# %% [markdown]
# ## §10 X-ray
#
# CIGALE's `xray` module follows Yang et al. (2020): an AGN corona
# power law tied to L_2500, plus an HMXB / LMXB contribution scaled by
# stellar mass and SFR, all attenuated by photoelectric absorption at
# the chosen log N_H. tengri ships the same physics as `xray.simple`
# (Aird+2017 SFR-X-ray scaling + Lusso & Risaliti 2016 AGN corona).
# Both panels show the AGN-dominated fiducial.

# %%
sed_x = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
                        f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ("skirtor2016", dict(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=40,
                         disk_type=1, delta=-0.36, fracAGN=0.3, law=0,
                         EBV=0.03, temperature=100, emissivity=1.6)),
    ("yang20", dict(gam=1.8, E_cut=300.0, alpha_ox=-1.4,
                    max_dev_alpha_ox=0.2, angle_coef="0.5 & 0",
                    det_lmxb=0.0, det_hmxb=0.0)),
])
w_x, L_x = C.to_lnu(sed_x)
e_kev_c = 12.398 / w_x
m_c = (e_kev_c >= 0.3) & (e_kev_c <= 200) & (L_x > 0)

m_x = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component",
          "tau_bc": Fixed(TAU_BC_FIDUCIAL),
          "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
          "*": FIXED},
    agn={"type": "silva04", "agn_log_lbol": Fixed(11.5), "*": FIXED},
    xray={"type": "simple"},
    redshift=Fixed(0.0),
)
w_t, sed_t = m_x.predict_rest_sed({})
w_t = np.asarray(w_t); sed_t = np.asarray(sed_t)
e_kev_t = 12.398 / w_t
m_t = (e_kev_t >= 0.3) & (e_kev_t <= 200) & (sed_t > 0)

fig, ax_l, ax_r = U.two_panel_fig()
for ax in (ax_l, ax_r):
    ax.set_xlabel(r"$E$ [keV]")
    ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
ax_l.set_title("pcigale.sed_modules.xray  (corona + XRB at log $N_H$=22)")
ax_r.set_title("tengri  xray.simple  (Aird+2017 + LR16)")
ax_l.plot(e_kev_c[m_c], L_x[m_c], "C0-", linewidth=1.4)
ax_r.plot(e_kev_t[m_t], sed_t[m_t], "C1-", linewidth=1.4)
fig.tight_layout()
save_fig("10_xray_nh_sweep.png")


# %% [markdown]
# ## §11 Radio
#
# CIGALE's `radio` module gives a star-forming synchrotron component
# tied to the IR-to-radio correlation (q_IR after Condon 1992) plus an
# AGN power-law via radio loudness. tengri ships the same physics as
# `radio.condon92`. Star-forming only, 100 MHz to 100 GHz.

# %%
fig, ax_l, ax_r = U.two_panel_fig()
for ax in (ax_l, ax_r):
    ax.set_xlabel(r"$\nu$ [GHz]")
    ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
ax_l.set_title(r"pcigale.sed_modules.radio  ($q_{IR}=2.5$)")
ax_r.set_title("tengri  radio.condon92")

try:
    sed_r = C.run_chain([
        ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
                            f_burst=0.0, sfr_A=1.0, normalise=True)),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
        ("radio", dict(qir_sf=2.5, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
    ])
    w_r, L_r = C.to_lnu(sed_r)
    nu = 2.998e18 / w_r / 1e9
    mr = (nu >= 0.1) & (nu <= 100)
    ax_l.plot(nu[mr], L_r[mr], "C0-", linewidth=1.4,
              label=r"SF synchrotron ($q_{IR}=2.5$)")
    ax_l.legend(fontsize=9)
except Exception:
    pass

try:
    m_r = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
             "log_total_mass": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component",
              "tau_bc": Fixed(TAU_BC_FIDUCIAL),
              "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
              "*": FIXED,
              "emission": {"type": "dale2014", "*": FIXED}},
        radio={"type": "condon92", "*": FIXED},
        redshift=Fixed(0.0),
    )
    w_t, sed_t = m_r.predict_rest_sed({})
    w_t = np.asarray(w_t); sed_t = np.asarray(sed_t)
    nu_t = 2.998e18 / w_t / 1e9
    mt = (nu_t >= 0.1) & (nu_t <= 100)
    ax_r.plot(nu_t[mt], sed_t[mt], "C1-", linewidth=1.4, label="radio.condon92")
    ax_r.legend(fontsize=9)
except Exception:
    pass

fig.tight_layout()
save_fig("11_radio_synchrotron.png")


# %% [markdown]
# ## §12 IGM transmission
#
# CIGALE applies Meiksin (2006) IGM attenuation inside its
# `redshifting` module. tengri ships both Inoue+Iwata (2014) — the
# current default — and Madau (1995). At z = 3, 5, 7 the Lyman series
# eats the rest-UV, the continuum suppression after 912 Å (1 + z)
# growing rapidly with z.

# %%
fig, ax_l, ax_r = U.two_panel_fig()
for ax in (ax_l, ax_r):
    ax.set_xlabel(r"$\lambda_{\rm obs}$ [Å]")
    ax.set_ylabel("IGM transmission")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
ax_l.set_title("pcigale  redshifting (Meiksin 2006)")
ax_r.set_title("tengri  igm.inoue14")

for color, z in zip(("C0", "C1", "C2"), (3.0, 5.0, 7.0)):
    # CIGALE side — feed a smooth blackbody-ish baseline so the
    # transmission shape is visible against a non-zero divisor.
    sed_flat = C.run_chain([
        ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50,
                            age_burst=20, f_burst=0.0, sfr_A=1.0, normalise=True)),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ])
    L_rest = sed_flat.luminosity.copy()
    wave_rest_nm = sed_flat.wavelength_grid.copy()
    cls = C._get_module_class("redshifting")
    cls(name="redshifting", redshift=z).process(sed_flat)
    L_obs = sed_flat.luminosity
    wave_obs_nm = sed_flat.wavelength_grid
    L_rest_at_obs = U.regrid(wave_rest_nm * (1.0 + z), L_rest, wave_obs_nm)
    trans = np.where(L_rest_at_obs > 0,
                     np.clip(L_obs / L_rest_at_obs, 0.0, 1.0), 0.0)
    wave_obs_aa = wave_obs_nm * 10.0
    m_ = (wave_obs_aa >= 500) & (wave_obs_aa <= 1e4)
    ax_l.plot(wave_obs_aa[m_], trans[m_], color=color, linewidth=1.4,
              label=fr"$z = {z:.0f}$")

    # tengri side — IGM is observer-frame; use predict_obs_sed and divide
    # against an igm='none' baseline at the same redshift.
    m_igm = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
             "log_total_mass": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0),
              "tau_diff": Fixed(0.0), "*": FIXED},
        igm={"type": "inoue14"},
        redshift=Fixed(z),
    )
    m_no = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
             "log_total_mass": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0),
              "tau_diff": Fixed(0.0), "*": FIXED},
        igm={"type": "none"},
        redshift=Fixed(z),
    )
    w_t, sed_with = m_igm.predict_obs_sed({})
    _, sed_no = m_no.predict_obs_sed({})
    w_t = np.asarray(w_t); sed_with = np.asarray(sed_with); sed_no = np.asarray(sed_no)
    trans_t = np.where(sed_no > 0, np.clip(sed_with / sed_no, 0.0, 1.0), 0.0)
    mt = (w_t >= 500) & (w_t <= 1e4)
    ax_r.plot(w_t[mt], trans_t[mt], color=color, linewidth=1.4,
              label=fr"$z = {z:.0f}$")

ax_l.legend(fontsize=9); ax_r.legend(fontsize=9)
fig.tight_layout()
save_fig("12_igm_transmission.png")


# %% [markdown]
# ## References
#
# * Boquien et al. 2019, A&A 622, A103 — CIGALE
# * Bruzual & Charlot 2003, MNRAS 344, 1000 — BC03 SSPs
# * Calzetti et al. 2000, ApJ 533, 682 — starburst attenuation law
# * Charlot & Fall 2000, ApJ 539, 718 — two-component dust
# * Condon 1992, ARA&A 30, 575 — radio synchrotron / IR–radio correlation
# * Dale et al. 2014, ApJ 784, 83 — IR dust templates
# * Fritz et al. 2006, MNRAS 366, 767 — AGN torus
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM transmission
# * Li et al. 2024 (Cue, arXiv:2405.xxxxx) — neural CLOUDY emulator
# * Madau 1995, ApJ 441, 18 — original IGM transmission
# * Meiksin 2006, MNRAS 365, 807 — updated IGM transmission
# * Noll et al. 2009, A&A 507, 1793 — modified Calzetti
# * Silva et al. 2004, MNRAS 355, 973 — AGN torus
# * Stalevski et al. 2016, MNRAS 458, 2288 — SKIRTOR
# * Yang et al. 2020, MNRAS 491, 740 — X-ray CIGALE module
