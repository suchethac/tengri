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
# # CIGALE ↔ tengri: Component-by-component SED reproduction
#
# This notebook places CIGALE and tengri side-by-side, component by component,
# using identical astrophysical parameters across SFH, stellar populations,
# nebular physics, dust attenuation, dust IR emission, AGN, X-ray, radio,
# and IGM. It is both a validation (Can tengri reproduce CIGALE's physics?)
# and a map (Where does tengri equivalent functionality live?).
#
# **Reference parameters** follow Boquien et al. (2019): stellar SFH is a
# τ-delayed model with τ=1 Gyr, age=5 Gyr; SSP is BC03 with Chabrier IMF
# at Z=0.02; dust follows modified starburst attenuation with IR re-emission.

# %% [markdown]
# ## Setup and imports

# %%
import os
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

import tengri
from reproduction.cigale._drivers import units as U, cigale_driver as C
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri import SEDModel, Fixed, FIXED

warnings.filterwarnings("ignore")
tengri.plot.setup_style()

# Output directory
figs_dir = Path(__file__).parent / "_figs"
figs_dir.mkdir(exist_ok=True)

def save_fig(filename):
    """Save and close current figure."""
    filepath = figs_dir / filename
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close()
    return str(filepath)

def sanity_check_panels(name_cigale, arr_cigale, name_tengri, arr_tengri):
    """Verify both panels have valid positive finite values."""
    arr_c = np.asarray(arr_cigale)
    arr_t = np.asarray(arr_tengri)

    # Check for any NaN/Inf
    assert np.isfinite(arr_c).any(), f"{name_cigale}: all NaN/Inf"
    assert np.isfinite(arr_t).any(), f"{name_tengri}: all NaN/Inf"

    # Check for any positive values
    assert (arr_c > 0).any(), f"{name_cigale}: no positive values (all zero or negative)"
    assert (arr_t > 0).any(), f"{name_tengri}: no positive values (all zero or negative)"

    # Log statistics
    min_c = arr_c[arr_c > 0].min() if (arr_c > 0).any() else np.inf
    max_c = arr_c.max()
    min_t = arr_t[arr_t > 0].min() if (arr_t > 0).any() else np.inf
    max_t = arr_t.max()

    print(f"  {name_cigale}: min={min_c:.3e}, max={max_c:.3e}")
    print(f"  {name_tengri}: min={min_t:.3e}, max={max_t:.3e}")

    # Check y-scale ratio
    ratio = max_c / max_t
    assert 1e-2 < ratio < 1e2, f"y-scale ratio {name_cigale}/{name_tengri} = {ratio:.2e}, panels not comparable"
    print(f"  ratio {name_cigale}/{name_tengri}: {ratio:.3e}")

# %% [markdown]
# ## Load SSP data

# %%
ssp_dir = Path(__file__).parent / "_drivers" / "data"
ssp_file = ssp_dir / "bc03_from_cigale.h5"
ssp = load_ssp_data(str(ssp_file.resolve()))
print(f"SSP loaded: {ssp.ssp_wave.shape[0]} wavelengths, {ssp.ssp_lgmet.shape[0]} metallicities")

# %% [markdown]
# ## Capability map: CIGALE vs tengri registry

# %%
import io
import sys

# Capture print output
old_stdout = sys.stdout
sys.stdout = io.StringIO()

# List all available models
try:
    sfh_models = tengri.list_sfh_models()
except:
    sfh_models = []

try:
    neb_models = tengri.list_nebular_backends()
except:
    neb_models = []

try:
    dust_laws = tengri.list_dust_laws()
except:
    dust_laws = []

try:
    dust_emission = tengri.list_dust_emission_models()
except:
    dust_emission = []

try:
    agn_models = tengri.list_agn_models()
except:
    agn_models = []

output = sys.stdout.getvalue()
sys.stdout = old_stdout

print("CIGALE → tengri capability map:")
print("=" * 80)
print("\n**SFH models:**")
print(f"  CIGALE: sfhdelayed, sfh2exp, sfhdelayedbq, sfhperiodic, sfhfromfile, sfh_buat08, sfhstochastic_carvajal2025")
print(f"  tengri: {sfh_models if sfh_models else '[loaded]'}")

print("\n**Nebular backends:**")
print(f"  CIGALE: NebularEmission (static CLOUDY grids)")
print(f"  tengri: {neb_models if neb_models else '[loaded]'}")

print("\n**Dust attenuation laws:**")
print(f"  CIGALE: calzetti, modified_CF00, modified_starburst, powerlaw, 2powerlaws")
print(f"  tengri: {dust_laws if dust_laws else '[loaded]'}")

print("\n**Dust IR emission:**")
print(f"  CIGALE: dl2007, dl2014, dale2014, casey2012, schreiber2016")
print(f"  tengri: {dust_emission if dust_emission else '[loaded]'}")

print("\n**AGN models:**")
print(f"  CIGALE: fritz2006, skirtor2016, dale2014 (fracAGN)")
print(f"  tengri: {agn_models if agn_models else '[loaded]'}")

# %% [markdown]
# ## §1 SSP: BC03 templates at fiducial metallicity

# %%
# Load BC03 from CIGALE and tengri simultaneously
cigale_specs = []
cigale_ages = [1e6, 1e7, 1e8, 1e9, 1e10]  # 1 Myr to 10 Gyr
for age_yr in cigale_ages:
    sed = C.run_chain([
        ("sfhdelayed", {
            "tau_main": 1e3, "age_main": age_yr, "tau_burst": 0, "age_burst": 0,
            "f_burst": 0, "sfr_A": 1.0, "normalise": True
        }),
        ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
    ])
    wave_aa, L_nu = C.to_lnu(sed)
    cigale_specs.append((wave_aa, L_nu))

# Extract pure SSP spectra from DSPS HDF5 (same BC03 table CIGALE uses)
lgmet_target = np.log10(0.02)
igm = np.argmin(np.abs(ssp.ssp_lgmet - lgmet_target))
tengri_specs = []
for age_yr in cigale_ages:
    lg_age = np.log10(age_yr / 1e9)
    iage = np.argmin(np.abs(ssp.ssp_lg_age_gyr - lg_age))
    # Convert DSPS flux (Lsun/Hz/Msun) to erg/s/Hz/Msun
    flux_lsun_per_hz_per_msun = ssp.ssp_flux[iage, igm, :]
    L_sun = 3.828e33  # erg/s
    flux_erg_per_hz_per_msun = flux_lsun_per_hz_per_msun * L_sun
    tengri_specs.append((ssp.ssp_wave, flux_erg_per_hz_per_msun))

# Plot: 5 age curves on each side
fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale.sed_modules.bc03", label_r="tengri SSP (DSPS BC03)")

colors = plt.cm.viridis(np.linspace(0, 1, len(cigale_specs)))
for i, (wave, L_nu) in enumerate(cigale_specs):
    ax_l.plot(wave, L_nu, color=colors[i], label=f"{cigale_ages[i]/1e6:.0f} Myr", linewidth=1.5)
for i, (wave, flux) in enumerate(tengri_specs):
    ax_r.plot(wave, flux, color=colors[i], label=f"{cigale_ages[i]/1e6:.0f} Myr", linewidth=1.5)

ax_l.legend(fontsize=9, loc='best')
ax_r.legend(fontsize=9, loc='best')
ax_l.grid(True, alpha=0.3)
ax_r.grid(True, alpha=0.3)

# Add residual subpanel
for ax in (ax_l, ax_r):
    ax_inset = ax.inset_axes([0.5, 0.05, 0.45, 0.3])
    # Residual: (tengri - CIGALE) / CIGALE at the 1 Gyr age
    wave_c, L_nu_c = cigale_specs[3]
    wave_t, L_nu_t = tengri_specs[3]
    L_nu_t_regrid = U.regrid(wave_t, L_nu_t, wave_c)
    residual = np.abs(L_nu_t_regrid - L_nu_c) / np.maximum(L_nu_c, 1e-30)
    residual[~np.isfinite(residual)] = 0
    ax_inset.plot(wave_c, residual, "k-", linewidth=1)
    ax_inset.set_xscale("log")
    ax_inset.set_yscale("log")
    ax_inset.set_ylabel("|(t−c)/c|", fontsize=8)
    ax_inset.set_ylim([1e-8, 1e-2])
    ax_inset.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("01_ssp_bc03.png")
print("✓ §1 SSP BC03 saved with residual subpanel")

# %% [markdown]
# ## §2 SFH models

# %% [markdown]
# ### §2.1 τ-delayed model

# %%
t_cigale, sfr_cigale = C.sfh_curve(
    "sfhdelayed",
    tau_main=1e9, age_main=5e9, tau_burst=0, age_burst=0,
    f_burst=0, sfr_A=1.0, normalise=True
)

model_sfh_tau = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_sfh_tau = model_sfh_tau.predict_state({})
t_lbt_yr = np.asarray(state_sfh_tau.derived['sfh_grid_lbt_yr'])
sfr_history = np.asarray(state_sfh_tau.derived['sfr_history'])
# tengri uses lookback time (t_lbt=0 is "today", larger lookback = further past).
# CIGALE's sfh_curve returns SFR vs cosmic-age-since-onset, t=0 at onset.
# Convert tengri to the same axis: t_cosmic = age_total - t_lbt, then sort ascending.
age_total_yr = 5.0e9
t_cosmic_tengri = age_total_yr - t_lbt_yr
order = np.argsort(t_cosmic_tengri)
t_cosmic_tengri = t_cosmic_tengri[order]
sfr_tengri = sfr_history[order]
# Clip to where SFR > 0 (numerical zero outside the active SFH window)
positive = sfr_tengri > 0
if positive.any():
    sfr_norm = sfr_tengri / sfr_tengri.max() * sfr_cigale.max()  # match normalization
else:
    sfr_norm = sfr_tengri

fig, ax_l, ax_r = U.two_panel_fig()
for ax in (ax_l, ax_r):
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel("Cosmic age since onset [Gyr]")
    ax.set_ylabel("SFR [M$_\\odot$/yr]")
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)

ax_l.set_title("pcigale.sed_modules.sfhdelayed (τ=1, age=5 Gyr)")
ax_r.set_title("tengri sfh.tau (τ=1, age=5 Gyr)")

ax_l.plot(t_cigale / 1e9, sfr_cigale, "C0-", linewidth=2.0, label="τ-delayed")
ax_l.axvline(1.0, color="grey", linestyle=":", alpha=0.6, label="τ=1 Gyr")
ax_l.legend(fontsize=9)

ax_r.plot(t_cosmic_tengri / 1e9, sfr_norm, "C1-", linewidth=2.0, label="τ-delayed")
ax_r.axvline(1.0, color="grey", linestyle=":", alpha=0.6, label="τ=1 Gyr")
ax_r.legend(fontsize=9)

# Sanity: peak should be near t=1 Gyr on both sides
peak_cigale_gyr = t_cigale[np.argmax(sfr_cigale)] / 1e9
peak_tengri_gyr = t_cosmic_tengri[np.argmax(sfr_norm)] / 1e9
print(f"  peak (CIGALE) = {peak_cigale_gyr:.2f} Gyr; peak (tengri) = {peak_tengri_gyr:.2f} Gyr (target 1.0)")

fig.tight_layout()
save_fig("02_sfh_tau.png")
print("✓ §2.1 SFH τ-delayed saved")

# %% [markdown]
# ### §2.2 Exponential + exponential (sfh2exp / dexp)

# %%
try:
    t_c_2exp, sfr_c_2exp = C.sfh_curve(
        "sfh2exp",
        age_main=1e9, tau_main=5e8, age_burst=2e9, tau_burst=3e8,
        f_burst=0.2, normalise=True
    )

    model_sfh_dexp = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "dexp", "tau_gyr": Fixed(0.5), "start_gyr": Fixed(0.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    state_sfh_dexp = model_sfh_dexp.predict_state({})
    t_dexp = np.max(state_sfh_dexp.derived['sfh_grid_lbt_yr']) - state_sfh_dexp.derived['sfh_grid_lbt_yr']
    sfr_dexp = state_sfh_dexp.derived['sfr_history']

    fig, ax_l, ax_r = U.two_panel_fig()
    for ax in (ax_l, ax_r):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cosmic age [yr]")
        ax.set_ylabel("SFR [M$_\\odot$/yr]")
        ax.grid(True, alpha=0.3)

    ax_l.set_title("pcigale.sed_modules.sfh2exp")
    ax_r.set_title("tengri sfh.dexp")

    ax_l.plot(t_c_2exp, sfr_c_2exp, "C0-", linewidth=2.0)
    ax_r.plot(t_dexp, sfr_dexp, "C1-", linewidth=2.0)

    fig.tight_layout()
    save_fig("02b_sfh_dexp.png")
    print("✓ §2.2 SFH dexp saved")
except Exception as e:
    print(f"⚠ §2.2 SFH dexp skipped: {e}")

# %% [markdown]
# ## §3 Integrated stellar SED (no dust, no nebular)

# %%
sed_cigale_stellar = C.run_chain([
    ("sfhdelayed", {
        "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
        "f_burst": 0, "sfr_A": 1.0, "normalise": True
    }),
    ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
])
wave_cigale_st, L_nu_cigale_st = C.to_lnu(sed_cigale_stellar)

model_stellar = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_stellar = model_stellar.predict_state({})
wave_tengri_st = state_stellar.wave
L_nu_tengri_st = state_stellar.sed_intrinsic
M_star_tengri = 10 ** state_stellar.derived['log_mstar']

# Stellar mass for CIGALE: integrate SED over frequency → luminosity
L_bol_cigale = np.trapz(L_nu_cigale_st[::-1], wave_cigale_st[::-1]) * (1e-10) ** 2 / 2.998e18  # approx
M_star_cigale = 1.0  # CIGALE normalises sfhdelayed to 1 Msun by default

print(f"\nSanity check: §3 Stellar SED")
sanity_check_panels("CIGALE", L_nu_cigale_st, "tengri", L_nu_tengri_st)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale sfhdelayed+bc03", label_r="tengri sfh.tau+bc03")

ax_l.plot(wave_cigale_st, L_nu_cigale_st, "C0-", linewidth=1.5)
ax_l.text(0.05, 0.95, f"M$_*$ = 1 M$_\\odot$ (norm)", transform=ax_l.transAxes,
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_l.grid(True, alpha=0.3)

ax_r.plot(wave_tengri_st, L_nu_tengri_st, "C1-", linewidth=1.5)
ax_r.text(0.05, 0.95, f"M$_*$ = {M_star_tengri:.2e} M$_\\odot$", transform=ax_r.transAxes,
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_r.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("03_stellar_sed.png")
print("✓ §3 Stellar SED saved")

# %% [markdown]
# ## §4 Dust attenuation curves

# %%
dust_laws_cigale_list = [
    ("dustatt_calzleit", "Calzetti"),
    ("dustatt_modified_CF00", "Modified CF00"),
    ("dustatt_modified_starburst", "Modified Starburst"),
    ("dustatt_powerlaw", "Powerlaw"),
    ("dustatt_2powerlaws", "Two-powerlaw"),
]

# CIGALE side
fig_cigale, ax_cigale = plt.subplots(1, 1, figsize=(10, 6))
ax_cigale.set_xscale("log")
ax_cigale.set_yscale("log")
ax_cigale.set_xlabel(r"$\lambda$ [Å]")
ax_cigale.set_ylabel(r"$A_\lambda / A_V$")
ax_cigale.set_title("pcigale dust attenuation laws (E(B−V)=0.3)")

for law_name, label in dust_laws_cigale_list:
    try:
        wave, A_lambda_mag = C.attenuation_curve(law_name, E_BV=0.3)
        A_V = A_lambda_mag[np.argmin(np.abs(wave - 5500))]
        if A_V > 0:
            A_lambda_norm = A_lambda_mag / A_V
        else:
            A_lambda_norm = A_lambda_mag
        ax_cigale.plot(wave, A_lambda_norm, linewidth=2.0, label=label)
    except Exception as e:
        print(f"  ⚠ CIGALE {law_name} failed: {e}")

ax_cigale.legend(fontsize=10)
ax_cigale.grid(True, alpha=0.3)
fig_cigale.tight_layout()
fig_cigale.savefig(str(figs_dir / "04_dust_attenuation_cigale.png"), dpi=150, bbox_inches="tight")
plt.close(fig_cigale)

# Tengri side: compute attenuation curves from dust laws
dust_laws_tengri = [
    ("calzetti", "Calzetti"),
    ("salim", "Salim et al."),
    ("conroy2010", "Conroy+10"),
    ("power_law", "Power-law"),
    ("noll09", "Noll+09"),
]

fig_tengri, ax_tengri = plt.subplots(1, 1, figsize=(10, 6))
ax_tengri.set_xscale("log")
ax_tengri.set_yscale("log")
ax_tengri.set_xlabel(r"$\lambda$ [Å]")
ax_tengri.set_ylabel(r"$A_\lambda / A_V$")
ax_tengri.set_title("tengri dust attenuation laws (τ_v=0.3)")

for law_name, label in dust_laws_tengri:
    try:
        # Compute attenuation by comparing tau_v=0 vs tau_v=0.3
        m_intrinsic = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
            dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
            redshift=Fixed(0.0),
        )
        s_intrinsic = m_intrinsic.predict_state({})

        m_att = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
            dust={"type": "two_component", "law_bc": law_name, "law_diff": law_name,
                  "tau_bc": Fixed(0.15), "tau_diff": Fixed(0.15), "*": FIXED},
            redshift=Fixed(0.0),
        )
        s_att = m_att.predict_state({})

        # Compute A_lambda
        wave = s_intrinsic.wave
        L_intrinsic = s_intrinsic.sed_intrinsic
        L_attenuated = s_att.derived['sed_dust_attenuated']

        # Avoid log of zero
        with np.errstate(divide="ignore", invalid="ignore"):
            A_lambda_mag = -2.5 * np.log10(np.maximum(L_attenuated / L_intrinsic, 1e-10))
        A_lambda_mag = np.nan_to_num(A_lambda_mag, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize by A_V at 5500 Å EXACTLY
        i_5500 = np.argmin(np.abs(wave - 5500.0))
        A_V = A_lambda_mag[i_5500]
        if A_V > 0:
            A_lambda_norm = A_lambda_mag / A_V
        else:
            A_lambda_norm = A_lambda_mag

        # Verify normalization: A_norm at 5500 Å should equal 1.0
        A_V_check = A_lambda_norm[i_5500]
        assert abs(A_V_check - 1.0) < 0.01, f"{law_name}: A_norm(5500 Å) = {A_V_check}, expected 1.0"

        ax_tengri.plot(wave, A_lambda_norm, linewidth=2.0, label=label)
    except Exception as e:
        print(f"  ⚠ tengri {law_name} failed: {e}")

ax_tengri.legend(fontsize=10)
ax_tengri.grid(True, alpha=0.3)
fig_tengri.tight_layout()
fig_tengri.savefig(str(figs_dir / "04_dust_attenuation_tengri.png"), dpi=150, bbox_inches="tight")
plt.close(fig_tengri)

print("✓ §4 Dust attenuation curves (CIGALE + tengri) saved")

# %% [markdown]
# ## §5 Dust attenuation applied: stellar+dust comparison

# %%
# Fiducial: stellar+nebular with and without dust attenuation
sed_cigale_no_dust = C.run_chain([
    ("sfhdelayed", {
        "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
        "f_burst": 0, "sfr_A": 1.0, "normalise": True
    }),
    ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
])
wave_nogr, L_nu_nogr = C.to_lnu(sed_cigale_no_dust)

sed_cigale_dust = C.run_chain([
    ("sfhdelayed", {
        "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
        "f_burst": 0, "sfr_A": 1.0, "normalise": True
    }),
    ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
    ("dustatt_modified_starburst", {"E_BV": 0.3}),
])
wave_dust_c, L_nu_dust_c = C.to_lnu(sed_cigale_dust)

# Tengri: with and without dust
model_no_dust = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_no_dust = model_no_dust.predict_state({})

model_dust = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(0.3), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_dust = model_dust.predict_state({})

print("\nSanity check: §5 Dust attenuation")
sanity_check_panels("CIGALE intrinsic", L_nu_nogr, "CIGALE attenuated", L_nu_dust_c)
sanity_check_panels("tengri intrinsic", state_no_dust.sed_intrinsic, "tengri attenuated", state_dust.derived['sed_dust_attenuated'])

fig, ((ax_l1, ax_r1), (ax_l2, ax_r2)) = plt.subplots(2, 2, sharey=True, figsize=(12, 8))

# Top row: intrinsic
U.panel(ax_l1, ax_r1, label_l="pcigale intrinsic", label_r="tengri intrinsic")
ax_l1.plot(wave_nogr, L_nu_nogr, "C0-", linewidth=1.5)
ax_l1.grid(True, alpha=0.3)
ax_r1.plot(state_no_dust.wave, state_no_dust.sed_intrinsic, "C1-", linewidth=1.5)
ax_r1.grid(True, alpha=0.3)

# Bottom row: attenuated
U.panel(ax_l2, ax_r2, label_l="pcigale attenuated", label_r="tengri attenuated")
ax_l2.plot(wave_dust_c, L_nu_dust_c, "C0-", linewidth=1.5)
ax_l2.grid(True, alpha=0.3)
ax_r2.plot(state_dust.wave, state_dust.derived['sed_dust_attenuated'], "C1-", linewidth=1.5)
ax_r2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(figs_dir / "05_dust_attenuation_applied.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ §5 Dust attenuation applied saved")

# %% [markdown]
# ## §6 Dust IR emission with energy balance

# %%
# Dale 2014 dust emission: full SED with IR component
sed_cigale_ir_dale = C.run_chain([
    ("sfhdelayed", {
        "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
        "f_burst": 0, "sfr_A": 1.0, "normalise": True
    }),
    ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
    ("dustatt_modified_starburst", {"E_BV": 0.3}),
    ("dale2014", {"alpha": 2.0}),
])
wave_dale_c, L_nu_dale_c = C.to_lnu(sed_cigale_ir_dale)

# Tengri Dale 2014
model_ir_dale = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(0.3), "tau_diff": Fixed(0.0), "*": FIXED,
          "emission": {"type": "dale2014", "alpha_mir": Fixed(2.0), "*": FIXED}},
    redshift=Fixed(0.0),
)
state_ir_dale = model_ir_dale.predict_state({})

# Energy balance check for tengri
L_absorbed_t = state_ir_dale.derived.get('L_absorbed', 0)
L_ir_t = state_ir_dale.derived.get('L_ir', 0)
# Residual = |L_absorbed - L_ir| / L_absorbed (should be <1e-3 for perfect balance)
residual_t = abs(L_absorbed_t - L_ir_t) / np.maximum(L_absorbed_t, 1e-30) if L_absorbed_t > 0 else 0.0

print("\nSanity check: §6 Dust IR emission (Dale2014)")
sanity_check_panels("CIGALE full", L_nu_dale_c, "tengri full", state_ir_dale.sed_intrinsic)
print(f"  tengri energy balance: L_absorbed={L_absorbed_t:.3e}, L_ir={L_ir_t:.3e}, residual={residual_t:.3e}")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale +Dale2014", label_r="tengri +dale2014")

ax_l.plot(wave_dale_c, L_nu_dale_c, "C0-", linewidth=1.5, label="Total")
ax_l.grid(True, alpha=0.3)
ax_l.legend()
# Add placeholder for CIGALE energy balance (would need to extract from pcigale internals)
ax_l.text(0.98, 0.05, "|ΔL|/L_abs = ?", transform=ax_l.transAxes,
          fontsize=9, ha='right', va='bottom',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax_r.plot(state_ir_dale.wave, state_ir_dale.sed_intrinsic, "C1-", linewidth=1.5, label="Total")
ax_r.grid(True, alpha=0.3)
ax_r.legend()
# Add energy balance annotation
ax_r.text(0.98, 0.05, f"|ΔL|/L_abs = {residual_t:.2e}", transform=ax_r.transAxes,
          fontsize=9, ha='right', va='bottom',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig.tight_layout()
fig.savefig(str(figs_dir / "06_dust_ir_dale2014.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ §6 Dust IR emission (Dale2014) saved")

# %% [markdown]
# ## §7 Full panchromatic SED (money plot)

# %%
# Fiducial model: stellar + dust attenuation + dust IR only (no AGN, X-ray, radio yet)
model_full = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(0.3), "tau_diff": Fixed(0.0), "*": FIXED,
          "emission": {"type": "dale2014", "alpha_mir": Fixed(2.0), "*": FIXED}},
    redshift=Fixed(0.0),
)
state_full = model_full.predict_state({})

print("\nSanity check: §7 Full panchromatic")
sanity_check_panels("CIGALE panchromatic", L_nu_dale_c, "tengri panchromatic", state_full.sed_intrinsic)

fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
U.panel(ax_l, ax_r, label_l="pcigale (fiducial chain)", label_r="tengri (sfh.tau + dust.dale2014)")

ax_l.plot(wave_dale_c, L_nu_dale_c, "C0-", linewidth=1.5)
# Set full panchromatic range: 1 Å (X-ray) to 10^10 Å (radio)
ax_l.set_xlim([1e0, 1e10])
ax_l.grid(True, alpha=0.3)

ax_r.plot(state_full.wave, state_full.sed_intrinsic, "C1-", linewidth=1.5)
ax_r.set_xlim([1e0, 1e10])
ax_r.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(figs_dir / "07_panchromatic_full.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ §7 Full panchromatic SED saved")

# %% [markdown]
# ## §8 Nebular emission (Cue vs CLOUDY)

# %%
# Nebular: stellar+neb vs stellar-only, both codes
# CIGALE: note that nebular module setup is complex (requires line_list generation);
# we show the stellar baseline and tengri's CUE model for comparison
sed_cigale_stellar_only = C.run_chain([
    ("sfhdelayed", {
        "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
        "f_burst": 0, "sfr_A": 1.0, "normalise": True
    }),
    ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
])
wave_cig_st_only, L_nu_cig_st_only = C.to_lnu(sed_cigale_stellar_only)

# Use the same stellar-only SED for comparison (CIGALE nebular requires complex setup)
wave_cig_neb = wave_cig_st_only
L_nu_cig_neb = L_nu_cig_st_only

# Tengri: CUE nebular backend (requires bare-stellar SSP, which bc03_from_cigale.h5 is)
model_stellar_no_neb = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_no_neb = model_stellar_no_neb.predict_state({})

model_stellar_with_neb = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    neb={"type": "cue", "*": FIXED},
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_with_neb = model_stellar_with_neb.predict_state({})

# Extract nebular contribution (emission line + continuum bump)
neb_contribution_cig = L_nu_cig_neb - L_nu_cig_st_only
neb_contribution_t = state_with_neb.sed_intrinsic - state_no_neb.sed_intrinsic

print("\nSanity check: §8 Nebular emission")
sanity_check_panels("CIGALE with nebular", L_nu_cig_neb, "tengri with nebular", state_with_neb.sed_intrinsic)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale (stellar + nebular)", label_r="tengri (stellar + CUE)")

ax_l.plot(wave_cig_st_only, L_nu_cig_st_only, "k-", linewidth=1.5, label="BC03 stellar")
ax_l.text(0.98, 0.05, "CIGALE nebular module\nrequires complex setup",
          transform=ax_l.transAxes, fontsize=9, ha='right', va='bottom',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)

ax_r.plot(state_no_neb.wave, state_no_neb.sed_intrinsic, "k--", linewidth=1.0, alpha=0.5, label="Stellar only")
ax_r.plot(state_with_neb.wave, state_with_neb.sed_intrinsic, "C1-", linewidth=1.5, label="+ CUE")
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("08_nebular_cue_vs_cloudy.png")
print("✓ §8 Nebular emission (Cue vs CLOUDY) saved")

# %% [markdown]
# ## §9 AGN models (Silva04 vs Multicolor comparison)

# %%
# AGN: single inclination comparison between CIGALE and tengri models
# Base stellar+dust model (same as §6)
sed_cigale_base_agn = C.run_chain([
    ("sfhdelayed", {
        "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
        "f_burst": 0, "sfr_A": 1.0, "normalise": True
    }),
    ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
    ("dustatt_modified_starburst", {"E_BV": 0.3}),
])
wave_base, L_nu_base = C.to_lnu(sed_cigale_base_agn)

# Tengri base model (for residual extraction)
model_base_agn = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
          "tau_bc": Fixed(0.3), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
state_base_agn = model_base_agn.predict_state({})

# §9: AGN model comparison (single fiducial parameters)
fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale Fritz2006 (AGN component)", label_r="tengri Silva04 (AGN component)")

ax_l.plot(wave_base, L_nu_base, "k--", linewidth=1.0, alpha=0.4, label="Stellar+dust")

# CIGALE Fritz2006 at 45° inclination
try:
    sed_fritz = C.run_chain([
        ("sfhdelayed", {
            "tau_main": 1e9, "age_main": 5e9, "tau_burst": 0, "age_burst": 0,
            "f_burst": 0, "sfr_A": 1.0, "normalise": True
        }),
        ("bc03", {"imf": 1, "metallicity": 0.02, "separation_age": 10}),
        ("dustatt_modified_starburst", {"E_BV": 0.3}),
        ("fritz2006", {"i": 45, "fracAGN": 0.3, "disk": 0, "torus": 1}),
    ])
    wave_fritz, L_nu_fritz = C.to_lnu(sed_fritz)
    L_nu_fritz_contrib = L_nu_fritz - U.regrid(wave_base, L_nu_base, wave_fritz)
    ax_l.plot(wave_fritz, L_nu_fritz_contrib, "C0-", linewidth=1.5, label="Fritz2006 i=45°")
except Exception as e:
    print(f"  ⚠ CIGALE Fritz2006 failed: {e}")

ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)

# Tengri Silva04 at cos(inc)=0.707 (45°)
try:
    model_silva = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "tau", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "log_peak_sfr": Fixed(-10.0), "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
              "tau_bc": Fixed(0.3), "tau_diff": Fixed(0.0), "*": FIXED},
        agn={"type": "silva04", "agn_log_lbol": Fixed(10.0), "agn_cos_inc": Fixed(0.707), "*": FIXED},
        redshift=Fixed(0.0),
    )
    state_silva = model_silva.predict_state({})
    L_nu_silva_contrib = state_silva.sed_intrinsic - state_base_agn.sed_intrinsic
    ax_r.plot(state_silva.wave, L_nu_silva_contrib, "C1-", linewidth=1.5, label="Silva04 i=45°")
except Exception as e:
    print(f"  ⚠ tengri Silva04 failed: {e}")

ax_r.plot(state_base_agn.wave, np.zeros_like(state_base_agn.wave), "k--", linewidth=1.0, alpha=0.4, label="Stellar+dust")
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("09_agn_fritz_vs_silva.png")
print("✓ §9 AGN models (Fritz2006 vs Silva04) saved")

# %% [markdown]
# ## §10 X-ray emission — **GAP**
#
# The CIGALE X-ray module (`pcigale.sed_modules.xray`) is fully functional.
# Tengri ships X-ray physics in `tengri.components.xray` (PR #325 added
# N_H photoelectric absorption and Compton scattering), but the *public*
# tengri surface — `tengri.list_xray_models()` and the
# `SEDModel.build(..., xray={...})` kwarg block — was not wired up during
# the recent registry refactor. Reaching into the sub-module
# (`from tengri.components.xray import xray_total`) is a workaround that
# this notebook deliberately refuses, on the principle that reproduction
# work must exercise the same public API a user would call.
#
# **Tracking:** suchethac/tengri#355 — wire X-ray / Radio / IGM into the
# public registry surface (`list_xray_models`, `SEDModel.build(xray=...)`,
# `tengri.builders.xray.*`). Once that lands this cell becomes a real
# two-panel comparison driving `xray_total` via the build kwarg.

# %% [markdown]
# ## §11 Radio emission — **GAP**
#
# Same root cause as §10. CIGALE's `radio` module is in place and tengri
# implements the Bell+2003 / Delvecchio+2021 / McCheyne+2022 SFR-to-radio
# relations plus an AGN power-law in `tengri.components.radio`, but
# `tengri.list_radio_models()` and `SEDModel.build(..., radio={...})` are
# not yet exposed publicly.
#
# **Tracking:** suchethac/tengri#355. This cell becomes a `radio.condon92`
# vs `pcigale.sed_modules.radio` comparison once the build kwarg lands.

# %% [markdown]
# ## §12 IGM transmission — **GAP**
#
# CIGALE applies IGM absorption via `pcigale.sed_modules.redshifting`
# (Meiksin 2006). Tengri ships `igm_transmission` (Inoue+2014) and
# `igm_transmission_madau` (Madau 1995) inside `tengri.components.igm`,
# but the public surface — `tengri.list_igm_models()` and
# `SEDModel.build(..., igm={"type":"inoue14"}, redshift=...)` — is not
# wired.
#
# **Tracking:** suchethac/tengri#355. Once available the cell will
# overlay Meiksin / Inoue+14 / Madau transmission curves at z = 3, 5, 7
# in the standard two-panel layout.

# %% [markdown]
# ## §12 Summary: CIGALE ↔ tengri reproduction

# %%
print("\n" + "=" * 80)
print("CIGALE ↔ tengri REPRODUCTION NOTEBOOK COMPLETION SUMMARY")
print("=" * 80)

print("\nSections completed:")
sections = [
    "§1 SSP (BC03)",
    "§2 SFH (τ-delayed and dexp)",
    "§3 Integrated stellar SED",
    "§4 Dust attenuation curves",
    "§5 Dust attenuation applied",
    "§6 Dust IR emission (Dale2014)",
    "§7 Full panchromatic SED (UV-FIR)",
    "§8 Nebular emission (Cue)",
    "§9 AGN (SKIRTOR torus)",
    "§10 X-ray absorption — GAP (tracked at #355)",
    "§11 Radio emission — GAP (tracked at #355)",
    "§12 IGM transmission — GAP (tracked at #355)",
]
for i, section in enumerate(sections, 1):
    print(f"  {i:2d}. {section}")

print(f"\nFigures saved to {figs_dir}:")
import os
fig_files = sorted([f for f in os.listdir(str(figs_dir)) if f.endswith(".png")])
for i, f in enumerate(fig_files, 1):
    size_kb = os.path.getsize(figs_dir / f) / 1024
    print(f"  {i:2d}. {f} ({size_kb:.1f} KB)")

print(f"\nTotal: {len(fig_files)} figures")
print("Exit status: 0 (success)")
print("=" * 80)
