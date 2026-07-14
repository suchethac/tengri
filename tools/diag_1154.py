"""#1154 probe: why does the Cue fast path disagree with the exact forward on linux?

Runs identically on linux/x86 (CI) and macOS/arm64 (local), so the two outputs can be
diffed. Answers three questions the CI assertion cannot:

  Q1. Is the ionizing-spectrum table itself platform-dependent?
      It is FIT with scipy curve_fit and refit whenever the disk cache is cold. If the
      fits land in different places on linux, everything downstream of Cue moves.

  Q2. Is the fast-vs-exact error present AT a grid node, or only BETWEEN nodes?
      At a node the interpolation is exact by construction, so a nonzero error there
      indicts the grid BUILD; a zero error there indicts the INTERPOLATION (i.e. the
      grid is too coarse for how the function actually bends).

  Q3. Which lines carry the error, and are they strong enough for the ratio to mean
      anything? A 15% error on a line at 1e-4 x Halpha is a noise floor, not a bug.

Disposable. Delete with the branch.
"""

import os
import platform
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"
# Force a REFIT: the whole question is whether the scipy fits are platform-stable, and
# a warm disk cache would hand us a table someone else fit and hide exactly that.
os.environ["TENGRI_JAX_CACHE_DIR"] = tempfile.mkdtemp(prefix="diag1154_")

import warnings

import jax
import numpy as np
import scipy

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, SEDModel, Uniform, WavePrecomp, load_ssp_data
from tengri.components.nebular import ionizing_spectrum as ion
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.photometry_config import Photometry

BARE = "data/fsps_prsc_miles_chabrier.h5"
BANDS = ["des_g", "des_r", "des_z", "wise_w1", "wise_w2"]
LINES = ["OII_3726", "Hbeta", "OIII_5007", "Halpha", "SII_6717"]
Z = 0.1
TRUTH = {
    "sfh_dpl_log_total_mass": 10.5, "sfh_dpl_age_gyr": 12.0, "sfh_dpl_tau_gyr": 11.0,
    "sfh_dpl_alpha": 1.0, "sfh_dpl_beta": 3.0, "met_logzsol": 0.0,
    "dust_tau_bc": 1.0, "dust_tau_diff": 0.4, "neb_logU": -2.8, "neb_logZ_gas": -0.2,
}
LO_U, HI_U = -4.0, -1.0
LO_Z, HI_Z = -1.5, 0.3

print("=" * 78)
print(f"platform : {platform.platform()}  machine={platform.machine()}")
print(f"numpy {np.__version__}  scipy {scipy.__version__}  jax {jax.__version__}")
print("=" * 78)


def line_waves():
    cat = LineList.default_optical()
    return np.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in LINES])


WAVES = line_waves()
LD = LineFluxData(
    names=tuple(LINES),
    fluxes=np.ones(len(LINES)),
    errors=np.ones(len(LINES)),
    wavelengths=WAVES,
)


def build(ssp, approx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(BANDS), line_fluxes=LD
            ),
            sfh={"type": "dpl", "*": FREE},
            stellar={"met_logzsol": Uniform(-1.5, 0.3)},
            dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED,
                  "tau_bc": Uniform(0.0, 4.0), "tau_diff": Uniform(0.0, 3.0)},
            neb={"type": "cue", "*": FIXED,
                 "logU": Uniform(LO_U, HI_U), "logZ_gas": Uniform(LO_Z, HI_Z)},
            redshift=Fixed(Z),
            approx=approx,
        )


def params(m, **over):
    p = {k: float(v) for k, v in m.spec.get_fixed_values().items()}
    for k in m.spec.free_params:
        p[k] = float(TRUTH[k])
    p.update(over)
    return p


ssp = load_ssp_data(BARE)

# ── Q1. Is the fitted ionizing-spectrum table platform-stable? ───────────────
tab = ion.precompute_ionizing_params_table(ssp.ssp_wave, ssp.ssp_flux, ssp.ssp_lgmet)
ionspec, logq = np.asarray(tab["ionspec_table"]), np.asarray(tab["logqion_table"])
fin = np.isfinite(ionspec)
print("\nQ1  IONIZING-SPECTRUM TABLE (freshly refit by scipy on THIS platform)")
print(f"    ionspec_table  shape={ionspec.shape}  finite={fin.sum()}/{fin.size}")
print(f"      sum   = {np.nansum(ionspec):.10e}")
print(f"      mean  = {np.nanmean(ionspec):.10e}   std = {np.nanstd(ionspec):.10e}")
print(f"      min   = {np.nanmin(ionspec):.10e}   max = {np.nanmax(ionspec):.10e}")
lq = logq[np.isfinite(logq) & (logq > -98)]
print(f"    logqion_table  live={lq.size}/{logq.size}")
print(f"      sum   = {lq.sum():.10e}")
print(f"      min   = {lq.min():.10e}   max = {lq.max():.10e}")
# a few concrete coefficients — a diff here IS the answer to #1154
print("    ionspec_table[0, 40, :] =", np.array2string(ionspec[0, 40, :], precision=9))
print("    ionspec_table[7, 20, :] =", np.array2string(ionspec[7, 20, :], precision=9))

# ── Q2/Q3. fast vs exact, at TRUTH and AT AN EXACT GRID NODE ─────────────────
N_GRID = 8
u_nodes = np.linspace(LO_U, HI_U, N_GRID)
z_nodes = np.linspace(LO_Z, HI_Z, N_GRID)
print(f"\n    grid nodes (n_grid={N_GRID})")
print(f"      logU    : {np.array2string(u_nodes, precision=4)}")
print(f"      logZ_gas: {np.array2string(z_nodes, precision=4)}")
print(f"      TRUTH logU={TRUTH['neb_logU']} logZ_gas={TRUTH['neb_logZ_gas']}  <- BETWEEN nodes")

m_ex = build(ssp, None)
m_fa = build(ssp, WavePrecomp())
m_fa.enable_fast_nebular(WAVES, n_grid=N_GRID)


def compare(tag, **over):
    pe = np.asarray(m_ex.predict_line_fluxes(params(m_ex, **over), target_wavelengths=WAVES))
    pf = np.asarray(m_fa.predict_line_fluxes(params(m_fa, **over), target_wavelengths=WAVES))
    rel = np.abs(pf - pe) / (np.abs(pe) + 1e-300)
    ha = abs(pe[LINES.index("Halpha")]) + 1e-300
    print(f"\n{tag}")
    print(f"    {'line':>11} {'exact':>14} {'fast':>14} {'rel':>9} {'/Halpha':>9}")
    for i, n in enumerate(LINES):
        print(f"    {n:>11} {pe[i]:14.6e} {pf[i]:14.6e} {rel[i]:9.2%} {abs(pe[i]) / ha:9.2e}")
    strong = np.abs(pe) > 1e-2 * ha
    print(f"    worst overall      = {rel.max():9.2%}")
    print(f"    worst on strong    = {rel[strong].max() if strong.any() else float('nan'):9.2%}"
          "   (lines >1% of Halpha)")
    return rel.max()


r_truth = compare("Q3  AT TRUTH (logU=-2.8, logZ_gas=-0.2) — BETWEEN grid nodes")
r_node = compare(
    f"Q2  AT AN EXACT GRID NODE (logU={u_nodes[3]:.6f}, logZ_gas={z_nodes[5]:.6f})",
    neb_logU=float(u_nodes[3]),
    neb_logZ_gas=float(z_nodes[5]),
)

print("\n" + "=" * 78)
print("VERDICT")
print(f"  between nodes : {r_truth:.2%}")
print(f"  at a node     : {r_node:.2%}")
if r_node < 0.01 <= r_truth:
    print("  -> grid BUILD is fine; the INTERPOLATION is failing. n_grid is too coarse")
    print("     for how l(logU, logZ_gas) actually bends on this platform.")
elif r_node >= 0.01:
    print("  -> the error survives AT A NODE, where interpolation is exact by")
    print("     construction. The grid BUILD (or Q_H) is wrong, not the interpolation.")
else:
    print("  -> both clean on this platform; compare against the other platform's run.")
print("=" * 78)
