"""
The IRX–β relation emerges from the dust model
==============================================

The infrared excess (IRX = L_IR / L_FUV) versus UV-continuum slope β
diagram (Meurer+1999) is the standard tool for inferring attenuation
in unresolved star-forming galaxies. We mock a population of
star-forming galaxies with a fixed SFH and a range of diffuse dust
optical depths, measure each galaxy's β by fitting a power-law to its
rest-frame UV continuum (1268–2580 Å, Calzetti+1994 windows), and
overplot the empirical Meurer+1999 starburst relation.

References:
- Meurer, Heckman & Calzetti 1999, ApJ, 521, 64
- Calzetti, Kinney & Storchi-Bergmann 1994, ApJ, 429, 582
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


C_AA_PER_S = 2.998e18

# Calzetti+1994 windows for fitting the UV slope: pairs of (lam_lo, lam_hi)
# in Å. β is the slope of f_λ ∝ λ^β fit through the windows.
CAL94_WINDOWS = np.array(
    [
        [1268, 1284],
        [1309, 1316],
        [1342, 1371],
        [1407, 1515],
        [1562, 1583],
        [1677, 1740],
        [1760, 1833],
        [1866, 1890],
        [1930, 1950],
        [2400, 2580],
    ]
)


def _measure_beta(wave_aa: np.ndarray, l_nu: np.ndarray) -> float:
    """Return Calzetti+1994 UV slope β from f_λ ∝ λ^β."""
    f_lam = l_nu * C_AA_PER_S / wave_aa**2  # L_λ ∝ L_ν / λ^2
    mask = np.zeros_like(wave_aa, dtype=bool)
    for lo, hi in CAL94_WINDOWS:
        mask |= (wave_aa >= lo) & (wave_aa <= hi)
    x = np.log10(wave_aa[mask])
    y = np.log10(f_lam[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _bolometric_lir(wave_aa: np.ndarray, l_nu: np.ndarray) -> float:
    """Integrate L_ν from 8–1000 μm to obtain L_IR [erg s⁻¹]."""
    mask = (wave_aa >= 8.0e4) & (wave_aa <= 1.0e7)
    nu = C_AA_PER_S / wave_aa[mask]
    order = np.argsort(nu)
    return float(np.trapz(l_nu[mask][order], nu[order]))


def _lfuv(wave_aa: np.ndarray, l_nu: np.ndarray) -> float:
    """ν L_ν at the FUV (1600 Å) — proxy for unattenuated FUV power."""
    i = int(np.argmin(np.abs(wave_aa - 1600.0)))
    return float(C_AA_PER_S / wave_aa[i] * l_nu[i])


ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 0.5,  # young starburst -> strong UV
        "log_peak_sfr": 1.5,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 4.0),
        "tau_bc": 0.5,
        "slope": -0.7,
        "emission": {"type": "dale2014", "*": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

tau_grid = np.linspace(0.0, 3.0, 18)
beta_arr = np.empty_like(tau_grid)
irx_arr = np.empty_like(tau_grid)

for i, tau in enumerate(tau_grid):
    out = model.predict_rest_sed({**baseline, "dust_tau_diff": jnp.float64(tau)})
    wave = np.asarray(out.wavelength)
    l_nu = np.asarray(out.sed)
    beta_arr[i] = _measure_beta(wave, l_nu)
    irx_arr[i] = _bolometric_lir(wave, l_nu) / _lfuv(wave, l_nu)

# Meurer+1999 starburst relation: log10 IRX = log10(10^(0.4*(4.43+1.99*beta)) - 1)
beta_emp = np.linspace(-2.5, 0.5, 200)
A_FUV_meurer = 4.43 + 1.99 * beta_emp
irx_meurer = 10 ** (0.4 * A_FUV_meurer) - 1.0

fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.plot(beta_emp, irx_meurer, color="0.55", lw=1.0, ls="--", label="Meurer+1999 starburst")
sc = ax.scatter(
    beta_arr,
    irx_arr,
    c=tau_grid,
    cmap="viridis",
    s=42,
    lw=0.5,
    edgecolor="0.2",
    zorder=3,
    label="tengri models",
)
ax.set_yscale("log")
ax.set_xlim(-2.6, 0.7)
ax.set_ylim(1e-2, 2e2)
ax.set_xlabel(r"UV continuum slope $\beta$")
ax.set_ylabel(r"$\mathrm{IRX} \equiv L_{\rm IR}\,/\,L_{\rm FUV}$")
ax.legend(frameon=False, fontsize=9, loc="lower right")
cbar = fig.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label(r"$\tau_{\rm diff}$  [mag]")

fig.tight_layout()
fig.savefig("plot_usecase_uv_slope_beta.png", dpi=150, bbox_inches="tight")
