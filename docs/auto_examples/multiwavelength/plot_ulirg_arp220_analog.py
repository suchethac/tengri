"""
Arp 220 analog: panchromatic SED of a heavily obscured ULIRG
==============================================================

An archetype obscured starburst built up from the tengri component
toolkit: a 100 Myr ongoing burst with ``τ_diff = 2 mag`` of diffuse
dust and a 1 mag birth-cloud opacity. The Dale+2014 IR template
re-emits the absorbed UV/optical power into a 60 μm peak; the
Condon-92 radio extends to the GHz with the right FIR–radio ratio;
the SDSS r-band sits ~3 magnitudes below the FIR peak because almost
all the UV/optical has been reprocessed.

Anchor parameters chosen to land at ``L_IR ≈ 10^{12.3} L_sun``,
similar to Arp 220's bolometric output.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18

model = tengri.SEDModel.build(
    tengri.load_ssp(),
    sfh={"type": "const", "*": tengri.FIXED, "log_sfr": 2.3,  # ~200 Msun/yr
         "start_gyr": 0.1, "end_gyr": 0.0},                   # 100 Myr burst
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 2.0, "tau_bc": 1.0,
          "emission": {"type": "dale2014", "*": tengri.FIXED}},
    radio={"type": "condon92", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.018),                # Arp 220 redshift
)
p = dict(model.spec.sample(jax.random.PRNGKey(0)))
out = model.predict_rest_sed(p)
wave = np.asarray(out.wavelength)
nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.loglog(wave, nu_l_nu, color="0.15", lw=1.2)

# Annotate the bolometric components
for x, name, dy in [
    (3000,   "UV (attenuated)",      0.7),
    (6000,   "stellar continuum",    0.9),
    (1.5e5,  "warm dust",            0.85),
    (8e5,    "FIR peak (Dale+14)",   0.7),
    (3e7,    "free-free knee",       0.8),
    (3e8,    "synchrotron (Condon-92)", 0.9),
]:
    ax.text(x, 5e46 * dy, name, fontsize=7, color="0.45", ha="center")

ax.set(xlim=(1e3, 3e9), ylim=(1e40, 8e46),
       xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
       ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]")

# L_IR(8-1000 μm) for context
ir = (wave > 8e4) & (wave < 1e7)
nu_ir = C_AA_PER_S / wave[ir]
order = np.argsort(nu_ir)
L_ir = np.trapezoid(np.asarray(out.sed)[ir][order], nu_ir[order])
ax.text(0.97, 0.05,
        rf"$L_{{\rm IR}}^{{(8-1000\,\mu\mathrm{{m}})}} \approx 10^{{{np.log10(L_ir/3.84e33):.1f}}}\,L_\odot$",
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

fig.tight_layout()
plt.savefig("plot_ulirg_arp220_analog.png", dpi=150, bbox_inches="tight")
