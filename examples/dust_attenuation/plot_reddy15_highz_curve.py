"""
Reddy+2015 high-redshift attenuation curve
===========================================

Reddy et al. (2015) derived a dust attenuation curve from Balmer decrements
of ``z ~ 1.4–2.6`` star-forming galaxies in the MOSDEF survey. It is shallower
in the UV than the SMC curve but has a lower total-to-selective ratio
(``R_V = 2.505``) than Calzetti's local starburst law (``R_V = 4.05``) — a
combination relevant when fitting rest-UV/optical SEDs of high-z galaxies.
FSPS exposes this curve; tengri provides it as the ``reddy15`` dust law.

This figure overlays ``reddy15`` against three reference curves on a common
``k(λ)`` normalization (``k(5500 Å) = 1``), using the public
``tengri.dust.resolve_dust_law`` accessor — no stellar population or SSP file
is required to inspect an attenuation curve.

References
----------
Reddy, N. A., Kriek, M., Shapley, A. E., et al. 2015, ApJ, 806, 259
("The MOSDEF Survey: ... the Dust Attenuation Curve at Redshifts z ~ 1.4–2.6").
Calzetti, D. et al. 2000, ApJ, 533, 682.
Gordon, K. D. et al. 2003, ApJ, 594, 279 (SMC).
Cardelli, J. A. et al. 1989, ApJ, 345, 245 (MW).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.dust import resolve_dust_law

setup_style()

# Common rest-frame wavelength grid spanning the curves' validity (0.12-2.85 um).
wave = jnp.geomspace(1300.0, 25000.0, 600)
wave_um = np.asarray(wave) / 1e4

CURVES = [
    ("reddy15", "Reddy+2015 (z~2, R_V=2.505)", "#d62728", 2.4),
    ("calzetti", "Calzetti+2000 (starburst)", "#1f77b4", 1.6),
    ("smc", "SMC (Gordon+2003)", "#2ca02c", 1.6),
    ("cardelli", "Cardelli+1989 (MW, R_V=3.1)", "#7f7f7f", 1.4),
]

fig, ax = plt.subplots(figsize=(6.6, 4.3))
for name, label, color, lw in CURVES:
    law = resolve_dust_law(name)
    k = np.asarray(law(wave))
    ax.plot(wave_um, k, color=color, lw=lw, label=label)

ax.axhline(1.0, color="0.7", ls=":", lw=0.9)
ax.axvline(0.55, color="0.7", ls=":", lw=0.9)
ax.text(0.56, 0.12, "V band", color="0.45", fontsize=8)

ax.set(
    xscale="log",
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mu$m]",
    ylabel=r"$k(\lambda)\;/\;k(5500\,\mathrm{\AA})$",
    title="Dust attenuation curves: Reddy+2015 vs local references",
    xlim=(0.13, 2.6),
    ylim=(0, 6),
)
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.show()
