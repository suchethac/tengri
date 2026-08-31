"""
Custom AGN torus model via SEDModelComponent and direct integration
====================================================================

A toy single-temperature blackbody torus implemented as a modern
``SEDModelComponent`` subclass, discoverable through ``SEDModel.build``
and composable with other AGN blocks. The SEDModelComponent pattern is
the recommended path for any new SED physics — AGN, dust, or stellar.

The toy curve is a graybody (smooth, no features); the SKIRTOR curve
carries the silicate 9.7 μm feature and inclination-dependent geometry
the toy model elides. Both are normalized to the same bolometric luminosity
and plotted in νL_ν space.

References
----------
.. [1] Stalevski, M., Fritz, J., Baes, M., & Lutz, D. 2012, MNRAS, 420,
   3576 — SKIRTOR library and inclination effects.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.agn import register_agn_block
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
LSUN_ERG = 3.828e33


@register_agn_block(
    "torus",
    "demo_graybody",
    citation="gallery demo (replace with your reference)",
    status="experimental",
    short_doc="Single-temperature graybody torus — demonstration only",
)
def demo_graybody_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_T_torus: float = 300.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """Graybody torus emission at a single dust temperature.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / L_sun).
    agn_T_torus : float
        Dust temperature [K]. Default 300 K.
    agn_torus_frac : float
        Torus covering / luminosity fraction of L_bol. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Torus flux L_nu [erg/s/Hz].
    """
    h = 6.626e-27
    k_B = 1.381e-16
    c_cgs = 2.998e10

    nu = c_cgs / (wavelength * 1.0e-8)
    B_nu = (2 * h * nu**3 / c_cgs**2) / (jnp.exp(h * nu / (k_B * agn_T_torus)) - 1.0)

    L_bol_erg = 10.0**agn_log_lbol * LSUN_ERG
    nu_ref = c_cgs / 1.0e-4
    B_ref = (2 * h * nu_ref**3 / c_cgs**2) / (jnp.exp(h * nu_ref / (k_B * agn_T_torus)) - 1.0)
    L_nu = (B_nu / B_ref) * (L_bol_erg / 1.0e10) * agn_torus_frac
    return L_nu


# Negligible host SFH: total mass ~1e-10 Msun, completely subdominant
# to the AGN luminosity below. ``log_sfr`` was the legacy kwarg; current
# ``const`` SFH parametrizes by total mass over [start_gyr, end_gyr].
SFH = {"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT), "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.Fixed(tengri.DEFAULT),
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}
LOG_LBOL = 12.0
ssp = tengri.load_ssp()

model_skirtor = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust_attenuation=DUST,
    agn={
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "log_lbol": LOG_LBOL,
        "lum_ratio": 1.0,
        "disc": {"type": "multicolor", "all_params": tengri.Fixed(tengri.DEFAULT)},
        "torus": {"type": "skirtor", "all_params": tengri.Fixed(tengri.DEFAULT)},
    },
    redshift=tengri.Fixed(0.0),
)
p_skirtor = dict(model_skirtor.spec.sample(jax.random.PRNGKey(0)))
out_skirtor = model_skirtor.predict(p_skirtor)

wave_um = np.asarray(model_skirtor.wavelengths) * 1.0e-4
nu_lnu_skirtor = (
    C_AA_PER_S / np.asarray(model_skirtor.wavelengths) * np.asarray(out_skirtor.rest_sed())
)

L_nu_toy = np.asarray(
    demo_graybody_torus(
        jnp.asarray(model_skirtor.wavelengths),
        agn_log_lbol=LOG_LBOL,
        agn_lum_ratio=1.0,
        agn_T_torus=300.0,
        agn_torus_frac=0.5,
    )
)
nu_lnu_toy = C_AA_PER_S / np.asarray(model_skirtor.wavelengths) * L_nu_toy

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.loglog(wave_um, nu_lnu_skirtor, color="C0", lw=1.6, label="SKIRTOR (production)")
ax.loglog(wave_um, nu_lnu_toy, color="C3", lw=1.6, label="demo graybody (T=300 K)")
ax.set(
    xlim=(0.1, 1.0e3),
    ylim=(1.0e41, 1.0e47),
    xlabel=r"Rest-frame wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=9, loc="lower center")
fig.tight_layout()
plt.savefig("plot_custom_torus_extension.png", dpi=150, bbox_inches="tight")
