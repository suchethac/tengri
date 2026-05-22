"""
Composable AGN: per-block contribution breakdown
=================================================

Single recipe (all-GRAHSP), but each pipeline stage rendered independently
on top of the disc continuum. Demonstrates how the five blocks
(``disc → lines → feii → torus → attenuation``) contribute to the total
SED — useful for understanding which knob controls which feature.

Each panel highlights one block; the others are switched to ``"none"`` to
isolate the contribution. The dashed grey curve is the all-blocks-on
reference.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.agn.blocks import composable_agn_l_nu
from tengri.plot import setup_style

setup_style()

wave_aa = jnp.logspace(np.log10(500.0), np.log10(1.0e6), 1500)
wave_um = np.asarray(wave_aa) / 1e4
C_AA_PER_S = 2.99792458e18


def nu_l_nu(kw):
    """Return νL_ν in erg/s, NaN-masking non-positive values for log plotting."""
    l_nu = np.asarray(composable_agn_l_nu(wave_aa, agn_log_lbol=12.0, **kw))
    out = l_nu * C_AA_PER_S / np.asarray(wave_aa)
    return np.where(out > 0, out, np.nan)


# Full all-GRAHSP recipe (reference).
FULL = dict(
    agn_disc_block="grahsp_sbpl",
    agn_lines_block="grahsp",
    agn_feii_block="grahsp",
    agn_torus_block="grahsp",
    agn_attenuation_block="grahsp_biatten",
    agn_grahsp_a_feii=5.0,
    agn_grahsp_a_lines=1.0,
    agn_grahsp_fcov=0.4,
    agn_grahsp_ebv=0.3,
    agn_grahsp_ebv_agn=0.1,
)
full_sed = nu_l_nu(FULL)

# Per-stage isolations: turn everything off except the named stage.
# The disc panel always keeps the disc on (it's the normalisation anchor).
STAGES = [
    ("disc only", "grahsp_sbpl-disc", {"agn_disc_block": "grahsp_sbpl"}),
    (
        "+ lines",
        "grahsp-lines",
        {"agn_disc_block": "grahsp_sbpl", "agn_lines_block": "grahsp"},
    ),
    (
        "+ FeII forest",
        "grahsp-feii",
        {
            "agn_disc_block": "grahsp_sbpl",
            "agn_lines_block": "grahsp",
            "agn_feii_block": "grahsp",
        },
    ),
    (
        "+ torus",
        "grahsp-torus",
        {
            "agn_disc_block": "grahsp_sbpl",
            "agn_lines_block": "grahsp",
            "agn_feii_block": "grahsp",
            "agn_torus_block": "grahsp",
        },
    ),
    (
        "+ attenuation",
        "grahsp-biatten",
        {
            "agn_disc_block": "grahsp_sbpl",
            "agn_lines_block": "grahsp",
            "agn_feii_block": "grahsp",
            "agn_torus_block": "grahsp",
            "agn_attenuation_block": "grahsp_biatten",
        },
    ),
]

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(STAGES)))

for ax, (title, _block, kw), color in zip(axes, STAGES, colors):
    full_kw = dict(FULL)
    # Override only the selectors that this panel turns on; everything else
    # is forced to "none" so the cumulative effect is visible left-to-right.
    panel_kw = {
        **{
            k: "none"
            for k in (
                "agn_disc_block",
                "agn_lines_block",
                "agn_feii_block",
                "agn_torus_block",
                "agn_attenuation_block",
            )
        },
        **kw,
    }
    # Free params come from FULL but we strip the selector kwargs we just set.
    free_params = {
        k: v
        for k, v in full_kw.items()
        if k
        not in (
            "agn_disc_block",
            "agn_lines_block",
            "agn_feii_block",
            "agn_torus_block",
            "agn_attenuation_block",
        )
    }
    panel_sed = nu_l_nu({**panel_kw, **free_params})

    ax.loglog(wave_um, full_sed, lw=1.0, color="0.6", ls="--", label="full")
    ax.loglog(wave_um, panel_sed, lw=2.0, color=color, label=title)
    ax.set_xlim(5e-3, 1e2)
    ax.set_ylim(1e42, 1e47)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower center", fontsize=8, frameon=False)

axes[0].set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
fig.suptitle(
    "Composable AGN: cumulative block contributions (all-GRAHSP recipe)",
    fontsize=11,
)
fig.tight_layout()
plt.savefig("plot_composable_block_toggles.png", dpi=150, bbox_inches="tight")
plt.show()
