"""Switch between dust IR templates with one config-field change.

Demonstrates that
:class:`~tengri.components.dust.emission_component.DustEmissionSEDComponent`
unifies modified-blackbody, Draine+2021 PAHspec, and Hensley & Draine
2023 Astrodust+PAH behind a single ``config.template`` knob.

Plots the energy-balance-rescaled :math:`L_\\nu` for each template
side-by-side at a fixed absorbed luminosity ``L_ir = 1e44 erg/s``,
showing how the spectral shape changes while the bolometric output
is conserved.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.components.dust.emission_component import (
    DustEmissionSEDComponent,
    DustEmissionSEDComponentConfig,
)
from tengri.core.component import PipelineState

OUT = Path("examples/astrodust_hd23/fig_template_swap.png")


def _eval(comp: DustEmissionSEDComponent, wave_aa: jnp.ndarray, params: dict) -> np.ndarray:
    state = comp.precompute(wave_grid=wave_aa)
    pipeline = PipelineState(
        wave=wave_aa,
        sed_intrinsic=None,
        derived={"L_ir": 1.0e44},
    )
    out = comp.apply(pipeline, params, precomputed=state)
    return np.asarray(out.sed_intrinsic)


def main() -> None:
    wave_aa = jnp.asarray(np.geomspace(1.0e4, 1.0e7, 1000))  # 1 to 1000 um
    wave_um = np.asarray(wave_aa) / 1.0e4

    # 1. Modified blackbody (analytic; T=30 K, beta=1.8).
    mbb = DustEmissionSEDComponent()
    mbb_sed = _eval(
        mbb,
        wave_aa,
        {"dust_T": 30.0, "dust_beta_ir": 1.8, "redshift": 0.0},
    )

    # 2. Draine+2021 PAHspec (mMMP starlight, std/std, lgU=1).
    d21 = DustEmissionSEDComponent(
        config=DustEmissionSEDComponentConfig(
            template="draine2021_pah",
            pahspec_starlight="mMMP",
            pahspec_template_path="data/pahspec_draine2021.h5",
        ),
    )
    d21_sed = _eval(d21, wave_aa, {"dust_lgU": 1.0, "redshift": 0.0})

    # 3. Hensley & Draine 2023 Astrodust+PAH (lgU=0.2 fiducial).
    ad = DustEmissionSEDComponent(
        config=DustEmissionSEDComponentConfig(
            template="astrodust",
            astrodust_template_path="data/astrodust_templates.h5",
        ),
    )
    ad_sed = _eval(ad, wave_aa, {"dust_lgU": 0.2, "redshift": 0.0})

    # nu*L_nu makes the per-decade power obvious.
    c_aa_per_s = 2.99792458e18
    nu = c_aa_per_s / np.asarray(wave_aa)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=14)
    ax.set_ylabel(r"$\nu L_\nu\ [\mathrm{erg\,s^{-1}}]$", fontsize=14)
    ax.set_xlim(1.0, 1.0e3)

    ax.plot(
        wave_um, nu * mbb_sed, lw=2, color="#1f77b4", label="modified_blackbody (T=30K, beta=1.8)"
    )
    ax.plot(
        wave_um,
        nu * d21_sed,
        lw=2,
        color="#ff7f0e",
        label=r"draine2021_pah (mMMP, $\log_{10}U=1$)",
    )
    ax.plot(
        wave_um, nu * ad_sed, lw=2, color="#2ca02c", label=r"astrodust (HD23, $\log_{10}U=0.2$)"
    )
    ax.legend(loc="lower center", frameon=False, fontsize=10)
    ax.set_title(
        r"Same $L_{\rm ir}=10^{44}\,\mathrm{erg\,s^{-1}}$, three templates",
        fontsize=12,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
