"""Reproduce the emission/U panel from the H&D 2023 tutorial.

Plots :math:`\\lambda I_\\lambda / N_H / U` per H atom for several
:math:`\\log_{10} U` values via tengri's
:class:`DustEmissionSEDComponent` with ``template="astrodust"``.
The U-divided form makes the strong U-dependence of the PAH-vs-FIR
ratio visible.

Reference
---------
* Notebook: brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tengri.components.dust.astrodust_hd23 import load_astrodust_hd23_or_raise

HDF5 = "data/astrodust_templates.h5"
OUT = Path("examples/astrodust_hd23/fig_emission_vs_lgU.png")


def main() -> None:
    tpl = load_astrodust_hd23_or_raise(HDF5)
    wave_um = np.asarray(tpl.wavelength_um)
    lgU = np.asarray(tpl.lgU)
    L_nu_total = np.asarray(tpl.L_nu_total)  # (91, 1000), erg/s/Hz/H

    # Convert L_nu (erg/s/Hz/H) back to lambda*I_lambda (erg/s/sr/H)
    # for direct comparison with the H&D tutorial figure.  The
    # conversion is L_nu = 4*pi * lambda^2 * I_lambda / c, so
    # lambda*I_lambda = (L_nu * c) / (4*pi*lambda).
    c_cgs = 2.99792458e10
    lam_cm = wave_um * 1.0e-4
    lam_I_lam = L_nu_total * c_cgs / (4.0 * np.pi * lam_cm[None, :])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
    ax.set_ylabel(
        r"$\lambda I_\lambda / N_{\rm H} / U\ "
        r"[\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
        fontsize=11,
    )
    # Match the notebook's explicit clipping (cell 24): wavelength
    # range 2-1000 μm and y-range 1e-28 to 5e-25.  Beyond these
    # bounds the high-U curves drop steeply because the FIR
    # continuum has shifted to the MIR.
    ax.set_xlim(2.0, 1000.0)
    ax.set_ylim(1.0e-28, 5.0e-25)

    # Notebook's lgU sweep (cell 24): np.arange(-3, 6, 1.15) -> 8 values.
    cmap = plt.get_cmap("viridis")
    targets = np.arange(-3.0, 6.0, 1.15)
    for k, tg in enumerate(targets):
        i = int(np.argmin(np.abs(lgU - tg)))
        U = 10.0 ** lgU[i]
        ax.plot(
            wave_um,
            lam_I_lam[i] / U,
            color=cmap(k / max(1, len(targets) - 1)),
            lw=1.4,
            label=rf"$\log_{{10}} U={lgU[i]:+.2f}$",
        )
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=8)
    ax.set_title(
        "Astrodust+PAH emission per H per U  (HD23 fig 8 in tutorial)",
        fontsize=11,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
