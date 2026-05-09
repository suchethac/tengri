"""Extinction, polarized extinction, and albedo — H&D 2023 fiducial.

Reproduces three panels from the model_file_tutorial.ipynb that
characterise the wavelength dependence of dust opacity:

* Total extinction :math:`\\tau_\\lambda / N_H` decomposed into
  Astrodust and PAH contributions.
* Polarized extinction :math:`(p_\\lambda/N_H)^{\\rm max}` from
  Astrodust grains (PAHs are unaligned).
* Albedo :math:`\\tau^{\\rm sca}_\\lambda / \\tau^{\\rm ext}_\\lambda`
  for both compositions.

Reference
---------
* Notebook: brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HDF5 = Path("data/astrodust_templates.h5")
OUT = Path("examples/astrodust_hd23/fig_extinction_and_scattering.png")


def main() -> None:
    with h5py.File(HDF5, "r") as f:
        ext = np.asarray(f["extinction"])  # (1000, 4)
        scatt = np.asarray(f["scattering"])  # (1000, 4)
        extpol = np.asarray(f["polarized_extinction"])  # (1000, 2)

    wave_um = ext[:, 0]
    tau_Ad = ext[:, 1]
    tau_PAH = ext[:, 2]
    tau_total = ext[:, 3]
    sca_Ad = scatt[:, 1]
    sca_PAH = scatt[:, 2]
    pol_Ad_max = extpol[:, 1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

    # ── extinction ─────────────────────────────────────────────────
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=11)
    ax1.set_ylabel(r"$\tau_\lambda/N_{\rm H}\ [\mathrm{cm}^2\,\mathrm{H}^{-1}]$", fontsize=11)
    ax1.set_xlim(0.1, 40.0)
    ax1.set_ylim(5.0e-25, 3.0e-21)
    ax1.plot(wave_um, tau_Ad, color="#e41a1c", ls="--", label="Astrodust")
    ax1.plot(wave_um, tau_PAH, color="#0868ac", ls="--", label="PAHs")
    ax1.plot(wave_um, tau_total, color="k", lw=1.5, label="Total", zorder=0)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax1.set_title("Extinction (HDU 2)", fontsize=10)

    # ── polarized extinction ───────────────────────────────────────
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=11)
    ax2.set_ylabel(
        r"$(p_\lambda/N_{\rm H})^{\rm max}\ [\mathrm{cm}^2\,\mathrm{H}^{-1}]$",
        fontsize=11,
    )
    ax2.set_xlim(0.1, 40.0)
    ax2.set_ylim(5.0e-25, 3.0e-23)
    ax2.plot(wave_um, pol_Ad_max, color="k", lw=1.5)
    ax2.set_title("Polarized extinction (Astrodust, HDU 4)", fontsize=10)

    # ── albedo (notebook fig 6 uses 1/lambda in 1/μm as x-axis) ────
    ax3.set_xscale("linear")
    ax3.set_yscale("linear")
    ax3.set_xlabel(r"$\lambda^{-1}\ [\mu\mathrm{m}^{-1}]$", fontsize=11)
    ax3.set_ylabel("Albedo  $\\omega$", fontsize=11)
    ax3.set_xlim(0.0, 8.0)
    ax3.set_ylim(0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        albedo_Ad = np.where(tau_Ad > 0, sca_Ad / tau_Ad, np.nan)
        albedo_PAH = np.where(tau_PAH > 0, sca_PAH / tau_PAH, np.nan)
        sca_total = scatt[:, 3]
        albedo_total = np.where(tau_total > 0, sca_total / tau_total, np.nan)
    inv_lam = 1.0 / wave_um
    ax3.plot(inv_lam, albedo_Ad, color="#e41a1c", lw=1.5, label="Astrodust")
    ax3.plot(inv_lam, albedo_PAH, color="#0868ac", lw=1.5, label="PAHs")
    ax3.plot(inv_lam, albedo_total, color="k", lw=1.2, label="Total", zorder=0)
    ax3.legend(loc="upper left", frameon=False, fontsize=9)
    ax3.set_title("Albedo (HDU 3 / HDU 2)", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
