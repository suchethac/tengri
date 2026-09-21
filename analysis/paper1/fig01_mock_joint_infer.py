# SPDX-License-Identifier: BSD-3-Clause
"""Kitchen-sink mock joint spectro-photometric inference figure (fig:multiwavelength).

Builds ``figures/fig01_mock_joint_infer.pdf`` from the mock written by
``fig_mock_joint_infer.py`` and the posterior written by ``fit_mock_joint.py``.

Kept separate from the generator on purpose: that script is what regenerates
the truth file, and it runs on another machine while this one is edited. A
figure that imports the generator cannot break the generator.

Two frame facts this figure depends on, both measured rather than assumed
(:func:`observed_components` asserts the second on every run):

1. ``predict(params).sed.components["wavelength"]`` is **rest-frame**. Plotting
   the components against it as if it were observed wavelength shifts every
   feature by ``1 + z`` -- at this mock's z=1, a factor of two -- and produces
   a figure that looks entirely reasonable.
2. The components are luminosity densities, not observed flux. The conversion
   is one constant, and deriving it by interpolating the total at each filter's
   pivot wavelength does NOT recover it: that spread across this mock's sixteen
   bands is a factor of 36, because broadband photometry is a bandpass integral
   and not a point evaluation of the SED. Derived from the spectrum instead,
   which samples the SED directly, the constant is exact -- max/min 1.0000 over
   all 1500 pixels.

Without a posterior on disk the figure is still produced, from truth alone, but
under a DIFFERENT filename and with a banner. An incomplete paper figure that
carries the final name is one ``\\includegraphics`` away from being published as
the real thing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tengri

from ._posterior_gate import posterior_gate
from ._posterior_utils import posterior_output_paths
from .fig_mock_joint_infer import TRUTH_NPZ
from .verify_mock_listing import (
    REDSHIFT,
    SSP_NAME,
    build_joint_observation,
    build_mock_model,
)

FIG_DIR = Path(__file__).parent / "figures"
#: The posterior this figure reads, derived from the SAME helper the fit uses to
#: write it (``_posterior_utils.posterior_output_paths``) rather than retyped.
#: The two used to disagree -- the fit wrote ``mock_joint_mcmc_nuts.npz`` and
#: this script looked for ``mock_joint_nuts.npz`` -- and because a missing
#: posterior is a legitimate state, nothing raised: the truth-only figure was
#: rendered and the posterior reported NOT FOUND, which looks exactly like not
#: having run the fit.
POSTERIOR_METHOD = "mcmc_nuts"
POSTERIOR_NPZ, _POSTERIOR_JSON = posterior_output_paths(
    Path(__file__).parent / "results", POSTERIOR_METHOD
)

#: Detection threshold the mock applied; censored bands plot as limits at this level.
DETECTION_SIGMA = 3.0

#: Component key -> (label, color, linestyle). Order is draw order.
COMPONENT_STYLE = {
    "sed_attenuated": ("Stellar (attenuated)", "#1f77b4", "-"),
    "sed_nebular": ("Nebular", "#2ca02c", "-"),
    "sed_agn_disc": ("AGN disc (QSOgen)", "#d62728", "--"),
    "sed_agn_torus": ("AGN torus (SKIRTOR)", "#ff7f0e", "--"),
    "sed_agn_lines": ("AGN BLR + NLR", "#9467bd", ":"),
    "sed_dust_ir": ("Dust IR (Draine-Li)", "#8c564b", "-"),
    "sed_radio": ("Radio (SF + AGN)", "#7f7f7f", "-."),
    "sed_xray": ("X-ray (yang20)", "#17becf", "-"),
}


def pivot_wavelength(filter_curve) -> float:
    r"""Pivot wavelength [A], :math:`\sqrt{\int T\lambda\,d\lambda / \int T/\lambda\,d\lambda}`."""
    wave = np.asarray(filter_curve.wave)
    trans = np.asarray(filter_curve.trans)
    return float(np.sqrt(np.trapezoid(trans * wave, wave) / np.trapezoid(trans / wave, wave)))


def observed_components(model, params, truth, z=REDSHIFT):
    """Components as observed flux density, on an observed-frame wavelength axis.

    The scale factor is measured against the mock's own noiseless spectrum
    rather than assumed from a cosmology, so this cannot disagree with the data
    it is plotted against. Its constancy is the test that the rest-frame
    reading of the components grid is right: on the observed-frame reading the
    same ratio spans a factor of 776.
    """
    pred = model.predict(params)
    comp = pred.sed.components
    wave_rest = np.asarray(comp["wavelength"])
    total_rest = np.asarray(comp["sed_total"])

    spec_wave_obs = np.asarray(truth["wave_obs"])
    spec_true = np.asarray(truth["spec_true"])
    ratio = spec_true / np.interp(spec_wave_obs / (1.0 + z), wave_rest, total_rest)
    scale = float(np.median(ratio))
    spread = float(ratio.max() / ratio.min())
    if not (0.999 < spread < 1.001):
        raise ValueError(
            f"luminosity-to-flux ratio spans {spread:.4f} across the spectrum, so it is "
            "not one constant. Either the components grid is not rest-frame, or the "
            "spectrum and the components are no longer the same model."
        )

    return (
        wave_rest * (1.0 + z),
        {key: np.asarray(comp[key]) * scale for key in COMPONENT_STYLE if key in comp},
        np.asarray(comp["sed_total"]) * scale,
    )


def load_posterior(path: Path):
    """Posterior draws, or ``None`` when the fit has not been run yet."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as handle:
        return {k: np.asarray(handle[k]) for k in handle.files}


def plot_sed(ax, ax_res, model, truth, params, obs):
    """Decomposed SED, the photometry it is fit to, and the spectrum."""
    wave_obs, comps, total = observed_components(model, params, truth)
    nu_f_nu = lambda w, f: f * (2.99792458e18 / w)  # noqa: E731  -- c[A/s]/lambda * f_nu

    for key, (label, color, ls) in COMPONENT_STYLE.items():
        if key not in comps:
            continue
        flux = comps[key]
        if not np.any(flux > 0):
            continue  # a component that is off contributes no line and no legend entry
        ax.plot(
            wave_obs, nu_f_nu(wave_obs, flux), ls, color=color, lw=1.1, alpha=0.85, label=label
        )
    ax.plot(wave_obs, nu_f_nu(wave_obs, total), "-", color="k", lw=1.8, label="Total", zorder=5)

    # Spectroscopic channel, on the same axes as the photometry it is fit with.
    spec_w = np.asarray(truth["wave_obs"])
    spec_o = np.asarray(truth["spec_obs"])
    ax.plot(
        spec_w,
        nu_f_nu(spec_w, spec_o),
        "-",
        color="#e377c2",
        lw=0.6,
        alpha=0.9,
        label="Spectrum (observed)",
        zorder=4,
    )

    piv = np.array([pivot_wavelength(f) for f in obs.photometry.filters])
    flux = np.asarray(truth["phot_obs"])
    sig = np.asarray(truth["phot_sig"])
    det = np.asarray(truth["detected"], dtype=bool)

    # The band-integrated model, in the same space as the data. Without it the
    # X-ray points read as a bad fit: a broadband point is an integral over the
    # bandpass, not the SED's value at the pivot, and where the SED is steep
    # across a band the two differ a lot -- on this mock by up to a factor of 36.
    # The residual panel below says those same bands sit within 1 sigma.
    model_phot_plot = np.asarray(model.predict_photometry(params))
    ax.plot(
        piv,
        nu_f_nu(piv, model_phot_plot),
        "s",
        ms=7,
        mfc="none",
        mec="#d62728",
        mew=1.2,
        zorder=7,
        label="Model, band-integrated",
    )
    ax.errorbar(
        piv[det],
        nu_f_nu(piv[det], flux[det]),
        yerr=nu_f_nu(piv[det], sig[det]),
        fmt="o",
        ms=5,
        color="k",
        mfc="white",
        mew=1.3,
        capsize=2,
        lw=1.2,
        zorder=6,
        label="Photometry",
    )
    if (~det).any():
        limit = DETECTION_SIGMA * sig[~det]
        ax.errorbar(
            piv[~det],
            nu_f_nu(piv[~det], limit),
            yerr=nu_f_nu(piv[~det], limit) * 0.45,
            uplims=True,
            fmt="v",
            ms=6,
            color="k",
            mfc="0.6",
            lw=1.2,
            zorder=6,
            label=rf"{DETECTION_SIGMA:.0f}$\sigma$ upper limit",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\nu f_\nu$  [erg s$^{-1}$ cm$^{-2}$]")
    finite = nu_f_nu(wave_obs, total)
    finite = finite[finite > 0]
    ax.set_ylim(finite.max() * 3e-9, finite.max() * 6)
    ax.set_xlim(1.0, 1e8)
    ax.legend(
        ncol=3,
        fontsize=6.2,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.8",
        borderpad=0.5,
        columnspacing=1.1,
        handlelength=1.8,
    )
    ax.set_title(
        f"Mock Type 1 AGN + star-forming host at $z={REDSHIFT:g}$, "
        f"{int(det.sum())}/{det.size} bands detected",
        fontsize=9,
    )

    model_phot = np.asarray(model.predict_photometry(params))
    resid = (flux - model_phot) / sig
    ax_res.axhline(0.0, color="k", lw=0.8)
    for band in (1, 2):
        ax_res.axhspan(-band, band, color="0.85" if band == 2 else "0.7", zorder=0, lw=0)
    ax_res.plot(piv[det], resid[det], "o", ms=4, color="k", mfc="white", mew=1.1)
    if (~det).any():
        ax_res.plot(piv[~det], resid[~det], "v", ms=5, color="0.5")
    ax_res.set_xscale("log")
    ax_res.set_xlim(1.0, 1e8)
    ax_res.set_ylim(-4.2, 4.2)
    ax_res.set_xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
    ax_res.set_ylabel(r"$\chi$")


def plot_sfh(ax, model, params):
    """Star formation history on the model's own SFH grid."""
    derived = model.predict_state(params).derived
    lbt_gyr = np.asarray(derived["sfh_grid_lbt_yr"]) / 1e9
    sfr = np.asarray(derived["sfr_history"])
    ax.plot(lbt_gyr, sfr, "-", color="#1f77b4", lw=1.6, label="Truth")
    ax.set_xscale("log")
    ax.set_xlabel("Lookback time  [Gyr]")
    ax.set_ylabel(r"SFR  [$M_\odot$ yr$^{-1}$]")
    ax.set_title("Star formation history", fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_marginals(ax, posterior, truth_values, free_names):
    """Posterior marginals for the parameters the section makes claims about."""
    wanted = [
        ("sfh_cont_log_total_mass", r"$\log M_\star$"),
        ("agn_log_lbol", r"$\log L_{\rm bol}$"),
        ("xray_log_nh", r"$\log N_{\rm H}$"),
        ("dust_tau_diff", r"$\tau_{\rm diff}$"),
        ("met_logzsol", r"$\log Z/Z_\odot$"),
        ("neb_logU", r"$\log U$"),
    ]
    present = [(k, lab) for k, lab in wanted if k in posterior]
    if not present:
        ax.text(
            0.5,
            0.5,
            "no overlapping parameters in the posterior",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_axis_off()
        return
    offsets = []
    labels = []
    for key, label in present:
        draws = np.asarray(posterior[key]).ravel()
        idx = free_names.index(key)
        t = float(truth_values[idx])
        sigma = float(np.std(draws))
        offsets.append((float(np.median(draws)) - t) / sigma if sigma > 0 else np.nan)
        labels.append(label)
    y = np.arange(len(labels))
    ax.axvspan(-1, 1, color="0.85", lw=0, zorder=0)
    ax.axvline(0.0, color="k", lw=0.9)
    ax.plot(offsets, y, "o", ms=6, color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"(median $-$ truth) / posterior $\sigma$")
    ax.set_xlim(-3.2, 3.2)
    ax.set_title("Recovery, in units of the posterior width", fontsize=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=POSTERIOR_NPZ)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the figure into; defaults to this script's figures/",
    )
    args = parser.parse_args()
    fig_dir = args.out_dir or FIG_DIR

    if not TRUTH_NPZ.exists():
        print(f"no mock at {TRUTH_NPZ}; run `python -m paper1.fig_mock_joint_infer` first")
        return 1
    with np.load(TRUTH_NPZ, allow_pickle=True) as handle:
        truth = {k: handle[k] for k in handle.files}
    free_names = [str(x) for x in truth["free_params"]]
    truth_values = [float(v) for v in truth["truth_values"]]
    params = dict(zip(free_names, truth_values, strict=True))

    ssp = tengri.load_ssp(SSP_NAME)
    obs = build_joint_observation()
    model = build_mock_model(ssp, obs)
    print(f"model rebuilt at D = {len(model.spec.free_params)}")

    posterior = load_posterior(args.posterior)
    have_post = posterior is not None
    print(f"posterior: {'loaded from ' + str(args.posterior) if have_post else 'NOT FOUND'}")

    gate_passed, gate_reasons = False, ["no posterior on disk"]
    if have_post:
        gate_passed, gate_reasons, _ = posterior_gate(args.posterior)
        print("gate: PASSED" if gate_passed else "gate: FAILED -- " + "; ".join(gate_reasons))

    if not have_post:
        # Distinguish "not fitted yet" from "fitted, and this script is looking
        # in the wrong place". Both render the truth-only figure, but only one
        # of them is a defect, and the quiet version costs whatever the fit cost.
        others = sorted(args.posterior.parent.glob("mock_joint_*.npz"))
        others = [q for q in others if q.name not in {"mock_joint_truth.npz", args.posterior.name}]
        if others:
            print(
                f"  but {len(others)} other mock_joint_*.npz exist: "
                f"{[q.name for q in others]}\n"
                f"  If one of those is a sampler posterior, this script is reading the "
                f"wrong name and the figure below is truth-only by accident. Pass it "
                f"explicitly with --posterior."
            )

    fig = plt.figure(figsize=(7.4, 8.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3.0, 0.75, 1.9], hspace=0.38, wspace=0.30)
    ax_sed = fig.add_subplot(gs[0, :])
    ax_res = fig.add_subplot(gs[1, :], sharex=ax_sed)
    ax_sfh = fig.add_subplot(gs[2, 0])
    ax_mar = fig.add_subplot(gs[2, 1])

    plot_sed(ax_sed, ax_res, model, truth, params, obs)
    plot_sfh(ax_sfh, model, params)
    if have_post:
        plot_marginals(ax_mar, posterior, truth_values, free_names)
        if not gate_passed:
            fig.text(
                0.5,
                0.985,
                "PROVISIONAL -- posterior has not converged: " + "; ".join(gate_reasons),
                ha="center",
                fontsize=7.5,
                color="#b22222",
                weight="bold",
            )
    else:
        ax_mar.text(
            0.5,
            0.5,
            "awaiting NUTS posterior",
            ha="center",
            va="center",
            transform=ax_mar.transAxes,
            fontsize=9,
            color="0.4",
        )
        ax_mar.set_axis_off()
        fig.text(
            0.5,
            0.985,
            "INCOMPLETE: truth only, no posterior",
            ha="center",
            fontsize=10,
            color="#b22222",
            weight="bold",
        )

    fig_dir.mkdir(parents=True, exist_ok=True)
    if not have_post:
        name = "fig01_mock_joint_infer_truthonly.pdf"
    elif not gate_passed:
        name = "fig01_mock_joint_infer_provisional.pdf"
    else:
        name = "fig01_mock_joint_infer.pdf"
    out = fig_dir / name
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    if not gate_passed:
        print(f"NOTE: written as {name}. Do not wire this into the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
