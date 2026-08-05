# SPDX-License-Identifier: BSD-3-Clause
"""Truth-and-posterior contours for the shared PSD block (sigma, tau).

Re-evaluates the two-step shared posterior on a ``(sigma, tau)`` grid at several
population sizes and contours it against the injected truth. No refitting: the
interim draws are read from an existing bank, so the whole figure is a
re-weighting and takes ~90 s.

Two reporting choices here are deliberate, because the obvious alternatives
would each overstate what was measured:

* **The joint-NUTS result is drawn as 68% MARGINAL error bars, not a contour.**
  That run stored only interval summaries; rendering a 2D ellipse from two
  marginals would invent a correlation it never measured.
* **A posterior that rails into a grid corner is annotated, not just plotted.**
  Its contour degenerates to a sliver at the axis limit and reads as a missing
  curve rather than as the failure it is.

Run::

  PYTHONPATH=<worktree>/src:. JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_contours.py --bank psd_bank_fixed \\
      --ns 4,16,32,64 --out psd_contours.png

``PYTHONPATH`` is mandatory: the editable install resolves ``tengri`` to
whichever checkout its ``.pth`` names, which may lack
``tengri.inference.population``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from tengri.inference.population.estimator import SharedGrid, shared_log_posterior
from tengri.inference.population.reconstruct import centered_fields

# Joint NUTS (N=4, D=98) 68% marginals and median, transcribed from
# docs/dev/hierarchical-psd-preliminary-results.md 5.2. Hard-coded because that
# 5.7 h run wrote only a summary; regenerate with hierarchical_psd_joint_fit.py
# --out and update both places together if it is ever rerun.
JOINT = {
    "sigma": (0.497, 0.725, 0.897),
    "tau_myr": (104.5, 184.7, 429.7),
}
# Prior 68% under the same interim bounds. The width ratio against these is the
# figure's actual claim: truth-coverage alone does not separate "learned" from
# "returned the prior".
PRIOR = {
    "sigma": (0.168, 0.842),
    "tau_myr": (88.4, 421.6),
}


def load_bank(bank_dir, n_max):
    """Load contiguous galaxies ``0..n_max-1`` from a bank directory.

    Stops at the first missing index rather than skipping it: the bank is a
    prefix of a keyed stream, so ``0..M-1`` is the M-galaxy population only if
    none is missing.

    Parameters
    ----------
    bank_dir : str
        Directory holding ``bank_meta.json`` and ``gal_*.npz``.
    n_max : int
        Stop after this many galaxies [count].

    Returns
    -------
    meta : dict
        Bank metadata, including truth values and interim prior bounds.
    xi : ndarray, shape (N, K, n)
        Interim latent draws [dimensionless].
    sigma : ndarray, shape (N, K)
        Interim ``sigma`` draws [dex].
    tau_myr : ndarray, shape (N, K)
        Interim ``tau`` draws [Myr].
    """
    with open(os.path.join(bank_dir, "bank_meta.json")) as fh:
        meta = json.load(fh)
    xi, sig, tau = [], [], []
    for i in range(n_max):
        path = os.path.join(bank_dir, f"gal_{i:04d}.npz")
        if not os.path.exists(path):
            break
        with np.load(path) as d:
            xi.append(d["xi"])
            sig.append(d["sigma"])
            tau.append(d["tau_myr"])
    if not xi:
        raise SystemExit(f"No gal_*.npz found in {bank_dir}")
    return meta, np.stack(xi), np.stack(sig), np.stack(tau)


def posterior_mass(log_posterior, grid):
    """Normalized posterior mass per grid node.

    Parameters
    ----------
    log_posterior : array_like, shape (A * B,)
        Unnormalized log-posterior [nats], C-ordered so node ``a * B + b`` is
        ``(sigma[a], tau_yr[b])``.
    grid : SharedGrid
        Quadrature grid.

    Returns
    -------
    mass : ndarray, shape (A, B)
        Posterior mass summing to 1 [dimensionless].

    Notes
    -----
    The max is subtracted before exponentiating: ``log_posterior`` sums one term
    per galaxy and reaches the thousands at large N, where a bare ``np.exp``
    underflows every node to zero and the normalization returns NaN.
    """
    n_sigma = len(np.asarray(grid.sigma))
    n_tau = len(np.asarray(grid.tau_yr))
    lp = np.asarray(log_posterior, dtype=float).reshape(n_sigma, n_tau)
    m = np.exp(lp - lp.max())
    return m / m.sum()


def hpd_levels(mass, levels=(0.68, 0.95)):
    """Contour values enclosing the given highest-posterior-density fractions.

    Parameters
    ----------
    mass : ndarray, shape (A, B)
        Normalized posterior mass [dimensionless].
    levels : tuple of float, optional
        Enclosed-mass fractions [dimensionless].

    Returns
    -------
    values : list of float
        Ascending contour levels, deduplicated. Fewer entries than ``levels``
        when a railed posterior puts several fractions inside one node.
    """
    flat = np.sort(mass.ravel())[::-1]
    csum = np.cumsum(flat)
    out = []
    for lv in levels:
        idx = min(int(np.searchsorted(csum, lv)), flat.size - 1)
        out.append(flat[idx])
    return sorted(set(out))


def edge_mask(mass):
    """Boolean mask of the grid boundary.

    Parameters
    ----------
    mass : ndarray, shape (A, B)
        Posterior mass, used only for its shape.

    Returns
    -------
    mask : ndarray of bool, shape (A, B)
        True on the boundary. A mask, not a sum of four slices: the slice form
        counts each corner twice and can report an edge fraction above 1 when
        the posterior rails into a corner.
    """
    m = np.zeros_like(mass, dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ns", default="4,16,32,64")
    ap.add_argument("--n-sigma", type=int, default=60)
    ap.add_argument("--n-tau", type=int, default=60)
    ap.add_argument("--node-chunk", type=int, default=4)
    ap.add_argument("--out", default="psd_contours.png")
    args = ap.parse_args()

    n_values = [int(x) for x in args.ns.split(",")]
    meta, xi, sig, tau_myr = load_bank(args.bank, max(n_values))
    log_age = np.asarray(meta["log_age_grid"])
    times_yr = 10.0**log_age
    t_sigma = float(meta["truth_sigma"])
    t_tau = float(meta["truth_tau_myr"])

    grid = SharedGrid.uniform(
        tau_prior="uniform",
        sigma_bounds=tuple(meta["interim_sigma_bounds"]),
        tau_bounds_yr=(
            meta["interim_tau_bounds_myr"][0] * 1e6,
            meta["interim_tau_bounds_myr"][1] * 1e6,
        ),
        n_sigma=args.n_sigma,
        n_tau=args.n_tau,
    )
    s_nodes = np.asarray(grid.sigma)
    t_nodes = np.asarray(grid.tau_yr) / 1e6

    # Same field-scale rejection the scaling driver uses.
    sigma_max = float(meta["interim_sigma_bounds"][1])
    ceiling = 5.0 * sigma_max * np.log(10.0)
    allf = centered_fields(xi, sig, tau_myr * 1e6, log_age)
    keep = np.asarray(np.std(np.asarray(allf), axis=(1, 2))) <= ceiling
    if not keep.all():
        print(f"dropped {int((~keep).sum())} galaxies with implausible field scale")
        xi, sig, tau_myr = xi[keep], sig[keep], tau_myr[keep]

    results = {}
    for n in n_values:
        if n > xi.shape[0]:
            print(f"skip N={n}: only {xi.shape[0]} galaxies available")
            continue
        fields = centered_fields(xi[:n], sig[:n], tau_myr[:n] * 1e6, log_age)
        lp, _ = shared_log_posterior(
            fields, times_yr, grid, method="b2", node_chunk=args.node_chunk
        )
        mass = posterior_mass(np.asarray(lp), grid)
        results[n] = mass
        ms, mt = mass.sum(axis=1), mass.sum(axis=0)
        print(
            f"N={n:>4}  sigma_mode={s_nodes[ms.argmax()]:.3f}  "
            f"tau_mode={t_nodes[mt.argmax()]:.1f} Myr  "
            f"edge_mass={float(mass[edge_mask(mass)].sum()):.3f}"
        )

    np.savez(
        os.path.splitext(args.out)[0] + "_data.npz",
        sigma_nodes=s_nodes,
        tau_nodes_myr=t_nodes,
        truth_sigma=t_sigma,
        truth_tau_myr=t_tau,
        **{f"mass_N{n}": m for n, m in results.items()},
    )
    _plot(results, s_nodes, t_nodes, t_sigma, t_tau, meta, args.out)


def _plot(results, s_nodes, t_nodes, t_sigma, t_tau, meta, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    S, T = np.meshgrid(s_nodes, t_nodes, indexing="ij")
    colors = plt.cm.viridis(np.linspace(0.12, 0.86, len(results)))

    railed = []
    for (n, mass), c in zip(sorted(results.items()), colors):
        lv = hpd_levels(mass, (0.95, 0.68))
        ax.contour(S, T, mass, levels=lv, colors=[c], linewidths=[1.3, 2.1][: len(lv)])
        ax.contourf(S, T, mass, levels=[lv[-1], mass.max()], colors=[c], alpha=0.16)
        if float(mass[edge_mask(mass)].sum()) > 0.5:
            railed.append((n, c))

    # A contour pinned to the grid corner shows only a sliver and reads as a
    # missing curve. Say what it is instead.
    for n, c in railed:
        ax.annotate(
            f"N = {n}: railed into the\ngrid corner (mode 1.00, 500 Myr)",
            xy=(s_nodes[-1], t_nodes[-1]),
            xytext=(0.30, 0.62),
            textcoords="axes fraction",
            fontsize=9,
            color=c,
            fontweight="bold",
            ha="left",
            arrowprops=dict(arrowstyle="->", color=c, lw=1.6, connectionstyle="arc3,rad=-0.25"),
        )

    # Joint NUTS 68% marginals as a cross - NOT a contour (see module docstring).
    js, jt = JOINT["sigma"], JOINT["tau_myr"]
    ax.errorbar(
        js[1],
        jt[1],
        xerr=[[js[1] - js[0]], [js[2] - js[1]]],
        yerr=[[jt[1] - jt[0]], [jt[2] - jt[1]]],
        fmt="o",
        ms=7,
        color="crimson",
        ecolor="crimson",
        elinewidth=2.0,
        capsize=4,
        zorder=6,
    )

    ax.plot(t_sigma, t_tau, marker="*", ms=24, color="gold", mec="black", mew=1.3, zorder=8)
    ax.axvline(t_sigma, color="black", lw=0.7, ls=":", alpha=0.55)
    ax.axhline(t_tau, color="black", lw=0.7, ls=":", alpha=0.55)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$  [dex]")
    ax.set_ylabel(r"$\tau$  [Myr]")
    ax.set_xlim(s_nodes.min(), s_nodes.max())
    ax.set_ylim(t_nodes.min(), t_nodes.max())
    ax.set_title("Shared PSD posterior, two-step estimator\n(68% and 95% HPD)")

    handles = [
        Line2D([], [], color=c, lw=2.1, label=f"N = {n}")
        for (n, _), c in zip(sorted(results.items()), colors)
    ]
    handles += [
        Line2D(
            [],
            [],
            color="crimson",
            marker="o",
            lw=2,
            label="joint NUTS, N=4\n(68% marginals)",
        ),
        Line2D(
            [],
            [],
            color="gold",
            marker="*",
            ms=15,
            mec="black",
            lw=0,
            label=f"truth ({t_sigma:g}, {t_tau:g} Myr)",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.92)

    # ---- right panel: interval widths against the PRIOR ----
    # Support comes from the bank, not from literals: a bank built with
    # different interim bounds would otherwise be normalized against the wrong
    # range and silently misplace every bar.
    rows = [
        (
            r"$\sigma$  [dex]",
            tuple(meta["interim_sigma_bounds"]),
            PRIOR["sigma"],
            (JOINT["sigma"][0], JOINT["sigma"][2]),
            t_sigma,
        ),
        (
            r"$\tau$  [Myr]",
            tuple(meta["interim_tau_bounds_myr"]),
            PRIOR["tau_myr"],
            (JOINT["tau_myr"][0], JOINT["tau_myr"][2]),
            t_tau,
        ),
    ]
    # Each row is normalized onto [0, 1] of its own interim prior support, so
    # sigma [dex] and tau [Myr] are comparable on one axis.
    for i, (label, support, pr, po, truth) in enumerate(rows):
        lo, hi = support
        span = hi - lo
        base = i * 1.6
        pr_lo, pr_hi = (pr[0] - lo) / span, (pr[1] - lo) / span
        po_lo, po_hi = (po[0] - lo) / span, (po[1] - lo) / span
        truth_x = (truth - lo) / span
        ratio = (po[1] - po[0]) / (pr[1] - pr[0])

        axb.barh(
            base + 0.34, pr_hi - pr_lo, left=pr_lo, height=0.42, color="0.80", edgecolor="0.45"
        )
        axb.barh(base - 0.12, po_hi - po_lo, left=po_lo, height=0.42, color="crimson", alpha=0.78)
        axb.plot(
            [truth_x],
            [base + 0.11],
            marker="*",
            ms=21,
            color="gold",
            mec="black",
            mew=1.2,
            zorder=6,
        )
        axb.text(
            (pr_lo + pr_hi) / 2,
            base + 0.34,
            "prior 68%",
            ha="center",
            va="center",
            fontsize=9,
            color="0.2",
        )
        axb.text(
            (po_lo + po_hi) / 2,
            base - 0.12,
            f"posterior  {ratio:.2f}x prior",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        axb.text(0.0, base + 0.80, label, fontsize=11.5, va="bottom")
    axb.set_yticks([])
    axb.set_xlim(-0.02, 1.02)
    axb.set_ylim(-0.75, 2.75)
    axb.set_xlabel("fraction of the interim prior range")
    axb.set_title(
        "Joint NUTS (N=4) against the prior it started from\n"
        r"$\sigma$ is learned; $\tau$ returns the prior"
    )
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    fig.suptitle(
        "Constraining SFH burstiness PSD parameters "
        f"(bank N={meta['n']}, truth $\\sigma$={t_sigma:g} dex, $\\tau$={t_tau:g} Myr)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
