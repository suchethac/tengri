#!/usr/bin/env python3
"""Build precomputed nthcomp warm Comptonization templates for tengri.

Solves the Kompaneets equation (Zdziarski, Johnson & Magdziarz 1996,
MNRAS 283, 193) on a 4-D grid of (gamma, kTe, kTbb, nu) and saves the
normalised spectral shapes to ``data/nthcomp_templates.npz``.

Once built, tengri loads the table at import and uses JAX trilinear
interpolation in place of the per-call Kompaneets solve.  Build time is
~30–120 s depending on grid density and CPU speed.

Usage
-----
    python scripts/build_nthcomp_templates.py

Options
-------
    --output PATH       Output npz file (default: data/nthcomp_templates.npz)
    --n-gamma INT       Number of gamma grid points (default: 11, range 1.5–3.5)
    --n-kte INT         Number of kTe grid points (default: 8, range 0.05–0.5 keV)
    --n-ktbb INT        Number of kTbb grid points (default: 25, log-spaced)
    --n-nu INT          Number of output frequency points (default: 300)

References
----------
Kubota & Done (2018) MNRAS 480 1247 Section 2.2
Zdziarski, Johnson & Magdziarz (1996) MNRAS 283 193
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so we can import the solver
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tengri.models.agn._nthcomp import donthcomp_nu


def build_table(
    n_gamma: int = 11,
    n_kte: int = 8,
    n_ktbb: int = 25,
    n_nu: int = 300,
) -> dict:
    """Compute the nthcomp normalised spectral shape table.

    Parameters
    ----------
    n_gamma : int
        Number of photon index grid points.
    n_kte : int
        Number of electron temperature grid points [keV].
    n_ktbb : int
        Number of seed blackbody temperature grid points [keV].
    n_nu : int
        Number of output frequency grid points.

    Returns
    -------
    dict with keys:
        gamma_grid, kte_grid, ktbb_grid, nu_grid, table
    """
    gamma_grid = np.linspace(1.5, 3.5, n_gamma)
    kte_grid = np.linspace(0.05, 0.5, n_kte)
    ktbb_grid = np.logspace(-5, np.log10(0.3), n_ktbb)
    # Frequency range: IR (1e13 Hz) to ~20 keV soft X-ray (5e18 Hz)
    nu_grid = np.logspace(13.0, np.log10(5e18), n_nu)

    total = n_gamma * n_kte * n_ktbb
    table = np.zeros((n_gamma, n_kte, n_ktbb, n_nu), dtype=np.float32)

    k_boltz_keV = 8.617333262e-8  # keV/K
    h_planck_erg = 6.62607015e-27  # erg·s
    k_boltz_erg = 1.380649e-16  # erg/K

    done = 0
    t0 = time.time()
    for ig, gamma in enumerate(gamma_grid):
        for it, kte in enumerate(kte_grid):
            for ib, ktbb in enumerate(ktbb_grid):
                done += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(
                        f"  {done}/{total}  gamma={gamma:.2f}  kTe={kte:.3f}  "
                        f"kTbb={ktbb:.2e}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                        end="\r",
                        flush=True,
                    )

                # Degenerate: seed temperature >= electron temperature — use blackbody
                if ktbb >= kte:
                    T_K = ktbb / k_boltz_keV
                    x = h_planck_erg * nu_grid / (k_boltz_erg * T_K)
                    x_safe = np.clip(x, 1e-10, 500.0)
                    bnu = nu_grid**3 / (np.exp(x_safe) - 1.0)
                    norm = np.trapezoid(bnu, nu_grid)
                    if norm > 0:
                        table[ig, it, ib] = (bnu / norm).astype(np.float32)
                    continue

                shape = donthcomp_nu(nu_grid, gamma, kte, ktbb)
                norm = np.trapezoid(shape, nu_grid)
                if norm > 0:
                    table[ig, it, ib] = (shape / norm).astype(np.float32)

    print()  # newline after \r progress
    return {
        "gamma_grid": gamma_grid.astype(np.float32),
        "kte_grid": kte_grid.astype(np.float32),
        "ktbb_grid": ktbb_grid.astype(np.float32),
        "nu_grid": nu_grid.astype(np.float32),
        "table": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "data" / "nthcomp_templates.npz"),
        help="Output npz file path",
    )
    parser.add_argument("--n-gamma", type=int, default=11)
    parser.add_argument("--n-kte", type=int, default=8)
    parser.add_argument("--n-ktbb", type=int, default=25)
    parser.add_argument("--n-nu", type=int, default=300)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = (args.n_gamma, args.n_kte, args.n_ktbb, args.n_nu)
    print(
        f"Building nthcomp table: gamma={args.n_gamma} × kTe={args.n_kte} "
        f"× kTbb={args.n_ktbb} × nu={args.n_nu} = {args.n_gamma * args.n_kte * args.n_ktbb} solves"
    )

    result = build_table(*grid)

    np.savez_compressed(out_path, **result)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path}  ({size_mb:.1f} MB)")
    print("Reload tengri to use the new templates.")


if __name__ == "__main__":
    main()
