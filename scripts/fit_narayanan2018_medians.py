#!/usr/bin/env python3
r"""Fit the Kriek & Conroy (2013) curve to Narayanan et al. (2018) median curves.

Produces the ``(delta, bump_strength)`` table that
``tengri.components.dust.attenuation.narayanan_z`` interpolates in redshift.
Every number the law ships is reproduced by running this script, so the
redshift scaling is traceable to the paper's own published product rather than
to a remembered coefficient.

Source
------
    data/attenuation/narayanan2018_median_curves.dat

    The paper's best-fit median attenuation curves at integer z = 0 to 6,
    repackaged from https://bitbucket.org/desika/narayanan_attenuation_laws/
    with attribution. They come from the 25 Mpc MUFASA cosmological
    hydrodynamic simulation post-processed with dust radiative transfer, and
    are normalized by the 3000 Angstrom optical depth tau_3000, not by A_V.

Reference
---------
    D. Narayanan, C. Conroy, R. Dave, B. D. Johnson and G. Popping,
    "A Theory for the Variation of Dust Attenuation Laws in Galaxies",
    ApJ, 869, 70 (2018). arXiv:1805.06905.
    https://doi.org/10.3847/1538-4357/aaed25

    M. Kriek and C. Conroy, "The Dust Attenuation Law in Distant Galaxies:
    Evidence for Variation with Spectral Type", ApJL, 775, L16 (2013).
    https://doi.org/10.1088/2041-8205/775/1/L16

The model
---------
Three free parameters per redshift, fitted by unweighted least squares on the
tabulated points inside the fit range:

    A(lambda) / tau_3000  =  norm * k_KC13(lambda; delta, bump_strength)

``k_KC13`` is tengri's own :func:`~tengri.components.dust.attenuation.kriek_conroy`
- the runtime function, called here so the fitted form is exactly the form the
law evaluates - a Calzetti (2000) baseline plus a 2175 Angstrom Drude bump of
amplitude ``E_b = bump_strength * (0.85 - 1.9 * delta)`` divided by
``R_V = 4.05``, the sum tilted by ``(lambda / 5500 A) ** delta`` and
renormalized to ``k(5500 A) = 1``. ``norm`` absorbs the difference between that
normalization and the data's: it is the fitted A(5500 A) / tau_3000.

Fit range
---------
1250 Angstrom to 1 micron. The tabulation reaches 1005 Angstrom, and the rows
blueward of 1250 Angstrom are excluded for two reasons. The Calzetti (2000)
starburst curve the Kriek & Conroy form tilts is calibrated over 0.12 to
2.2 micron, so the form is an extrapolation there and cannot represent the
medians however the parameters are chosen; and that region sits inside
Lyman-alpha and the Lyman continuum, where tengri applies IGM transmission and
an optional Lyman-limit clip as separate steps, so it is not the attenuation
shape this law is asked to reproduce.

Usage
-----
    python scripts/fit_narayanan2018_medians.py
    python scripts/fit_narayanan2018_medians.py --out /tmp/fits.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from tengri import data_path
from tengri.components.dust.attenuation import kriek_conroy

#: Repackaged median curves, relative to any directory ``data_path`` searches.
DATA_FILE = "attenuation/narayanan2018_median_curves.dat"

#: Integer redshifts the paper tabulates.
REDSHIFTS = (0, 1, 2, 3, 4, 5, 6)

#: Fit range [Angstrom]; see the module docstring for why the blue end is 1250.
FIT_MIN_AA = 1250.0
FIT_MAX_AA = 10000.0

#: Free-parameter bounds ``(delta, bump_strength, norm)``. Wide enough that a
#: solution on a bound is a signal rather than a constraint of the fit.
LOWER = (-3.0, 0.0, 1e-3)
UPPER = (3.0, 20.0, 10.0)

#: Starting points for the multi-start check. The fit is reported degenerate
#: unless every start lands on the same residual.
STARTS = (
    (-0.2, 1.0, 0.4),
    (0.0, 0.0, 0.5),
    (-1.0, 5.0, 0.3),
    (0.5, 2.0, 0.6),
    (-0.5, 3.0, 0.4),
)


def load_medians(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the repackaged table and return it on an ascending wavelength axis.

    Parameters
    ----------
    path : pathlib.Path
        The repackaged ``.dat`` file. Column 0 is 1/lambda [1/micron];
        columns 1 to 7 are the medians at z = 0 to 6.

    Returns
    -------
    wavelength : ndarray, shape (n_wave,)
        Ascending wavelength grid. [Angstrom]
    medians : ndarray, shape (7, n_wave)
        Median A(lambda) / tau_3000 at each redshift. [dimensionless]
    """
    table = np.loadtxt(path)
    wavelength = 1e4 / table[:, 0]
    medians = table[:, 1:].T
    order = np.argsort(wavelength)
    return wavelength[order], medians[:, order]


def model(theta: tuple[float, float, float], wavelength: np.ndarray) -> np.ndarray:
    """Evaluate ``norm * kriek_conroy(wavelength; delta, bump_strength)``.

    Parameters
    ----------
    theta : tuple of float
        ``(delta, bump_strength, norm)``.
    wavelength : ndarray, shape (n_wave,)
        Wavelength grid. [Angstrom]

    Returns
    -------
    ndarray, shape (n_wave,)
        The curve on the data's normalization. [dimensionless]
    """
    delta, bump_strength, norm = theta
    curve = kriek_conroy(
        jnp.asarray(wavelength),
        dust_bump_strength=float(bump_strength),
        dust_delta=float(delta),
    )
    return float(norm) * np.asarray(curve)


def fit_one(wavelength: np.ndarray, median: np.ndarray) -> dict:
    """Least-squares fit of the Kriek & Conroy form to one median curve.

    Parameters
    ----------
    wavelength : ndarray, shape (n_wave,)
        Fit-range wavelength grid. [Angstrom]
    median : ndarray, shape (n_wave,)
        Median A(lambda) / tau_3000 at one redshift. [dimensionless]

    Returns
    -------
    dict
        ``dust_delta``, ``dust_bump_strength``, ``E_b``, ``norm``, ``rms``,
        ``max_abs_residual``, ``n_distinct_solutions`` and ``on_bound``.
    """

    def residual(theta):
        return model(theta, wavelength) - median

    solutions = []
    for start in STARTS:
        result = least_squares(
            residual,
            x0=start,
            bounds=(LOWER, UPPER),
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        rms = float(np.sqrt(np.mean(result.fun**2)))
        solutions.append((rms, tuple(float(v) for v in result.x), result.fun))
    solutions.sort(key=lambda item: item[0])
    best_rms, best_theta, best_residual = solutions[0]
    distinct = {
        tuple(np.round(theta, 6)) for rms, theta, _ in solutions if abs(rms - best_rms) < 1e-9
    }
    delta, bump_strength, norm = best_theta
    on_bound = [
        name
        for name, value, low, high in zip(
            ("dust_delta", "dust_bump_strength", "norm"), best_theta, LOWER, UPPER
        )
        if abs(value - low) < 1e-8 or abs(value - high) < 1e-8
    ]
    return {
        "dust_delta": delta,
        "dust_bump_strength": bump_strength,
        "E_b": bump_strength * (0.85 - 1.9 * delta),
        "norm": norm,
        "rms": best_rms,
        "max_abs_residual": float(np.max(np.abs(best_residual))),
        "n_distinct_solutions": len(distinct),
        "on_bound": on_bound,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"Median-curve table (default: {DATA_FILE} via tengri.data_path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the fit JSON (default: beside the data file).",
    )
    args = parser.parse_args()

    data_file = args.data if args.data is not None else data_path(DATA_FILE)
    default_out = data_file.parent / "narayanan2018_kc13_fits.json"
    out_file = args.out if args.out is not None else default_out

    wavelength, medians = load_medians(Path(data_file))
    inside = (wavelength >= FIT_MIN_AA) & (wavelength <= FIT_MAX_AA)
    wave_fit = wavelength[inside]

    print(f"data      {data_file}")
    print(f"fit range {FIT_MIN_AA:.0f} to {FIT_MAX_AA:.0f} A, {wave_fit.size} tabulated points")
    print()
    print(
        f"{'z':>2}  {'delta':>10}  {'bump':>10}  {'E_b':>8}  {'norm':>8}"
        f"  {'rms':>8}  {'max|res|':>8}"
    )

    fits = []
    for index, redshift in enumerate(REDSHIFTS):
        entry = fit_one(wave_fit, medians[index][inside])
        entry["z"] = redshift
        fits.append(entry)
        print(
            f"{redshift:>2}  {entry['dust_delta']:>+10.6f}  {entry['dust_bump_strength']:>10.6f}"
            f"  {entry['E_b']:>8.4f}  {entry['norm']:>8.6f}  {entry['rms']:>8.6f}"
            f"  {entry['max_abs_residual']:>8.6f}"
        )
        if entry["n_distinct_solutions"] != 1:
            print(f"    WARNING z={redshift}: {entry['n_distinct_solutions']} distinct optima")
        if entry["on_bound"]:
            print(f"    WARNING z={redshift}: on bound {entry['on_bound']}")

    payload = {
        "source_data": DATA_FILE,
        "generated_by": "scripts/fit_narayanan2018_medians.py",
        "form": "tengri.components.dust.attenuation.kriek_conroy",
        "fit_range_angstrom": [FIT_MIN_AA, FIT_MAX_AA],
        "n_fit_points": int(wave_fit.size),
        "normalization": (
            "norm is the fitted A(5500 A) / tau_3000: the data are normalized by the "
            "3000 A optical depth, kriek_conroy by k(5500 A) = 1."
        ),
        "fits": fits,
    }
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
