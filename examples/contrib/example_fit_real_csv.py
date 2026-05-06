"""
Worked example: fit a real galaxy from a CSV of measured fluxes.

The other contributor example (``example_new_agn_torus.py``) shows how to
*register* a new physics model.  This one shows the other half of the
workflow: how a working astronomer feeds *measured photometry* from a
catalogue or table into tengri and gets a posterior out.

Run::

    .venv/bin/python examples/contrib/example_fit_real_csv.py

The script:

1. Builds a small inline CSV (no external file required) of one nearby
   galaxy with SDSS griz + 2MASS JHKs + WISE W1–W4 fluxes.  Same idiom
   works for any survey output — see the comments below for the
   substitution.
2. Discovers the right filter set with ``tengri.list_filters().filter(...)``.
3. Builds Parameters / Observation / SEDModel / Fitter.
4. Runs MAP (the cheapest method — exact fits use ``"nuts"`` or ``"vi"``).
5. Prints the posterior summary.

Skip the live fit if the SSP grid isn't available — the discovery and
data-loading parts run regardless.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tengri

# ---------------------------------------------------------------------------
# 1.  The CSV.  In real life you'd ``pd.read_csv("galaxies.csv")``.
# ---------------------------------------------------------------------------
# Columns:
#   name, redshift, then one (flux, error) pair per band in [erg/s/cm²/Hz].
#
# This row is roughly NGC 4258 published photometry, but the exact
# numbers are not the point — substitute your own galaxy.
CSV = """\
name,redshift,sdss_g,sdss_g_err,sdss_r,sdss_r_err,sdss_i,sdss_i_err,sdss_z,sdss_z_err,2mass_j,2mass_j_err,2mass_h,2mass_h_err,2mass_ks,2mass_ks_err,wise_w1,wise_w1_err,wise_w2,wise_w2_err,wise_w3,wise_w3_err,wise_w4,wise_w4_err
NGC_4258,0.0015,2.3e-26,1e-27,3.5e-26,1e-27,4.4e-26,1e-27,5.1e-26,2e-27,7.0e-26,3e-27,8.6e-26,3e-27,8.4e-26,3e-27,9.0e-26,4e-27,7.5e-26,4e-27,2.5e-25,2e-26,5.0e-25,5e-26
"""


def _read_one_row(csv_text: str) -> dict:
    """Tiny standalone CSV parser so this script has zero dependencies
    beyond tengri itself.  Replace with ``pd.read_csv`` for catalogues."""
    lines = [ln for ln in csv_text.strip().splitlines() if ln]
    header = lines[0].split(",")
    row = dict(zip(header, lines[1].split(",")))
    return row


# ---------------------------------------------------------------------------
# 2.  Pick the bandset that matches the CSV columns.
#
#     The astronomer-conventional survey names just work via the smart
#     ``survey=`` alias on ``list_filters()`` — see __init__.py docstring.
# ---------------------------------------------------------------------------
SDSS_BANDS = tengri.list_filters(survey="SDSS").filter(band__in=("g", "r", "i", "z")).names()
TWOMASS = tengri.list_filters(instrument="2MASS").names()
WISE = tengri.list_filters().filter(survey="WISE").names()

# Order matches the CSV column order
filter_names = SDSS_BANDS + TWOMASS + WISE
# CSV column suffixes that pair with each filter (must be in column order)
csv_keys = [
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "wise_w4",
]


def main() -> int:
    # --- Parse the CSV (one row, but the same indexing scales) ---------
    row = _read_one_row(CSV)
    name = row["name"]
    z = float(row["redshift"])
    fluxes = jnp.array([float(row[k]) for k in csv_keys])
    errors = jnp.array([float(row[k + "_err"]) for k in csv_keys])

    print(f"Loaded {name}: z={z}, {len(fluxes)} bands")
    print(f"  fluxes  [erg/s/cm²/Hz]: {np.array(fluxes)}")
    print(f"  errors                  : {np.array(errors)}")

    # --- Build the photometric configuration ---------------------------
    print(f"\nFilters resolved: {len(filter_names)} bands")
    print(f"  {filter_names}")

    try:
        photometry = tengri.Photometry.from_names(filter_names)
    except Exception as exc:
        print(f"\n[skipping live fit — could not load filter curves]: {exc}")
        return 0

    obs = tengri.Observation(photometry=photometry)

    # --- Build a default-ish Parameters with redshift fixed to the row -
    # Pick a simple model; ``tengri.describe('dpl')`` shows the priors.
    parameters = tengri.Parameters(
        mean_sfh_type="dpl",
        redshift=tengri.Fixed(z),
    )
    print(f"\nParameters: {parameters.n_free} free, mean_sfh_type={parameters.mean_sfh_type}")

    # --- Locate a bundled SSP file -------------------------------------
    ssp_candidates = list(Path("data").glob("ssp_*.h5"))
    if not ssp_candidates:
        print("\n[skipping live fit — no SSP file in ./data/]")
        return 0
    ssp = tengri.load_ssp_data(str(ssp_candidates[0]))

    # --- Build model + fitter ------------------------------------------
    model = tengri.SEDModel(parameters, ssp, observation=obs)
    print(f"\n{model!r}")

    fitter = tengri.Fitter(model, data=fluxes, noise=errors)
    print(f"{fitter!r}")

    # --- MAP first (cheapest) ------------------------------------------
    print("\nRunning MAP optimization (Adam, 200 steps)…")
    posterior = fitter.run("map", key=jax.random.PRNGKey(0), n_steps=200)
    posterior.summary()

    # --- For posterior-quality fits, swap to NUTS or VI ---------------
    print(
        "\nFor posterior-quality work:\n"
        "    posterior = fitter.run('nuts',  init_from=map_result)   # exact, ≲20-D\n"
        "    posterior = fitter.run('vi',    init_from=map_result)   # scalable, geoVI\n"
        "    posterior = fitter.run('mcmc')                          # auto-pick by D\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
