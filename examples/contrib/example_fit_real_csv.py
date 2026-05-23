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
# This row is mock SDSS photometry. In real usage, substitute your own CSV.
CSV = """\
name,redshift,sdss_u,sdss_u_err,sdss_g,sdss_g_err,sdss_r,sdss_r_err,sdss_i,sdss_i_err,sdss_z,sdss_z_err
mock_galaxy,0.05,3.0e-27,5e-28,2.3e-26,1e-27,3.5e-26,1e-27,4.4e-26,1e-27,5.1e-26,2e-27
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
# ---------------------------------------------------------------------------
filter_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
csv_keys = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


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

    # --- Build model using SEDModel.build with recipes ---
    # Prefer tengri.recipes for standard workflows; use nested-dict API
    # for custom parameter choices.
    ssp_candidates = list(Path("data").glob("ssp_*.h5"))
    if not ssp_candidates:
        print("\n[skipping live fit — no SSP file in ./data/]")
        return 0
    ssp = tengri.load_ssp_data(str(ssp_candidates[0]))

    model = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "dpl", "*": tengri.FREE},
        redshift=tengri.Fixed(z),
    )
    print(f"\nModel: {model.spec.n_free} free parameters")

    # --- Fit with MAP (the workhorse for fast iteration) ----
    print("\nRunning MAP optimization (Adam, 200 steps)…")
    forward = tengri.ForwardModel.build(sed=model, observation=obs)
    posterior = forward.fit(
        fluxes, errors, method="map", optimizer="adam", n_steps=200, verbose=False
    )
    posterior.summary()

    # --- For posterior-quality fits, swap to NUTS or VI -----
    print(
        "\nFor posterior-quality work:\n"
        "    forward.fit(data, noise, method='nuts',   n_steps=1000)  # exact, ≲20-D\n"
        "    forward.fit(data, noise, method='vi_native', n_iter=50)   # scalable\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
