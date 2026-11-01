# SPDX-License-Identifier: BSD-3-Clause
"""Capstone fit: four Paper I galaxies through BEAGLE's photometric fitter.

The galaxies are GOODS-S workshop objects at their catalog redshifts (``zz``, read as a Dirac
prior). Photometry comes from :mod:`reproduction.beagle._drivers.beagle_driver`: AB magnitudes
are converted to microJy, and a band a galaxy did not measure is written with the negative flux
and negative error that BEAGLE ignores. Each band's throughput is an ASCII curve written from the
committed filter group. The priors follow BEAGLE's example fit, with two differences noted below.

Priors
------
* ``tau``, ``mass``, ``metallicity``, ``max_stellar_age``, ``specific_sfr`` and
  ``nebular_logU`` are log-valued parameters, so they take a uniform prior on their stated range.
  BEAGLE's ``log-uniform`` distribution is uniform in the log of a linear range; applying it to a
  logarithm would be wrong, so it is used only for ``tauV_eff`` (linear).
* ``max_stellar_age`` takes the age of the universe at the highest catalog redshift among the
  four galaxies. BEAGLE checks that the age does not exceed the age at each galaxy's redshift,
  and a single prior range cannot differ per object, so the upper bound is conservative for the
  lower-redshift galaxies.
* ``current_sfr_timescale`` is fixed at 7 (log yr) and ``specific_sfr`` sets the recent SFR,
  as in the manual's star-formation table.
"""

from __future__ import annotations

import h5py
import numpy as np
from astropy.table import Table

from tengri.utils.cosmology import age_at_z

from . import beagle_driver as D
from .beagle_run import FilterSpec, FitRun, Param, fit_fitted, fit_fixed

CAPSTONE_IDS = (79, 7837, 15775, 1826)
REDSHIFT_COLUMN = "zz"
SAMPLER_FILE = "MCMC_example.param"  # BEAGLE's MultiNest settings, read from $BEAGLE_PARAM_DIR
MICROJY_PER_CGS = 1.0e29  # 1 microJy = 1e-29 erg/s/cm^2/Hz
MASKED_VALUE = -99.0  # negative flux and negative error: BEAGLE ignores the band
TAU_V_RANGE = (1.0e-3, 5.0)
MASS_RANGE = (5.0, 12.0)
TAU_RANGE = (7.0, 10.5)
METALLICITY_RANGE = (-2.2, 0.24)
SSFR_RANGE = (-12.0, -7.0)
LOG_U_RANGE = (-4.0, -1.0)
MIN_STELLAR_AGE_LOG = 8.0
TEMPLATES = "ineb_june16"


def _single_row(table: Table, galaxy_id: int):
    rows = table[table["ID"] == galaxy_id]
    if len(rows) != 1:
        raise ValueError(f"galaxy {galaxy_id}: expected one catalog row, found {len(rows)}")
    return rows[0]


def _curve(curve_name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(D.WORKSHOP_PATH, "r") as handle:
        group = handle["filters"][curve_name]
        return np.asarray(group["wave_aa"][:]), np.asarray(group["response"][:])


def _catalog_table(table: Table, bands: list[str]) -> Table:
    """Photometric catalog for the four galaxies, fluxes and errors in microJy."""
    columns: dict[str, np.ndarray] = {
        "ID": np.array([int(_single_row(table, g)["ID"]) for g in CAPSTONE_IDS]),
        REDSHIFT_COLUMN: np.array([float(_single_row(table, g)[REDSHIFT_COLUMN]) for g in CAPSTONE_IDS]),
    }
    per_galaxy = [D.photometry_for_row(_single_row(table, g), bands) for g in CAPSTONE_IDS]
    for j, band in enumerate(bands):
        flux = [np.ma.filled(f[j] * MICROJY_PER_CGS, MASKED_VALUE) for _, f, _ in per_galaxy]
        err = [np.ma.filled(s[j] * MICROJY_PER_CGS, MASKED_VALUE) for _, _, s in per_galaxy]
        columns[f"flux_{band}"] = np.array(flux, dtype=float)
        columns[f"fluxerr_{band}"] = np.array(err, dtype=float)
    return Table(columns)


def _transmission(curve_name: str, response: np.ndarray) -> np.ndarray:
    """Dimensionless throughput: HAWKI_KS is stored as published (below 100), so it is divided
    by 100 as the builder's note records; the small negative published values are set to zero."""
    scaled = response / 100.0 if curve_name == "HAWKI_KS" else response
    return np.clip(scaled, 0.0, None)


def _filters(meta: dict, bands: list[str]) -> tuple[FilterSpec, ...]:
    curve_for_band = meta["curve_for_band"]
    specs = []
    for band in bands:
        curve_name = curve_for_band[band]
        wave, response = _curve(curve_name)
        specs.append(
            FilterSpec(
                name=band,
                wl_aa=wave,
                transmission=_transmission(curve_name, response),
                flux_col=f"flux_{band}",
                err_col=f"fluxerr_{band}",
            )
        )
    return tuple(specs)


def _params(max_age_log: float) -> tuple[Param, ...]:
    return (
        fit_fixed("sfh_type", char_value="delayed"),
        fit_fitted("tau", *TAU_RANGE),
        fit_fitted("mass", *MASS_RANGE),
        fit_fitted("metallicity", *METALLICITY_RANGE),
        fit_fitted("max_stellar_age", MIN_STELLAR_AGE_LOG, max_age_log),
        fit_fitted("specific_sfr", *SSFR_RANGE),
        fit_fixed("current_sfr_timescale", value=7.0),
        fit_fixed("attenuation_type", char_value="CF00"),
        Param(
            "tauV_eff",
            "type:fitted",
            prior=f"order_priority:0 prior:distribution:log-uniform prior:range:[{TAU_V_RANGE[0]},{TAU_V_RANGE[1]}]",
        ),
        fit_fixed("mu", value=0.4),
        fit_fitted("nebular_logU", *LOG_U_RANGE),
        fit_fixed("nebular_xi", value=0.3),
        Param("nebular_Z", "type:dependent"),
    )


def capstone_run() -> FitRun:
    """The four-galaxy ``--fit`` run (materialize and fit with ``--fit-capstone --allow-docker``).

    Returns
    -------
    FitRun
        Section ``capstone_paper1``; the galaxy IDs are in ``CAPSTONE_IDS``.

    Raises
    ------
    FileNotFoundError
        If the committed workshop catalog is missing.
    """
    table = D.workshop_catalog(1)
    bands = list(table.meta["bands"])
    z_max = max(float(_single_row(table, g)[REDSHIFT_COLUMN]) for g in CAPSTONE_IDS)
    max_age_log = float(np.log10(age_at_z(z_max) * 1.0e9))
    return FitRun(
        section="capstone_paper1",
        galaxy_ids=tuple(str(g) for g in CAPSTONE_IDS),
        catalog_table=_catalog_table(table, bands),
        filters=_filters(table.meta, bands),
        params=_params(max_age_log),
        templates=TEMPLATES,
        nebular=True,
        redshift_column=REDSHIFT_COLUMN,
        print_spectrum=True,
        sampler=SAMPLER_FILE,
    )
