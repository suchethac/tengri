# SPDX-License-Identifier: BSD-3-Clause
"""Run matrix for the BEAGLE reproduction: one :class:`MockRun` per physics section.

Every entry writes one BEAGLE parameter file with the same SSP or SFH, matched
across sections, so the comparison isolates one physics block at a time. Rows are
per-object ``from_file`` values in BEAGLE's log units (:math:`\\log Z/Z_\\odot` with
:math:`Z_\\odot=0.01524`, :math:`\\log M_\\odot`, :math:`\\log` yr).

IGM: every section with :math:`z > 0` carries ``IGM ABSORPTION = Inoue`` (the default in
:class:`~reproduction.beagle._drivers.beagle_run.MockRun`); the two no-IGM controls
(``s12_igm_none_*``) omit the keyword. The CCWW13_full section is for information only (see
``_TUFFS_GEOMETRY``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from .beagle_run import (
    ATTENUATION_LAWS,
    ZSUN,
    FilterSpec,
    MockRun,
    mock_fixed,
)


# ── Run matrix ──────────────────────────────────────────────────────────────
def _logz(z_abs: float) -> float:
    return math.log10(z_abs / ZSUN)


def _delayed_rows(taus: Sequence[float], z: float, z_abs: float) -> tuple[dict[str, float], ...]:
    return tuple(
        {"redshift": z, "mass": 10.0, "sfr": 1.0, "metallicity": _logz(z_abs), "tau": tau} for tau in taus
    )


def _ssp_rows(ages: Sequence[float], z_abs: float) -> tuple[dict[str, float], ...]:
    return tuple(
        {"redshift": 0.001, "mass": 10.0, "metallicity": _logz(z_abs), "ssp_age": age} for age in ages
    )


def _nebular_rows() -> tuple[dict[str, float], ...]:
    log_u = (-3.5, -3.0, -2.5, -2.0)
    z_abs = (0.001, 0.004, 0.014, 0.03)
    xi = (0.1, 0.3, 0.5)
    rows = []
    for u, z, x in itertools.product(log_u, z_abs, xi):
        rows.append(
            {
                "redshift": 0.001,
                "mass": 10.0,
                "sfr": 1.0,
                "metallicity": _logz(z),
                "nebular_logU": u,
                "nebular_xi": x,
                "nebular_Z": _logz(z),
            }
        )
    return tuple(rows)


_SAMPLE_Z = 1.0
_DELAYED = mock_fixed("sfh_type", char_value="delayed")

IGM_REST_CENTERS_AA = tuple(float(v) for v in np.linspace(850.0, 1216.0, 12))
IGM_BAND_FRACTION = 0.02  # top-hat full width as a fraction of the observed center


def _tophat(name: str, center: float) -> FilterSpec:
    """A top-hat band of full width ``IGM_BAND_FRACTION * center`` (observed frame, Angstrom)."""
    half = 0.5 * IGM_BAND_FRACTION * center
    wl = np.linspace(center - 1.5 * half, center + 1.5 * half, 201)
    transmission = (np.abs(wl - center) <= half).astype(float)
    return FilterSpec(name, wl, transmission)


def _igm_filters(z: float) -> tuple[FilterSpec, ...]:
    """Twelve top-hats whose rest-frame centers span 850-1216 A, observed at ``1 + z``."""
    return tuple(
        _tophat(f"th{i:02d}", rest * (1.0 + z)) for i, rest in enumerate(IGM_REST_CENTERS_AA)
    )


def _igm_sections() -> tuple[MockRun, ...]:
    """Per redshift, an Inoue run and a no-IGM control; BEAGLE filters are per run, so one run per z."""
    runs = []
    for z in (1.0, 3.0, 6.0):
        row = {"redshift": z, "mass": 10.0, "sfr": 1.0, "metallicity": _logz(0.02), "tau": 9.0}
        for label, igm in (("inoue", "Inoue"), ("none", None)):
            runs.append(
                MockRun(
                    section=f"s12_igm_{label}_z{int(z)}",
                    rows=(row,),
                    templates="bc03",
                    params=(_DELAYED,),
                    filters=_igm_filters(z),
                    igm=igm,
                )
            )
    return tuple(runs)

# CCWW13_full solves the Tuffs et al. (2004) radiative transfer for a galaxy geometry, which BEAGLE
# reads from these two parameters. The values are arbitrary fixed choices. The full variant is shown
# for information only: tengri implements the universal relation (CCWW13_universal), not the
# geometry-resolved one.
_TUFFS_GEOMETRY = {
    "CCWW13_full": (mock_fixed("inclination", value=60.0), mock_fixed("tau_b_perp", value=4.0)),
}

RUN_MATRIX: tuple[MockRun, ...] = (
    MockRun(
        section="s1_ssp_bc03",
        rows=_ssp_rows((7.0, 8.0, 9.0), 0.02),
        templates="bc03",
        params=(mock_fixed("sfh_type", char_value="ssp"),),
    ),
    MockRun(
        section="s1a_ssp_cb16",
        rows=_ssp_rows((7.0, 8.0, 9.0), 0.02),
        templates="cb_june16",
        params=(mock_fixed("sfh_type", char_value="ssp"),),
    ),
    MockRun(
        section="s2_sfh_delayed",
        rows=_delayed_rows((8.0, 9.0, 10.0), _SAMPLE_Z, 0.02),
        templates="bc03",
        params=(mock_fixed("sfh_type", char_value="delayed"),),
    ),
    MockRun(
        section="s3_fiducial",
        rows=_delayed_rows((9.0,), _SAMPLE_Z, 0.02),
        templates="bc03",
        params=(_DELAYED,),
    ),
    *(
        MockRun(
            section=f"s4_{law.lower()}",
            rows=_delayed_rows((9.0,), _SAMPLE_Z, 0.02),
            templates="bc03",
            dust=law,
            params=(
                _DELAYED,
                mock_fixed("tauV_eff", value=1.0),
                mock_fixed("mu", value=0.4),
                *_TUFFS_GEOMETRY.get(law, ()),
            ),
        )
        for law in ATTENUATION_LAWS
    ),
    MockRun(
        section="s5_cf00_tauv",
        rows=tuple({**_delayed_rows((9.0,), _SAMPLE_Z, 0.02)[0], "tauV_eff": t} for t in (0.3, 1.0, 2.0)),
        templates="bc03",
        dust="CF00",
        params=(_DELAYED, mock_fixed("mu", value=0.4)),
    ),
    MockRun(
        section="s8_nebular",
        rows=_nebular_rows(),
        templates="ineb_june16",
        nebular=True,
        params=(mock_fixed("sfh_type", char_value="constant"), mock_fixed("max_stellar_age", value=9.0)),
    ),
    *_igm_sections(),
)



def by_section(section: str) -> list[MockRun]:
    found = [r for r in RUN_MATRIX if r.section == section]
    if not found:
        known = ", ".join(r.section for r in RUN_MATRIX)
        raise ValueError(f"unknown section {section!r}; known: {known}")
    return found
