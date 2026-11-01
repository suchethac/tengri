# SPDX-License-Identifier: BSD-3-Clause
"""Run matrix for the BEAGLE reproduction: one :class:`MockRun` per physics section.

Every entry writes one BEAGLE parameter file with the same SSP or SFH, matched
across sections, so the comparison isolates one physics block at a time. Rows are
per-object ``from_file`` values in BEAGLE's log units (:math:`\\log Z/Z_\\odot` with
:math:`Z_\\odot=0.01524`, :math:`\\log M_\\odot`, :math:`\\log` yr).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from .beagle_run import (
    ATTENUATION_LAWS,
    ZSUN,
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
    MockRun(
        section="s12_igm",
        rows=tuple(
            {"redshift": z, "mass": 10.0, "sfr": 1.0, "metallicity": _logz(0.02), "tau": 9.0}
            for z in (1.0, 3.0, 6.0)
        ),
        templates="bc03",
        params=(mock_fixed("sfh_type", char_value="delayed"),),
        igm="Inoue",
    ),
)


def by_section(section: str) -> list[MockRun]:
    found = [r for r in RUN_MATRIX if r.section == section]
    if not found:
        known = ", ".join(r.section for r in RUN_MATRIX)
        raise ValueError(f"unknown section {section!r}; known: {known}")
    return found
