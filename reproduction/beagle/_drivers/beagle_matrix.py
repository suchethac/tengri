# SPDX-License-Identifier: BSD-3-Clause
"""Run matrix for the BEAGLE reproduction: one :class:`MockRun` per physics section.

Every entry writes one BEAGLE parameter file with the same SSP or SFH, matched
across sections, so the comparison isolates one physics block at a time. Rows are
per-object ``from_file`` values in BEAGLE's log units (:math:`\\log Z/Z_\\odot` with
:math:`Z_\\odot=0.01524`, :math:`\\log M_\\odot`, :math:`\\log` yr).

IGM: every section with :math:`z > 0` carries ``IGM ABSORPTION = Inoue`` (the default in
:class:`~reproduction.beagle._drivers.beagle_run.MockRun`); the two no-IGM controls
(``s12_igm_none_*``) omit the keyword. The CCWW13_full variant is not run: its radiative-transfer
tables are not distributed with the public BEAGLE package, so tengri implements the universal
relation only (see the comment above ``RUN_MATRIX``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from .beagle_run import (
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
        {"redshift": z, "mass": 10.0, "sfr": 1.0, "metallicity": _logz(z_abs), "tau": tau}
        for tau in taus
    )


def _ssp_rows(ages: Sequence[float], z_abs: float) -> tuple[dict[str, float], ...]:
    return tuple(
        {"redshift": 0.001, "mass": 10.0, "metallicity": _logz(z_abs), "ssp_age": age}
        for age in ages
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
    """Per redshift, an Inoue run and a no-IGM control, at native template resolution.

    BEAGLE filters are per run, so one run per redshift. ``rebin=None`` keeps the native
    grid: the 2 % top-hats at rest 1216 A sit on a feature that ``REBIN TEMPLATES = 10``
    smooths (by -0.14 against Inoue+2014 at z = 6; by -0.003 at native resolution). The
    z = 6 pair is repeated with ``REBIN TEMPLATES = 10`` so the rebinned result stays on record.
    """
    runs = []
    cases = [(z, None) for z in (1.0, 3.0, 6.0)] + [(6.0, 10)]
    for z, rebin in cases:
        row = {"redshift": z, "mass": 10.0, "sfr": 1.0, "metallicity": _logz(0.02), "tau": 9.0}
        suffix = "" if rebin is None else f"_rebin{rebin}"
        for label, igm in (("inoue", "Inoue"), ("none", None)):
            runs.append(
                MockRun(
                    section=f"s12_igm_{label}_z{int(z)}{suffix}",
                    rows=(row,),
                    templates="bc03",
                    params=(_DELAYED,),
                    filters=_igm_filters(z),
                    igm=igm,
                    rebin=rebin,
                )
            )
    return tuple(runs)


# ── Nebular physics and AGN sections ────────────────────────────────────────
# Fiducial: delayed history tau 9 Gyr, current episode 7 yr (log), Z = 0.014, log U = -2.5,
# xi = 0.3, SFR 1 Msun/yr at 10 log Msun formed. Each section varies one thing.
_FIDUCIAL_EPISODE = (
    _DELAYED,
    mock_fixed("tau", value=9.0),
    mock_fixed("max_stellar_age", value=9.0),
    mock_fixed("current_sfr_timescale", value=7.0),
)


def _fid_row(**overrides: float) -> dict[str, float]:
    """One fiducial nebular row; ``overrides`` replace a key in place, keeping the key order."""
    base = {
        "redshift": 0.001,
        "mass": 10.0,
        "sfr": 1.0,
        "metallicity": _logz(0.014),
        "nebular_logU": -2.5,
        "nebular_xi": 0.3,
        "nebular_Z": _logz(0.014),
    }
    return {**base, **overrides}


def _episode_sections() -> tuple[MockRun, ...]:
    """Current-episode timescale (s8b) and the HII-dust switch (s8c), both on the fiducial."""
    rows = tuple(
        _fid_row(sfr=sfr, current_sfr_timescale=7.0) for sfr in (-1.0, 0.0, 0.5, 1.0, 1.5)
    )
    rows += tuple(_fid_row(sfr=0.5, current_sfr_timescale=t) for t in (7.5, 8.0))
    episode = MockRun(
        section="s8b_episode",
        rows=rows,
        templates="ineb_june16",
        nebular=True,
        igm=None,
        # BEAGLE 0.29.2 aborts (array index out of range) when a per-row current_sfr_timescale is
        # combined with PRINT SF AND ChE HISTORIES; the section needs no printed history.
        print_sfh=False,
        params=(_DELAYED, mock_fixed("tau", value=9.0), mock_fixed("max_stellar_age", value=9.0)),
    )
    hii = tuple(
        MockRun(
            section=f"s8c_hiidust_{label}",
            rows=tuple(_fid_row(tauV_eff=tau_v) for tau_v in (0.3, 1.0, 2.0)),
            templates="ineb_june16",
            nebular=True,
            igm=None,
            dust="CF00",
            params=(*_FIDUCIAL_EPISODE, mock_fixed("mu", value=0.4)),
            hii_dust=flag,
        )
        for label, flag in (("on", True), ("off", False))
    )
    return (episode, *hii)


def _nebular_physics_sections() -> tuple[MockRun, ...]:
    """Escape fraction (s8d), SFH against template version (s8e) and population age (s8f)."""
    fesc = MockRun(
        section="s8d_fesc",
        rows=tuple(_fid_row(LyC_f_esc=f) for f in (0.0, 0.1, 0.3, 0.6)),
        templates="ineb_june16",
        nebular=True,
        igm=None,
        params=_FIDUCIAL_EPISODE,
    )
    const100 = MockRun(
        section="s8e_const100",
        # No current episode: a pure constant 100 Myr history is the public table's population.
        rows=tuple({k: v for k, v in r.items() if k != "sfr"} for r in _nebular_rows()),
        templates="ineb_june16",
        nebular=True,
        igm=None,
        params=(
            mock_fixed("sfh_type", char_value="constant"),
            mock_fixed("max_stellar_age", value=8.0),
        ),
    )
    age = MockRun(
        section="s8f_age",
        # No current episode, so the lines follow the age of the constant history alone.
        rows=tuple(
            {k: v for k, v in _fid_row(max_stellar_age=age).items() if k != "sfr"}
            for age in (6.5, 7.0, 7.5, 8.0, 9.0)
        ),
        templates="ineb_june16",
        nebular=True,
        igm=None,
        params=(mock_fixed("sfh_type", char_value="constant"),),
    )
    return (fesc, const100, age)


def _agn_sections() -> tuple[MockRun, ...]:
    """The AGN narrow-line region on the fiducial (s9_agn) and alone (s9_agn_only)."""
    grid = tuple(
        _fid_row(agn_lacc=lacc, agn_logU=log_u, agn_Z=log_z)
        for lacc, log_u, log_z in itertools.product(
            (43.0, 44.0, 45.0, 46.0), (-3.5, -2.5, -1.5), (-0.5, 0.0)
        )
    )
    combos = tuple(itertools.product((-3.5, -2.5, -1.5), (-0.5, 0.0)))
    agn_only = tuple(
        _fid_row(sfr=-3.0, mass=8.0, agn_logU=log_u, agn_Z=log_z) for log_u, log_z in combos
    )
    common = {
        "templates": "ineb_june16",
        "nebular": True,
        "agn": True,
        "rebin": None,
        "igm": None,
    }
    return (
        MockRun(
            section="s9_agn",
            rows=grid,
            params=(*_FIDUCIAL_EPISODE, mock_fixed("agn_xi", value=0.3)),
            **common,
        ),
        MockRun(
            section="s9_agn_only",
            rows=agn_only,
            params=(
                *_FIDUCIAL_EPISODE,
                mock_fixed("agn_lacc", value=45.0),
                mock_fixed("agn_xi", value=0.3),
            ),
            **common,
        ),
    )


# CCWW13_full is not run: its Tuffs et al. (2004) radiative-transfer tables are not distributed
# with the public BEAGLE package, and BEAGLE stops with "Radiative transfer calculations not
# initialized" without them. tengri implements the universal relation (CCWW13_universal). For anyone
# who obtains the tables, the geometry keywords are ``inclination`` and ``tauB_perp`` (the output
# column name; BEAGLE rejects ``tau_b_perp``), e.g. inclination 60 deg and tauB_perp 4.0.


# (section, attenuation_type, needs mu). The manual requires mu for cf00, ccww13_universal and
# ccww13_full only; the Calzetti and Pei (1992) single-screen curves take tauV_eff alone.
_ATTENUATION_SECTIONS = (
    ("s4_cf00", "CF00", True),
    ("s4_calzetti", "Calzetti", False),
    ("s4_ccww13_universal", "CCWW13_universal", True),
    ("s4_mw", "mw", False),
    ("s4_lmc", "lmc", False),
    ("s4_smc", "smc", False),
)

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
            section=section,
            rows=_delayed_rows((9.0,), _SAMPLE_Z, 0.02),
            templates="bc03",
            dust=law,
            params=(
                _DELAYED,
                mock_fixed("tauV_eff", value=1.0),
                *((mock_fixed("mu", value=0.4),) if needs_mu else ()),
            ),
        )
        for section, law, needs_mu in _ATTENUATION_SECTIONS
    ),
    MockRun(
        section="s5_cf00_tauv",
        rows=tuple(
            {**_delayed_rows((9.0,), _SAMPLE_Z, 0.02)[0], "tauV_eff": t} for t in (0.3, 1.0, 2.0)
        ),
        templates="bc03",
        dust="CF00",
        params=(_DELAYED, mock_fixed("mu", value=0.4)),
    ),
    MockRun(
        section="s8_nebular",
        rows=_nebular_rows(),
        templates="ineb_june16",
        nebular=True,
        params=(
            mock_fixed("sfh_type", char_value="constant"),
            mock_fixed("max_stellar_age", value=9.0),
        ),
    ),
    *_episode_sections(),
    *_nebular_physics_sections(),
    *_agn_sections(),
    *_igm_sections(),
)


def by_section(section: str) -> list[MockRun]:
    found = [r for r in RUN_MATRIX if r.section == section]
    if not found:
        known = ", ".join(r.section for r in RUN_MATRIX)
        raise ValueError(f"unknown section {section!r}; known: {known}")
    return found
