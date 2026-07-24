# SPDX-License-Identifier: BSD-3-Clause
"""Parameters that have units must declare them (#1296).

Measured before this landed: **53 of 350** parameters declared any units at
all, and whole physics groups declared none — `agn` 0/87, `dust` 0/30,
`neb` 0/13, `radio` 0/16, `xray` 0/7, `igm` 0/5, `shock` 0/5. The units were
usually *already in the prose* (``"AGN torus temperature (K)"``,
``"Exponential cutoff energy [keV]"``), so ``describe_parameter`` showed them
on screen while ``record.units`` stayed empty and nothing programmatic could
read them.

CLAUDE.md makes units in brackets non-negotiable, and #1286 gave all seven
distributions somewhere to put them.

Two guards here:

* **the ratchet** — the number of parameters declaring units may rise, never
  fall, so a new physics block cannot quietly ship unit-less;
* **the prose check** — a parameter whose *description* states a unit must
  declare that unit, with an explicit allowlist for the cases where the prose
  unit belongs to context rather than to the parameter.

The second is the one with teeth. The auto-extraction that produced the
initial mapping got 12 of 88 wrong, every one of them the same mistake: the
description names a unit that bounds a *wavelength range*, labels a *band*, or
gives a *reference frequency*, none of which is the parameter's own unit.
"""

from __future__ import annotations

import re

import pytest

import tengri

pytestmark = pytest.mark.contract

ROWS = {r["name"]: r for r in tengri.list_parameters()}

#: Shrink-only ratchet. Raise it when you declare more units; never lower it.
MIN_DECLARING_UNITS = 134

#: Parameters whose description mentions a unit that is NOT the parameter's own.
#: Each entry states why, because "it looked wrong" is not a reason.
PROSE_UNIT_IS_CONTEXT: dict[str, str] = {
    "ionspec_index1": "spectral slope; the Angstroms bound the segment",
    "ionspec_index2": "spectral slope; the Angstroms bound the segment",
    "ionspec_index3": "spectral slope; the Angstroms bound the segment",
    "ionspec_index4": "spectral slope; the Angstroms bound the segment",
    "xray_alpha_irx": "log of a luminosity ratio; keV labels the band",
    "radio_delv_q0": "FIRRC q is dimensionless; GHz is the reference band",
    "radio_mcch_q0": "FIRRC q is dimensionless; MHz is the reference band",
}

#: Unit tokens that, appearing bracketed in a description, imply the parameter
#: is dimensional. Deliberately conservative — bare words like "mag" or "yr"
#: appear too often inside prose to be evidence on their own.
BRACKETED_UNIT = re.compile(
    r"[\[(]\s*(K|keV|GHz|MHz|km/s|Myr|Gyr|yr|mag|deg|degrees|erg/s|erg/s/Hz|"
    r"Msun|Msun/yr|Angstrom)\s*[\])]"
)


def _description(name: str) -> str:
    try:
        return (tengri.describe_parameter(name).description or "").strip()
    except Exception:
        return ""


def test_the_census_is_not_empty():
    """Guard the guard: an empty registry would satisfy everything below."""
    assert len(ROWS) >= 300, f"only {len(ROWS)} parameters registered — census rotted"


def test_units_coverage_only_grows():
    declaring = sum(1 for r in ROWS.values() if (r["units"] or "").strip())
    assert declaring >= MIN_DECLARING_UNITS, (
        f"only {declaring} parameters declare units, down from "
        f"{MIN_DECLARING_UNITS}. Units coverage is a shrink-only ratchet: if a "
        "parameter legitimately lost its units, lower the constant in this PR "
        "and say why."
    )


def test_a_description_that_states_a_unit_must_declare_it():
    """The specific gap: units visible in prose but unreadable by code."""
    offenders = []
    for name, row in sorted(ROWS.items()):
        if (row["units"] or "").strip():
            continue
        if name in PROSE_UNIT_IS_CONTEXT:
            continue
        desc = _description(name)
        match = BRACKETED_UNIT.search(desc)
        if match:
            offenders.append((name, match.group(1), desc[:60]))
    assert not offenders, (
        "parameters whose description states a unit but which declare none:\n"
        + "\n".join(f"  {n:30s} [{u}]  {d}" for n, u, d in offenders)
        + "\n\nSet units= on the declaration. If the unit in the prose belongs "
        "to context (a wavelength range, a band label, a reference frequency) "
        "rather than to the parameter, add it to PROSE_UNIT_IS_CONTEXT with "
        "the reason."
    )


def test_the_context_allowlist_is_still_needed():
    """A stale allowlist entry hides a real gap once the parameter is fixed."""
    stale = [n for n in PROSE_UNIT_IS_CONTEXT if (ROWS.get(n, {}).get("units") or "").strip()]
    assert not stale, f"these now declare units, so their allowlist entries are stale: {stale}"
    missing = [n for n in PROSE_UNIT_IS_CONTEXT if n not in ROWS]
    assert not missing, f"allowlisted parameters that no longer exist: {missing}"


def test_log_quantities_declare_a_log_unit():
    """``log10(M/Msun)`` is dimensionless — declaring ``Msun`` is a units error.

    The convention is ``log10(<unit>)`` when there is an underlying unit and
    ``dex`` when the log is of a ratio or an offset. Naming the bare unit on a
    log quantity is the mistake the first extraction pass made 28 times.
    """
    wrong = []
    for name, row in sorted(ROWS.items()):
        units = (row["units"] or "").strip()
        if not units:
            continue
        looks_log = bool(re.search(r"_log[a-z0-9_]|_lg[A-Z]", name)) or "log10" in _description(
            name
        )
        if not looks_log:
            continue
        if units.startswith("log10(") or units == "dex":
            continue
        wrong.append((name, units))
    assert not wrong, (
        "log-valued parameters declaring a bare unit (they are dimensionless):\n"
        + "\n".join(f"  {n:30s} units={u!r}" for n, u in wrong)
        + "\n\nUse log10(<unit>), or dex for a log ratio/offset."
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("agn_T_torus", "K"),
        ("radio_T_e", "K"),
        ("xray_E_cut", "keV"),
        ("agn_oa_skirtor", "deg"),
        ("shock_velocity", "km/s"),
        ("neb_log_nH", "log10(cm^-3)"),
        ("xray_log_nh", "log10(cm^-2)"),
        ("sfh_dpl_log_total_mass", "log10(Msun)"),
        ("agn_log_mbh", "log10(Msun)"),
        ("sfh_db_log_sfr_inst", "log10(Msun/yr)"),
        ("dust_log_ssfr", "log10(1/yr)"),
    ],
)
def test_spot_checks(name, expected):
    """Hand-verified values, pinned so a re-derivation cannot drift them."""
    assert ROWS[name]["units"] == expected, (
        f"{name} declares {ROWS[name]['units']!r}, expected {expected!r}"
    )
