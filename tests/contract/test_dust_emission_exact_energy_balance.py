# SPDX-License-Identifier: BSD-3-Clause
"""Every energy-balanced dust emission type integrates to L_ir on the public grid.

The contract: ``integral(sed_dust_ir d nu) == L_ir`` for the spectrum
``predict_state`` publishes, evaluated on the model's own wavelength grid.
Template models used to normalize on their native grid and resample
afterwards, so the delivered spectrum carried the resampling error
(3-13% for Dale+2014, BOSA, THEMIS, Draine & Li on coarse grids; 7e-6 for
astrodust from the spinning-dust term riding outside the budget). They now
resample first and normalize on the evaluation grid.

Tolerances come from a measured run at z = 0 (2026-09-05): every template
model is 1.00000000 to eight digits; the analytic closures sit at
0.9999984-0.9999994 because the 2.725 K CMB contrast removes a physical
1e-6 of a 35 K blackbody. ``pah_drude`` is a building block that is not
energy balanced by design (0.00019) and ``energy_balance_split`` is a
partition, so both are excluded and the census below is the record of what
must run.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract

_C_AA_PER_S = 2.99792458e18

# The names as a measured run listed them; a rename or removal must edit this on purpose.
_REQUIRED = (
    "astrodust",
    "bosa",
    "casey2012",
    "dale2014",
    "dale2014_cigale",
    "dh02_ce01",
    "draine2021_pah_ir",
    "draine_li2007",
    "draine_li2014",
    "graybody",
    "modified_blackbody",
    "schreiber2016",
    "schreiber2018",
    "themis",
)
_NOT_BALANCED_BY_DESIGN = ("pah_drude", "energy_balance_split")
# Menu spellings that select a component already covered under its canonical name.
# ``draine2021_pah`` -> ``draine2021_pah_ir`` joined this list when the canonical
# spelling became a menu row of its own: the component publishes ``sed_dust_ir``
# and so is listed, and running both spellings would run one component twice.
_ALIASES_OF_REQUIRED = ("dl07", "dl14", "draine2021_pah", "mbb")
_ANALYTIC = ("casey2012", "graybody", "modified_blackbody", "schreiber2016")
_RTOL_TEMPLATE = 1e-7
_RTOL_ANALYTIC = 5e-6

_COMPLETED: list[str] = []


@functools.lru_cache(maxsize=1)
def _ssp():
    return tengri.load_ssp()


def _all_balanced_types() -> list[str]:
    excluded = _NOT_BALANCED_BY_DESIGN + _ALIASES_OF_REQUIRED
    return sorted(
        m["name"]
        for m in tengri.list_dust_emission_models()
        if m["status"] in ("production", "experimental") and m["name"] not in excluded
    )


def _integral_over_l_ir(dust_type: str) -> float:
    model = tengri.SEDModel.build(
        _ssp(),
        sfh={"all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_emission={"type": dust_type, "all_params": tengri.Fixed(tengri.DEFAULT)},
        redshift=tengri.Fixed(0.0),
    )
    state = model.predict_state({})
    wave_aa = np.asarray(state.wave, dtype=np.float64)
    sed = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    l_ir = float(np.asarray(state.derived["L_ir"]))
    assert l_ir > 0.0, f"{dust_type}: no absorbed luminosity (L_ir={l_ir})"
    nu = _C_AA_PER_S / wave_aa
    # nu descends with wavelength, so the signed trapezoid is negative.
    return float(-np.trapezoid(sed, nu)) / l_ir


@pytest.mark.parametrize("dust_type", _all_balanced_types())
def test_integral_of_sed_dust_ir_equals_l_ir(dust_type: str):
    ratio = _integral_over_l_ir(dust_type)
    rtol = _RTOL_ANALYTIC if dust_type in _ANALYTIC else _RTOL_TEMPLATE
    assert abs(ratio - 1.0) <= rtol, (
        f"{dust_type}: integral(sed_dust_ir)/L_ir = {ratio:.8f}, |1 - ratio| > {rtol:g}"
    )
    _COMPLETED.append(dust_type)


def test_census_every_required_type_completed():
    """Runs last in file order; a skip, rename, or removal above fails here, not silently."""
    missing = sorted(set(_REQUIRED) - set(_COMPLETED))
    assert not missing, (
        f"required dust emission types did not complete the balance check: {missing}"
    )
    unexpected = sorted(set(_all_balanced_types()) - set(_REQUIRED))
    assert not unexpected, f"new energy-balanced types must be added to _REQUIRED: {unexpected}"
