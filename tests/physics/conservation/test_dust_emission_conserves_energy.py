# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: every tabulated dust emitter re-emits exactly what it absorbed.

The dust IR templates are energy-balance normalized — the whole forward model
depends on ``int L_nu dnu == L_absorbed``. This test pins that invariant *at the
closure*, which is the only place it can be measured cleanly.

Why not rely on ``tests/contract/test_dust_energy_balance.py``: that test infers
``L_abs`` from ``int (intrinsic - attenuated) dnu`` over a 912 A - 3 um window,
which undercounts the true absorbed energy by 1-3%. That systematic is *larger*
than the kind of leak this guards against, so it cannot resolve one. Feeding the
closure a known ``L_absorbed`` and integrating its output has no such proxy.

Regression: the da Cunha et al. (2013) CMB contrast factor used to be applied to
the *emitted* SED of THEMIS / Astrodust / BOSA, after the unit-integral
renormalization. It is an observational suppression (flux measured above the CMB
background), so it silently destroyed up to 1.6% of the absorbed energy — worst
for the aromatic-rich, mm-bright THEMIS templates — while DL07/DL14 never applied
it and CIGALE conserves to 0.01%. It also never received a real redshift (no
component plumbs one through), so it only ever imposed the z=0 suppression.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.conservation

_C_AA_S = 2.998e18  # [Angstrom/s]


def _bolometric(l_nu: np.ndarray, wave_aa: np.ndarray) -> float:
    """Integrate L_nu over frequency."""
    nu = _C_AA_S / wave_aa
    order = np.argsort(nu)
    return float(np.trapezoid(np.asarray(l_nu)[order], nu[order]))


def _emitters():
    """(name, callable, wave grid) for every tabulated dust emitter."""
    from tengri.components.dust.emission import (
        dale2014,
        draine_li2007,
        draine_li2014,
        themis,
    )

    # Each grid must span its model's full template range, or the integral
    # truncates the Rayleigh-Jeans tail and reports a false deficit.
    wide = np.logspace(np.log10(4.0e3), np.log10(2.0e10), 60000)
    themis_grid = np.logspace(np.log10(9.93e3), np.log10(3.057e7), 40000)

    return [
        (
            "themis",
            lambda w: themis(
                w, 1.0, dust_umin=1.0, dust_gamma_dl=0.1, dust_qhac=0.17, dust_alpha=1.0
            ),
            themis_grid,
        ),
        (
            "themis_alpha3",
            lambda w: themis(
                w, 1.0, dust_umin=1.0, dust_gamma_dl=0.1, dust_qhac=0.17, dust_alpha=3.0
            ),
            themis_grid,
        ),
        (
            "draine_li2007",
            lambda w: draine_li2007(w, 1.0, dust_umin=1.0, dust_gamma_dl=0.1, dust_qpah=3.0),
            wide,
        ),
        (
            "draine_li2014",
            lambda w: draine_li2014(w, 1.0, dust_umin=1.0, dust_gamma_dl=0.1, dust_qpah=3.0),
            wide,
        ),
        ("dale2014", lambda w: dale2014(w, 1.0, dust_alpha=2.0), wide),
    ]


@pytest.mark.parametrize(
    "name,fn,wave", _emitters(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_dust_emitter_conserves_absorbed_energy(name, fn, wave) -> None:
    """Feeding L_absorbed = 1 must give back a unit bolometric luminosity."""
    emitted = _bolometric(fn(wave), wave)
    assert abs(emitted - 1.0) < 0.01, (
        f"{name} re-emits {emitted:.4f} of the absorbed energy (expected 1.0 +/- 1%). "
        f"A factor applied after the unit-integral renormalization (e.g. an "
        f"observational CMB contrast) leaks energy out of the dust budget."
    )
