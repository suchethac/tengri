# SPDX-License-Identifier: BSD-3-Clause
"""Regression: tengri ``noll09`` reproduces the ``dust_attenuation`` N09 law.

tengri's ``noll09`` attenuation curve is a port of the N09 model from the
``dust_attenuation`` package (Noll et al. 2009): a Calzetti+Leitherer base,
a 2175 Å Drude bump of amplitude ``E_b``, and a power-law slope ``delta``,
combined as ``(base + bump) * (lambda / 5500 A)**delta`` and normalized by a
fixed ``R_V = 4.05``.

This pins the parity so the bump amplitude / slope conventions cannot drift.
It is the authoritative answer to the reproduction-notebook question "is the
2175 Å bump faithful?" — the CIGALE ``modified_CF00`` law carries *no* bump
(it is a pure ``(lambda/550)**delta`` power law), so it is not a valid bump
reference; ``dust_attenuation`` N09 is.

References
----------
.. [1] S. Noll, D. Burgarella, E. Giovannoli, et al., "Analysis of galaxy
   spectral energy distributions from far-UV to far-IR with CIGALE,"
   A&A, 507, 1793 (2009). https://doi.org/10.1051/0004-6361/200912497
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.dust.attenuation import noll09

pytestmark = pytest.mark.regression_paper


# (E_b bump amplitude, delta slope) — bump-free, MW-like, and tilted cases.
_PARAM_CASES = [(1.0, 0.0), (3.0, -0.3), (2.0, 0.2)]


@pytest.mark.parametrize("bump_strength,delta", _PARAM_CASES)
def test_noll09_matches_dust_attenuation_n09(bump_strength, delta):
    """tengri noll09 == dust_attenuation N09 to <1% over its valid range.

    Both curves are normalized to A_V at 5500 Å, so the test isolates the
    *shape* (bump + slope), independent of the overall A_V scaling.
    """
    n09_shapes = pytest.importorskip("dust_attenuation.shapes")
    import astropy.units as u

    # N09 is defined for 0.097-2.2 micron; stay safely inside.
    wave_aa = np.linspace(1000.0, 21000.0, 3000)
    i_v = int(np.argmin(np.abs(wave_aa - 5500.0)))

    k_tengri = np.asarray(noll09(wave_aa, dust_bump_strength=bump_strength, dust_delta=delta))
    k_tengri = k_tengri / k_tengri[i_v]

    ref = n09_shapes.N09(Av=1.0, ampl=bump_strength, slope=delta)
    a_ref = np.asarray(ref((wave_aa / 1e4) * u.micron))
    a_ref = a_ref / a_ref[i_v]

    rel = np.abs(k_tengri - a_ref) / np.clip(np.abs(a_ref), 1e-6, None)
    assert rel.max() < 1.0e-2, (
        f"noll09 vs dust_attenuation N09 (E_b={bump_strength}, delta={delta}): "
        f"max relative deviation {rel.max():.2e} exceeds 1%"
    )


def test_noll09_bump_is_real_when_enabled():
    """The 2175 Å Drude bump is present only when E_b > 0.

    Guards the reproduction-notebook footgun: a baseline-subtracted "bump
    excess" on the *default* (E_b=0) curve measures power-law curvature, not a
    feature. With E_b > 0 the curve rises above the local continuum at 2175 Å.
    """
    wave_aa = np.linspace(1900.0, 2500.0, 601)
    i_bump = int(np.argmin(np.abs(wave_aa - 2175.0)))
    continuum = 0.5 * (
        noll09(wave_aa, dust_bump_strength=2.0)[0] + noll09(wave_aa, dust_bump_strength=2.0)[-1]
    )

    no_bump = np.asarray(noll09(wave_aa, dust_bump_strength=0.0))
    with_bump = np.asarray(noll09(wave_aa, dust_bump_strength=2.0))

    excess_off = no_bump[i_bump] - 0.5 * (no_bump[0] + no_bump[-1])
    excess_on = with_bump[i_bump] - continuum
    assert excess_on > 10.0 * abs(excess_off), (
        "Enabling E_b must produce a 2175 Å excess far above the bump-free baseline"
    )
