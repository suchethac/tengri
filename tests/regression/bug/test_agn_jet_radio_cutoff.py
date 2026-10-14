# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the AGN radio jet must not hard-cut at 300 GHz.

Both AGN-jet functions (:func:`radio_agn` SPL and :func:`radio_agn_dpl` DPL)
applied a ``_RADIO_WAVE_MIN_AA = 1e7`` (1 mm = 299.8 GHz) hard wavelength floor
that returned exactly zero above it. That truncated the jet 30x below its own
physical synchrotron-aging cutoff (``log_nu_cut``, default 10 THz) — an AGN jet
extends well into the sub-mm/IR (AGNfitter-rX carries it to 10^15 Hz, 10^18.5
for blazars), and the DPL's ``exp(-nu/nu_cut)`` rollover never got a chance to
act. ``radio_agn`` (SPL) additionally lacked the aging cutoff entirely, so it
was a bare power law truncated at 300 GHz rather than AGNfitter-rX's
``(nu/nu_t)^alpha * exp(-nu/1e13)``.

The fix drops the hard floor from the two AGN-jet functions and lets the
exponential aging cutoff govern the high-frequency rolloff (and adds the
missing cutoff to the SPL). The star-formation radio kept a 1 mm floor then,
on the grounds that it is tied to the dust FIR and must not double-count there.
That floor moved to 100 um (pcigale's radio-module edge) with the synchrotron fix:
the dust-emission SED holds no synchrotron, so adding the power law below 1 mm
double-counts nothing, and every ``L_ir`` window is integrated from the dust SED
alone. The thermal free-free shares the 100 um edge.

References
----------
.. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX," A&A 688, A46 (2024),
   ``functions/MODEL_AGNfitter.py`` ``AGN_RAD`` (agnrad_nu spans 10^7-10^15 Hz;
   SPL branch carries ``exp(-nu/1e13)``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.radio import radio_agn, radio_agn_dpl, radio_sfr_bell2003
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

_GHZ = 1.0e9


def _wave(ghz: float) -> jnp.ndarray:
    """Wavelength [Angstrom] for a frequency in GHz."""
    return jnp.asarray([C_AA / (ghz * _GHZ)])


@pytest.mark.parametrize(
    "fn,kw",
    [
        (radio_agn, dict(radio_loudness=1.0, alpha_agn=0.75)),
        (radio_agn_dpl, dict(radio_loudness=1.0, alpha1=-0.75, alpha2=-0.1, log_nu_cut=13.0)),
    ],
)
def test_agn_jet_extends_past_300ghz(fn, kw):
    """The jet must be finite and positive well above the old 300 GHz floor."""
    for ghz in (350.0, 1000.0, 3000.0):
        val = float(fn(_wave(ghz), 1.0e45, **kw)[0])
        assert np.isfinite(val) and val > 0.0, (
            f"{fn.__name__} is {val} at {ghz} GHz — the jet must not hard-cut at "
            "the 1 mm / 300 GHz floor (it should roll over at its aging cutoff)."
        )


@pytest.mark.parametrize(
    "fn,kw",
    [
        (radio_agn, dict(radio_loudness=1.0, alpha_agn=0.75)),
        (radio_agn_dpl, dict(radio_loudness=1.0, alpha1=-0.75, alpha2=-0.1, log_nu_cut=13.0)),
    ],
)
def test_agn_jet_rolls_over_smoothly(fn, kw):
    """No discontinuity across 300 GHz; monotone decline into the sub-mm."""
    ghz = np.array([100.0, 200.0, 300.0, 500.0, 1000.0, 3000.0, 1.0e4])
    lum = np.array([float(fn(_wave(g), 1.0e45, **kw)[0]) for g in ghz])
    assert np.all(lum > 0.0), "jet must be positive across the turnover"
    # Monotone non-increasing on the falling side (250 GHz upward), no hard step.
    hi = ghz >= 250.0
    assert np.all(np.diff(lum[hi]) < 0.0), "jet must decline smoothly, not step to zero"
    # The aging cutoff must suppress the 10 THz flux far below the 100 GHz level.
    assert lum[-1] < 0.1 * lum[0], "exp aging cutoff should govern the high-nu rolloff"


def test_sf_radio_edge_is_100_um():
    """The SF synchrotron reaches 100 um (pcigale's radio edge) and is zero shortward.

    Dust emission carries no synchrotron, so the power law adds to it without
    double-counting; the 1 mm floor of #1071 is superseded (see module docstring).
    """
    l_ir = 1.0e45
    ref = float(radio_sfr_bell2003(_wave(100.0), l_ir)[0])  # 3 mm, in support
    sub_mm = float(radio_sfr_bell2003(_wave(1000.0), l_ir)[0])  # 0.3 mm
    edge = float(radio_sfr_bell2003(jnp.asarray([1.0e6]), l_ir)[0])  # 100 um
    shortward = float(radio_sfr_bell2003(jnp.asarray([9.9e5]), l_ir)[0])  # 99 um
    assert ref > 0.0, "SF radio must emit within its support"
    assert sub_mm == pytest.approx(ref * (1000.0 / 100.0) ** -0.8, rel=1e-12)
    assert edge > 0.0
    assert shortward == 0.0, "SF radio must stay zero shortward of 100 um"


def test_agn_jet_gradient_flows():
    """The reworked SPL cutoff stays differentiable in log_nu_cut."""

    def total(log_nu_cut):
        w = jnp.asarray([C_AA / (1000.0 * _GHZ)])
        return jnp.sum(radio_agn(w, 1.0e45, radio_loudness=1.0, log_nu_cut=log_nu_cut))

    grad = float(jax.grad(total)(13.0))
    assert np.isfinite(grad) and grad != 0.0, f"non-finite/zero gradient {grad}"
