# SPDX-License-Identifier: BSD-3-Clause
"""The two GOODS-S U-band curves must be the responses they claim to be.

``ctio_u`` and ``vimos_u`` were added so the CANDELS demonstration could fit the
two bluest columns in the workshop catalog. Two things downstream depend on
these curves being the right ones, and neither notices if they are not.

The first is scientific: at z ~ 1 these bands sample the rest-frame ultraviolet
near 1500 A, where the leverage on unobscured young stars and on the
attenuation slope lives. A substituted U response shifts that leverage without
failing anything.

The second is a guard. ``candels_io.assert_igm_node_fold_adequate`` decides
whether the ``WavePrecomp`` IGM node fold is adequate by comparing the observed
Lyman break against the blue edge of the bluest fitted band, and ``ctio_u`` is
that band. The guard reads the same file this test reads, so it cannot tell a
wrong curve under the right filename from a right one -- it would compute a
confident threshold from bad data. Checking the curve against its source is the
only thing that discriminates that case.

Reference values are from the SVO Filter Profile Service entries the registry
names, ``CTIO/MosaicII.U`` and ``Paranal/VIMOS.U``, read at the time the curves
were committed (2026-09-20). Tolerances are a few Angstrom: wide enough for a
resampling or a header convention, far too narrow to admit a different U-band
response. Generic Johnson U starts near 3100 A, SDSS u near 3055, MegaCam u
near 3160 and Swift UVOT U near 3017, so a 5 A window on a 3044 A edge
separates the intended curve from every near neighbor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.observation.filters import FILTER_REGISTRY, load_filter

pytestmark = [pytest.mark.contract]

#: name -> (svo_id, peak_wavelength_A, blue_edge_A, red_edge_A)
#: Measured from the SVO transmission tables when the curves were committed.
EXPECTED = {
    "ctio_u": ("CTIO/MosaicII.U", 3644.0, 3044.0, 4139.0),
    "vimos_u": ("Paranal/VIMOS.U", 3851.0, 3329.0, 4003.8),
}

EDGE_TOL_A = 5.0
PEAK_TOL_A = 10.0


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_registry_names_the_expected_svo_source(name):
    """The registry entry points at the GOODS-S response, not a near neighbor."""
    svo_id = EXPECTED[name][0]
    assert FILTER_REGISTRY[name] == svo_id, (
        f"{name} resolves to {FILTER_REGISTRY[name]!r}, expected {svo_id!r}. "
        f"Another observatory's U band is not a substitute: these two are the "
        f"actual GOODS-S U-band sources and the catalog fluxes were measured "
        f"through them."
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_curve_matches_its_source(name):
    """Peak and span identify the response; a swapped curve fails here."""
    _, peak_expected, blue_expected, red_expected = EXPECTED[name]
    curve = load_filter(name)
    wave = np.asarray(curve.wave, dtype=float)
    trans = np.asarray(curve.trans, dtype=float)

    assert trans.max() > 0, f"{name}: curve has no positive transmission"
    support = wave[trans > 0]
    peak = float(wave[trans.argmax()])

    assert peak == pytest.approx(peak_expected, abs=PEAK_TOL_A), (
        f"{name}: peak transmission at {peak:.1f} A, expected "
        f"{peak_expected:.1f} A. The committed curve is not the response the "
        f"registry names."
    )
    assert float(support.min()) == pytest.approx(blue_expected, abs=EDGE_TOL_A), (
        f"{name}: blue edge {support.min():.1f} A, expected {blue_expected:.1f} A. "
        f"This edge sets the IGM node-fold threshold in "
        f"candels_io.assert_igm_node_fold_adequate; a wrong value there is a "
        f"confident answer computed from the wrong curve."
    )
    assert float(support.max()) == pytest.approx(red_expected, abs=EDGE_TOL_A), (
        f"{name}: red edge {support.max():.1f} A, expected {red_expected:.1f} A."
    )


def test_the_two_u_bands_are_distinct_responses():
    """They are kept as separate measurements, so they must not be one curve.

    Both catalog U columns are fit, unlike the Ks pair where ISAAC and HAWK-I
    collapse onto one VISTA stand-in and only the first detected is taken. That
    asymmetry is only defensible while these really are different bandpasses.
    """
    ctio = load_filter("ctio_u")
    vimos = load_filter("vimos_u")
    ctio_peak = float(np.asarray(ctio.wave)[np.asarray(ctio.trans).argmax()])
    vimos_peak = float(np.asarray(vimos.wave)[np.asarray(vimos.trans).argmax()])
    assert abs(ctio_peak - vimos_peak) > 100.0, (
        f"CTIO U peaks at {ctio_peak:.0f} A and VIMOS U at {vimos_peak:.0f} A. "
        f"If these were the same response, fitting both would enter one "
        f"bandpass twice with correlated errors, which is exactly what the Ks "
        f"first-detected-wins rule exists to avoid."
    )
