# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the astropy-cosmology adapter + PLANCK18 drift fix (#401).

Three changes are pinned here:

1. ``tengri.cosmology.PLANCK18`` matches astropy's canonical
   ``Planck18`` to floating-point precision (drift bug from the
   pre-#401 hard-coded values).
2. ``cosmo_from_astropy`` correctly extracts ``Om0``, ``h``, ``w0``,
   ``wa`` for the supported flat-cosmology families.
3. ``_resolve_cosmo`` accepts ``w0`` and ``wa`` scalar kwargs.
"""

from __future__ import annotations

import pytest

from tengri.cosmology import PLANCK18, CosmoParams, cosmo_from_astropy
from tengri.utils.cosmology import _resolve_cosmo

pytestmark = pytest.mark.contract


def test_planck18_matches_astropy_within_floating_point():
    """``tengri.PLANCK18`` and ``astropy.cosmology.Planck18`` must agree
    on ``Om0`` and ``h`` to <1e-4 relative — closes the drift bug from
    #401 where tengri shipped (0.315, 0.674) while astropy has
    (0.30966, 0.6766)."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    Planck18 = astropy_cosmo.Planck18

    h_astropy = float(Planck18.H0.value) / 100.0
    om0_astropy = float(Planck18.Om0)

    assert abs(PLANCK18.h - h_astropy) < 1e-4, (
        f"PLANCK18.h = {PLANCK18.h}, astropy.Planck18.h = {h_astropy}"
    )
    assert abs(PLANCK18.Om0 - om0_astropy) < 1e-4, (
        f"PLANCK18.Om0 = {PLANCK18.Om0}, astropy.Planck18.Om0 = {om0_astropy}"
    )


def test_cosmo_from_astropy_planck18_round_trip():
    """``cosmo_from_astropy(Planck18)`` reproduces tengri.PLANCK18 exactly."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    cp = cosmo_from_astropy(astropy_cosmo.Planck18)

    assert abs(cp.Om0 - PLANCK18.Om0) < 1e-6
    assert abs(cp.h - PLANCK18.h) < 1e-6
    assert cp.w0 == -1.0
    assert cp.wa == 0.0


def test_cosmo_from_astropy_extracts_w0_wa():
    """For Flatw0waCDM, the adapter must pull ``w0`` and ``wa`` correctly."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    de = astropy_cosmo.Flatw0waCDM(H0=70, Om0=0.3, w0=-0.95, wa=-0.05)
    cp = cosmo_from_astropy(de)

    assert abs(cp.h - 0.7) < 1e-6
    assert abs(cp.Om0 - 0.3) < 1e-6
    assert abs(cp.w0 - (-0.95)) < 1e-6
    assert abs(cp.wa - (-0.05)) < 1e-6


def test_cosmo_from_astropy_rejects_non_flat():
    """LambdaCDM with Ode0 ≠ 1 - Om0 must raise — DSPS only does flat."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    non_flat = astropy_cosmo.LambdaCDM(H0=70, Om0=0.3, Ode0=0.5)
    with pytest.raises(ValueError, match="flat cosmologies only"):
        cosmo_from_astropy(non_flat)


def test_cosmo_from_astropy_accepts_planck18_with_neutrinos():
    """Planck18's flatness includes a small Onu0; the adapter must
    recognize this as flat (Ok0 = 0) and not be confused by
    ``Ode0 + Om0 ≈ 0.9985`` (which would falsely trip a naïve
    flatness check)."""
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    # Planck18 has Onu0 ≈ 0.00144 — should still pass.
    cp = cosmo_from_astropy(astropy_cosmo.Planck18)
    assert isinstance(cp, CosmoParams)


def test_resolve_cosmo_accepts_w0_wa_kwargs():
    """``_resolve_cosmo(w0=..., wa=...)`` constructs a CosmoParams with
    the supplied values and PLANCK18 defaults for the rest."""
    cp = _resolve_cosmo(w0=-0.9, wa=0.1)
    assert isinstance(cp, CosmoParams)
    assert abs(cp.w0 - (-0.9)) < 1e-12
    assert abs(cp.wa - 0.1) < 1e-12
    # Fell back to PLANCK18 defaults for Om0/h.
    assert abs(cp.Om0 - PLANCK18.Om0) < 1e-12
    assert abs(cp.h - PLANCK18.h) < 1e-12


def test_resolve_cosmo_rejects_mixed_cosmo_and_scalar():
    """Passing both a ``cosmo`` object and scalar kwargs is an error."""
    with pytest.raises(ValueError, match="not both"):
        _resolve_cosmo(cosmo=PLANCK18, w0=-0.9)


def test_resolve_cosmo_combines_h0_w0():
    """``_resolve_cosmo(h0=70, w0=-0.9)`` builds the full CosmoParams,
    pulling Om0/wa from defaults — verifies the issue's example."""
    cp = _resolve_cosmo(h0=70, w0=-0.9)
    assert abs(cp.h - 0.7) < 1e-12
    assert abs(cp.w0 - (-0.9)) < 1e-12
    assert cp.wa == 0.0  # default
    assert abs(cp.Om0 - PLANCK18.Om0) < 1e-12  # default
