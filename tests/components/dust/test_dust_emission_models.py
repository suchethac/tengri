# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust emission template models (MBB, DL07, DL14, Casey).

Every dust-emission template must peak in IR and conserve energy (∫L_ν dν ≈ L_absorbed).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


class TestDustEmissionCombinations:
    """Every dust-emission template should peak in IR and conserve L_absorbed."""

    @pytest.mark.parametrize(
        "name", ["modified_blackbody", "casey2012", "dale2014", "draine_li2007"]
    )
    def test_dust_emission_peaks_in_ir(self, name):
        """Peak λ must be in IR (8–1000 μm) for cold-dust parameters."""
        import tengri.dust as dust_mod

        fn = getattr(dust_mod, name, None)
        if fn is None:
            pytest.skip(f"{name} not available in tengri.dust")
        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        try:
            if name == "modified_blackbody":
                L = np.array(fn(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8))
            elif name == "casey2012":
                L = np.array(fn(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0))
            elif name == "dale2014":
                L = np.array(fn(wl, 1e44, dust_alpha_dale=2.0))
            else:
                L = np.array(fn(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        except FileNotFoundError:
            pytest.skip(f"{name} requires data files not present")
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 8.0 < peak_um < 1000.0, f"{name}: peak at {peak_um:.1f} μm"

    @pytest.mark.parametrize("name", ["modified_blackbody", "casey2012", "draine_li2007"])
    def test_dust_emission_energy_balance(self, name):
        """∫L_ν dν ≈ L_absorbed (within 10% trapezoid tolerance)."""
        import tengri.dust as dust_mod

        fn = getattr(dust_mod, name, None)
        if fn is None:
            pytest.skip(f"{name} not available")
        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 5000)
        try:
            if name == "modified_blackbody":
                L = np.array(fn(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8))
            elif name == "casey2012":
                L = np.array(fn(wl, 1e44, dust_T=35.0))
            else:
                L = np.array(fn(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        except FileNotFoundError:
            pytest.skip(f"{name} requires data files not present")
        nu = 2.998e18 / np.array(wl)
        L_int = -np.trapezoid(L, nu)
        assert 0.9 < L_int / 1e44 < 1.1, (
            f"{name}: ∫L_ν dν / L_abs = {L_int / 1e44:.3f}, expected ≈1"
        )
