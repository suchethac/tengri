# SPDX-License-Identifier: BSD-3-Clause
"""Tests for NEV variability and bolometric quantities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


# ---- NEV ----
def test_nev_low_lum_capped_at_0p1():
    from tengri.components.agn.grahsp.variability import normalised_excess_variance

    out = float(normalised_excess_variance(1.0e40))
    assert out == pytest.approx(0.1)


def test_nev_at_l45_eq_1_matches_paper():
    """At L_bol = 1e45, NEV = 10^-1.43 = 0.0372 (Buchner+ 2024 Eq. NEV)."""
    from tengri.components.agn.grahsp.variability import normalised_excess_variance

    out = float(normalised_excess_variance(1.0e45))
    assert out == pytest.approx(10.0**-1.43, rel=1e-12)


def test_nev_decreases_with_luminosity():
    from tengri.components.agn.grahsp.variability import normalised_excess_variance

    L = jnp.array([1.0e44, 1.0e45, 1.0e46, 1.0e47])
    out = np.asarray(normalised_excess_variance(L))
    # Strictly decreasing in the unsaturated regime.
    assert np.all(np.diff(out) < 0)


def test_nev_jit():
    import jax

    from tengri.components.agn.grahsp.variability import normalised_excess_variance

    fn = jax.jit(normalised_excess_variance)
    assert float(fn(1.0e45)) == pytest.approx(10.0**-1.43, rel=1e-12)


# ---- Bolometric ----
def test_lumbol_bbb_excludes_below_lyman_limit():
    from tengri.components.agn.grahsp.bolometric import (
        LYMAN_LIMIT_NM,
        bolometric_luminosity_bbb,
    )

    wave = jnp.linspace(50.0, 1000.0, 1001)
    L = jnp.ones_like(wave)
    out = float(bolometric_luminosity_bbb(wave, L))
    expected = 1000.0 - LYMAN_LIMIT_NM
    # Trapezoidal mask edge has 1-bin-width slop; allow 1 grid spacing.
    dlam = float(wave[1] - wave[0])
    assert abs(out - expected) < dlam


def test_lumbol_torus_full_integral():
    from tengri.components.agn.grahsp.bolometric import bolometric_luminosity_torus

    wave = jnp.linspace(1000.0, 100000.0, 1001)
    L = jnp.ones_like(wave)
    out = float(bolometric_luminosity_torus(wave, L))
    expected = 100000.0 - 1000.0
    assert out == pytest.approx(expected, rel=1e-6)


def test_fracagn_dale_zero_when_no_agn():
    from tengri.components.agn.grahsp.bolometric import agn_fraction_dale

    wave = jnp.linspace(3000.0, 30000.0, 1001)
    L_agn = jnp.zeros_like(wave)
    L_gal = jnp.ones_like(wave)
    out = float(agn_fraction_dale(wave, L_agn, L_gal))
    assert out == 0.0


def test_fracagn_dale_one_when_no_galaxy():
    from tengri.components.agn.grahsp.bolometric import agn_fraction_dale

    wave = jnp.linspace(3000.0, 30000.0, 1001)
    L_agn = jnp.ones_like(wave)
    L_gal = jnp.zeros_like(wave)
    out = float(agn_fraction_dale(wave, L_agn, L_gal))
    assert out == pytest.approx(1.0)


def test_fracagn_dale_half():
    from tengri.components.agn.grahsp.bolometric import agn_fraction_dale

    wave = jnp.linspace(3000.0, 30000.0, 1001)
    L_agn = jnp.full_like(wave, 2.0)
    L_gal = jnp.full_like(wave, 2.0)
    out = float(agn_fraction_dale(wave, L_agn, L_gal))
    assert out == pytest.approx(0.5)
