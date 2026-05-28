# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE skirtor2016 piecewise-power-law disc-block regressions (#487).

Three empirical disc spectra ported from CIGALE
(``pcigale.sed_modules.skirtor2016``) and registered as composable-AGN
disc blocks:

* ``disc/skirtor``          — SKIRTOR analytic disc (``disk_type=0``)
* ``disc/schartmann2005``   — Schartmann (2005) disc (``disk_type=1``, default)
* ``disc/adaf_lopez2024``   — ADAF↔thin-disc blend (``disk_type=2``)

These tests pin (a) registration, (b) energy conservation
:math:`\\int L_\\lambda\\, d\\lambda = L_{\\rm bol}`, (c) positivity,
(d) JIT-compatibility, and (e) ``delta`` parameter sensitivity.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri.components.agn.blocks  # noqa: F401 — triggers registrations
from tengri.components.agn.blocks._protocol import AGN_BLOCKS, resolve_agn_block

_L_SUN_ERG = 3.828e33
_CIGALE_BLOCKS = ("skirtor", "schartmann2005", "adaf_lopez2024")


@pytest.mark.parametrize("name", _CIGALE_BLOCKS)
def test_block_registered(name: str) -> None:
    assert name in AGN_BLOCKS["disc"]


@pytest.mark.parametrize("name", _CIGALE_BLOCKS)
def test_energy_conservation(name: str) -> None:
    """\\int L_lambda dlambda must equal L_bol (Lsun -> erg/s)."""
    block = resolve_agn_block("disc", name)
    wave_aa = jnp.geomspace(100.0, 1.0e7, 600)  # 10 nm -> 1 mm
    log_lbol = 10.0
    L_lambda = block(wave_aa, log_lbol)
    L_int = float(jnp.trapezoid(L_lambda, wave_aa))
    L_expected = (10.0**log_lbol) * _L_SUN_ERG
    np.testing.assert_allclose(L_int, L_expected, rtol=0.01)


@pytest.mark.parametrize("name", _CIGALE_BLOCKS)
def test_positivity(name: str) -> None:
    block = resolve_agn_block("disc", name)
    wave_aa = jnp.geomspace(100.0, 1.0e7, 300)
    L_lambda = block(wave_aa, 10.0)
    assert jnp.all(L_lambda >= 0.0)
    chex.assert_equal_shape([L_lambda, wave_aa])


@pytest.mark.parametrize("name", _CIGALE_BLOCKS)
def test_lbol_scales_linearly(name: str) -> None:
    """Doubling L_bol must exactly double L_lambda at every wavelength."""
    block = resolve_agn_block("disc", name)
    wave_aa = jnp.geomspace(100.0, 1.0e7, 200)
    L1 = block(wave_aa, 10.0)
    L2 = block(wave_aa, 10.0 + jnp.log10(2.0))
    np.testing.assert_allclose(np.asarray(L2), 2.0 * np.asarray(L1), rtol=1e-5)


@pytest.mark.parametrize("name", _CIGALE_BLOCKS)
def test_jit_compatible(name: str) -> None:
    block = resolve_agn_block("disc", name)
    wave_aa = jnp.geomspace(100.0, 1.0e7, 200)
    jitted = jax.jit(lambda wl, lb: block(wl, lb))
    L = jitted(wave_aa, 10.0)
    assert jnp.all(jnp.isfinite(L))


@pytest.mark.parametrize("name", ("skirtor", "schartmann2005"))
def test_delta_modulates_slope(name: str) -> None:
    """delta must change the spectrum shape for slope-modulator discs."""
    block = resolve_agn_block("disc", name)
    wave_aa = jnp.geomspace(100.0, 1.0e7, 300)
    L0 = block(wave_aa, 10.0, agn_cigale_disk_delta=0.0)
    Lp = block(wave_aa, 10.0, agn_cigale_disk_delta=0.5)
    Lm = block(wave_aa, 10.0, agn_cigale_disk_delta=-0.5)
    # Both perturbations must move the spectrum.
    assert float(jnp.mean(jnp.abs(Lp - L0)) / jnp.mean(L0)) > 1e-3
    assert float(jnp.mean(jnp.abs(Lm - L0)) / jnp.mean(L0)) > 1e-3


def test_adaf_blend_endpoints_differ() -> None:
    """ADAF-Lopez2024 must produce distinct spectra at delta=0 and delta=1."""
    block = resolve_agn_block("disc", "adaf_lopez2024")
    wave_aa = jnp.geomspace(100.0, 1.0e7, 300)
    L_adaf = block(wave_aa, 10.0, agn_cigale_disk_delta=0.0)
    L_disc = block(wave_aa, 10.0, agn_cigale_disk_delta=1.0)
    rel_diff = float(jnp.mean(jnp.abs(L_adaf - L_disc)) / jnp.mean(L_adaf))
    assert rel_diff > 0.1  # the two extremes are far apart


def test_skirtor_and_schartmann_differ() -> None:
    """The two disc shapes must be distinguishable.

    The two power laws share the optical slope (-1.5) but differ in the
    short-wavelength breakpoints (8/10/100 nm vs 8/50/125 nm), so we
    look at the FUV/NUV regime where the disagreement is largest.
    """
    sk = resolve_agn_block("disc", "skirtor")
    sc = resolve_agn_block("disc", "schartmann2005")
    wave_aa = jnp.geomspace(100.0, 2000.0, 200)  # 10-200 nm
    L_sk = sk(wave_aa, 10.0)
    L_sc = sc(wave_aa, 10.0)
    rel = float(jnp.max(jnp.abs(L_sk - L_sc)) / jnp.max(L_sk + L_sc))
    assert rel > 0.05
