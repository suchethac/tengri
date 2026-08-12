# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE skirtor2016 piecewise-power-law disc-block regressions (#487).

Three empirical disc spectra as implemented in CIGALE
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
import jax.numpy as jnp
import numpy as np
import pytest

import tengri.components.agn.blocks  # noqa: F401 — triggers registrations
from tengri.components.agn.blocks._protocol import AGN_BLOCKS, resolve_agn_block
from tests._jit_parity import assert_jit_matches_eager

# Module taxonomy: most cases verify the registry/adapter contract; the
# energy-conservation test below carries an explicit ``conservation`` marker.
pytestmark = pytest.mark.contract

_L_SUN_ERG = 3.828e33
_CIGALE_BLOCKS = ("skirtor", "schartmann2005", "adaf_lopez2024")


@pytest.mark.parametrize("name", _CIGALE_BLOCKS)
def test_block_registered(name: str) -> None:
    assert name in AGN_BLOCKS["disc"]


@pytest.mark.conservation
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


@pytest.mark.bounds
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
    L = assert_jit_matches_eager(lambda wl, lb: block(wl, lb), wave_aa, 10.0)
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


# ──────────────────────────────────────────────────────────────────────
# Polar dust integration in torus/skirtor (CIGALE skirtor2016 parity)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_skirtor_torus_polar_dust_on_by_default() -> None:
    """agn_polar_ebv default = 0.03 (CIGALE skirtor2016 default); the
    block's keyword default must match the param-spec default and
    produce a non-zero polar-dust contribution."""
    torus = resolve_agn_block("torus", "skirtor")
    wave_aa = jnp.geomspace(1e3, 1e7, 300)
    L_default = torus(wave_aa, agn_log_lbol=-0.42, l5100_disc=jnp.zeros_like(wave_aa))
    L_off = torus(
        wave_aa,
        agn_log_lbol=-0.42,
        l5100_disc=jnp.zeros_like(wave_aa),
        agn_polar_ebv=0.0,
    )
    # Default must DIFFER from explicit-off (proves polar dust is on).
    assert float(jnp.max(jnp.abs(L_default - L_off))) > 0.0


@pytest.mark.conservation
def test_skirtor_torus_polar_dust_redistributes_energy() -> None:
    """Polar dust redistributes energy from SKIRTOR thermal-dust peak
    to the FIR tail — total integrated IR luminosity is conserved
    (matches CIGALE ``skirtor2016.py:389`` where ``norm = 1/∫(dust +
    polar)`` includes both contributions). Polar-on lifts the FIR,
    polar-off lifts the MIR peak; total stays the same.
    """
    torus = resolve_agn_block("torus", "skirtor")
    wave_aa = jnp.geomspace(1e3, 1e7, 300)
    L_off = torus(
        wave_aa,
        agn_log_lbol=-0.42,
        l5100_disc=jnp.zeros_like(wave_aa),
        agn_polar_ebv=0.0,
    )
    L_on = torus(
        wave_aa,
        agn_log_lbol=-0.42,
        l5100_disc=jnp.zeros_like(wave_aa),
        agn_polar_ebv=0.03,
        agn_polar_T=100.0,
        agn_polar_beta=1.6,
        agn_oa_skirtor=40.0,
    )
    # Total IR luminosity should be conserved (within numerical precision)
    int_off = float(jnp.trapezoid(L_off, wave_aa))
    int_on = float(jnp.trapezoid(L_on, wave_aa))
    np.testing.assert_allclose(int_on, int_off, rtol=0.01)
    # FIR (100 µm) gets the polar bump
    i100 = int(np.argmin(np.abs(np.asarray(wave_aa) - 1.0e6)))
    assert float(L_on[i100]) > float(L_off[i100])


@pytest.mark.conservation
def test_skirtor_torus_polar_dust_lifts_fir_tail() -> None:
    """At the §9 CIGALE fiducial, polar dust must lift the 100 µm tail
    by a factor >2 — the regression that motivated the audit."""
    torus = resolve_agn_block("torus", "skirtor")
    wave_aa = jnp.geomspace(1e3, 1e7, 400)
    L_off = torus(
        wave_aa,
        agn_log_lbol=-0.42,
        l5100_disc=jnp.zeros_like(wave_aa),
        agn_polar_ebv=0.0,
    )
    L_on = torus(
        wave_aa,
        agn_log_lbol=-0.42,
        l5100_disc=jnp.zeros_like(wave_aa),
        agn_polar_ebv=0.03,
        agn_polar_T=100.0,
        agn_polar_beta=1.6,
        agn_oa_skirtor=40.0,
    )
    # Index nearest 100 µm:
    i100 = int(np.argmin(np.abs(np.asarray(wave_aa) - 1.0e6)))
    ratio = float(L_on[i100] / L_off[i100])
    assert ratio > 2.0, f"100 um lift {ratio:.2f}x — expected >2x"
