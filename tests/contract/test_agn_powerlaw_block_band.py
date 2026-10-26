# SPDX-License-Identifier: BSD-3-Clause
"""The power-law AGN disc is band-limited at its source, and every caller inherits it.

``disc.powerlaw_disc`` normalizes its shape over 10 A to 1e8 A (``_PL_BAND_LO_AA`` to
``_PL_BAND_HI_AA``, both edges inclusive) and is zero outside that band: the band is the model's
definition, and beyond it the power-law tail carried up to 1e4 times the budget, counted by
nothing. The three callers are the standalone ``PowerLawDisc``, the composable
``disc='powerlaw'`` block, and its photometry lookup table (``disc_precompute``). This file
holds all three to the band:

* edges: nonzero at exactly 10 A and 1e8 A, exactly zero at 9.999 A and 1.0001e8 A;
* the block's ledger: its power on the normalization's own nodes is the budget
  ``10**agn_log_lbol L_sun`` to 1e-10 (C2), and a dense integral over 1e-3 A to 1e10 A, far
  past both edges, is the budget to 1e-6 (C3); its registered closed-form power is 1;
* the lookup table agrees with the band-limited SED on filters straddling either edge, and
  holds nothing for a filter outside the band;
* downstream, a model whose only AGN emitter is this disc puts nothing at 1.4 GHz or in the
  hard X-ray: by design, since other components own those regimes.
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import disc_precompute
from tengri.components.agn._params import DEFAULT_AGN_LUM_RATIO
from tengri.components.agn._publication import log10_native_bolometric
from tengri.components.agn.blocks import _protocol, resolve_agn_block
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.components.agn.disc import _PL_BAND_HI_AA, _PL_BAND_LO_AA, powerlaw_disc
from tengri.components.agn.powerlaw_disc_model import PowerLawDisc
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = [
    pytest.mark.contract,
    pytest.mark.filterwarnings("ignore:powerlaw_disc is deprecated:DeprecationWarning"),
]

#: The band edges, inclusive [Angstrom].
_EDGES = np.array([_PL_BAND_LO_AA, _PL_BAND_HI_AA])
#: Just outside each edge [Angstrom].
_JUST_OUTSIDE = np.array([9.999, 1.0001e8])
#: Far outside: hard X-ray and radio [Angstrom].
_FAR_OUTSIDE = np.array([1.0e-3, 1.0, 1.0e9, 1.0e10])

#: (agn_log_lbol, agn_alpha, agn_T_max) across the declared box: alpha in [-2, 0], T_max in
#: [1e4, 1e6] K. T_max = 1e6 keeps the 10 A edge well above float64 underflow.
_POINTS = [
    (10.0, -1.0, 1.0e5),
    (12.5, -2.0, 1.0e6),
    (8.3, 0.0, 1.0e6),
    (11.0, -0.5, 3.0e5),
]

#: The normalization's own nodes: 8001 log-uniform from 10 A to 1e8 A, end points exact.
_BAND_NODES = np.geomspace(_PL_BAND_LO_AA, _PL_BAND_HI_AA, 8001)
#: 1e-3 A to 1e10 A, dense, with both edges and a node a relative 1e-12 outside each, so the
#: step at either edge costs the trapezoid nothing.
_WIDE = np.unique(
    np.concatenate(
        [
            np.geomspace(1.0e-3, 1.0e10, 13001),
            np.geomspace(_PL_BAND_LO_AA, _PL_BAND_HI_AA, 100_001),
            _EDGES * np.array([1.0 - 1.0e-12, 1.0 + 1.0e-12]),
        ]
    )
)

_C2_RTOL = 1.0e-10
_C3_RTOL = 1.0e-6


def _shared(wave, lbol, alpha, t_max):
    """``powerlaw_disc`` L_nu [erg/s/Hz] in float64, ``lum_ratio = 1``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return np.asarray(
            powerlaw_disc(jnp.asarray(wave, dtype=jnp.float64), lbol, 1.0, alpha, t_max)
        )


def _block_l_nu(wave, lbol, alpha, t_max):
    """The composable ``disc='powerlaw'`` block as L_nu [erg/s/Hz] in float64."""
    w = jnp.asarray(wave, dtype=jnp.float64)
    l_lam = resolve_agn_block("disc", "powerlaw")(w, lbol, agn_alpha=alpha, agn_T_max=t_max)
    return np.asarray(l_lam) * np.asarray(wave) ** 2 / C_AA


def _standalone(wave, lbol, alpha, t_max):
    """The standalone ``PowerLawDisc`` SED L_nu [erg/s/Hz] in float64, ``lum_ratio = 1``."""
    w = jnp.asarray(wave, dtype=jnp.float64)
    p = {k: jnp.float64(v) for k, v in zip(("log_lbol", "alpha", "T_max"), (lbol, alpha, t_max))}
    sed, _ = PowerLawDisc().predict({**p, "lum_ratio": jnp.float64(1.0)}, jnp.zeros_like(w), w)
    return np.asarray(sed)


def _ln_nu_power(l_nu, wave):
    """Trapezoid in ln(nu) of L_nu nu [erg/s] on ``wave`` [Angstrom]."""
    nu = C_AA / np.asarray(wave)
    order = np.argsort(nu)
    return float(np.trapezoid((np.asarray(l_nu) * nu)[order], np.log(nu[order])))


_CALLERS = {"shared": _shared, "block": _block_l_nu, "standalone": _standalone}


@pytest.mark.bounds
@pytest.mark.parametrize("caller", sorted(_CALLERS))
def test_band_edges_are_inclusive(caller):
    """Nonzero at exactly 10 A and 1e8 A; exactly zero at 9.999 A, 1.0001e8 A and beyond."""
    fn = _CALLERS[caller]
    for point in _POINTS:
        at_edges = fn(_EDGES, *point)
        assert np.all(at_edges > 0.0), f"{caller} at {point}: edge L_nu {at_edges}"
        outside = fn(np.concatenate([_JUST_OUTSIDE, _FAR_OUTSIDE]), *point)
        assert np.all(outside == 0.0), f"{caller} at {point}: outside-band L_nu {outside}"


@pytest.mark.conservation
def test_block_power_on_the_normalization_nodes_is_the_budget():
    """C2: on the normalization's own nodes the block's power is ``10**log_lbol L_sun``."""
    for lbol, alpha, t_max in _POINTS:
        l_nu = jnp.asarray(_block_l_nu(_BAND_NODES, lbol, alpha, t_max))
        log_power = float(log10_native_bolometric(l_nu, jnp.asarray(_BAND_NODES)))
        err = abs(10.0 ** (log_power - lbol - math.log10(L_SUN)) - 1.0)
        assert err < _C2_RTOL, f"block at {(lbol, alpha, t_max)}: power / budget - 1 = {err:.2e}"


@pytest.mark.conservation
def test_block_integral_far_past_both_edges_is_the_budget():
    """C3: integrated from 1e-3 A to 1e10 A the block's power is still the budget, to 1e-6."""
    for lbol, alpha, t_max in _POINTS:
        power = _ln_nu_power(_block_l_nu(_WIDE, lbol, alpha, t_max), _WIDE)
        err = abs(power / (10.0**lbol * L_SUN) - 1.0)
        assert err < _C3_RTOL, f"block at {(lbol, alpha, t_max)}: power / budget - 1 = {err:.2e}"


@pytest.mark.contract
def test_block_registers_its_closed_form_power():
    """The conserving line debit reads the block's power in closed form: exactly ``L_acc``."""
    assert float(_protocol.DISC_POWER_BLOCKS["powerlaw"](11.0)) == 1.0


@pytest.mark.gradient
def test_block_gradients_stay_finite_across_the_edges():
    """The truncation leaves finite, nonzero gradients in log_lbol, alpha and T_max."""
    wave = jnp.asarray(np.geomspace(1.0, 1.0e9, 400))
    block = resolve_agn_block("disc", "powerlaw")

    def total(lbol, alpha, t_max):
        l_lam = block(wave, lbol, agn_alpha=alpha, agn_T_max=t_max)
        return jnp.sum(jnp.log(jnp.where(l_lam > 0.0, l_lam, 1.0)))

    grads = jax.grad(total, argnums=(0, 1, 2))(
        jnp.float64(10.0), jnp.float64(-1.0), jnp.float64(2.0e5)
    )
    assert all(np.isfinite(float(g)) and float(g) != 0.0 for g in grads), grads


# ── The lookup table ────────────────────────────────────────────────────────────────


def _tophat(lo, hi):
    """A top-hat filter [Angstrom], observed frame, 4001 log-uniform nodes."""
    wave = np.geomspace(lo, hi, 4001)
    return wave, np.ones_like(wave)


def _band_limited_photometry(fw, z, alpha):
    """Photon-weighted band average of the band-limited SED [erg/s/Hz per L_sun].

    Integrated over the in-band part of the filter only, with that part's ends as nodes:
    independent of the table's grid and of any interpolation across the step.
    """
    lo, hi = max(fw[0], _PL_BAND_LO_AA * (1 + z)), min(fw[-1], _PL_BAND_HI_AA * (1 + z))
    if lo >= hi:
        return 0.0
    x = np.geomspace(lo, hi, 200_001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        l_nu = np.asarray(
            powerlaw_disc(jnp.asarray(x / (1 + z)), 0.0, DEFAULT_AGN_LUM_RATIO, alpha, 1.0e5)
        )
    return np.trapezoid(l_nu / x, x) / np.trapezoid(1.0 / fw, fw)


#: Rest-frame filter ranges [Angstrom] and the relative tolerance of each.
_LUT_FILTERS = {
    # Away from the low edge the floor is linear interpolation of L_nu ~ lambda**-alpha between
    # nodes 0.004 dex apart: zero for alpha = -1, measured at most 1.2e-5 (alpha = -1.5).
    "optical (in band)": ((2.0e3, 3.0e3), 3e-5),
    "far-IR to radio (in band, past 1e5 A)": ((3.0e5, 1.0e6), 3e-5),
    "straddles 1e8 A": ((5.0e7, 2.0e8), 3e-5),
    "straddles 1e8 A, narrow": ((9.0e7, 1.1e8), 3e-5),
    # At the low edge it is the linear interpolant of the exponential cutoff
    # (h nu / k T_max = 144 at 10 A), measured 6.5e-5.
    "straddles 10 A": ((8.0, 12.0), 2e-4),
    "straddles 10 A, wide": ((5.0, 20.0), 2e-4),
}
_LUT_OUTSIDE = {"hard X-ray": (1.0, 9.0), "radio": (2.0e8, 5.0e8)}


@pytest.mark.conservation
@pytest.mark.parametrize("z", [0.0, 1.0, 2.7])
def test_lookup_table_matches_the_band_limited_sed(z):
    """The table agrees with the band-limited SED on filters straddling either edge."""
    names = list(_LUT_FILTERS) + list(_LUT_OUTSIDE)
    ranges = [_LUT_FILTERS[n][0] for n in _LUT_FILTERS] + list(_LUT_OUTSIDE.values())
    filters = [_tophat(lo * (1 + z), hi * (1 + z)) for lo, hi in ranges]
    alpha_grid = np.linspace(-2.0, 0.0, 9)
    table = disc_precompute.precompute(
        [f[0] for f in filters],
        [f[1] for f in filters],
        redshift=z,
        parameters=None,
        model="powerlaw_disc",
        alpha_grid=alpha_grid,
    )
    lookup = disc_precompute.build_lookup(table, model="powerlaw_disc")
    for alpha in (-1.5, -1.0, -0.25):  # on the table's alpha nodes
        lut = np.asarray(lookup(jnp.float64(0.0), jnp.float64(alpha)))
        scale = float(np.max(lut))
        for i, name in enumerate(names):
            if name in _LUT_OUTSIDE:
                assert lut[i] < 1e-280 * scale, f"z={z} a={alpha} {name}: table holds {lut[i]}"
                continue
            ref = _band_limited_photometry(filters[i][0], z, alpha)
            err = abs(lut[i] / ref - 1.0)
            rtol = _LUT_FILTERS[name][1]
            assert err < rtol, f"z={z} a={alpha} {name}: table / exact - 1 = {err:.2e}"


@pytest.mark.contract
def test_lookup_table_grid_spans_the_band_exactly():
    """The table's nodes reach both edges, with a zero-valued node just outside each."""
    wave = disc_precompute._POWERLAW_REST_WAVE_AA
    assert np.all(np.diff(wave) > 0.0)
    assert _PL_BAND_LO_AA in wave and _PL_BAND_HI_AA in wave
    assert wave[0] < _PL_BAND_LO_AA and wave[-1] > _PL_BAND_HI_AA
    outer = _shared(wave[[0, -1]], 0.0, -1.0, 1.0e5)
    assert np.all(outer == 0.0)


# ── Downstream: radio and hard X-ray ───────────────────────────────────────────────

#: 1.4 GHz, 10 GHz [Angstrom]; 2 keV, 10 keV, 30 keV [Angstrom].
_RADIO_AA = C_AA / np.array([1.4e9, 1.0e10])
_HARD_XRAY_AA = 12.398419843320026 / np.array([2.0, 10.0, 30.0])


@pytest.mark.limit
def test_powerlaw_disc_puts_nothing_in_the_radio_or_hard_xray():
    """With the power-law disc as the only AGN emitter, radio and hard X-ray are exactly zero.

    By design: the band 10 A - 1e8 A is the model's definition, and the radio and hard X-ray
    belong to other components (radio, corona, X-ray binaries).
    """
    wave = np.sort(np.concatenate([_RADIO_AA, _HARD_XRAY_AA, [5.0e3]]))
    probe = (wave < _PL_BAND_LO_AA) | (wave > _PL_BAND_HI_AA)
    for lbol, alpha, t_max in _POINTS:
        composed = np.asarray(
            composable_agn_l_nu(
                jnp.asarray(wave),
                agn_log_lbol=lbol,
                agn_disc_block="powerlaw",
                agn_alpha=alpha,
                agn_T_max=t_max,
            )
        )
        standalone = _standalone(wave, lbol, alpha, t_max)
        for name, sed in (("composable", composed), ("standalone", standalone)):
            assert np.all(sed[probe] == 0.0), f"{name} at {(lbol, alpha, t_max)}: {sed[probe]}"
            assert np.all(sed[~probe] > 0.0), f"{name} at {(lbol, alpha, t_max)}: optical off"
