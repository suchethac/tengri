# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2662 (item 3): the ADAF blend weight is its own key, unclipped.

``agn_cigale_disk_delta`` used to carry two meanings: the skirtor/schartmann2005 slope
modulator on [-0.5, 0.5], and the ``adaf_lopez2024`` ADAF-to-thin-disc blend weight. The
blend weight was clipped to [0, 1] inside the block, so half of the declared prior was a
flat plateau (``delta=-0.5`` equal to 0 and ``delta=1.5`` equal to 1, exactly). The blend
weight now has its own key, ``agn_cigale_adaf_delta``, on [0, 1].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.agn.blocks import resolve_agn_block
from tengri.components.agn.disc_cigale import adaf_disk_parts, adaf_disk_spectrum
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.regression_bug

_WAVE_AA = jnp.geomspace(100.0, 1.0e7, 300)


def _rel_diff(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))


def _adaf_block(delta: float):
    block = resolve_agn_block("disc", "adaf_lopez2024")
    return block(_WAVE_AA, 10.0, agn_cigale_adaf_delta=delta)


def test_adaf_blend_is_live_across_the_range() -> None:
    """(a) The blend moves at every interior value; the endpoints are the pure parts."""
    spectra = {d: np.asarray(_adaf_block(d)) for d in (0.0, 0.3, 0.7, 1.0)}
    values = list(spectra)
    for i, da in enumerate(values):
        for db in values[i + 1 :]:
            assert _rel_diff(spectra[da], spectra[db]) > 1e-3, (da, db)

    # The block is the block's own nm-grid spectrum times a constant (L_bol / 10). delta=0 must
    # then be the pure ADAF part and delta=1 the pure thin-disc part of the block's split.
    wave_nm = _WAVE_AA / 10.0
    adaf_part, _ = adaf_disk_parts(wave_nm, 0.0)
    _, disc_part = adaf_disk_parts(wave_nm, 1.0)
    for delta, part in ((0.0, adaf_part), (1.0, disc_part)):
        part = np.asarray(part, dtype=float)
        live = part > 0.0
        ratio = spectra[delta][live] / part[live]
        assert np.ptp(ratio) / np.mean(ratio) < 1e-5, delta


@pytest.mark.parametrize("bad", [1.5, -0.5])
def test_out_of_range_fixed_blend_weight_is_refused(bad: float) -> None:
    """(b) A Fixed blend weight outside the declared [0, 1] is refused at build.

    The declaration's ``bound_check`` is applied to Fixed values by
    ``Parameters._validate_bounds`` (``lo = hi = value``); the block never clamps.
    """
    with pytest.raises(ValueError, match=r"agn_cigale_adaf_delta.*must be in \[0, 1\]"):
        tengri.parse_groups(
            redshift=tengri.Fixed(0.1),
            agn={
                "type": "composable",
                "disc": {"type": "adaf_lopez2024", "agn_cigale_adaf_delta": tengri.Fixed(bad)},
                "torus": {"type": "none"},
            },
        )


@pytest.mark.parametrize(
    ("key", "bad"),
    [("agn_adaf_delta", 5.0), ("agn_adaf_beta", 1.5), ("agn_adaf_alpha", 0.0)],
)
def test_out_of_range_fixed_adaf_fraction_is_refused(key: str, bad: float) -> None:
    """The ADAF plasma fractions (heating fraction, pressure ratio, viscosity) share the rule."""
    with pytest.raises(ValueError, match=rf"{key}.*must be in \(0, 1\]"):
        tengri.parse_groups(
            redshift=tengri.Fixed(0.1),
            agn={
                "type": "composable",
                "disc": {"type": "adaf", key: tengri.Fixed(bad)},
                "torus": {"type": "none"},
            },
        )


def test_old_disk_delta_key_is_refused_on_adaf_disc() -> None:
    """(c) The slope key on an adaf_lopez2024 disc is refused as a key the block does not read."""
    with pytest.raises(ParameterError, match=r"agn_cigale_disk_delta"):
        tengri.parse_groups(
            redshift=tengri.Fixed(0.1),
            agn={
                "type": "composable",
                "disc": {"type": "adaf_lopez2024", "agn_cigale_disk_delta": tengri.Fixed(0.3)},
                "torus": {"type": "none"},
            },
        )


def test_disk_delta_still_moves_the_skirtor_slope() -> None:
    """(d) On a skirtor disc the slope key is unchanged and still moves the spectrum."""
    block = resolve_agn_block("disc", "skirtor")
    L0 = block(_WAVE_AA, 10.0, agn_cigale_disk_delta=0.0)
    Lp = block(_WAVE_AA, 10.0, agn_cigale_disk_delta=0.3)
    assert _rel_diff(L0, Lp) > 1e-3


def test_adaf_blend_weight_has_a_nonzero_gradient() -> None:
    """(e) d(spectrum)/d(agn_cigale_adaf_delta) at 0.5 is non-zero (chi-square form)."""
    reference = jnp.asarray(adaf_disk_spectrum(_WAVE_AA, 0.0))

    def chi2(d):
        return jnp.sum((adaf_disk_spectrum(_WAVE_AA, d) - reference) ** 2)

    grad = float(jax.grad(chi2)(0.5))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0


def test_block_extrapolates_beyond_unit_blend_weight() -> None:
    """(f) The block does not clamp the weight: delta=1.005 differs from delta=1.0.

    This is the mutant-sensitive test: re-inserting ``jnp.clip(delta, 0.0, 1.0)`` in
    ``adaf_disk_parts`` makes delta=1.005 collapse onto delta=1.0 and this test fails.
    The probe sits inside the range where the thin-disc breakpoint 2000 - 1875 delta
    stays positive. Past delta of about 1.0133 that breakpoint is negative, its log is
    NaN, and the spectrum is NaN (see the #2662 report); that edge is not pinned here.
    """
    at_one = np.asarray(adaf_disk_spectrum(_WAVE_AA, 1.0))
    beyond = np.asarray(adaf_disk_spectrum(_WAVE_AA, 1.005))
    assert np.all(np.isfinite(beyond))
    assert _rel_diff(at_one, beyond) > 1e-3
