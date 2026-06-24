# SPDX-License-Identifier: BSD-3-Clause
"""Stellar Lyman continuum is absorbed (not leaked / negated) under two-component dust.

Regression for #824. With a photoionised nebular backend and ``neb_fesc < 1``,
the stellar Lyman continuum (λ < 912 Å) is absorbed by the same gas that powers
the nebular emission; only the escaping fraction ``neb_fesc`` survives. CIGALE
removes ``stellar_LyC × (1 − fesc)`` below the Lyman break
(``pcigale.sed_modules.nebular``), so at ``fesc = 0`` the observed stellar
continuum below 912 Å is zero — same as FSPS / bagpipes.

tengri's nebular component masks the *summed* ``sed_intrinsic`` by ``fesc``
below 912 Å, but the two-component :class:`DustSEDComponent` rebuilds the stellar
SED from the **unmasked** per-age ``lnu_age`` cube. Before the fix the mask was
bypassed: ``sed_dust_attenuated`` leaked attenuated stellar LyC, and the
``non_stellar_other = state.sed_intrinsic − Σ lnu_age`` bookkeeping picked up a
phantom ``−stellar_LyC`` that drove ``predict()`` **negative** below 912 Å.

The nebular component now publishes ``lyc_transmission = where(λ<912, fesc, 1)``
and dust applies it to the per-age reconstruction, so the LyC scales cleanly
with ``fesc``: 0 → fully absorbed, 1 → full continuum.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig
from tengri.protocols.component import ForwardState

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

# Grid straddling the Lyman limit (first two points are LyC, λ < 912 Å).
_WAVE = jnp.array([500.0, 800.0, 1000.0, 1500.0, 5500.0])
_AGES = jnp.array([1.0e6, 1.0e8, 1.0e10])
_LYC = np.asarray(_WAVE) < 912.0


def _state(fesc: float, *, publish_lyc: bool) -> ForwardState:
    """Mimic the photoionised-nebular output at a given escape fraction.

    Stellar light lives in the old bin of ``lnu_age`` (unmasked). The nebular
    component masks the summed ``sed_intrinsic`` LyC by ``fesc`` and (after the
    fix) publishes ``lyc_transmission``.
    """
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    full = jnp.sum(lnu_age, axis=0)
    lyc_mask = _WAVE < 912.0
    sed_intrinsic = jnp.where(lyc_mask, full * fesc, full)
    derived = {
        "lnu_age": lnu_age,
        "ssp_ages_yr": _AGES,
        "sed_nebular": jnp.zeros_like(_WAVE),
    }
    if publish_lyc:
        derived["lyc_transmission"] = jnp.where(lyc_mask, fesc, jnp.ones_like(_WAVE))
    return ForwardState(wave=_WAVE, sed_intrinsic=sed_intrinsic, derived=derived)


def _apply(state):
    comp = DustSEDComponent(
        config=DustSEDComponentConfig(law_bc="calzetti", law_diff="calzetti", emission_model=None)
    )
    return comp.apply(state, {"dust_tau_bc": 0.0, "dust_tau_diff": 1.0})


def test_fesc0_absorbs_lyc_no_negative_no_leak():
    """fesc=0: below 912 Å the model output and sed_dust_attenuated are ~0."""
    out = _apply(_state(0.0, publish_lyc=True))
    total = np.asarray(out.sed_intrinsic)
    sda = np.asarray(out.derived["sed_dust_attenuated"])
    # No negative flux anywhere (the phantom -stellar_LyC is gone).
    assert np.all(total >= 0.0)
    # Stellar LyC fully absorbed: both the total and the attenuated stellar
    # are ~0 below 912 Å (no leak).
    assert np.allclose(total[_LYC], 0.0)
    assert np.allclose(sda[_LYC], 0.0)
    # Above 912 Å is untouched (stellar present).
    assert np.all(total[~_LYC] > 0.0)


@pytest.mark.parametrize("fesc", [0.0, 0.3, 1.0])
def test_lyc_scales_linearly_with_fesc(fesc):
    """Below 912 Å the surviving stellar continuum is ∝ fesc (CIGALE convention)."""
    out = _apply(_state(fesc, publish_lyc=True))
    full_escape = _apply(_state(1.0, publish_lyc=True))
    sda = np.asarray(out.derived["sed_dust_attenuated"])
    sda_full = np.asarray(full_escape.derived["sed_dust_attenuated"])
    np.testing.assert_allclose(sda[_LYC], fesc * sda_full[_LYC], rtol=1e-6)


def test_absent_lyc_transmission_passes_through():
    """No published lyc_transmission (BakedIn) -> LyC unmasked, no error."""
    # publish_lyc=False and fesc=1 so sed_intrinsic carries the full LyC.
    out = _apply(_state(1.0, publish_lyc=False))
    total = np.asarray(out.sed_intrinsic)
    assert np.all(total >= 0.0)
    assert np.all(total[_LYC] > 0.0)  # LyC present (gas absorption not applied)
