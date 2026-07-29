# SPDX-License-Identifier: BSD-3-Clause
"""Q_H interpolation floors negatives on every nebular backend (#1405).

``_get_qh_at`` — bilinear interpolation of the ionizing photon rate Q_H —
was implemented three times. Two copies ended in ``jnp.maximum(..., 0.0)``;
:class:`CB19Backend` did not, so it could hand a **negative** Q_H to the line
kernel. All three consume it multiplicatively
(``weight_i * qh_i * lum_per_qh * k_factor``), so a negative Q_H becomes a
negative line luminosity — unphysical, and silent.

Two behaviors are pinned deliberately, because they are easy to "fix" wrongly:

* **The per-backend fallback for a missing table is preserved, not unified.**
  CB19 returns ``1.0`` (identity for a multiplier — "no Q_H scaling"), the other
  two return ``0.0`` ("no ionizing photons, no lines"). Both are defensible and
  the difference is intentional.
* **NaN propagates.** ``jnp.maximum(nan, 0.0)`` is ``nan``, so the original floor
  never removed NaN despite its comment claiming "negative/NaN". That is the
  correct behavior and is kept: a NaN Q_H means the table upstream is broken,
  and propagating it makes that visible, where flooring it to 0.0 would hide it
  (the failure mode cataloged in #1404).

The backends are instantiated via ``object.__new__`` with the three attributes
``_get_qh_at`` actually reads. Building a real backend needs CLOUDY grid files,
which would make this test data-gated and therefore invisible to CI.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend

pytestmark = pytest.mark.regression_bug

# (class, fallback returned when _qh_table is None)
BACKENDS = [
    pytest.param(CB19Backend, 1.0, id="cb19"),
    pytest.param(CloudyGridBackend, 0.0, id="cloudy_grid"),
    pytest.param(MappingsPhotoStellarBackend, 0.0, id="mappings_photo"),
]

LOG_MET = jnp.asarray([-4.0, -3.0, -2.0])
LOG_AGE = jnp.asarray([6.0, 7.0, 8.0])


def _make(cls, table):
    """A backend carrying only what ``_get_qh_at`` reads."""
    obj = object.__new__(cls)
    obj._qh_table = None if table is None else jnp.asarray(table)
    obj._qh_log_met = LOG_MET
    obj._qh_log_age = LOG_AGE
    return obj


@pytest.mark.parametrize("cls,_fallback", BACKENDS)
def test_negative_table_entries_are_floored(cls, _fallback):
    """A negative Q_H never reaches the caller. CB19 failed this before #1405."""
    table = np.full((3, 3), -7.0)
    backend = _make(cls, table)

    # Query the interior of the grid so every corner is the negative value.
    qh = float(backend._get_qh_at(-3.0, 7.0))

    assert np.isfinite(qh), f"{cls.__name__} returned non-finite {qh}"
    assert qh >= 0.0, f"{cls.__name__} returned a negative Q_H: {qh}"


@pytest.mark.parametrize("cls,_fallback", BACKENDS)
def test_mixed_sign_table_is_floored_not_canceled(cls, _fallback):
    """A partly-negative table cannot produce a negative interpolant.

    Guards the subtler case: the bilinear blend of a positive and a negative
    corner can land below zero without any single corner dominating.
    """
    table = np.array([[-10.0, -10.0, -10.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float)
    backend = _make(cls, table)

    # Sit near the negative row so the blend is negative pre-floor.
    qh = float(backend._get_qh_at(-3.9, 7.0))
    assert qh >= 0.0, f"{cls.__name__} returned a negative Q_H: {qh}"


@pytest.mark.parametrize("cls,fallback", BACKENDS)
def test_missing_table_fallback_is_preserved_per_backend(cls, fallback):
    """The 1.0-vs-0.0 split is intentional; unifying it would be a behavior change."""
    backend = _make(cls, None)
    assert float(backend._get_qh_at(-3.0, 7.0)) == pytest.approx(fallback)


@pytest.mark.parametrize("cls,_fallback", BACKENDS)
def test_positive_table_is_interpolated_unchanged(cls, _fallback):
    """The floor must not perturb ordinary values — the no-regression check."""
    # Linear in the metallicity axis so the expected value is exact.
    table = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=float)
    backend = _make(cls, table)

    assert float(backend._get_qh_at(-4.0, 7.0)) == pytest.approx(0.0)
    assert float(backend._get_qh_at(-3.0, 7.0)) == pytest.approx(10.0)
    assert float(backend._get_qh_at(-2.0, 7.0)) == pytest.approx(20.0)
    # Halfway between grid nodes.
    assert float(backend._get_qh_at(-3.5, 7.0)) == pytest.approx(5.0)


@pytest.mark.parametrize("cls,_fallback", BACKENDS)
def test_nan_propagates_rather_than_being_zeroed(cls, _fallback):
    """NaN is the loud signal for a broken table; flooring it would hide it (#1404)."""
    table = np.full((3, 3), np.nan)
    backend = _make(cls, table)
    assert np.isnan(float(backend._get_qh_at(-3.0, 7.0)))


def test_all_three_backends_share_one_implementation():
    """The assertion that cannot rot: re-forking the helper fails here."""
    from tengri.components.nebular._shared import _qh_bilinear

    table = np.array([[-10.0, -10.0, -10.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float)
    expected = float(_qh_bilinear(jnp.asarray(table), LOG_MET, LOG_AGE, -3.9, 7.0, missing=0.0))
    for cls, _ in [(c.values[0], c.values[1]) for c in BACKENDS]:
        got = float(_make(cls, table)._get_qh_at(-3.9, 7.0))
        assert got == pytest.approx(expected), f"{cls.__name__} diverged from _qh_bilinear"
