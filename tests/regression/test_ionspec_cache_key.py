# SPDX-License-Identifier: BSD-3-Clause
"""The ionizing-spectrum cache must key on the SSP's FLUX, not just its shape.

``_ssp_fingerprint`` hashed ``ssp_wave`` and ``ssp_lgmet`` byte-for-byte but reduced
``ssp_flux`` to ``(shape, dtype)``. A bare-stellar SSP and its with-nebular-emission
twin share a wavelength axis, a metallicity axis, a shape and a dtype, and differ
only in the flux — by ~100x in the Lyman continuum, which is exactly the region this
table integrates. So they collided in ``_IONSPEC_TABLE_CACHE``, and whichever grid a
process loaded first silently supplied the ionizing spectrum for the other.

The poison also reaches disk, under a filename hashed from the same colliding key, so it
outlives the process — on CI and on a laptop — until someone clears the cache. When the
wNE table wins the race, a bare model's Q_H is wrong by 4–7 dex, and the guard that
exists for exactly that (``CueWNESSPError``) cannot catch it: the model under fit is the
*bare* one, so only the Q_H heuristic branch could fire, and that branch is still
downgraded suite-wide by conftest's ``TENGRI_ALLOW_WNE_CUE=1``. (#1579 narrowed that
switch to the heuristic — the metadata branch is now unbypassable — which does not help
here, because an unflagged grid was never reaching the metadata branch.)

**What this bug is NOT.** The first version of this docstring claimed the collision was
the cause of #1154 (fast-vs-exact photometry drifting 20.9%). It is not, and the way that
was established is worth recording, because the story was extremely plausible:

* the corruption is *symmetric*. Poison the table and the fast and exact paths move
  TOGETHER — they end up agreeing on the same wrong answer. Verified by neutering
  ``_ssp_fingerprint`` and re-running the end-to-end comparison with the disk cache
  isolated: agreement is unchanged. A fast-vs-exact parity test is therefore
  structurally BLIND to this bug, which is precisely what makes it dangerous, and why
  the guard below tests the CACHE rather than the physics downstream of it.
* an end-to-end "poisoned model drifts" test was written, passed, and was then found to
  pass just as happily with the fix stripped out. It was deleted. A regression test that
  survives having its bug reintroduced is not a regression test.

Regression for the silent-failure found while sharding the contract tier (2026-07).
"""

import numpy as np
import pytest

from tengri.components.nebular import ionizing_spectrum as ion

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]


def _grids(n_met=2, n_age=3, n_wave=64):
    wave = np.linspace(100.0, 10000.0, n_wave)
    lgmet = np.linspace(-2.0, 0.0, n_met)
    flux = np.ones((n_met, n_age, n_wave))
    return wave, flux, lgmet


def test_flux_values_change_the_fingerprint():
    """The bug, stated minimally: same axes and shape, different flux → different key."""
    wave, flux_a, lgmet = _grids()
    flux_b = flux_a.copy()
    flux_b[0, 0, 0] *= 2.0  # one element — a bare/wNE pair differs far more

    key_a = ion._ssp_fingerprint(wave, flux_a, lgmet)
    key_b = ion._ssp_fingerprint(wave, flux_b, lgmet)

    assert key_a != key_b, (
        "two SSP grids differing in flux share an ionspec cache key: whichever loads "
        "first supplies the ionizing spectrum for the other"
    )
    assert ion._fingerprint_hash(key_a) != ion._fingerprint_hash(key_b), (
        "the DISK filename collides too — the poison outlives the process"
    )


def test_identical_grids_still_share_a_key():
    """The fix must not defeat the cache: the same grid must still hit."""
    wave, flux, lgmet = _grids()
    key_1 = ion._ssp_fingerprint(wave, flux, lgmet)
    key_2 = ion._ssp_fingerprint(wave.copy(), flux.copy(), lgmet.copy())

    assert key_1 == key_2, "an identical grid missed its own cache entry — cache defeated"
    assert ion._fingerprint_hash(key_1) == ion._fingerprint_hash(key_2)


def test_a_wne_like_twin_does_not_collide_with_its_bare_grid():
    """The real shape of the bug: nebular emission added on top of the same axes.

    A wNE grid IS its bare twin plus nebular continuum and lines — identical
    wavelength and metallicity axes, identical shape and dtype. Only the flux moves.
    """
    wave, bare, lgmet = _grids(n_wave=256)
    wne = bare.copy()
    lyc = wave < 912.0  # the Lyman continuum — what the ionspec table integrates
    wne[:, :, lyc] *= 100.0

    assert ion._ssp_fingerprint(wave, bare, lgmet) != ion._ssp_fingerprint(wave, wne, lgmet)


def test_cache_lookup_does_not_serve_one_grid_from_the_other(monkeypatch, tmp_path):
    """End-to-end on the cache itself: a poisoned entry must not be handed out.

    Fill the table for the wNE-like grid, then ask for the bare grid. Before the fix
    the bare grid got the wNE table back; now it must miss and build its own.
    """
    monkeypatch.setattr(ion, "_ionspec_disk_cache_dir", lambda: tmp_path)
    ion._IONSPEC_TABLE_CACHE.clear()

    wave, bare, lgmet = _grids(n_wave=256)
    wne = bare.copy()
    wne[:, :, wave < 912.0] *= 100.0

    ion.precompute_ionizing_params_table(wave, wne, lgmet)
    poisoned = ion._IONSPEC_TABLE_CACHE[ion._ssp_fingerprint(wave, wne, lgmet)]

    served = ion.precompute_ionizing_params_table(wave, bare, lgmet)

    assert served is not poisoned, "the bare grid was served the wNE grid's ionspec table"
    assert not np.allclose(served["logqion_table"], poisoned["logqion_table"]), (
        "the bare and wNE grids produced identical ionizing photon rates despite a "
        "100x difference in the Lyman continuum — the cache is still colliding"
    )

    ion._IONSPEC_TABLE_CACHE.clear()
