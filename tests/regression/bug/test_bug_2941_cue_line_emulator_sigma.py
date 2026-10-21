# SPDX-License-Identifier: BSD-3-Clause
"""Cue ships a per-line emulator error, and tengri carries it with the weights (#2941).

Upstream Cue publishes ``nn_stats_v0.pkl``, whose per-line signal-to-noise
quantiles give the fractional 1-sigma emulator error ``1 / SN_quantile[1]``
(the convention of upstream ``cue.utils.sigma_line_for_fsps``). tengri dropped
it. This module pins the shipped array, its alignment with the published
wavelengths, the pytree round trip, and the back-compatible load of a weights
file that predates the key.

Values are dimensionless fractional errors. The line wavelengths compared in
the tests are the published VACUUM catalog wavelengths.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tests._data_skip import CUE_WEIGHTS, requires_cue_weights

pytestmark = pytest.mark.regression_bug

#: Tolerance [Angstrom] for matching a catalog line to a vacuum target.
_MATCH_AA = 0.5

#: (vacuum target [Angstrom], expected sigma_frac to 3 significant figures, label).
#: Targets are the published vacuum catalog values, not the upstream air labels.
_PINNED_LINES = (
    (4862.68, 0.00865, "H I 4862.68 (H-beta)"),
    (6564.61, 0.0119, "H I 6564.61 (H-alpha)"),
    (5008.24, 0.0452, "[O III] 5008.24"),
    (1640.41, 0.0159, "He II 1640.41"),
    (1548.19, 1.44, "C IV 1548.19"),
    (1550.77, 1.42, "C IV 1550.77"),
    (2425.02, 4.39, "[Ne IV] 2425.02"),
)


@pytest.fixture(scope="module")
def cue_weights():
    from tengri.components.nebular.cue import load_cue_weights

    return load_cue_weights(str(CUE_WEIGHTS))


@pytest.fixture(scope="module")
def cue_backend():
    from tengri.components.nebular.cue import CueBackend

    return CueBackend(str(CUE_WEIGHTS))


def _sigma_at(wave_all: np.ndarray, sigma_all: np.ndarray, target: float) -> float:
    """Sigma of the catalog line nearest ``target``, which must lie within the match tolerance."""
    i = int(np.argmin(np.abs(wave_all - target)))
    assert abs(wave_all[i] - target) < _MATCH_AA, (
        f"no catalog line within {_MATCH_AA} A of {target}; nearest is {wave_all[i]:.3f}"
    )
    return float(sigma_all[i])


@requires_cue_weights
def test_loaded_sigma_is_finite_positive_float32_per_line(cue_weights):
    sigma = cue_weights.line_emul_sigma_frac

    assert isinstance(sigma, np.ndarray)
    assert sigma.shape == (138,)
    assert sigma.dtype == np.float32
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


@requires_cue_weights
@pytest.mark.parametrize(
    ("target", "expected", "label"),
    _PINNED_LINES,
    ids=[p[2] for p in _PINNED_LINES],
)
def test_pinned_line_sigma_values(cue_backend, target, expected, label):
    wave = cue_backend.published_line_wavelengths(cloudyfsps_only=False)
    sigma = cue_backend.published_line_model_sigma(cloudyfsps_only=False)

    assert _sigma_at(wave, sigma, target) == pytest.approx(expected, rel=0.02), label


@requires_cue_weights
def test_model_sigma_is_the_128_line_subset_aligned_with_the_wavelengths(cue_backend, cue_weights):
    full = cue_backend.published_line_model_sigma(cloudyfsps_only=False)
    subset = cue_backend.published_line_model_sigma(cloudyfsps_only=True)
    wave_full = cue_backend.published_line_wavelengths(cloudyfsps_only=False)
    wave_sub = cue_backend.published_line_wavelengths(cloudyfsps_only=True)

    old_idx = np.asarray(cue_weights.line_old_idx)
    assert subset.shape == (128,)
    assert subset.dtype == np.float32
    np.testing.assert_array_equal(subset, full[old_idx])

    # Alignment: the subset's wavelengths and sigmas come from the same catalog rows.
    np.testing.assert_array_equal(wave_sub, wave_full[old_idx])
    assert _sigma_at(wave_sub, subset, 4862.68) == pytest.approx(0.00865, rel=0.02)
    assert _sigma_at(wave_sub, subset, 5008.24) == pytest.approx(0.0452, rel=0.02)


@requires_cue_weights
def test_pytree_round_trip_preserves_the_sigma_array(cue_weights):
    from tengri.components.nebular.cue import _cue_weights_flatten

    children, _ = _cue_weights_flatten(cue_weights)
    assert any(child is cue_weights.line_emul_sigma_frac for child in children)

    leaves, treedef = jax.tree_util.tree_flatten(cue_weights)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    out = rebuilt.line_emul_sigma_frac
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, cue_weights.line_emul_sigma_frac)


@requires_cue_weights
def test_pytree_round_trip_with_sigma_none(cue_weights):
    bare = cue_weights._replace(line_emul_sigma_frac=None)

    leaves, treedef = jax.tree_util.tree_flatten(bare)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.line_emul_sigma_frac is None
    np.testing.assert_array_equal(rebuilt.sorted_line_wav, bare.sorted_line_wav)


@requires_cue_weights
def test_weights_file_without_the_key_loads_with_sigma_none(tmp_path):
    from tengri.components.nebular.cue import CueBackend, load_cue_weights

    with np.load(CUE_WEIGHTS, allow_pickle=True) as src:
        arrays = {k: src[k] for k in src.files if k != "line_emul_sigma_frac"}
    legacy = tmp_path / "cue_weights_pre2941.npz"
    np.savez(legacy, **arrays)

    weights = load_cue_weights(str(legacy))
    assert weights.line_emul_sigma_frac is None

    backend = CueBackend(str(legacy))
    assert backend.published_line_model_sigma(cloudyfsps_only=True) is None
    assert backend.published_line_model_sigma(cloudyfsps_only=False) is None
    assert backend.published_line_wavelengths(cloudyfsps_only=True).shape == (128,)
