# SPDX-License-Identifier: BSD-3-Clause
"""The nebular grid build is chunked, and chunking does not change the grid (#1361).

The build vmaps one Cue forward per node. vmap is *batched*, not streamed, so a
single call over every node holds every node's intermediates live at once and peak
memory scales with the node count: a three-axis grid at ``n_grid=8``
(16 x 8 x 8 = 1024 nodes) peaked at **11.7 GB** — enough to OOM a 16 GB CI runner
or an ordinary laptop. Evaluating in batches of ``_BUILD_CHUNK_NODES`` bounded that
at **3.8 GB** with a bit-identical result (the same measurement reported 0.4185 %
line accuracy before and after).

The risk chunking introduces is *ordering*: the per-node rows are concatenated back
into a flat array that is then reshaped onto the grid, so a wrong concatenation
scrambles the interpolation table silently — every value present, every value in the
wrong cell. These tests pin the chunked result against the single-call result.

Deliberately cheap on two axes. Rather than rebuild the 11.7 GB case, they force a
small chunk size on a small grid, which exercises the same multi-chunk concatenation
path. And each (single-call, chunked) pair is built **once** per channel in a
module-scoped fixture: an earlier revision rebuilt them per test and cost +1.6 GB
peak RSS, which was enough — stacked with the #1311 file on the sibling xdist
worker — to OOM the 16 GB `regression` runner at ~94 % of the suite (#1346).
A memory-regression test that is itself a memory hog defeats its own fix.
"""

import numpy as np
import pytest

from tengri import FIXED, FREE, SEDModel, Uniform, WavePrecomp
from tengri.components.nebular import nebular_grid_precompute as ngp

pytestmark = pytest.mark.regression_bug

_LINES = (4862.68, 5008.24, 6564.61)  # Hbeta, [OIII]5007, Halpha — vacuum

#: Points on the requested axis. The grid is **two**-dimensional — the model
#: below leaves ``met_logzsol`` griddable too, and it is snapped to the SSP's
#: own metallicity nodes (#1020), so its length is fixed by the SSP (13 for the
#: FSPS fixture) and ignores this number. Node count is therefore ``13 x _N_GRID``.
#: Keep this as small as the assertions allow: the single-call reference vmaps
#: every node at once, so it is the memory floor of this module.
_N_GRID = 2

#: Forced chunk size for the chunked build. Must be < the node count, or the
#: multi-chunk concatenation this module exists to test is never exercised.
_CHUNK = 3


def _cue_model(ssp, obs, **extra):
    """A Cue model whose grid is genuinely 2-D: ``met_logzsol`` x ``neb_logU``.

    Two axes matter here. The concatenation this module guards reassembles a flat
    per-node array and *reshapes* it onto the grid, so a one-axis grid would make
    a transposed or mis-ordered reshape indistinguishable from a correct one.

    Note: after #1796 (#1915), sfh wildcards no longer free met_* without an
    explicit met block. This fixture opts in to metallicity freedom (#1926 pattern)
    to restore the two-axis grid that the chunking assertions depend on.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "none"},
        met={"logzsol": FREE},
        redshift=0.05,
        neb={"type": "cue", "all_params": FIXED, "logU": Uniform(-3.5, -1.5)},
        **extra,
    )


def _build_pair(model):
    """Build the same grid twice — one vmapped call, then several chunks.

    Sets ``_BUILD_CHUNK_NODES`` directly rather than via ``monkeypatch``, because
    that fixture is function-scoped and this runs once per module.
    """
    saved = ngp._BUILD_CHUNK_NODES
    try:
        ngp._BUILD_CHUNK_NODES = 10**9  # never chunks: the reference
        one_shot = ngp.precompute_nebular_grid(model, np.asarray(_LINES), n_grid=_N_GRID)
        ngp._BUILD_CHUNK_NODES = _CHUNK
        chunked = ngp.precompute_nebular_grid(model, np.asarray(_LINES), n_grid=_N_GRID)
    finally:
        ngp._BUILD_CHUNK_NODES = saved
    return one_shot, chunked


@pytest.fixture(scope="module")
def line_pair(ssp_data_fsps, synthetic_tophat_obs):
    """(single-call, chunked) for the line channel — built once for the module."""
    return _build_pair(_cue_model(ssp_data_fsps, synthetic_tophat_obs))


@pytest.fixture(scope="module")
def phot_pair(ssp_data_fsps, synthetic_tophat_obs):
    """(single-call, chunked) for the photometry channel — a different code path.

    With ``WavePrecomp`` the build returns (line, phot) per node and takes the
    tuple branch of ``_in_chunks``, whose concatenation is separate from the
    line-only one.
    """
    return _build_pair(_cue_model(ssp_data_fsps, synthetic_tophat_obs, approx=WavePrecomp()))


class TestChunkingDoesNotChangeTheGrid:
    def test_the_forced_chunk_size_actually_splits_the_grid(self, line_pair):
        """Guard the setup: if the grid fits in one chunk, everything below is vacuous."""
        one_shot, _ = line_pair
        n_nodes = int(np.prod([a.shape[0] for a in one_shot.axes]))
        assert n_nodes > _CHUNK, (
            f"probe setup failed: {n_nodes} nodes does not exceed the forced chunk "
            f"size {_CHUNK}, so the chunked path was never taken"
        )

    def test_chunked_matches_single_call(self, line_pair):
        """LOAD-BEARING: chunking is a memory optimization, not a numerical one.

        Neuter: swap the last two entries of ``parts`` in ``_in_chunks`` and this
        fails — every value is still present, but in the wrong grid cell. (Reversing
        *all* parts instead trips the builder's own first-node sanity guard, which
        would pass this test for the wrong reason.)
        """
        one_shot, chunked = line_pair
        a = np.asarray(one_shot.log_line_per_qh, dtype=np.float64)
        b = np.asarray(chunked.log_line_per_qh, dtype=np.float64)
        assert a.shape == b.shape, f"chunking changed the table shape: {a.shape} vs {b.shape}"
        # atol=0: these are log10 luminosities, and a default atol would mask a
        # genuine per-cell disagreement in the small-magnitude entries.
        assert np.allclose(a, b, rtol=1e-12, atol=0.0), (
            f"chunked grid differs from the single-call grid by "
            f"{np.max(np.abs(b - a)):.3e} in log10 — the concatenation is not "
            "reassembling the nodes in grid order"
        )

    def test_axes_are_unchanged_by_chunking(self, line_pair):
        """The axes themselves must not depend on how the nodes were batched."""
        one_shot, chunked = line_pair
        assert one_shot.axis_names == chunked.axis_names
        for name, x, y in zip(one_shot.axis_names, one_shot.axes, chunked.axes):
            assert np.array_equal(np.asarray(x), np.asarray(y)), f"axis {name} changed"

    def test_chunking_also_holds_for_the_photometry_channel(self, phot_pair):
        """The tuple branch of ``_in_chunks`` has its own concatenation.

        A bug there would scramble ``log_phot_per_qh`` — the table
        ``predict_photometry`` actually consumes — while every line assertion
        above still passed.
        """
        one_shot, chunked = phot_pair
        assert one_shot.log_phot_per_qh is not None, (
            "probe setup failed: no photometry channel, so the tuple branch of "
            "_in_chunks was never exercised"
        )
        a = np.asarray(one_shot.log_phot_per_qh, dtype=np.float64)
        b = np.asarray(chunked.log_phot_per_qh, dtype=np.float64)
        assert a.shape == b.shape, f"chunking changed the phot table shape: {a.shape} vs {b.shape}"
        assert np.allclose(a, b, rtol=1e-12, atol=0.0), (
            f"chunked phot grid differs by {np.max(np.abs(b - a)):.3e} in log10 — "
            "the tuple branch is not reassembling the nodes in grid order"
        )

    def test_grids_at_or_below_the_chunk_size_are_finite(self, line_pair):
        """The common one-axis grid must be untouched by this change.

        The reference build ran with chunking effectively disabled, so this pins
        that the unchunked path still produces a usable table.
        """
        one_shot, _ = line_pair
        assert np.all(np.isfinite(np.asarray(one_shot.log_line_per_qh)))
