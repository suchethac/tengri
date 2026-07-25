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

Deliberately cheap: rather than rebuild the 11.7 GB case, they force a small chunk
size on a small grid, which exercises the same multi-chunk concatenation path.
"""

import numpy as np
import pytest

from tengri import FIXED, FREE, SEDModel, Uniform, WavePrecomp
from tengri.components.nebular import nebular_grid_precompute as ngp

pytestmark = pytest.mark.regression_bug

_LINES = (4862.68, 5008.24, 6564.61)  # Hbeta, [OIII]5007, Halpha — vacuum


@pytest.fixture
def cue_model(ssp_data_fsps, synthetic_tophat_obs):
    """One free gas axis, so the grid is small but genuinely multi-node."""
    return SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "none"},
        redshift=0.05,
        neb={"type": "cue", "all_params": FIXED, "logU": Uniform(-3.5, -1.5)},
    )


def _build(model, chunk, monkeypatch, n_grid=4):
    monkeypatch.setattr(ngp, "_BUILD_CHUNK_NODES", chunk)
    return ngp.precompute_nebular_grid(model, np.asarray(_LINES), n_grid=n_grid)


class TestChunkingDoesNotChangeTheGrid:
    def test_chunked_matches_single_call(self, cue_model, monkeypatch):
        """LOAD-BEARING: chunking is a memory optimization, not a numerical one.

        Neuter: reverse the concatenated parts in ``_in_chunks`` and this fails —
        every value is still present, but in the wrong grid cell.
        """
        one_shot = _build(cue_model, 10**9, monkeypatch)  # single vmapped call
        chunked = _build(cue_model, 3, monkeypatch)  # several chunks

        n_nodes = int(np.prod([a.shape[0] for a in one_shot.axes]))
        assert n_nodes > 3, (
            f"probe setup failed: {n_nodes} nodes does not exceed the forced chunk "
            "size, so the chunked path was never taken"
        )

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

    def test_axes_are_unchanged_by_chunking(self, cue_model, monkeypatch):
        """The axes themselves must not depend on how the nodes were batched."""
        one_shot = _build(cue_model, 10**9, monkeypatch)
        chunked = _build(cue_model, 3, monkeypatch)
        assert one_shot.axis_names == chunked.axis_names
        for name, x, y in zip(one_shot.axis_names, one_shot.axes, chunked.axes):
            assert np.array_equal(np.asarray(x), np.asarray(y)), f"axis {name} changed"

    def test_chunking_also_holds_for_the_photometry_channel(
        self, ssp_data_fsps, synthetic_tophat_obs, monkeypatch
    ):
        """The tuple branch of ``_in_chunks`` has its own concatenation.

        With ``WavePrecomp`` the build returns (line, phot) per node and takes a
        different reassembly path than the line-only case above. A bug there
        would scramble ``log_phot_per_qh`` — the table
        ``predict_photometry`` actually consumes — while every line assertion
        above still passed.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "none"},
            redshift=0.05,
            neb={"type": "cue", "all_params": FIXED, "logU": Uniform(-3.5, -1.5)},
            approx=WavePrecomp(),
        )
        one_shot = _build(model, 10**9, monkeypatch)
        chunked = _build(model, 3, monkeypatch)

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

    def test_small_grids_take_the_single_call_path(self, cue_model, monkeypatch):
        """A grid at or below the chunk size must be untouched by this change.

        The common one-axis grid should behave exactly as before, so the chunking
        branch must not fire for it.
        """
        table = _build(cue_model, 10**9, monkeypatch)
        n_nodes = int(np.prod([a.shape[0] for a in table.axes]))
        assert n_nodes <= 10**9
        assert np.all(np.isfinite(np.asarray(table.log_line_per_qh)))
