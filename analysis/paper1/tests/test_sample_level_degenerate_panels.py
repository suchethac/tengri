# SPDX-License-Identifier: BSD-3-Clause
"""A one-row grid must not draw as twenty galaxies agreeing across six models.

``fig09_sample_level`` measures each cell's offset against that galaxy's own
cross-configuration median, so a galaxy with a single cell contributes exactly
zero -- its value minus itself. While only one configuration has finished,
every marker in both offset panels therefore sits on the zero line, inside a
shaded tolerance band, under an axis labeled "galaxy, ordered by
cross-configuration median".

Nothing about that is a computational error, which is what makes it worth a
test. The figure is stamped INCOMPLETE, and the stamp is true, but it says the
grid is partial rather than that these two panels cannot yet mean anything;
a reader seeing a flat row of points inside a band reads agreement. Row III
finished first, so this is not hypothetical -- it is what the figure drew.

The annotation must also disappear once a second configuration lands, or it
becomes a permanent caveat on a panel that has started working.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

pytestmark = pytest.mark.contract

DEGENERATE_NOTE = "zero by construction"


def _cell(gal_id: int, config: str, mstar: float, sfr: float):
    from fig09_sample_level import Cell

    return Cell(
        gal_id=gal_id,
        config=config,
        adopted=True,
        low_ess=None,
        log_mstar=(mstar - 0.1, mstar, mstar + 0.1),
        log_sfr=(sfr - 0.1, sfr, sfr + 0.1),
    )


def _texts(fig) -> list[str]:
    found = []
    for ax in fig.axes:
        found.extend(t.get_text() for t in ax.texts)
    found.extend(t.get_text() for t in fig.texts)
    return found


def _build(cells):
    import matplotlib.pyplot as plt
    from fig09_sample_level import build_figure

    fig, _ = build_figure(cells)
    try:
        collected = _texts(fig)
    finally:
        plt.close(fig)
    return collected


def test_one_configuration_per_galaxy_is_refused():
    """What row III actually produced.

    This used to draw a note on the canvas saying the offsets were zero by
    construction. A note asks the reader to notice; it still ships a panel
    showing twenty galaxies in perfect agreement across configurations the
    grid does not have. Developmental text does not belong on a published
    figure, so the figure is not produced at all.
    """
    cells = [
        _cell(gid, "III", 10.0 + i * 0.1, 0.5 + i * 0.05) for i, gid in enumerate(range(1, 9))
    ]

    with pytest.raises(SystemExit) as excinfo:
        _build(cells)

    message = str(excinfo.value)
    assert "zero by construction" in message, message


def test_a_second_configuration_lets_the_figure_build():
    """Otherwise the refusal would be a permanent block on a working panel."""
    cells = []
    for i, gid in enumerate(range(1, 9)):
        cells.append(_cell(gid, "III", 10.0 + i * 0.1, 0.5 + i * 0.05))
        cells.append(_cell(gid, "II", 10.0 + i * 0.1 + 0.07, 0.5 + i * 0.05 - 0.04))

    texts = _build(cells)

    assert not any(DEGENERATE_NOTE in t for t in texts), (
        "the degenerate-panel note survived into a figure holding two "
        f"configurations per galaxy, where the offsets are real.\ntexts: {texts}"
    )
    assert texts is not None, "the figure must build once the offsets are real"


def test_the_offsets_really_are_zero_with_one_configuration():
    """The premise the note describes, measured rather than assumed."""
    from fig09_sample_level import _offsets

    cells = [_cell(gid, "III", 10.0 + i * 0.3, 0.5 + i * 0.2) for i, gid in enumerate(range(1, 6))]

    _, _, offsets = _offsets(cells, "log_mstar")

    assert set(offsets.values()) == {0.0}, (
        "a single-configuration grid produced non-zero offsets; the note this "
        f"module adds would then be describing something untrue.\n{offsets}"
    )
