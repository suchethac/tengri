# SPDX-License-Identifier: BSD-3-Clause
"""Is a grid figure drawn from the whole sample, or from part of it?

Split out of the figure scripts so it can be tested without matplotlib, the
same reason ``_adoption.py`` and ``_figure_style.py`` are separate. Nothing
here imports anything beyond the standard library.

``fig09_sample_level.py`` decided its provenance stamp on directory identity:
PROVISIONAL when ``--results-dir`` was not the canonical path. That answers
"where did this come from", which is not the same question as "is all of it
here", and the two came apart the moment the production grid wrote its first
cells into the canonical directory. Five cells of a hundred and twenty then
rendered with no stamp at all, under a caption describing twenty galaxies by
six configurations -- a figure that looks finished and shows four percent of
the sample.

The expected sample is **read** from the committed selection rather than
restated as a literal here, so a change to the locked sample cannot leave a
stale 20 behind in a figure script.
"""

from __future__ import annotations

import json
from pathlib import Path

#: How many missing cells to name before summarizing the rest.
_NAMED_LIMIT = 6


def load_expected_galaxy_ids(selection_path: Path) -> list[int]:
    """The locked sample's galaxy ids, in the order the selection records them."""
    payload = json.loads(Path(selection_path).read_text())
    entries = payload["selected_galaxies"]
    if not entries:
        raise ValueError(f"{Path(selection_path).name} declares no galaxies")
    return [int(entry["id"]) for entry in entries]


def missing_cells(present, galaxy_ids, config_keys) -> list[str]:
    """Cells the sample declares that the render does not have.

    ``present`` is an iterable of ``(galaxy_id, config)`` pairs. Returns
    ``"<id>_<config>"`` names, sorted, so a caller can print them.
    """
    have = {(int(gal), str(cfg)) for gal, cfg in present}
    return sorted(
        f"{gal}_{cfg}"
        for gal in (int(g) for g in galaxy_ids)
        for cfg in (str(c) for c in config_keys)
        if (gal, cfg) not in have
    )


def completeness_note(present, galaxy_ids, config_keys) -> str | None:
    """One line naming the shortfall, or ``None`` when the grid is whole.

    Returning ``None`` for a complete grid is what lets the caller use this
    as the stamp itself: a stamp that is always present says nothing.
    """
    absent = missing_cells(present, galaxy_ids, config_keys)
    if not absent:
        return None
    total = len(list(galaxy_ids)) * len(list(config_keys))
    shown = total - len(absent)
    named = ", ".join(absent[:_NAMED_LIMIT])
    rest = "" if len(absent) <= _NAMED_LIMIT else f", and {len(absent) - _NAMED_LIMIT} more"
    return f"INCOMPLETE - {shown} of {total} cells rendered; missing {named}{rest}"


def present_on_disk(results_dir: Path, galaxy_ids, config_keys) -> list[tuple[int, str]]:
    """Declared cells that have both a JSON and an NPZ on disk.

    Deliberately a plain directory scan rather than a call into a figure's own
    result loader: a completeness check that reuses the loader inherits every
    reason the loader silently drops a cell, and would then agree with it.
    """
    root = Path(results_dir)
    found: list[tuple[int, str]] = []
    for gal in (int(g) for g in galaxy_ids):
        for cfg in (str(c) for c in config_keys):
            stem = root / f"{gal}_{cfg}"
            if stem.with_suffix(".json").is_file() and stem.with_suffix(".npz").is_file():
                found.append((gal, cfg))
    return found
