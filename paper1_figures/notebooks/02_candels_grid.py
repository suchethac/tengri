# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # The CANDELS grid
#
# Twenty galaxies by six configurations, and the figures that read it: the
# sample in one frame, three galaxies in detail, and the comparison against
# published per-code values.
#
# Every one of these figures stamps its own provenance on its face. While the
# grid is incomplete they say so, with the count and the missing cells named,
# because a figure drawn from part of a sample and captioned as the whole is
# the failure these stamps exist to prevent.

# %%
import os
import sys
from pathlib import Path

HERE = Path.cwd()
while not (HERE / "paper1_figures").is_dir() and HERE.parent != HERE:
    HERE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "paper1_figures"))

from _run import SKIPPED, repo_root, run_figure

REPO = repo_root(HERE)
sys.path.insert(0, str(REPO))
# Overridable, because the grid commits to a different branch from the one
# the paper is built on: a reader with the cells elsewhere should not have
# to edit a notebook to use them.
_DEFAULT_FITS = REPO / "analysis" / "paper1" / "results" / "fits"
RESULTS = Path(os.environ.get("PAPER1_FITS_DIR", _DEFAULT_FITS))
OUT = REPO / "paper1_figures" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

n_cells = len(list(RESULTS.glob("*_*.json"))) if RESULTS.is_dir() else 0
print(f"cells on disk: {n_cells}")

# The grid writes to a different branch from the one the paper is built on, so
# a clean checkout of this branch has no cells and these figures cannot be
# regenerated here. Say that once and stop, rather than emitting three empty
# figures that each stamp INCOMPLETE - 0 of N and look like output.
if n_cells == 0:
    print(
        "\nNo grid cells in analysis/paper1/results/fits/.\n"
        "The production grid commits to paper1/nss-profile-mass; this branch\n"
        "carries the scripts, not the cells. Fetch them, or point --results-dir\n"
        "at a directory that has them, and re-run. Skipping the grid figures."
    )
    raise SystemExit(SKIPPED)

# %% [markdown]
# ## The sample in one frame
#
# Stellar mass against star formation rate for every cell, and each galaxy's
# spread across configurations. A cell the bar adopted on too few effective
# samples is drawn ringed rather than filled, because the bar adopting it does
# not make its posterior one.

# %%
status = {}
status["fig09_sample_level"] = run_figure(
    "fig09_sample_level",
    ["--results-dir", str(RESULTS), "--out", str(OUT / "fig09_sample_level.pdf")],
)

# %% [markdown]
# ## Three galaxies in detail
#
# One per color class, under every configuration that produced a cell.

# %%
status["fig05_candels_galaxies"] = run_figure(
    "fig05_candels_galaxies",
    ["--results-dir", str(RESULTS), "--output-dir", str(OUT)],
)

# %% [markdown]
# ## Against the published per-code values
#
# This one refuses a results directory whose cells sample a different star
# formation history family than the configuration table declares, rather than
# failing later inside a model rebuild.

# %%
try:
    status["fig06_code_overlay"] = run_figure(
        "fig06_code_overlay",
        ["--results-dir", str(RESULTS), "--out-dir", str(OUT)],
    )
except Exception as exc:
    status["fig06_code_overlay"] = f"refused: {type(exc).__name__}: {exc}"

# %%
for name, code in status.items():
    print(f"{name}: {code}")
