# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Precomputation: what it costs and what it costs you
#
# Two panels from one script. The speed panel is the forward-model cost with
# and without the build-time lookup table; the accuracy panel is the error
# that buys, per band, against redshift.
#
# The accuracy panel is the one the appendix prints. Read it as a **bias**
# rather than a scatter: the lookup table's error does not average away over a
# catalog, and a constant forward offset enters the gradient multiplied by the
# signal-to-noise. That is why prediction defaults to the exact wavelength
# grid and only fits opt in.

# %%
import sys
from pathlib import Path

HERE = Path.cwd()
while not (HERE / "paper1_figures").is_dir() and HERE.parent != HERE:
    HERE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "paper1_figures"))

from _run import repo_root, run_figure

REPO = repo_root(HERE)
sys.path.insert(0, str(REPO))
RESULTS = REPO / "analysis" / "paper1" / "results"
OUT = REPO / "paper1_figures" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BENCH = RESULTS / "fig03_bench_forward_2026-08-30.json"
ACCURACY = RESULTS / "fig03_precompute_data.json"

# Both are committed, so a missing one means the checkout is wrong rather than
# the grid being unfinished -- a different situation from the CANDELS
# notebook, and worth a different message.
missing = [p.name for p in (BENCH, ACCURACY) if not p.is_file()]
if missing:
    print(f"committed input missing from this checkout: {missing}")
    raise SystemExit(1)

print(f"benchmark  : {BENCH.name}")
print(f"accuracy   : {ACCURACY.name}")

# %% [markdown]
# ## Both panels
#
# This script is the one entry point in `analysis/paper1/` with no `main()`:
# it parses its arguments and writes its figures at module scope. `run_figure`
# sets `sys.argv` before importing it for exactly that reason, and re-executes
# the module body if it has already been imported once in this session.

# %%
status = run_figure(
    "fig03_precompute",
    [
        "--bench-json",
        str(BENCH),
        "--accuracy-json",
        str(ACCURACY),
        "--out-dir",
        str(OUT),
    ],
)
print(f"fig03_precompute: {status}")

# %% [markdown]
# ## What landed
#
# `figB1_lut_accuracy.pdf` is the appendix figure. The speed panel is kept as
# output but the paper prints the timing table instead, so it is here for
# inspection rather than for inclusion.

# %%
for name in ("figB1_lut_accuracy.pdf", "fig03_precompute.pdf"):
    path = OUT / name
    print(f"{'ok ' if path.is_file() else 'MISSING'} {name}")
