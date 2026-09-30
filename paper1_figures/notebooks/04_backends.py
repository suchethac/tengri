# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # One galaxy, every recommended backend
#
# The same posterior target through MAP, Laplace, NUTS, HMC and nested slice
# sampling. This is the figure behind the design claim that a backend is one
# function of the standardized loss and its gradient plus a registry entry:
# nothing about the model changes between these panels.
#
# Read the marginals for agreement and the cost for what each method charges
# to get there. They are not interchangeable -- a point estimate and a
# posterior answer different questions -- so the comparison is of methods that
# agree where they overlap, not of one method winning.

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
SWEEP = REPO / "analysis" / "paper1" / "results" / "backend_sweep_pin"
OUT = REPO / "paper1_figures" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# The pinned sweep, not the older `backend_sweep/`: the two were run against
# different commits, and the figure's caption describes the pinned one.
if not SWEEP.is_dir():
    print(f"pinned backend sweep missing from this checkout: {SWEEP}")
    raise SystemExit(1)

runs = sorted(p.name for p in SWEEP.glob("*.json"))
print(f"sweep runs: {len(runs)} -> {runs}")

# %%
status = run_figure(
    "fig07_backends",
    ["--sweep-dir", str(SWEEP), "--out-dir", str(OUT)],
)
print(f"fig07_backends: {status}")

# %%
path = OUT / "fig07_backends.pdf"
print(f"{'ok ' if path.is_file() else 'MISSING'} fig07_backends.pdf")
