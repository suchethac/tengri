# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Bayesian model averaging
#
# Each galaxy is fit under every model in a set, and the models are weighted by
# their evidence. The evidence of one model is ln Z from an eight-start MAP
# optimization followed by a Laplace approximation at the best optimum. The
# stellar mass is integrated under its prior rather than profiled out, and the
# flux errors carry a 5% systematic floor.
#
# A model's evidence is valid when ln Z is finite, the Newton decrement at the
# optimum is at most 0.1, and the Hessian has no clipped eigenvalues. Invalid
# models are left out of the average. Within each weight set the model prior is
# flat, so the weights are proportional to Z (a softmax of ln Z): valid models do
# not contribute equally, and one model with a much higher evidence dominates.
#
# Three weight sets are averaged separately:
#
# - `named_grid`: grid configurations I to V.
# - `named_all`: configurations I to V plus the five X-like configurations.
# - `factorial`: 100 models, the product of 5 star formation histories,
#   5 SSP grids, and 4 attenuation laws. Axis marginals are compared with the
#   marginals the flat model prior implies.
#
# A galaxy is flagged close when its largest weight in a set is below 0.9. The
# model choice is then not settled by the evidence, and those galaxies are the
# candidates for a nested-sampling spot-check.

# %%
import json
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
RESULTS = REPO / "analysis" / "paper1" / "results"
OUT = REPO / "paper1_figures" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BMA_SUMMARY = RESULTS / "bma_summary.json"
BMA_EVIDENCE_DIR = RESULTS / "bma_evidence"

if not BMA_SUMMARY.is_file() or not BMA_EVIDENCE_DIR.is_dir():
    print(
        f"No BMA summary in {BMA_SUMMARY} or no evidence in {BMA_EVIDENCE_DIR}.\n"
        "Compute and combine evidence with:\n"
        "  python -m analysis.paper1.bma_evidence --galaxy <id> ...\n"
        "  python -m analysis.paper1.bma_combine\n"
        "and re-run. Skipping the BMA figures."
    )
    raise SystemExit(SKIPPED)

summary = json.loads(BMA_SUMMARY.read_text())
galaxies = summary["galaxies"]
SETS = ("named_grid", "named_all", "factorial")
print(f"{summary['n_galaxies']} galaxies; route: {summary['route']}; {summary['model_prior']}")

# %% [markdown]
# ## Models and weights per galaxy
#
# For each weight set: how many of the expected models have a valid evidence,
# the model with the largest weight, and that weight. A set with no cells for a
# galaxy shows a dash.

# %%
print(f"{'galaxy':<8}{'set':<12}{'valid':>9}  {'top model':<22}{'weight':>7}  close")
for gid, gal in sorted(galaxies.items(), key=lambda kv: int(kv[0])):
    for name in SETS:
        block = gal["sets"][name]
        valid = f"{block['n_valid']}/{block['n_expected']}"
        weighted = [m for m in block["models"] if m["weight"] is not None]
        if not weighted:
            print(f"{gid:<8}{name:<12}{valid:>9}  {'-':<22}{'-':>7}")
            continue
        top = max(weighted, key=lambda m: m["weight"])
        flag = "yes" if block["close"] else "no"
        print(f"{gid:<8}{name:<12}{valid:>9}  {top['model_key']:<22}{top['weight']:>7.3f}  {flag}")

# %% [markdown]
# ## Models left out of the average
#
# How many galaxies each model was excluded for, and why.

# %%
for name in SETS:
    excluded = {}
    for gal in galaxies.values():
        for model in gal["sets"][name]["models"]:
            if not model["valid"]:
                excluded.setdefault(model["model_key"], set()).add(model["excluded_reason"])
    counts = summary["invalid_counts"].get(name, {})
    if not counts:
        print(f"{name}: no excluded models")
        continue
    print(f"{name}:")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n} galaxies ({'; '.join(sorted(excluded.get(key, ())))})")

# %% [markdown]
# ## Galaxies for a nested-sampling spot-check
#
# Galaxies whose largest weight is below 0.9 in a set.

# %%
for name in SETS:
    close = [gid for gid, gal in galaxies.items() if gal["sets"][name]["close"]]
    print(f"{name}: {', '.join(sorted(close, key=int)) if close else 'none'}")

# %% [markdown]
# ## Model-averaged stellar mass and star formation rate
#
# The 16th, 50th, and 84th percentiles of the averaged posterior in the
# `named_all` set. Stellar mass is the formed mass; the star formation rate is
# the average over the last 100 Myr.

# %%
QUANTITIES = (
    ("log_stellar_mass_formed", "log M*,formed"),
    ("log_sfr_100myr", "log SFR 100 Myr"),
)
print(f"{'galaxy':<8}" + "".join(f"{label:>26}" for _, label in QUANTITIES))
for gid, gal in sorted(galaxies.items(), key=lambda kv: int(kv[0])):
    perc = gal["sets"]["named_all"]["bma_percentiles"]
    cells = []
    for key, _ in QUANTITIES:
        if key in perc:
            lo, med, hi = perc[key]
            cells.append(f"{med:6.2f} (+{hi - med:.2f} / -{med - lo:.2f})".rjust(26))
        else:
            cells.append("-".rjust(26))
    print(f"{gid:<8}" + "".join(cells))

# %% [markdown]
# ## BMA figure

# %%
status = {}
has_factorial = any(gal["sets"]["factorial"]["models"] for gal in galaxies.values())
for name in ("named_all", "factorial"):
    if name == "factorial" and not has_factorial:
        print("No factorial evidence cells yet; drawing the named_all set only.")
        continue
    stem = "fig11_bma" if name == "named_all" else "fig11_bma_factorial"
    status[f"fig11_bma[{name}]"] = run_figure(
        "fig11_bma",
        ["--summary", str(BMA_SUMMARY), "--set", name, "--out", str(OUT / f"{stem}.pdf")],
    )

# %%
for name, code in status.items():
    print(f"{name}: {code}")
