# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Tengri's X-like configurations
#
# CIGALE-, Prospector-, BAGPIPES-, BEAGLE-, and Dense Basis-like configurations,
# each set up to resemble the named code as closely as tengri's components allow.
# Where a parity check against the code exists it is flagged, and every remaining
# difference from the code's choices (Pacifici et al. 2023, Table 1) is listed.
#
# The adoption census counts, per code, the fits whose sampler diagnostics pass
# the adoption bar. Only adopted fits are drawn in the one-to-one comparison.

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

FITS_XLIKE = RESULTS / "fits_xlike"
n_cells = len(list(FITS_XLIKE.glob("*_*.json"))) if FITS_XLIKE.is_dir() else 0
print(f"X-like cells on disk: {n_cells}")

if n_cells == 0:
    print(
        f"\nNo X-like fit cells in {FITS_XLIKE}.\n"
        "Run the X-like configuration suite and re-run.\n"
        "Skipping the X-like figures."
    )
    raise SystemExit(SKIPPED)

# %% [markdown]
# ## Configuration summary
#
# Each X-like configuration's code, whether a parity check against that code
# exists, how the code defines stellar mass, and how many known differences remain.

# %%
from analysis.paper1.config_metadata import XLIKE_CONFIGS

print(f"{'configuration':<17}{'code':<13}{'parity':<8}{'M* definition':<15}{'differences'}")
for key, cfg in sorted(XLIKE_CONFIGS.items()):
    parity = "yes" if cfg["parity_check"] else "no"
    print(
        f"{key:<17}{cfg['code']:<13}{parity:<8}{cfg['mass_definition']:<15}"
        f"{len(cfg.get('mismatches', []))}"
    )

print()
for _, cfg in sorted(XLIKE_CONFIGS.items()):
    print(f"{cfg['code']}:{cfg['name']}")
    for text in cfg.get("mismatches_text", cfg.get("mismatches", [])):
        print(f"  - {text}")

# %% [markdown]
# ## Adoption census per code
#
# A fit is adopted when its sampler diagnostics pass the adoption bar: zero
# divergent transitions and split R-hat below 1.01.

# %%
from analysis.paper1._adoption import is_adopted

counts = {}
for path in sorted(FITS_XLIKE.glob("*_*.json")):
    meta = json.loads(path.read_text())
    config = meta["config"]
    if config not in XLIKE_CONFIGS:
        continue
    code = XLIKE_CONFIGS[config]["code"]
    tally = counts.setdefault(code, {"adopted": 0, "total": 0})
    tally["total"] += 1
    tally["adopted"] += int(is_adopted(meta, config).adopted)

print(f"{'code':<13}{'adopted':>8}{'fits':>6}")
for code, tally in sorted(counts.items()):
    print(f"{code:<13}{tally['adopted']:>8}{tally['total']:>6}")

# %% [markdown]
# ## One-to-one comparison
#
# Tengri's stellar mass and star formation rate for each X-like configuration
# against the values that code published for the same galaxies. Tengri's
# stellar mass is the formed mass; Prospector reports formed mass, while
# BAGPIPES, BEAGLE, CIGALE, and Dense Basis report surviving mass. The star
# formation rate axis is the average over the last 100 Myr.

# %%
DATA_OUT = RESULTS / "fig10_xlike_one_to_one_data.json"
status = {}
status["fig10_xlike_one_to_one"] = run_figure(
    "fig10_xlike_one_to_one",
    [
        "--results-dir",
        str(FITS_XLIKE),
        "--out",
        str(OUT / "fig10_xlike_one_to_one.pdf"),
        "--data-out",
        str(DATA_OUT),
    ],
)

# %% [markdown]
# ## Offsets and scatter
#
# Median offset (tengri minus published) and scatter in dex for each code, over
# its adopted fits.

# %%
if DATA_OUT.is_file():
    codes = json.loads(DATA_OUT.read_text()).get("codes", {})

    def fmt(value):
        return f"{value:8.3f}" if value is not None else f"{'n/a':>8}"

    print(f"{'code':<13}{'n':>3}{'M* off':>9}{'M* scat':>9}{'SFR off':>9}{'SFR scat':>9}")
    for code, row in sorted(codes.items()):
        print(
            f"{code:<13}{row['n_adopted']:>3} {fmt(row['mass_median_offset'])}"
            f" {fmt(row['mass_scatter'])} {fmt(row['sfr_median_offset'])}"
            f" {fmt(row['sfr_scatter'])}"
        )
else:
    print("No comparison data: the figure did not run.")

# %%
for name, code in status.items():
    print(f"{name}: {code}")
