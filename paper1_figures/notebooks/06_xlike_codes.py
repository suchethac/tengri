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
# each constrained by parity checks where available. Each configuration's choices
# are compared against Pacifici et al. (2023) Table 1 to document deviations.
#
# The adoption census records which grid cells passed the inspection bar (no
# divergent transitions, split R-hat within threshold, minimum effective samples)
# for each code, so a figure reading the grid can note when a cell was drawn
# against a bar that was relaxed for that configuration.

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
        "Run the X-like configuration suite or point PAPER1_FITS_DIR\n"
        "at a directory that has them, and re-run. Skipping the X-like figures."
    )
    raise SystemExit(SKIPPED)

# %% [markdown]
# ## Configuration mismatch summary
#
# Each X-like configuration's code name, parity check status, mismatches
# against Pacifici et al. (2023) Table 1, and their source documentation.

# %%
status = {}
MISMATCH_JSON = RESULTS / "xlike_mismatches.json"
if MISMATCH_JSON.is_file():
    with open(MISMATCH_JSON) as f:
        mismatch_data = json.load(f)
    print("\nX-like mismatch table (from JSON):\n")
    for key in sorted(mismatch_data.keys()):
        cfg = mismatch_data[key]
        parity = "Yes" if cfg["parity_check"] else "No"
        n_mismatches = cfg.get("n_mismatches", 0)
        print(f"{key:20} {cfg['code']:15} parity={parity:3} mismatches={n_mismatches}")
        for m in cfg.get("mismatches", [])[:2]:
            print(f"  - {m}")
        if n_mismatches > 2:
            print(f"  ... {n_mismatches - 2} more")
    status["xlike_mismatch_table"] = 0
else:
    status["xlike_mismatch_table"] = run_figure(
        "xlike_mismatch_table",
        ["--out-json", str(RESULTS / "xlike_mismatches.json")],
    )

# %% [markdown]
# ## Adoption census per code
#
# Count of adopted and unfit cells per code. A cell is adopted when it passes
# the bar: for relaxed configurations, the split R-hat and divergence rate
# thresholds; for others, zero divergent transitions and split R-hat below 1.01.

# %%
from analysis.paper1._adoption import is_adopted

adoption_counts = {}
for json_file in sorted(FITS_XLIKE.glob("*_*.json")):
    with open(json_file) as f:
        meta = json.load(f)
    code = meta.get("code", "unknown")
    verdict = is_adopted(meta)
    if code not in adoption_counts:
        adoption_counts[code] = {"adopted": 0, "unfit": 0}
    if verdict.adopted:
        adoption_counts[code]["adopted"] += 1
    else:
        adoption_counts[code]["unfit"] += 1

print("\nAdoption census by code:\n")
for code in sorted(adoption_counts.keys()):
    counts = adoption_counts[code]
    total = counts["adopted"] + counts["unfit"]
    frac = counts["adopted"] / total if total > 0 else 0
    print(f"{code:20} adopted={counts['adopted']:3}/{total:3} ({frac:.1%})")

# %% [markdown]
# ## Per-code statistics
#
# Median offset and scatter in stellar mass and star formation rate across
# the grid. Stellar mass is survival (present mass) except for Prospector
# which reports formed mass. SFR is measured over the last 100 Myr.

# %%
data_file = RESULTS / "fig10_xlike_one_to_one_data.json"
if data_file.is_file():
    with open(data_file) as f:
        fig_data = json.load(f)
    print("\nPer-code statistics from fig10:\n")
    hdr = (
        f"{'Code':<20} {'M* offset':>12} {'M* scatter':>12} "
        f"{'SFR offset':>12} {'SFR scatter':>12}"
    )
    print(hdr)
    print("-" * 68)
    measurements = fig_data.get("measurements", [])
    for code in sorted(set(m.get("code") for m in measurements if "code" in m)):
        code_meas = [m for m in measurements if m.get("code") == code]
        if code_meas:
            m_offsets = [m.get("offset_mstar", 0) for m in code_meas]
            m_scatters = [m.get("scatter_mstar", 0) for m in code_meas]
            sfr_offsets = [m.get("offset_sfr100", 0) for m in code_meas]
            sfr_scatters = [m.get("scatter_sfr100", 0) for m in code_meas]
            m_off_med = (
                sorted(m_offsets)[len(m_offsets) // 2] if m_offsets else 0
            )
            m_scat_med = (
                sorted(m_scatters)[len(m_scatters) // 2] if m_scatters else 0
            )
            sfr_off_med = (
                sorted(sfr_offsets)[len(sfr_offsets) // 2] if sfr_offsets else 0
            )
            sfr_scat_med = (
                sorted(sfr_scatters)[len(sfr_scatters) // 2]
                if sfr_scatters
                else 0
            )
            row = (
                f"{code:<20} {m_off_med:12.3f} {m_scat_med:12.3f} "
                f"{sfr_off_med:12.3f} {sfr_scat_med:12.3f}"
            )
            print(row)

# %% [markdown]
# ## One-to-one comparison figure
#
# Stellar mass and star formation rate, each code against Prospector and
# Tengri's configuration. Cells are marked by adoption status.

# %%
status["fig10_xlike_one_to_one"] = run_figure(
    "fig10_xlike_one_to_one",
    ["--results-dir", str(FITS_XLIKE), "--out", str(OUT / "fig10_xlike_one_to_one.pdf")],
)

# %%
for name, code in status.items():
    print(f"{name}: {code}")
