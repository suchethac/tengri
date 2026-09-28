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
# Evidence computed by eight-start MAP inference plus Laplace approximation for
# each galaxy across all named model configurations. Routes are weighted
# according to the validity of each model's evidence value: zero weight if the
# configuration had convergence issues or measurement barriers, full weight
# otherwise. Flat priors are applied within each weight set to form the combined
# posterior.
#
# Models that yield valid evidence contribute equally; those with barriers or
# convergence issues are excluded. The posterior under each weighting scheme is
# aggregated to report a single posterior median in stellar mass and SFR.

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

if not BMA_SUMMARY.is_file():
    print(
        f"No BMA summary in {BMA_SUMMARY}.\n"
        "Compute and combine evidence with:\n"
        "  python -m analysis.paper1.bma_evidence --galaxy <id> ...\n"
        "  python -m analysis.paper1.bma_combine\n"
        "and re-run. Skipping the BMA figures."
    )
    raise SystemExit(SKIPPED)

if not BMA_EVIDENCE_DIR.is_dir():
    print(
        f"No BMA evidence directory in {BMA_EVIDENCE_DIR}.\n"
        "Compute evidence with:\n"
        "  python -m analysis.paper1.bma_evidence --galaxy <id> ...\n"
        "and re-run. Skipping the BMA figures."
    )
    raise SystemExit(SKIPPED)

# %% [markdown]
# ## Per-galaxy evidence and validity summary
#
# Each row shows one galaxy across all model configurations. Validity flags
# indicate which configurations contributed to the BMA average: a blank cell
# means the evidence was excluded due to convergence issues (divergences,
# R-hat) or too few effective samples.

# %%
with open(BMA_SUMMARY) as f:
    bma_data = json.load(f)

galaxies = sorted(set(g["galaxy"] for g in bma_data.get("entries", [])))
print(f"\nBMA summary: {len(galaxies)} galaxies × {len(bma_data.get('configs', []))} models\n")
print(f"{'Galaxy':<10} ", end="")
for cfg in sorted(bma_data.get("configs", [])):
    print(f"{cfg[:8]:<10} ", end="")
print()
print("-" * (10 + 10 * len(bma_data.get("configs", []))))

for galaxy in galaxies[:10]:  # Show first 10 as a sample
    gal_entries = [e for e in bma_data.get("entries", []) if e["galaxy"] == galaxy]
    print(f"{galaxy:<10} ", end="")
    for cfg in sorted(bma_data.get("configs", [])):
        entry = next((e for e in gal_entries if e["config"] == cfg), None)
        if entry:
            valid = "✓" if entry.get("valid", False) else "✗"
            print(f"{valid:<10} ", end="")
        else:
            print(f"{'?':<10} ", end="")
    print()

if len(galaxies) > 10:
    print(f"... {len(galaxies) - 10} more galaxies")

# %% [markdown]
# ## Evidence weights and routes
#
# Models are grouped into weight sets. Within each set, all valid evidence
# values receive equal weight; models with barriers or convergence issues are
# zero-weighted. The aggregated posterior averages across all weight routes.

# %%
weight_sets = bma_data.get("weight_sets", {})
print(f"\nWeight routes: {len(weight_sets)}\n")
for route_name, route_data in sorted(weight_sets.items()):
    included = route_data.get("included_configs", [])
    excluded = route_data.get("excluded_configs", [])
    print(f"{route_name}:")
    print(f"  included: {len(included)} models")
    print(f"  excluded: {len(excluded)} models (validity: {excluded[:3]}...)")

# %% [markdown]
# ## Route agreement: close galaxies
#
# For galaxies marked as needing a spot-check, compare BMA evidence routes
# against independent evidence measurements if available.

# %%
BMA_EVIDENCE_NSS = RESULTS / "bma_evidence_nss"
close_galaxies = [g for g in bma_data.get("close_galaxies", [])]

if close_galaxies:
    print(f"\nSpot-check galaxies (close-fit cases): {len(close_galaxies)}\n")
    if BMA_EVIDENCE_NSS.is_dir():
        nss_files = sorted(BMA_EVIDENCE_NSS.glob("*.json"))
        print(f"NSS reference evidence found: {len(nss_files)} measurements\n")
        print("Comparing BMA routes against NSS:")
        for gid in close_galaxies[:5]:
            nss_file = BMA_EVIDENCE_NSS / f"galaxy_{gid}.json"
            if nss_file.is_file():
                with open(nss_file) as f:
                    nss_data = json.load(f)
                print(f"  Galaxy {gid}: NSS ln(Z)={nss_data.get('ln_z', float('nan')):.1f}")
        if len(close_galaxies) > 5:
            print(f"  ... {len(close_galaxies) - 5} more")
    else:
        print("NSS spot-check has not been run; the close-fit galaxies are:")
        for gid in close_galaxies:
            print(f"  {gid}")
        print("\nThis is expected on the first run. No failure.")
else:
    print("\nNo galaxies marked for spot-check.")

# %% [markdown]
# ## Averaged posterior statistics
#
# Median stellar mass and 100-Myr star formation rate across all BMA routes,
# per galaxy. These values represent the model-averaged posterior when
# integration is performed over all valid routes.

# %%
print("\nAveraged posterior (sample of galaxies):\n")
print(f"{'Galaxy':<10} {'M* (dex)':>12} {'SFR_100Myr':>12}")
print("-" * 35)
for galaxy in galaxies[:10]:
    gal_entries = [e for e in bma_data.get("entries", []) if e["galaxy"] == galaxy]
    if gal_entries:
        m_vals = [e.get("mstar_med", float("nan")) for e in gal_entries if e.get("valid")]
        sfr_vals = [e.get("sfr100_med", float("nan")) for e in gal_entries if e.get("valid")]
        m_avg = sum(m_vals) / len(m_vals) if m_vals else float("nan")
        sfr_avg = sum(sfr_vals) / len(sfr_vals) if sfr_vals else float("nan")
        print(f"{galaxy:<10} {m_avg:12.2f} {sfr_avg:12.2e}")

if len(galaxies) > 10:
    print(f"... {len(galaxies) - 10} more galaxies")

# %% [markdown]
# ## BMA posterior figure
#
# Scatter plot of stellar mass and star formation rate under Bayesian model
# averaging, showing both individual route posteriors and the combined result.

# %%
status = {}
status["fig11_bma"] = run_figure(
    "fig11_bma",
    ["--summary", str(BMA_SUMMARY), "--out", str(OUT / "fig11_bma.pdf")],
)

# %%
for name, code in status.items():
    print(f"{name}: {code}")
