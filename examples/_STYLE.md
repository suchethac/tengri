# Examples gallery — style + API contract

This file is the contract every script under `examples/` must satisfy. It exists so
contributors (human or AI) write scripts that read like scientist-to-scientist
demos rather than auto-generated boilerplate. The bar is **DSPS / Synthesizer demo
style**: one knob varied per figure, one physical effect made obvious, no chrome.

## Hard rules (never violate)

1. **Use the current public API.** Build models with `SEDModel.build(...)`.
   `SEDModel.from_groups(...)` was removed 2026-05-23 — calling it now raises
   `AttributeError`. Treat any reference in older docs as historical.
2. **No deprecated names.** `Parameters` (not `ParamSpec`), `PopulationFitter`
   (not `HierarchicalFitter`), `SEDModel` (not `Model`), `Spectroscopy`,
   `NoiseModel`, `LineList`. See `docs/dev/NAMING_CONTRACT.md`.
3. **Apply the house style.** Every script that draws a figure must call
   `tengri.analysis.plotting.setup_style()` exactly once at the top, after
   imports. Do not override fonts, ticks, or grid afterwards.
4. **Save the figure as `<script_stem>.png`** at `dpi=150`,
   `bbox_inches="tight"`. Sphinx-gallery picks up the matching name.
5. **No `plt.show()` in the final `if __name__ == "__main__"` path.** Sphinx-gallery
   handles rendering; `plt.show()` blocks builds when invoked headlessly.

## Figure recipe (sweep scripts — the majority)

A sweep script varies **one** physical parameter across a small grid (5–9 values)
and shows the resulting SEDs on a single axis, colored by the swept value.

```
fig, ax = plt.subplots(figsize=(6.5, 4.2))
cmap = plt.get_cmap("viridis")
values = np.linspace(lo, hi, 7)
norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
for v in values:
    params = {**baseline, "<the_knob>": v}
    pred = model.predict(params)
    wave = pred.wave_rest                      # the array does not carry its axis
    ax.loglog(wave, wave * pred.rest_sed(), color=cmap(norm(v)), lw=1.4)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    pad=0.01, label=r"<knob label with units>")
ax.set(xlabel=r"$\lambda$ [$\mu$m or $\mathrm{\AA}$, pick one]",
       ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]")
```

Defaults to enforce:
- Plot `nu * L_nu` (or `nu * F_nu`), not raw `L_nu`. Eyes read SED bumps in νL_ν.
- `loglog` for full-spectrum sweeps; `semilogy` for narrow ranges.
- Colorbar label includes units in brackets, e.g. `r"$\tau_V$ [mag]"`,
  `r"$\alpha_{\rm IR}$"`, `r"$\log U$"`.
- `lw=1.4`, `viridis` (sequential) or `coolwarm` (signed/centered), nothing else.
- One legend OR one colorbar — never both.

## Figure recipe (fit / workflow scripts)

For `quickstart/`, `inference/`, `workflows/`, `usecases/`: a script runs a real
fit and shows data + model + a recovery panel. Two-panel layout:

```
fig, (ax_sed, ax_res) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
# Top: data points, truth SED curve, MAP/median SED curve
# Bottom: (data - model) / sigma residuals
```

If the script produces a corner plot, use `corner.corner(..., color="C0",
hist_kwargs={"density": True})` and **drop the title**. Captions go in the
docstring.

## Docstring header (sphinx-gallery)

The very first triple-quoted block is the sphinx-gallery card. Title format:

```
"""
What the figure shows: the physics, the knob, the effect
========================================================

One sentence: what model, what's being varied, what the reader will see.
Cite the physics paper if a specific equation drives the plot. Avoid
mentioning ``tengri`` in the title — the gallery already lives in tengri.
"""
```

Bad titles (avoid):
- `"First Photometric Fit with tengri"`
- `"DPL alpha sweep"`
- `"AGN composable demo"`

Good titles:
- `"Recovering stellar mass from 5-band SDSS photometry"`
- `"Early-time SFH slope α controls the UV continuum"`
- `"Switching torus library: SKIRTOR vs CAT3D-WIND at fixed L_bol"`

## What NOT to do

- ❌ `plt.title("Some Title")` — the sphinx-gallery card title is the figure
  title. Setting another in matplotlib is duplication.
- ❌ Bare `plt.legend()` with default frame and serif mismatch. Use
  `ax.legend(frameon=False, fontsize=8)` only when a colorbar would be wrong.
- ❌ Hard-coded axis limits without a physical reason. If you set `xlim`, the
  comment must say why (e.g. `# zoom on Lyman break`).
- ❌ Axis ranges that hide the physics. The feature the docstring promises must
  sit fully inside the panel with continuum context on both sides — a Lyman-break
  page shows the blueward suppression *and* the red continuum; a silicate-feature
  zoom shows the full 9.7 μm trough plus wings. y-ranges frame the curves: νL_ν
  panels typically span ~4–6 dex around the data — wide enough that nothing
  clips, tight enough that a sweep's curves visibly separate. When in doubt,
  derive limits from the data with margins instead of hard-coding.
- ❌ Running a real fit (NUTS, geoVI) for the first time inside an example
  script that loads with the gallery. Pre-cache or use MAP with ≤300 steps.
- ❌ Inline comments narrating obvious steps (`# build the model`). Comments
  explain *why*, never *what*.

## Reference scripts

Copy from:

- Sweep template: `examples/sfh/plot_dexp_tau_sweep.py`
- Fit template: `examples/quickstart/plot_first_fit.py`

Both have been manually rewritten to the bar this file defines.

## After you edit an example: regenerate its render

The gallery users read is committed under `docs/auto_examples/` — CI never
executes examples (it has no SSP grids and none of the ~20 GB of optional
data), it ships what you committed. So an edited source with a stale render
means the docs site shows code that no longer exists. CI fails on this
(`tools/check_gallery_fresh.py --strict`, #805).

```bash
python tools/regen_gallery.py plot_your_example    # then commit what it writes
```

**Never a bare `make html` while anything is stale.** Sphinx-gallery rewrites a
page whose source no longer matches its committed stamp, but `filename_pattern`
stops it executing — so that page comes back *without* the output execution
produced. Measured with 60 examples stale, a single-example build deleted
45,204 lines across 195 files and exited 0 (#1236). `regen_gallery.py` runs the
same build behind a fence that restores every page that was not a target.

Pages that are already fresh are not rewritten at all, so on a gallery with
zero drift a full build changes nothing — which is why the freshness gate is
what keeps ordinary doc builds safe.

Regenerating requires the optional data grids. If you do not have them, say so
in the PR rather than committing a render built without them.
