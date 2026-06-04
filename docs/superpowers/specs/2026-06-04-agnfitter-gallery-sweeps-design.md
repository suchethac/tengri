# AGNfitter-rX model gallery — physics parameter sweeps

**Date:** 2026-06-04
**Branch:** `cs/agn-gallery-sweeps`
**Status:** approved (design), prototype-first build

## Goal

High-quality, scientist-facing sphinx-gallery examples that show what each
AGNfitter-rX disc/torus model's physical parameters do to the SED. Fills the
gaps left after the AGNfitter-rX parity PR (#656): `slone_netzer` and `silva04`
had no dedicated sweep, `skirtor_agnfitter` had only a vs-CIGALE comparison, and
`cat3d_wind` used a hardcoded constant and a different visual style.

These are physics-intuition examples (what each knob does), not parity/methods
plots.

## Deliverables

Four `examples/agn/plot_*.py` scripts, one cohesive family:

| File | Model (block) | Primary × secondary sweep | Physics |
|---|---|---|---|
| `plot_slone_netzer_disc_sweep.py` (new) | slone_netzer (disc) | log M_BH × Eddington ratio | massive BH → cooler/redder disc; high Eddington → hotter UV peak |
| `plot_silva04_nh_sweep.py` (new) | silva04 (torus) | N_H column density | obscuration deepens 9.7 µm silicate feature; reprocessed IR |
| `plot_skirtor_agnfitter_sweep.py` (new) | skirtor_agnfitter (torus) | optical depth τ × inclination | clumpy-torus IR shape; ~25 µm AGNfitter-averaged peak |
| `plot_cat3d_wind_sweep.py` (refresh) | cat3d_wind (torus) | wind fraction × inclination | polar wind vs clumpy disc; viewing-angle dependence |

## Shared template (the consistency that makes it a family)

- `νL_ν` vs rest-frame `λ` on log-log axes.
- `analysis.plotting.setup_style()` for publication defaults.
- Primary parameter swept through a **perceptually-uniform colormap + colorbar**
  (viridis/cividis family), secondary parameter across 2 panels (`sharey`).
- One concise physics annotation per panel (peak location / silicate feature).
- Constants from `tengri.utils.physics_constants` (e.g. `C_AA`) — no literals.
- Built via the public `SEDModel.build(...)` composable-AGN grammar; everything
  through the public `tengri.*` API (no private modules, no raw h5py).
- Sphinx-gallery module docstring with title, physics narrative, and references
  (verified citations).

## Approach

Prototype-first: build `plot_slone_netzer_disc_sweep.py`, render the actual PNG,
get visual approval on the "high quality" look, then replicate the exact
template to the other three.

## Out of scope

- Parity/validation overlays vs the AGNfitter reference grid (separate story).
- New physics or component changes — examples consume the merged #656 models
  unchanged (now node-exact via `interp_nd_pchip`).
- Wiring examples into CI execution (CI never runs the gallery; verify renders
  locally instead).

## Verification

Each script must run clean under the canonical venv and emit a PNG:
`JAX_PLATFORMS=cpu python examples/agn/plot_<name>.py`. Ruff check + format on
all four. Inspect each PNG before committing.
