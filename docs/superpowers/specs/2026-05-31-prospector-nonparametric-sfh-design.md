# Expand the Prospector reproduction with nonparametric SFH head-to-heads

**Date:** 2026-05-31
**Branch:** `cs/prosectorp-notebok-updateo`
**Notebook:** `reproduction/prospector/01_prospector.py` → `docs/reproduction/prospector.ipynb`

## Motivation

Prospector's defining feature is its family of *nonparametric* star formation
histories (Leja+2017, Leja+2019, Suess+2022). The reproduction notebook today
covers only the parametric delayed-τ SFH (`§2`, FSPS `sfh=4`). tengri already
implements every matching family on `main`, so a faithful matched-parameter
head-to-head is both possible and missing.

## Scope

Add, after the existing parametric `§2`:

| Section | Family | Prospector source | tengri model |
|---|---|---|---|
| §2a | continuity | `transforms.logsfr_ratios_to_masses` | `continuity` |
| §2b | continuity_flex | `transforms.logsfr_ratios_to_masses_flex` | `continuity_flex` |
| §2c | dirichlet | `transforms.zfrac_to_masses` | `dirichlet` |
| §2d | PSB (post-starburst) | `transforms.logsfr_ratios_to_masses_psb` | `psb_continuity` |
| §2e | IFT stochastic field | — (no Prospector counterpart) | `field` |

§2e is a tengri-only flourish showing a stochastic draw the binned families
cannot represent. Clearly labeled as having no Prospector counterpart.

## The matched-parameter contract (verified)

Both codes parametrize identically, so the *same* numbers feed both sides:

- **continuity**: `logsfr_ratios[j] = log10(SFR_j / SFR_{j+1})`, `j=0` =
  youngest bin in lookback time. tengri `ratio_i` uses the same sign and the
  same young→old ordering (`nonparametric.py:131`). Identical ratio array →
  both sides.
- **dirichlet**: stick-breaking `z_fraction` (Leja+2017) on both sides.
- **Shared bin grid**: pass an explicit `bin_edges_gyr` to tengri AND build
  Prospector `agebins` (in `log10(yr)`) from the same edges, so binning is
  identical. The grid must start at a small finite youngest age (e.g. 1 Myr),
  because Prospector's `log10(yr)` agebins cannot represent tengri's default
  `0.0 Gyr` youngest edge.

## Architecture

### 1. Driver — `reproduction/prospector/_drivers/prospector_driver.py`

Add a binned-SFH → FSPS tabular path (the parametric `sfh=4` path is
untouched):

- `csp_lnu_binned(*, agebins, masses, logzsol=0.0)` — convert per-bin masses to
  a step-function `(age, sfr)`, call `sp.set_tabular_sfh(age, sfr)` with
  `sp.params["sfh"] = 3`, return `L_ν` [erg/s/Hz] per the existing
  `_spectrum`/`U.lnu_lsun_to_erg` convention.
- Thin wrappers that call the verified `prospect.models.transforms` functions
  to turn family parameters into per-bin masses:
  `continuity_masses`, `flex_masses`, `dirichlet_masses`, `psb_masses`.
- A helper to build Prospector `agebins` (shape `(nbin, 2)`, `log10(yr)`) from
  shared `bin_edges_gyr`.

### 2. Notebook — `reproduction/prospector/01_prospector.py`

For each of §2a–§2d, one `U.two_panel_fig`:
- **left**: SFR(t) step function — Prospector (`C0-`) vs tengri (`C1-`) at
  identical bin masses, on cosmic-age axis.
- **right**: resulting stellar SED head-to-head with a fractional-residual
  strip, matching the existing capstone convention.
- `_assert_comparable` guards each panel; print `∫SFR dt` mass check.

§2e: a single tengri-only panel — a `field` SFH draw plus its SED, no FSPS
side.

### 3. Figures

`reproduction/prospector/_figs/prospector_02a_*.png … prospector_02e_*.png`
(code-prefixed per the series convention).

## Verification

- Numerically confirm tengri vs Prospector SFR-ratio convention match (feed a
  monotonic ramp, compare SFR(t)) before plotting.
- Render: `jupytext --to ipynb`, then execute with `SPS_HOME` set and
  `PYTHONPATH=<worktree>/src:<worktree>` so it imports the worktree's tengri
  and the `reproduction` package.
- Re-copy rendered `.ipynb` → `docs/reproduction/prospector.ipynb` and figs →
  `docs/reproduction/_figs/` (guarded by
  `tests/contract/test_reproduction_docs_sync.py`).
- Run the sync contract test + `ruff check`/`ruff format`.

## Out of scope

- Changes to tengri's SFH models (they already exist on `main`).
- X-ray / radio (Prospector has neither; covered by the CIGALE notebook).
- Prior-predictive comparison (a possible later section; this is point-estimate
  matched-parameter SED reproduction).
