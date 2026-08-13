# Physics reproduction notebook contract

Every comparison under `reproduction/` follows the rules below, so the
series reads as one document and the published copies never drift from
the sources. If you are adding a comparison or re-rendering an existing
one, this page is the checklist. The conventions here are load-bearing:
a contract test enforces the publishing rules, and the shared-helper
rules are what make cross-code figures comparable at a glance.

## 1. Layout

One comparison per subfolder: `reproduction/<slug>/`, holding the
notebook `01_<slug>.py` (jupytext percent format — the `.py` is the
source of truth, the `.ipynb` is a render), thin code-specific driver
modules under `_drivers/`, rendered figures under `_figs/`, and a
per-slug README stating prerequisites and data setup (external
installs, `SPS_HOME`, template downloads).

## 2. Sections

Each section sweeps one physics block — SFH, attenuation, IR emission,
nebular, AGN, IGM — with the same SSP on both sides and matched
parameters. The external code plots in `C0` solid, tengri in `C1`, on
shared axes in erg/s/Hz via `units.panel` / `units.two_panel_fig`.
State where the two codes agree, where they disagree, and why; a
residual without an explanation is an open question, not a result.

## 3. Shared helpers

`_drivers/units.py` is byte-identical across comparisons for `regrid`,
`verify_unit_conversion`, `panel`, and `two_panel_fig`; only the
luminosity converter and the `L_SUN` constant may differ, because the
reference codes themselves differ there. `save_fig` uses the single
`_FIG_DPI`; comparisons assert with `_assert_comparable(arr_ref,
arr_t, *, name)`. If you improve a shared helper, propagate it to every
comparison in the same PR.

## 4. Figures

Every saved PNG is prefixed with its slug (`cigale_*`, `bagpipes_*`,
...) so aggregated `_figs/` never collide. Every quantitative claim in
the prose sits next to the figure or printed number that shows it —
no residuals quoted from memory, and always with the grid and settings
that produced them.

## 5. The capstone

Each notebook closes with a full-SED head-to-head: tengri configured to
emulate the external code end to end, overlaid on that code's own
panchromatic output, with a fractional-residual panel and an optical
normalization ratio quoted with its 16–84% spread. The last two
sections are `## Summary` and `## References`, and they are **separate
cells** — agnfitter carried them fused into one 629-word cell, which is
how a bibliography ends up counted as prose. Cite the reference code's
paper exactly (title, journal, arXiv/DOI), not from memory. A citation
you have not checked is not a citation: cigale shipped
`arXiv:2405.xxxxx` and two different years for the same Cue paper.

## 6a. The devices

Six comparisons written over a year converged on six different habits,
each of which one notebook does better than the rest. They are the house
style; use them all, rather than each notebook keeping its own.

**Tables carry numbers; prose carries reasoning** (agnfitter). A
quantitative comparison written as a paragraph is the hardest thing to
read in this series. Any ladder of values that explains a discrepancy, and
any parameter mapping between the two codes, is a table. §9a's Hα/2500 Å
ladder was four numbers inside a nested `(1)…(2)` clause; as four rows it
needs no unpicking. Same for §4's two reddening conventions, and CIGALE
§8's matched H II region.

**Equations use `:math:` or `$…$`** (cigale). Not ASCII transliteration.
Define every symbol with its units on first use.

**Every non-obvious claim names its issue** (prospector, ten of them).
`#NNNN` beside a residual is what lets a reader find out whether it is a
known bug, a convention mismatch, or settled physics. A residual with no
explanation and no ticket is an open question, not a result (§2).

**Known limitations are `**Caveat:**` blocks** (synthesizer). Set them off
so they are scannable, rather than buried mid-paragraph. Synthesizer §9c's
hash-seed dependence is the model: state the effect, the numbers on both
sides, and why the conclusion survives it.

**Menus are enumerated live** (prospect_r, cigale's module map). Write
`tengri.list_agn_models()` and let the notebook print what the installed
version exposes; never hand-maintain a list of model names in prose, which
drifts silently the moment a registry gains an entry.

**The Summary is section by section** (bagpipes). One entry per `§`, in
notebook order, each saying what matched and what did not. It is the page
a reader checks before trusting anything else.

**Keep a markdown cell under ~200 words.** Where a cell runs long, split
it so each piece sits beside the figure it explains. Above roughly 300
words a cell stops being read and starts being skimmed.

## 6. Voice

Scientist to scientist, lean. No cookbook scaffolding, no narration of
the code ("now we will plot..."), no unexplained jargon from either
code's internals. The reader is an astronomer deciding whether to trust
tengri; show them the evidence and get out of the way.

Framing: these notebooks reproduce the physics in the reference codes.
tengri implements the same models independently, and the comparison is
the validation. Never write that a tengri component was ported or
copied from the reference code. External template and SSP data files
used as matched inputs are "repackaged" into tengri's formats, and
that is the word to use.

## 7. Rendering

Notebooks must render headless without losing figures. Two rules:

- Keep the inline-backend guard near the top of every notebook —
  `get_ipython().run_line_magic("matplotlib", "inline")` behind an
  `if get_ipython()` check. A non-inline backend (Agg is the usual
  ambient default) renders a figure-less `.ipynb` silently.
- Render with the repo's environment:

  ```bash
  cd reproduction/<slug>
  jupytext --to ipynb 01_<slug>.py
  PYTHONHASHSEED=0 PYTHONPATH=../..:../../src \
      jupyter nbconvert --to notebook --execute --inplace 01_<slug>.ipynb
  ```

  Worktrees need the gitignored `data/` files symlinked from the main
  checkout first. Prospector additionally needs `SPS_HOME` pointing at
  an FSPS checkout. `PYTHONHASHSEED=0` pins reference-side dict/set
  iteration order — Synthesizer's `UnifiedAGN` assembles its emission
  tree in hash order and its NLR spectrum genuinely changes with the
  seed (synthesizer §9c) — and the repo root on `PYTHONPATH` resolves
  the `reproduction.<slug>._drivers` imports.

Never run `ruff format` on `reproduction/*.py`. The percent-format
cells carry hand-tuned alignment that formatting destroys; lint scope
for this repo is `src/` and `tests/` only.

## 8. Publishing

The docs site shows *committed copies*, not the sources:
`docs/reproduction/<slug>.ipynb` plus the PNGs in
`docs/reproduction/_figs/`. nbsphinx renders stored outputs
(`nbsphinx_execute = "never"`), so there is no auto-sync. Whenever you
re-render a source notebook, re-copy the `.ipynb` and figures into
`docs/reproduction/` **in the same PR**.
`tests/contract/test_reproduction_docs_sync.py` pins this: the SHA-1 of
every embedded PNG, in document order, must match between source and
docs copy. It bit us once — a fixed attenuation panel (#552) kept
showing its pre-fix figure on the live site until #555.

## 9. Adding a comparison

1. New subfolder per §1; copy `_drivers/units.py` from an existing
   comparison unchanged (§3).
2. Write the notebook per §2–§6, with the §7 backend guard.
3. Render, then publish per §8 — the sync test discovers the new slug
   automatically once both `reproduction/<slug>/01_<slug>.ipynb` and
   `docs/reproduction/<slug>.ipynb` exist.
4. Add the comparison to the tree in `reproduction/readme.md` (with the
   reference paper), a `{doc}` bullet + toctree entry in
   `docs/reproduction/index.md`, and the list in the top-level README.
5. Withholding a notebook from the site while it is revised is fine:
   exclude `reproduction/<slug>.ipynb` in `docs/conf.py` and drop its
   toctree entry together, and restore both together (see the
   Synthesizer entry there for the pattern).
