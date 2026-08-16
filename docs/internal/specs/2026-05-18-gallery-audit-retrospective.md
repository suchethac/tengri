# Gallery audit — retrospective (what actually shipped)

Closes the loop on the gallery audit that ran 2026-05-16 → 2026-05-18
across five PRs (#15, #20, #24, #28, #31, #33). This file is the
*delivered* record; plan files at the top of each PR are the
*intended* record.

## What was the audit

The Sphinx-Gallery output under `docs/auto_examples/` had grown to
~130 scripts across 19 sections — the main visual face of the codebase
— and had real rot:

- A buggy `filename_pattern` in `docs/conf.py` (anchored on `^plot_`)
  was silently disabling almost every script's re-execution. The
  committed PNGs only stayed valid because nothing was forcing a
  re-render.
- Three sections (`recipes`, `usecases`, `workflows`) were orphaned
  from the gallery index for lack of a `README.rst`.
- Hand-rolled 4-level `_find_ssp()` path-walkers appeared at the top
  of **91 of 130** scripts.
- **71 scripts** used the legacy flat-kwarg `Parameters(...)`
  constructor; **zero** used the recommended `SEDModel.from_groups +
  recipes.*` path documented in CLAUDE.md as preferred since 2026-05.

## Plan files of record

- Original audit plan:
  `~/.claude/plans/go-through-all-the-zippy-hamster.md` — the four-
  phase plan (visual review / gap analysis / fix bugs / unified style).
  Drove PR #15.
- Dust_attenuation pilot spec:
  `docs/internal/specs/2026-05-17-dust-attenuation-pilot-design.md`
  — established the recipe-card rewrite style for a working-scientist
  audience. Drove PR #20 and propagated into #24, #28, #31, #33.

## The five PRs

| #   | Title                                                   | Scope                                     | Lines net |
| --- | ------------------------------------------------------- | ----------------------------------------- | --------- |
| #15 | gallery audit + restructure                             | All 19 sections: conf.py fix, dust split, orphan index, blank PNG, vacuum-wavelength fixes, audit reports | +44.6 k / −4.1 k (mostly auto-regen) |
| #20 | dust_attenuation pilot                                  | 8 scripts; introduced `load_ssp`, `recipes.dust_demo`, `dust.list_laws` | 619 → 392 lines (−37 %) |
| #24 | dust_emission pilot                                     | 21 scripts; added `data_path` helper                                | 1987 → 1411 lines (−29 %) |
| #28 | batch 2 — quickstart, recipes, sps, metallicity, sfh    | 32 scripts                                                          | ~30 % avg reduction |
| #31 | batch 3 — mechanical cleanup of 12 remaining sections   | ~67 of 86 scripts                                                   | substantial |
| #33 | batch 4 — hand cleanup of 13 try/except holdouts        | 13 scripts                                                          | substantial |

Net effect: **zero scripts** in the gallery now carry the
`_find_ssp()` boilerplate. All 19 sections use the same
`tengri.analysis.plotting.setup_style()` look. The pedagogical-
section order in the sidebar is anchored in `_GALLERY_SECTION_ORDER`
in `docs/conf.py`.

## The four reusable helpers

The audit landed four small library helpers that replaced
hand-rolled boilerplate in dozens of scripts:

1. **`tengri.load_ssp(name=None)`** — short-name SSP loader. `None`
   defaults to the wNE PRSC/MILES Chabrier grid. Walks parent dirs
   for `data/<file>.h5`. Replaces 12-line `_find_ssp()` blocks.
2. **`tengri.data_path(filename)`** — bundled-data-file lookup, same
   parent-dir walk. Returns `pathlib.Path`. For template HDF5s and
   filter caches.
3. **`tengri.recipes.dust_demo()`** — typical SF-galaxy recipe with
   every parameter `FIXED`. Drop into `SEDModel.from_groups(**recipe)`
   and override knobs via `recipe["sfh"].update(...)` when needed.
4. **`tengri.dust.list_laws()`** — returns a `{label: callable}` dict
   of the six headline attenuation laws (Calzetti, Charlot & Fall,
   Cardelli MW, SMC, Kriek & Conroy, Salim+2018) at their canonical
   kwargs. Drives `plot_attenuation_law_compare` and `plot_dust_curves`
   without restating any law's argument signature.

These helpers are the explanatory anchor of the recipe-card style:
a script using all four reads as

```python
from tengri import SEDModel, load_ssp, recipes

model = SEDModel.from_groups(ssp_data=load_ssp(), **recipes.dust_demo())
# sweep one knob, save, show
```

instead of the 30-line setup it used to carry.

## Final per-section line counts (post-audit)

| Section            | Scripts | Lines |
| ------------------ | ------: | ----: |
| advanced           |       6 |   856 |
| agn                |      21 |  2277 |
| dust_attenuation   |       8 |   426 |
| dust_emission      |      21 |  1411 |
| igm                |       4 |   387 |
| inference          |       6 |   900 |
| metallicity        |       5 |   352 |
| multiwavelength    |       6 |   669 |
| nebular            |      14 |  1563 |
| photometry         |       5 |   599 |
| quickstart         |       2 |   198 |
| radio              |       4 |   250 |
| recipes            |       5 |   672 |
| sfh                |      16 |  1371 |
| spectroscopy       |       4 |   437 |
| sps                |       4 |   297 |
| usecases           |       6 |  1343 |
| workflows          |       5 |   787 |
| xray               |       5 |   410 |
| **TOTAL**          | **147** | **15205** |

Pre-audit total was ~17 800 lines (roll-up of the per-PR diffs). Net
reduction ≈ **2.6 k lines, 13 %**, while expanding `dust/` into two
better-organized sections (`dust_attenuation`, `dust_emission`) and
adding the four library helpers.

## Drift from the original plan

1. **Per-section audit reports went unsaved.** The original plan
   called for `docs/dev/gallery_audit/<section>.md` × 19 plus an
   `INDEX.md` rollup. The 19 Haiku agents produced these reports in
   #15, but they lived in pruned worktrees and were never committed.
   The information made it into the conf.py and dust-split fixes;
   the reports themselves are gone. **No follow-up planned** — the
   value extracted from those reports is in the code changes.

2. **Plan scope vs final scope.** The original plan stopped at
   "Phase D: unified plotting style". We kept going into a full
   pedagogical recipe-card rewrite anchored on the working-scientist
   audience. That extension was negotiated in the brainstorming
   dialog before #20 and produced PRs #20/#24/#28/#31/#33.

3. **A fourth helper emerged.** `tengri.data_path` wasn't in either
   plan; it surfaced during #24 as the natural counterpart to
   `load_ssp` for template HDF5s and filter caches. It is now used
   in 30+ scripts.

## Known follow-ups (not blocked, just not in scope)

- `sweep_parameter()` has a sphinx-gallery interaction bug where its
  internally-created Figure sometimes isn't the one the scraper
  captures. Workaround in 8+ scripts: pass `ax=ax` from an
  explicitly-created `plt.subplots(...)`. A clean fix lives in
  `sweep_parameter` itself, not the gallery scripts.
- One pre-existing `B007` ruff warning in
  `examples/nebular/plot_bpt_cue_flexibility.py:181` (unused loop
  variable). Not from any audit PR.
- The gallery's "Examples gallery" landing page in
  `docs/auto_examples/index.rst` (the toctree above the
  thumbnail grid) is auto-generated by sphinx-gallery and
  re-ordered by `_fix_gallery_index_toctree` in `docs/conf.py`. If
  the section list ever needs another reorder, that's the single
  place to edit `_GALLERY_SECTION_ORDER`.

## Acceptance — this audit is closed

- 147 of 147 PNGs render after `cd docs && make html` on a clean
  worktree (`build succeeded, 383 warnings` last verified
  2026-05-18).
- `.venv/bin/ruff check examples/` reports only the one pre-existing
  B007 warning noted above.
- No `_find_ssp()` boilerplate remains anywhere under `examples/`.
- All 19 sections use `setup_style()` and the recommended
  `SEDModel.from_groups + recipes.*` path where applicable.
