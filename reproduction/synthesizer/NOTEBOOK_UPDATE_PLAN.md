<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Reproduction notebook update plan — `01_synthesizer.py`

Plan (not code) for the changes that follow from the AGN-parity work on
`cs/synthesizer-parity`. Render is deferred: the §9 panels need the gitignored
Synthesizer AGN test grids and a multi-minute execution, so this is authored in
the worktree and rendered later in the canonical repo (see *Rendering*).

tengri is an **independent** code; Synthesizer (Lovell et al. 2025, OJA; Roper
et al. 2026, JOSS) is a peer comparator, not a north-star. No lineage language —
the `compute_nlr_sed_synthesizer` adapter reading the same Cloudy grid is a
unit-matching convenience, not evidence of derivation.

## Stack the panels exercise

```
SEDModel.build(agn={...})                 ← grammar path (NEW: surface this)
        │
        ├─ composable runner → disc + torus + lines (nlr/blr) + feii + atten
        │      │
        │      └─ compute_nlr_sed_synthesizer(grid)   ← raw-function path (§9c/§9d today)
        │
        └─ predict_state({}) → derived["sed_agn"]      (exact, eager)
           predict_photometry(...) under WavePrecomp   (precompute path — NEW panel)
```

## What changed upstream (PR1 on this branch)

| Fix | Effect the notebook should now show |
|-----|-------------------------------------|
| Synthesizer NLR/BLR Cloudy line backends pre-warmed at factory time (commit `72df4dd7`) | `nlr_synthesizer`/`blr_synthesizer` now run under `predict_photometry`/`WavePrecomp`, not only eager `predict_state`. A precompute-vs-exact panel is now possible. |
| AGN params carry free `Uniform`/`LogUniform` priors + block-scoped `'*': FREE` (commit `757fc330`) | `recipes.agn_panchromatic()` and `agn={'*': FREE}` now actually free AGN params (was a silent no-op). The notebook can show a "what's free" provenance table that is no longer empty. |

## §9f can now use the grammar (no more hand-assembly)

The combined `lines` selectors `nlr_blr` (analytic) and `nlr_blr_synthesizer`
(grid-backed) now express a unified AGN — disc + torus + **both** line regions —
in one `SEDModel.build`. So §9f's hand-assembly (separately-built disc + torus +
raw `compute_nlr_sed_synthesizer` + raw `compute_blr_sed_synthesizer`) can be
replaced by:

```python
m = SEDModel.build(
    ssp_data=ssp,
    agn={'type': 'composable',
         'disc':  {'type': 'kubota_done'},
         'torus': {'type': 'simple'},          # 1000 K graybody, matches Synthesizer
         'lines': {'type': 'nlr_blr_synthesizer'},  # NLR + BLR from the same Cloudy grids
         'agn_log_lbol': Fixed(agn_log_lbol), '*': FIXED},
    redshift=Fixed(0.0),
)
L_tot_t = np.asarray(m.predict_state({}).derived['sed_agn'])
```

Note: the composable total will **not** bit-match the old hand-sum — the runner
energy-couples the disc into the torus (`compose_l_nu`), whereas the hand-sum
used an independent `torus_frac` split (a ~12% continuum difference). The
composable behavior is the more physical one; flag this in the panel caption.

## Changes (each maps to a numbered cell)

1. **§9c/§9d — surface the builder-grammar path beside the raw call.**
   Today these cells call `compute_nlr_sed_synthesizer(grid_path, ...)` directly.
   Add a short cell that builds the *same* unified AGN through the public grammar
   and shows the two SEDs overlay (they should agree — same grid, same
   normalization):
   ```python
   model = SEDModel.build(
       ssp_data=ssp,
       agn={'type': 'composable',
            'disc':  {'type': 'kubota_done'},
            'torus': {'type': 'nenkova'},
            'lines': {'type': 'nlr_synthesizer'},
            'agn_log_lbol': Fixed(agn_log_lbol),
            '*': FIXED},
       redshift=Fixed(0.0),
   )
   sed_grammar = np.asarray(model.predict_state({}).derived['sed_agn'])
   ```
   Narration: "the line regions a fit would use are the same Cloudy grids §9c
   reads directly — the grammar is the supported entry point; the raw function
   is the unit-matching probe."

2. **§9g (NEW) — precompute-vs-exact parity panel.**
   Predict the §9 model with `approx=None` vs `approx=WavePrecomp()`; plot
   per-band photometry residuals. Assert the documented bound: additive AGN
   emitters are exact filter-integrated via `lnu_filter_integral_batch`, so the
   line-region structure must agree to ~float precision (this is the PR #629
   reddest-band failure class — line regions are the high-risk case). Reuse the
   notebook's existing `_assert_comparable` rather than inventing a tolerance.

3. **§9h (NEW, short) — "what a fit would free".**
   Build `recipes.agn_panchromatic()` and print `model.spec.summary()` +
   `model.spec.free_params`. Before PR1 this freed **zero** AGN params; the cell
   documents the now-real free set and that `'*': FREE` is block-scoped (only the
   active disc/torus/lines blocks' consumed params move). One-line callout:
   "freeing an AGN param now changes `predict()` — verified by the contract test
   `tests/contract/test_agn_block_consumes.py`."

4. **§9f — inclination-mask parity note (no new code).**
   Add a markdown paragraph: tengri's smooth sigmoid disc-incident mask vs
   Synthesizer's hard `inclination + theta_torus > 90°` cliff is a deliberate
   gradient-safety choice, not a discrepancy. Point at the Type-1/Type-2
   transition panel already present.

## Out of scope for the notebook update
- Re-deriving stellar/SFH/dust panels (§1–§6) — recon shows them at/above parity.
- Forcing tengri defaults to match Synthesizer numerically (independent-code rule).
- The other five reproduction comparators.

## Rendering (deferred, canonical repo only)
The worktree lacks the gitignored grids. To render:
```bash
# in /Users/suchethacooray/Projects/tengri (canonical, has data/)
synthesizer-download --agn-test-grids        # if not already present (~340 MB)
PYTHONPATH=src .venv/bin/python reproduction/synthesizer/01_synthesizer.py
# then re-copy rendered _figs/ per project_reproduction_docs_publish (committed copies)
```
Record the §9g precompute residual and any §9c grammar-vs-raw agreement number in
the PR body; never claim a rendered figure that was not run.
