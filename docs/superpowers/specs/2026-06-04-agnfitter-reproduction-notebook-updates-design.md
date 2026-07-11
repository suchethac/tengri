# Reproduction-notebook updates after the AGNfitter-rX parity work

**Date:** 2026-06-04
**Status:** Proposed (awaiting review)
**Scope:** `reproduction/agnfitter/01_agnfitter.py` only (the five sibling
notebooks are provably unaffected — see §4).

## 1. What changed upstream this session

Three things merged across PRs #656 / #677 that touch the *AGNfitter-rX*
torus surface:

1. **`skirtor_agnfitter` is now a registered, correctly-normalized torus
   block** (`register_agn_block("torus", "skirtor_agnfitter")`,
   `torus_blocks.py:292`). It is the node-exact implementation of AGNfitter-rX's
   parameter-collapsed `SKIRTOR_mean_3p.pickle`, distinct from the
   full-grid X-CIGALE `skirtor` block. Its exact runtime path now applies
   the `L_SUN × ∫dν` normalization it previously skipped (was emitting
   ~1e-18; now physical).

2. **Five torus shape parameters are now wired through `SEDModel.build`**
   (were silent no-ops): `agn_log_nh_silva`, `agn_a_cat3d`,
   `agn_fwd_cat3d`, `agn_incl_skirtor`, `agn_tv_skirtor`. The nested-dict
   grammar now accepts them as short keys inside an `agn.torus` sub-block.

3. **`cat3d_wind` and `skirtor_agnfitter` now interpolate with node-exact
   PCHIP** (`interp_nd_pchip`) instead of the C²-smoothing triweight
   kernel, which had been smearing the mid-IR torus peak by ~29% median.

A fourth change — the `*_analytic` → `*_sed` AGN function rename — is
**not** notebook-relevant: no reproduction notebook calls those functions
by name (verified by grep). They go through builder block strings
(`"cat3d_wind"`, `"skirtor"`, …), which are unchanged.

## 2. Why the AGNfitter notebook should change

The notebook's §9c ("Torus library face-off") currently carries a deferred
caveat (lines 634–652):

> *"tengri instead keeps the full Stalevski grid … A future exact implementation of
> `SKIRTOR_mean_3p` (as was done for `silva04` and `cat3d_wind`) would give
> a node-exact AGNFITTER-RX-style SKIRTOR panel."*

That future implementation now exists (`skirtor_agnfitter`). The notebook should stop
describing it as future work and demonstrate it.

Separately, the §9c silva04 panel compares AGNfitter at `log_nh=23.0`
against tengri silva04 with **no** `log_nh` set — fine while the param was
a no-op, but now that `log_nh_silva` is wired, the panel can pin both sides
to the same column density and become a genuine matched-parameter check.

## 3. Proposed changes (AGNfitter notebook)

### 3a. §9c — turn the SKIRTOR caveat into a result
- Add a fifth comparison: `tengri_torus("skirtor_agnfitter", ...)` against
  AGNfitter's `SKIRTOR_mean_3p`, mapped to the same 3 retained axes
  (inclination → `incl_skirtor`, optical depth → `tv_skirtor`; opening
  angle handled per the block's signature). This panel is the
  **node-exact** comparison and should agree to node-exact tolerance.
- **Keep** the existing `skirtor` (full X-CIGALE grid) panel — it is the
  *intentional-difference* comparison (40 µm vs 25 µm peak). The two panels
  side by side tell the real story: tengri can reproduce *either* reduction
  of the Stalevski models, and the choice is the user's.
- Rewrite the §9c markdown: replace the "future port would give…"
  paragraph with "tengri ships both — `skirtor` (full X-CIGALE grid) and
  `skirtor_agnfitter` (the node-exact `SKIRTOR_mean_3p` average)", and
  re-measure the silva04 / cat3d_wind agreement numbers (the "≲1.6×" claim)
  against the new PCHIP curves before quoting them.

### 3b. §9c — pin silva04 column density on both sides
- Pass `log_nh=23.0` (→ `log_nh_silva`) to the tengri silva04 call so it
  matches the `af_kw=dict(log_nh=23.0)` already used on the AGNfitter side.
- Optionally pin cat3d via the now-wired `a_cat3d` / `fwd_cat3d` if the
  AGNfitter side fixes those; otherwise leave at defaults and say so.

### 3c. §9d and the combined-SED panels — re-render, re-verify prose
- No code change required: `tengri_torus("cat3d_wind")` is called in §9d
  and the two combined-SED figures. PCHIP makes the returned curve node-
  exact (sharper mid-IR peak than the previously-rendered triweight curve).
- **Re-execute** these cells and re-read any prose that quotes peak
  positions, ratios, or "agrees to" numbers; update wording to match the
  re-rendered figures. This is the main correctness risk — a stale prose
  number next to a re-rendered figure.

### 3d. Optional — a short note on now-settable torus shape params
- One markdown sentence (or a tiny inset) noting that `log_nh_silva`,
  `a_cat3d`, `fwd_cat3d`, `incl_skirtor`, `tv_skirtor` are now fittable via
  the nested-dict grammar, pointing readers to the new gallery sweeps
  (`examples/agn/plot_*_sweep.py`, #677) rather than duplicating them here.

### 3e. Re-render + re-publish (mandatory mechanics)
- Render the source `.py` via jupytext in a worktree with the canonical
  `data/` deps copied in (cue_weights + SSP grids), per
  `[[feedback_repro_render_pythonpath]]` / `[[feedback_worktree_data_deps]]`.
- **Copy the rendered `.ipynb` into `docs/reproduction/agnfitter.ipynb`**
  in the same PR — reproduction notebooks are committed copies, not
  auto-synced; the `test_reproduction_docs_sync.py` contract test guards
  drift (`[[project_reproduction_docs_publish]]`).

## 4. Notebooks that do NOT change (and why)

| Notebook | AGN torus used | Affected? |
|----------|----------------|-----------|
| `cigale` | `skirtor` / `skirtor2016` (X-CIGALE block) | No — triweight block untouched |
| `prospect_r` | `skirtor` (oa/tau/p/q, pre-existing params) | No — X-CIGALE block untouched |
| `synthesizer` | `nenkova`, `skirtor`, `two_temperature` | No — none use cat3d/skirtor_agnfitter |
| `prospector` | `nenkova` only | No |
| `bagpipes` | no AGN | No |

The PCHIP change and the 5 newly-wired params are confined to
`cat3d_wind` + `skirtor_agnfitter`, which only the AGNfitter notebook
exercises. The X-CIGALE `skirtor` block (cigale/prospect_r/synthesizer)
still uses triweight and is byte-for-byte unchanged.

## 5. Out of scope
- The cigale notebook's pre-existing "remaining torus-peak residual" note
  (line 1282) — unrelated to this session; leave as-is.
- Broader gallery consolidation (~300 examples) — separate effort.
- Stale SSP-gated test cleanup — tracked in #685.

## 6. Verification checklist
- [ ] §9c node-exact `skirtor_agnfitter` panel agrees to node-exact
      tolerance against `SKIRTOR_mean_3p`.
- [ ] silva04 panel uses matched `log_nh` on both sides.
- [ ] All cat3d_wind figures re-rendered; every quoted ratio/peak number
      re-measured against the new curves.
- [ ] §9c markdown no longer says the SKIRTOR port is "future".
- [ ] `docs/reproduction/agnfitter.ipynb` re-copied;
      `test_reproduction_docs_sync.py` passes.
- [ ] Notebook executes end-to-end without error (one process, OOM rules).
