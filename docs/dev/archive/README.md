# Archive: Completed & Historical Documentation

Files in this directory have been completed, superseded, or represent past design decisions. They are preserved for reference but should not guide new contributor decisions. Refer to the active docs in `docs/dev/` for current project state.

## Archived Files (2026-05-05)

### Refactor Plans (Completed Phases)
- `REFACTOR.md` — 7-phase refactor plan (Phases 1, 4–7 complete; Phases 2–3 partial). Superseded by `20260404-refactor.md`.
- `phase_ii_2_stellar_migration.md` — Phase II-2 stellar SEDComponent migration (completed 2026-05-05).
- `DOCS_REFACTOR.md` — First iteration of docs/notebooks restructure (superseded by `DOCS_REFACTOR_REFINED.md`).
- `DOCS_REFACTOR_REFINED.md` — Docs restructure plan v2 (completed; implementation tracked in `api_migration_v0.x.md`).

### Audits & Verification (Completed)
- `test_audit_2026_04.md` — April 2026 test suite audit; 32 duplicate tests removed (completed 2026-04-08).
- `CITATION_AUDIT.md` — Citation verification sweep (completed 2026-03–04).
- `CITATIONS_TODO.md` — TODO list from citation audit (completed).
- `CITATIONS_ADS_VERIFIED.md` — Final citation verification (completed).
- `ASTRONOMER_GAPS_FOLDED.md` — Gaps audit vs. CIGALE/BAGPIPES (incorporated into `MISSING_FEATURES.md`).

### Notebook & Gallery Polish (Deferred)
- `NOTEBOOK_AXIS_POLISH.md` — Notebook axis polish tracking (deferred beyond v1.0).
- `NOTEBOOK_EDITORIAL_REVIEW.md` — Editorial review checklist (deferred).
- `GALLERY_POLISH.md` — Gallery script polish (deferred).

### Performance Analysis (Completed, Data Preserved)
- `jit-optimization-report-2026-04-18.md` — JAX JIT compile-time analysis (completed).
- `performance-bottleneck-analysis-2026-04-18.md` — Forward-model bottleneck analysis (completed).
- `performance-diagnostics-2026-04-17.md` — Memory/performance diagnostic session (completed).
- `performance-fix-2026-04-18.md` — Performance fix summary (completed).

### Design & Physics (Historical Reference)
- `design/` — Design docs for AGN, dust, metallicity, paper sections, SSP reformatting (historical decisions; reference only).
- `roadmap/` — Future physics components (ADAF, MAGPHYS dust, BPASS, etc.; aspirational, not active development).
- `nebular-beagle-comparison.md` — Nebular backend comparison vs. BEAGLE (completed; Cue/CloudyGrid/MAPPINGS chosen).
- `AGN_MODEL_COMPARISON.md` — AGN model selection (completed; K&D disc + SKIRTOR + Cue chosen).
- `spdx-headers.md` — SPDX license header audit (completed).

### Session Notes (Completed Discussions)
- `sessions/` — All dated session notes (2026-03-20 through 2026-04-22). These are working notes from implementation sprints. Canonical project state always in `20260404-refactor.md`, `api_migration_v0.x.md`, and CLAUDE.md.

---

## Active Documentation (in `docs/dev/`)

**Keep using these for guidance:**
- `NAMING_CONTRACT.md` — Canonical class/parameter naming (MANDATORY read before any refactor).
- `docstring-standard.md` — Docstring and documentation tier rules (MANDATORY before writing functions).
- `design_philosophy.md` — Architecture principles and design decisions.
- `20260404-refactor.md` — Current refactor status and module inventory.
- `api_migration_v0.x.md` — Public API migration plan (Phase 1–6, Part II scaffold).
- `DEPRECATION_AUDIT.md` — Deprecated aliases and removal timeline.
- `MISSING_FEATURES.md` — Backlog of missing features vs. competitive codes.
- `benchmarks/` — Performance data and inference method comparisons.

---

## Archive Strategy

**Goal:** Reduce contributor confusion by separating active plans from completed/historical work.

**Old files moved on 2026-05-05** for:
1. **Completed refactor phases** → Phases fully landed; history preserved.
2. **Completed audits** → Verified; data captured; recommendations implemented.
3. **Deferred polish** → Out of scope for v1.0; preserved for future reference.
4. **Historical design discussions** → Decisions finalized; notes archived.
5. **Session working notes** → Transient; canonical state in permanent docs.
6. **Performance snapshots** — Data preserved; reports archived once recommendations acted on.

**When to consult archive:** Only if debugging a *historical* bug, understanding *why* a decision was made, or recovering work from a past sprint.
