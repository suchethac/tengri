# Archived notebooks

Superseded notebooks, kept for provenance. Nothing here is maintained, executed in CI,
or linted — `pyproject.toml` excludes `notebooks/archive` from ruff, and
`tools/check_british_spelling.py` skips any path with an `archive` part.

The live notebooks are the numeric-prefixed spine at `notebooks/` root
(`00_quickstart` … `12_simulation_populations`), which the published docs render via
`docs/spine/`, plus `notebooks/tutorials/`.

Expect these to be broken against the current API. They predate the v0.x parameter-name
migration (`docs/dev/api_migration_v0.x.md`) and the unified component dispatch
(ADR 0019), so a notebook here may call accessors that no longer exist or pass
parameter names that now raise. Read them as a record of what was tried, not as
runnable examples.

| Directory | Files | Span | What it is |
|---|---|---|---|
| `v1/` | 469 | 2026-03 → 2026-08 | The original pre-restructure tree, itself containing nested `_old_notebooks/`, `new_notebooks/`, and `notebooks/` generations. The messiest layer; oldest material. |
| `2026-04/` | 161 | 2026-04 → 2026-07 | Topic-organized set (`fitting/`, `models/`, `theory/`, `quickstart/`, `specialist/`, `demonstrations/`, `reference/`) retired when the spine notebooks took over the same ground. |
| `2026-05/` | 36 | 2026-05 → | Notebook-renewal pass; see `bench/reports/2026-05-06_notebook_renewal.md`. |
| `retired/` | 5 | 2026-04 → | Individually retired notebooks, no common theme. |
| `migrated_galleries/` | 10 | 2026-04 → | Notebooks whose content became sphinx-gallery examples under `examples/`; the rendered output lives in `docs/auto_examples/`. |

## Why these are still here

They are checked out rather than deleted because the physics comparisons in `v1/` and
`2026-04/` are the only record of several backend cross-checks that were never written
up elsewhere. Deleting them would leave the git history as the only copy, and history
is not where anyone looks.

## Directory names

`2026-04/` and `2026-05/` were `archive_2/` and `archive_2020506/`. The latter name
dropped a digit from `20260506`; the date is recovered from its first commit
(2026-05-06), not guessed.
