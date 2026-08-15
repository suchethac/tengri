# Git history bloat: measurement and options

`.git` is 2.1 GB packed, from 3629 MB of uncompressed blobs across 41 502 objects and
2569 commits. A fresh `git clone` pays that 2.1 GB before it can run a single test.

This is a measurement, not a proposal. **Nothing here has been executed.** The recommendation
at the bottom is to leave history alone and fix the inflow instead.

## Where the volume is

By file type, across all of history:

| Category | Size | Blobs | Share |
|---|---:|---:|---:|
| Executed notebooks (`*.ipynb`) | 1451.4 MB | 3580 | 40% |
| Binary grids (`*.h5`, `*.npz`, `*.npy`) | 646.7 MB | 121 | 18% |
| JAX compilation caches (`.nb_home/`) | 268.3 MB | 18 | 7% |
| Everything else (source, docs prose, tests, figures) | 1262.6 MB | 37 783 | 35% |

The intuition that data files are the problem is wrong by a factor of two. **Executed
notebooks cost more than every HDF5 grid combined.**

### Why notebooks dominate

An executed notebook embeds its figures as base64 PNG inside the JSON. Two consequences:

1. Git stores a whole new blob per revision. Re-running a notebook changes every embedded
   image, so a one-cell edit rewrites the entire multi-megabyte object.
2. Base64 delta-compresses poorly, so the pack file cannot recover much.

`docs/reproduction/cigale.ipynb` is the clearest case: **126.6 MB across 51 revisions**, or
about 2.5 MB per commit that touched it. The six reproduction notebooks together account
for ~484 MB — roughly a quarter of the packed repository — from six files.

### Top paths by cumulative history size

| Cumulative | Revisions | Path |
|---:|---:|---|
| 218.1 MB | 4 | `data/dl14_templates.h5` |
| 126.6 MB | 51 | `docs/reproduction/cigale.ipynb` |
| 96.8 MB | 40 | `docs/reproduction/agnfitter.ipynb` |
| 96.5 MB | 359 | `src/tengri/forward/sed_model.py` |
| 77.5 MB | 31 | `docs/reproduction/bagpipes.ipynb` |
| 67.1 MB | 27 | `docs/reproduction/synthesizer.ipynb` |
| 65.8 MB | 31 | `docs/reproduction/prospector.ipynb` |
| 63.9 MB | 1 | `data/fsps_prsc_miles_chabrier.h5` |
| 63.8 MB | 1 | `data/ssp_prsc_miles_chabrier_wNE...h5` |
| 63.8 MB | 1 | `data/fsps_prsc_miles_chabrier.h5.1` |
| 57.8 MB | 33 | `notebooks/00_quickstart.ipynb` |
| 51.5 MB | 4 | `data/themis_templates.h5` |

The four largest `.nb_home/.cache/tengri_jax_cache/jit_run_evi_geovi-*` blobs are 46.4,
45.1, 32.5 and 31.5 MB — 155.5 MB between them, one revision each. They are listed
separately because they are one accident rather than one path: 18 cache blobs totalling
268.3 MB, each with a content-hashed filename, committed before `.nb_home/` was ignored.

Two entries deserve comment.

**`src/tengri/forward/sed_model.py` — a source file in fourth place.** It is 8823 lines
and 403 KB, and each of its 359 revisions stored a fresh blob averaging 269 KB. The repo
sets no explicit file-length limit, so this is an observation rather than a rule violation:
a module this size is hard to review and hard to hold in context, and the history cost is
the measurable symptom of that. Splitting it would help reviewability first and history
second.

**`data/fsps_prsc_miles_chabrier.h5.1`** is a stray copy created by a download that did not
overwrite cleanly. It was removed in #1563, but its 63.8 MB is permanent in history.

## What a rewrite would recover

Each row assumes `git-filter-repo` purging that path from all history, then `gc --prune`.
Savings are of uncompressed blob volume; packed savings are typically 40–60% of these.

| Scenario | Recovers | Risk |
|---|---:|---|
| A. `.nb_home/` JAX caches only | 268.3 MB | Effectively none — machine-local compilation artifacts, referenced by nothing, gitignored since. |
| B. A + untracked `.h5` grids (`dl14`, `themis`, the `.h5.1` stray) | ~600 MB | Low — none are tracked at HEAD; all are regenerable from `scripts/`. |
| C. B + strip outputs from all historical `.ipynb` | ~2.0 GB | High — rewrites 3580 blobs across most of the history. |

## Recommendation: do not rewrite

The repository is public (since 2026-03-21). A rewrite changes every commit SHA after the
earliest touched commit, which means:

- every existing clone and fork diverges irrecoverably and must be re-cloned;
- every SHA cited in an issue, a PR discussion, or a `CITATION.cff` release note becomes a
  dead reference;
- for a package meant to be cited in papers, reproducibility of "the version I ran" is
  the whole point of the history, and breaking SHA references undermines it.

Scenario A is the one arguably worth it — 268 MB for essentially zero semantic risk. Even
there, the cost is not the risk of the rewrite but the force-push to a public repo, which
is disruptive out of proportion to 268 MB.

**Fix the inflow instead.**

### Not by stripping outputs

An earlier version of this page recommended `nbstripout` on `docs/reproduction/*.ipynb`
as the way to stop the largest inflow. **That is wrong and would break the published
site.** `docs/conf.py` sets `nbsphinx_execute = "never"`, so the committed outputs *are*
the figures the site displays — `docs/reproduction/cigale.ipynb` carries 18 cells of
embedded PNG. Stripping them blanks every reproduction and spine page, and breaks
`tests/contract/test_reproduction_docs_sync.py`, which asserts the embedded figure hashes
match between each source notebook and its docs copy.

The `.gitignore` precedent that suggested it does not generalize: `notebooks/*.ipynb` is
ignored *because* those are jupytext pairs whose `.py` is the source of truth, and the
numeric-prefixed spine notebooks are un-ignored precisely so their outputs survive for
rendering. Every committed notebook in this repository carries outputs on purpose.

### What actually applies

1. **A size ceiling, enforced.** `tools/check_notebook_size.py` runs in the `lint` job and
   fails any committed notebook over 4 MB (largest today: 3.00 MB). This does not shrink
   history — nothing does, short of the rewrite this page argues against — but it turns the
   next 3 MB notebook into a decision instead of a surprise.
2. **Externalize the figures.** The structural fix. Write figures to a `_figs/` directory
   and reference them instead of embedding base64: an unchanged figure then re-commits as
   the *same* blob rather than a new one, and PNG on disk is ~⅓ smaller than its base64
   form. The directories already exist on both sides — `reproduction/<name>/_figs/`, which
   `scripts/_render_s9.py` and `scripts/_render_audit_radio_xray.py` already write into, and
   `docs/reproduction/_figs/` — so the pattern is established; the notebooks just do not use
   it for their inline figures. This is a real refactor of six published notebooks plus the
   sync test, so it wants its own change with a docs build to verify.
3. **Stop committing each reproduction notebook twice.** `reproduction/<name>/01_<name>.ipynb`
   and `docs/reproduction/<name>.ipynb` are near-identical by construction — the sync test
   asserts their figures match — so every re-render costs ~5 MB, not ~2.5 MB. Generating the
   docs copy at build time, the way `scripts/sync_spine_notebooks_for_docs.py` already does
   for the spine, would halve reproduction inflow with no visual change.
4. **Split `src/tengri/forward/sed_model.py`.** 8823 lines is the reviewability problem;
   the history cost is a bonus.
5. **Keep `.nb_home/` and `data/*.h5` gitignored.** Both rules exist and work. The caches
   entered history before the rule did.

If the clone size becomes a real barrier to contributors, revisit scenario B — and do it
once, announced, alongside a tagged release, rather than incrementally.

## Reproducing these numbers

```bash
git rev-list --objects --all > /tmp/objs.txt
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  < /tmp/objs.txt > /tmp/sized.txt

# by cumulative size per path
awk '$1=="blob" && $4!="" {s[$4]+=$3; n[$4]++} END \
  {for (x in s) printf "%8.1f MB  %4d revs  %s\n", s[x]/1048576, n[x], x}' \
  /tmp/sized.txt | sort -rn | head -25
```
