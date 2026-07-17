# Consolidating the approximation state and the data bootstrap

## Motivation

A novice-proxy documentation audit (agents restricted to public docs after a fresh
install) surfaced a cluster of defects that share one shape: **a second way of asking a
question that already had an answer.** The duplicate spelling drifts, because nothing
exercises it, and then it answers confidently and wrongly.

Two instances are proven below. Both are shipped bugs on the front-page path.

This spec covers only what was *verified against `main`*. The audit's raw list was longer;
most of it turned out to be deliberate design (see "Explicitly out of scope").

### Instance 1: the approximation state is unobservable (#1222)

`Fitter` puts a photometry fit **on** the WavePrecomp look-up table by default
(`approx="auto"` → `_resolve_fit_approx` → `with_approx(WavePrecomp())`), and then warns
the user that they are *not* on it:

| `Fitter(approx=)` | inner `wave_precomp` | guard warns | verdict |
|---|---|---|---|
| `"auto"` — the default | `True` | `True` | **false positive** |
| `None` — forced exact | `False` | `True` | correct by accident |

The guard fires unconditionally for any photometry `ForwardModel`, so it carries no
information. The mechanism is a single line:

```python
approx = getattr(model, "_approx", None) or {}     # inference/fitter.py
```

`_approx` is the *lowered flag dict* on `SEDModel`, an internal detail with ~35 readers.
`ForwardModel` — the canonical model argument — has no such attribute, so the probe
returns `{}` and the guard concludes "exact path". Twenty lines away in the same file,
`_resolve_fit_approx` asks the *same question correctly*, via `_has_modern_approx()`, a
method `ForwardModel` explicitly delegates to its inner SED.

One question, two spellings, one file. The spelling used by the hot loop stayed correct
because it is exercised; the spelling used only by a warning drifted, because nothing
checks a warning.

The deeper cost is not the spurious warning. It is that **no instrument could answer
"is the LUT live?"** — timing cannot (fit wall-clock is ~28:1 compile-dominated), the
plausibly-named entry points are never called (the LUT is a build-time state publication),
and the one runtime signal is a constant. Two independent audits reached opposite
confident conclusions from that fog.

### Instance 2: the two documented bootstrap steps do not connect (#1209)

```
tengri.download_ssp()   ->  fsps_prsc_miles_chabrier.h5                    (bare-stellar)
tengri.load_ssp()       ->  ssp_prsc_miles_chabrier_wNE_logGasU-3.0_...h5  (wNE)
```

Different files. The `load_ssp()` default is absent from `list_known_ssps()` **and** from
the remote catalog — whose own comment states that wNE grids are not shipped from it. A
fresh user therefore downloads one grid and the loader looks for another that cannot be
obtained, and the resulting `FileNotFoundError` points to `tengri.download_ssp('<name>')`,
which cannot fetch it either.

Contributing fragmentation, all verified:

- **Two `download_ssp` functions**, different objects, incompatible signatures:
  `tengri.download_ssp(name="fsps_prsc_miles_chabrier", dest=None, force=False)` from
  `_data_setup.py` (exported), and `tengri.data.download_ssp(name, dest_dir="data", *,
  overwrite=False, progress=True)` (self-contained, owns the remote catalog, and is the
  one the Cue error message points users at). Neither calls the other.
- **Two environment variables for one concept**: `TENGRI_DATA_DIR` governs *writes*
  (`_data_setup.py`, `bench/`); `TENGRI_DATA` governs *finds* (`facade.py`,
  `dsps_wrapper.py`). They never meet.
- `doctor()` globs `ssp_*.h5`, which cannot match the `fsps_*.h5` files
  `download_ssp()` actually writes.
- Each of `download_ssp()` and `load_ssp()` hardcodes its own default filename. Nothing
  forces agreement, so they drifted.

## Principle

**One question, one accessor. Wrappers delegate explicitly. Never `getattr`-probe for
state.**

This is the codebase's own documented first defect class — a guard that fails open — in a
new costume. `getattr(model, "_approx", None) or {}` cannot fail: a missing attribute
becomes "no LUT" rather than "I do not know."

## Design

Three independent changes, each branching from `main` so each receives real CI (a stacked
PR gets none — `tests.yml` gates on `pull_request: branches: [main]`).

### PR 1 — `model.approx`: make the approximation state observable

Add a frozen, introspectable view over state that already exists:

```python
@dataclass(frozen=True)
class ApproxState:
    """The effective approximation state of a model. Read-only."""
    wave_precomp: bool
    spectrum_precomp: bool
    feature_precomp: bool
    ztable: bool
    n_subbands: int
    def __bool__(self) -> bool:      # any LUT active
```

- `SEDModel.approx` — property, computed from the existing `_approx` lowered flags and
  `_approx_config_*` objects. Adds no state; changes no routing.
- `ForwardModel.approx` — delegates through the existing `_inner_sed_for_delegation()`
  seam, the same shape as `_has_modern_approx`.
- `_warn_if_exact_forward_path` asks `model.approx.wave_precomp`. The `getattr` probe
  dies.

The ~35 internal `_approx` readers are untouched. `_approx` (lowered flags) and
`_approx_config_*` (user configs) are two *layers*, not two spellings, and both stay.

Why an accessor rather than a one-line guard fix: a corrected guard leaves the state
unobservable, so the next guard re-invents its own probe and the class of bug survives.
Exposing `model.approx` retires it, and answers the question in #1222's title.

**Tests.** A LUT-equipped `ForwardModel` must not warn; an `approx=None` model must warn.
Both directions mutation-tested — a guard that has not been deliberately broken is not
verified.

**Risk: low.** Purely additive.

### PR 2 — one download path, one environment variable

- Collapse the two `download_ssp` implementations to one. `tengri.download_ssp` keeps its
  exported signature (it is the documented one, in `README.md` and `installation.md`) and
  delegates to the catalog logic in `tengri/data/`. `tengri.data.download_ssp` keeps
  working.
- `TENGRI_DATA_DIR` becomes canonical for **both** reads and writes. `TENGRI_DATA` is
  honored with a `DeprecationWarning` naming its replacement.
- `load_ssp()`, `doctor()`, and the SSP finders honor `TENGRI_DATA_DIR`.
- `doctor()`'s globs match the filenames `download_ssp()` actually writes.

**Risk: low.** One behavior change (an env var gains an alias), no physics.

### PR 3 — one default SSP

A single `DEFAULT_SSP` constant, consumed by **both** `download_ssp()` and `load_ssp()`,
so the two defaults cannot drift apart again. Value: `fsps_prsc_miles_chabrier` —
bare-stellar, present in `list_known_ssps()` and in the remote catalog, and the grid the
`recipes.*` configs require.

This is physics-affecting. `load_ssp()` has ~40 bare callers; a handful depend on the wNE
default explicitly and become explicit `load_ssp("prsc_miles_chabrier_wNE")`:

| Caller | Why it needs wNE |
|---|---|
| `tests/contract/test_feature_precomp_api.py` | comment: "wNE: lines baked into the templates" |
| `tests/regression/bug/test_bug_302_bakedin_lines_error.py` | comment: "default wNE → BakedInBackend" |
| `src/tengri/recipes/__init__.py` (`dust_demo`) | docstring: "Uses the BakedIn nebular path" |

An implicit default is the wrong dependency for a regression test regardless; making it
explicit is an improvement independent of this change.

**Risk: medium.** Ships with a full fast-tier run before and after, and an explicit list of
every caller changed.

## Explicitly out of scope

Verified as deliberate or already settled. Listed so they are not re-litigated:

- **The three forward branches in `loss_functions.py`** are load-bearing. Threading keeps
  the SSP grid an XLA `Parameter` op rather than a closure-captured `Constant`;
  collapsing them regresses compile time (#1201).
- **`predict()` vs `predict_photometry()`** is the #1043 contract's two-surface design,
  working as specified.
- **`Fitter(sed_model)` vs `Fitter(forward)`** is already soft-deprecated (#211). The
  remaining work is migrating ~13 docs (#1219) — a documentation task, not a code one.
- **`'*'` vs `all_params`** is not a contradiction: `'*'` is internally canonical,
  `all_params` is the preferred user-facing spelling. A known false positive.
- **`sed_model.py`'s size** — the project accepts long files; splits are declined.
- **Public API consolidation (#1043)** is closed. It consolidated *how you ask the model
  for an answer*. This spec addresses *how you build the thing* and *where the fast path
  lives* — untouched by that campaign, and where all verified redundancy sits.

## Success criteria

1. `print(model.approx)` answers "is the LUT live?" for both `SEDModel` and
   `ForwardModel`. The question that defeated two audits becomes a one-liner.
2. The `>100x` warning fires when and only when the fit is genuinely on the exact path;
   mutation-tested in both directions.
3. `tengri.download_ssp()` followed by `tengri.load_ssp()` works on a fresh install.
4. One `download_ssp`, one data environment variable, one default SSP name.
5. No `getattr`-probe for approximation state remains in `src/`.
