# The float32 emission-line channel: three overflows, one class, all fixed

**Date:** 2026-09-05
**Verdict:** All three float32 line-channel blockers measured in
`bench/reports/2026-09-05_float32_spectroscopy_lines.md` are repaired, and **a
float32 line fit now runs on the default configuration** — `Fitter(approx="auto")`,
either nebular backend family. The three were **one class and three separate bugs**:
in each case a quantity outside float32 range was *materialized* on the way to an
answer that sits comfortably inside it. They were not, however, one bug: fixing
`predict_line_fluxes` alone left the Cue fast grid still failing, and the
`FeaturePrecomp` window path overflowed on something else entirely — not a line
luminosity at all, but the **stellar mass scale constant**.

**float64 is bit-identical on 71 of 116 arrays across a 13-seam cross-tree A/B**
(`array_equal`, not a tolerance), including **every photometry seam** and the whole
baked-in window-measurement path. On the remaining seams — the ones that carry a
line luminosity — it moves by at most **3.8e-14 relative**, and that movement is
**toward** the exact value: adjudicated against a 60-digit `decimal` reference over
60 points, the new spelling is closer on 28, the old on 15, tied on 17, with mean
relative error **3.36e-15 against the old 4.73e-15**. Bit-identity is unattainable
there in principle, and the reason is stated below rather than waved at.

**Platform:** Linux 6.8, AMD Ryzen 9 5900X (`JAX_PLATFORMS=cpu`), JAX 0.11.0 /
jaxlib 0.11.0. Branch `float32-line-channel-fix`, **stacked on
`float32-spectroscopy-lines`** (PR #2143) at `eb66d8f34`, which is itself rebased
onto `origin/main`. The measurement branch carries the strict `xfail`s this work
converts; stacking rather than cherry-picking keeps a single copy of those
assertions, so the diff here shows the fix and the flipped assertions and nothing
else. Every run set `TENGRI_DISABLE_JAX_CACHE=1` (the box was shared).

**Every number below was re-taken after that rebase**, in a private scratch
directory, rather than carried over: the branch this work stacks on moved under it,
and a second agent was writing to the scratch directory the first pass had used. The
re-take reproduced the earlier run **to the digit on all 13 seams** — which is the
evidence that neither the rebase nor the shared directory disturbed anything, and
not something that was assumed.

**Precision** is proven throughout on **the dtype of the array that came back**,
never on `jax.config.jax_enable_x64` — #1840: `tengri/__init__.py` re-enables x64 on
import, so the flag lies. Every new assertion checks `dtype` first.

**Every new assertion is non-zero, not merely finite.** Three of the four repairs
end in a `pow10` of a large negative exponent, whose natural failure mode is a flush
to exactly `0.0` — and zero is finite. PR #2100's still-open defect is precisely a
gradient that was identically zero on both backends while its guard pinned it
*finite*; a repair of this shape must not inherit that hole.

---

## The mechanism, established by measurement

The report being answered here named the root cause correctly for the first defect
and left the other two open. Measuring each before writing anything is what
separated them — and it is what PR #2100's campaign got wrong when it misdiagnosed a
structurally identical `multicolor_disc` overflow as an unreachable cancellation.

### Defect 1 — `predict_line_fluxes`: the line luminosity

The operator `loss_functions.py` selects for Cue and every other line-publishing
backend. Instrumented at the standardized origin, stellar+dust+Cue, *z* = 0.1:

| quantity | float64 | float32 |
|---|---|---|
| `state.derived["line_lums"]` (linear) | max 7.24e+41, 0/128 non-finite | max **`inf`**, **84/128 non-finite** |
| `state.derived["log_line_lums"]` | max 41.860 | max 41.860, **0/128 non-finite** |
| `state.derived["log_line_lums_attenuated"]` | max 40.896 | max 40.896, **0/128 non-finite** |
| `predict_line_fluxes` | `[4.81e-16 5.88e-16 1.70e-15 5.31e-16]` | **`[nan nan nan nan]`** |
| the same, computed in log | `[4.81e-16 5.88e-16 1.70e-15 5.31e-16]` | `[4.81e-16 5.88e-16 1.70e-15 5.31e-16]` |

The reading is unambiguous and it is not the one the code implied. **The log form
was already published and already finite**; the operator powered it back to a linear
~1e40 erg/s catalog — `pow10(_log_atten)`, one line — which is `inf` above 3.4e38,
and `apply_log10_scale(inf, -55.4)` is `inf/inf` = `nan`. `apply_log10_scale` is not
at fault: its peak factorization is sound, and by the time it is called the
information is already gone.

This is exactly #1859's account, on the operator #1859 did not reach. The last row
of the table is the fix, before it was written.

### Defect 2 — the `FeaturePrecomp` window LUT: the *mass scale*, not a luminosity

This is the one a "same bug, second site" assumption would have got wrong, and the
reason the brief's warning about #2100's misdiagnosis is worth taking literally. The
baked-in family has no line catalog, so the default fit routes to
`measure_line_fluxes(approx=True)` — which runs through `_line_flux_from_means` and
therefore **already carries #1859's grouping**. It still returned `nan`.

Instrumented at the same point (baked-in wNE, *z* = 0.1):

| quantity | float64 | float32 |
|---|---|---|
| `total_mass` | 1.0000e+10 | 1.0000e+10 |
| `scale = total_mass * L_sun` | 3.8280e+43 | **`inf`** |
| `window_integrals` | max 4.87e-10 | max 4.87e-10, finite |
| window mean **before** the scale | max 1.63e-15 | max 1.63e-15, finite |
| `window_means` (after the scale) | max 6.24e+28 | **`inf`, 12/12** |
| `measure_line_fluxes(approx=True)` | `[2.77e-16 4.09e-16 1.35e-15 3.22e-18]` | **`[nan nan nan nan]`** |

The overflow is in the **scale constant on its own**. Both the array it multiplies
(1.6e-15) and the product (6.2e28) are comfortably inside float32; only
`total_mass * L_sun` is not. `feat_mean - cont_mean` was then `inf - inf` = `nan`.
`derived_state.py` already says so in a comment — *"`stellar_mass_scale`, which is
~1e43 and so overflows float32"* — and publishes `log_stellar_mass_scale` for the
purpose; the window LUT is a consumer that never used it.

The exact projector survives because the stellar component applies the same scale
through `apply_log10_scale`, so the SED it hands `measure_line_flux_jax` is an
in-range ~1e28 `L_nu`. One consumer had the grouping and its sibling did not — the
same one-sidedness as defect 1, one layer down.

### Defect 3 — the Cue fast grid: Q_H, and a guard that was telling the truth

The measurement report recorded the build failing in float32 and building in
float64, flagged the guard's own message as suspect — *"the guard's message asserts
it is 'not a rounding gap' … on this evidence that wording is wrong"* — and listed
"why the Cue fast-grid guard fires only in float32" as unresolved.

**The guard was right, and so was its wording.** It compares the vmapped first node
against the eager reference with `jnp.allclose(..., rtol=1e-5)`. Its inputs were
`nan`, and `allclose(nan, nan)` is `False`. It was neither a rounding gap nor a
tracer regression: it was a `nan` arriving from upstream, reported as faithfully as
that guard can report anything. It should not be weakened, and it has not been.

Two materializations feed it, and **fixing defect 1 alone did not clear either**:

* **Build** (`nebular_grid_precompute.py`): `apply_log10_scale(flux, log10_ref_divisor)`
  recovers the ~1e40 luminosity *back out of* the in-range flux — `inf` — and then
  multiplies by `inv_qh = 1/Q_H`, which is `1/inf = 0`. `inf * 0` = `nan`. The two
  offsets are +55 and −53 dex and cancel to an O(1e-13) answer; applying them
  separately is what put a 1e40 number between them. The same `inv_qh` is ~1e-53,
  **below float32's smallest subnormal (1.4e-45)**, so it also flushed the
  photometry and rest-band per-Q_H columns to zero — silently, since a zeroed
  column logs to −300 rather than raising.
* **Reconstruction**: `reconstruct_nebular_line_lums` computes `nion * 10**log_lpq`,
  with `nion` ~1e53 (`inf` in float32) and the table value ~1e-13. The broadband
  twin `reconstruct_nebular_phot` **already** takes `log_nion` and adds exponents;
  the line sibling was the only one that did not.
* And one more, found only by running it: with no `state` passed,
  `predict_line_fluxes` fell back to `self._compute_nion(params)` → `inf` →
  `jnp.log10(inf)` = `inf`. `StellarSEDComponent.compute_log_nion` is the log-domain
  core that `compute_nion` itself wraps, so the round trip through the linear value
  was both lossy and longer.

**So the answer to the report's open question "whether the two `nan` paths are one
bug or two" is: three bugs, one class.** Defect 3 is *downstream* of defect 1 —
its guard fires because `predict_line_fluxes` handed it `nan` — but it is not
*cured* by it: two further materializations sit between them.

---

## What changed

Four sites, one principle: **a quantity outside float32 range is carried as an
exponent, never as a value.**

| file | change |
|---|---|
| `src/tengri/forward/sed_model.py` | `predict_line_fluxes` carries a `log_all_lums` log10 catalog wherever a producer publishes one (`log_line_lums`, `log_line_lums_attenuated`, the grid), and the cosmology tail exponentiates **once**, with the −55 dex distance already inside. Reddening in the log carrier is an add, taken from the same `_attenuate_line_catalog` screen on a unit catalog so the #1867 single-sourcing is preserved. Linear producers keep the old `apply_log10_scale` path unchanged. |
| `src/tengri/forward/sed_model.py` | New `_compute_log_nion`, the log-domain sibling of `_compute_nion`; the grid branch uses it instead of `log10(compute_nion(...))`. |
| `src/tengri/observation/line_measurement.py` | `measure_line_fluxes_from_window_lut` takes `total_mass` rather than a pre-multiplied `scale`, and applies `L_sun` as `_LSUN_MANTISSA` × `2**_LSUN_EXP2` via `jnp.ldexp`. |
| `src/tengri/components/nebular/nebular_grid_precompute.py` | Build applies `−log10 Q_H` as an offset instead of dividing by `inv_qh`, for the line, photometry and rest-band columns alike; new `_log_nion_of_state` (logsumexp for a multi-component Q_H); new `reconstruct_nebular_line_log_lums`, mirroring the existing `reconstruct_nebular_phot` closing step, with the linear sibling kept as a wrapper. |

### The float64 constraint, and where it can and cannot be met

The requirement was bit-identity, verified per seam with `array_equal`. It is met
where it is attainable and it is not attainable everywhere, so both halves are
stated.

**Where it is met, it is met by construction.** The window-LUT repair splits `L_sun`
by an **exact power of two**. Scaling a float by a power of two is exact and
commutes with rounding, so `ldexp(fl(m·x), k)` and `fl(2^k·m·x)` are the same
float64 — the identical property that lets `DEFAULT_COTANGENT_BOOST` be divided back
out without perturbing a float64 result. `test_the_mass_scale_power_of_two_split_is_exact_in_float64`
asserts it with `array_equal` over formed masses from 1e4 to 1e13 and window means
from 1e-22 to 1e-10, and the cross-tree A/B confirms it end-to-end: **30/30 forward
values bit-identical**.

**Where it is not met, it cannot be.** `apply_log10_scale(arr, s)` computes
`(arr/peak)·10^(s + log10 peak)`. Any grouping that avoids materializing `arr`
changes what `log10` is applied to, and a power-of-two rescale does not survive a
logarithm: `log10(peak/2^k)` is not `log10(peak) − k·log10(2)` in floating point.
A line luminosity is genuinely 1e40 and float32's ceiling is genuinely 3.4e38, so
not materializing it is not optional — and not materializing it necessarily
re-associates the float64 arithmetic. The measured cost:

| seam | arrays | bit-identical | max relative move |
|---|---|---|---|
| `predict_line_fluxes` (Cue, exact) | 30 | 0/30 | 1.44e-14 |
| `predict_line_fluxes` (Cue, fast grid) | 6 | 0/6 | 3.82e-14 |
| grid `log_line_per_qh` table | 2 | 0/2 | 4.32e-16 |
| grid `log_phot_per_qh` table | 2 | 0/2 | 1.37e-16 |
| grad `predict_line_fluxes` (exact) | 2 | 1/2 | 1.96e-15 |
| grad `predict_line_fluxes` (fast grid) | 2 | 0/2 | 2.24e-14 |
| grad `measure_line_fluxes(approx=True)` | 2 | 0/2 | 5.37e-16 |
| **`measure_line_fluxes`, both projectors** | **30** | **30/30** | **0** |
| **`predict_photometry` (baked-in)** | **15** | **15/15** | **0** |
| **`predict_photometry` (Cue, fast grid)** | **6** | **6/6** | **0** |
| **grid `log_restband_per_qh` table** | **2** | **2/2** | **0** |
| **grad `measure_line_fluxes(approx=False)`** | **2** | **2/2** | **0** |
| **`predict(...).lines.halpha`** | **15** | **15/15** | **0** |
| **total** | **116** | **71/116** | **3.82e-14** |

Every move is ≤3.8e-14 relative — inside the 1e-12 budget this repo's own precision
tests already use for "beyond XLA reassociation", and the same order as the 1.4e-14
float64 shift PR #2104 accepted for the cotangent boost on CUDA. **No photometric
result moves at all.**

**And the movement is an improvement, not a cost.** Both spellings approximate
`10^(log L − log 4πd_L²)` from the same inputs; the old one detours through a linear
1e40 intermediate and back. Adjudicated against a 60-significant-digit `decimal`
reference at 60 points (5 redshifts × 3 parameter offsets × 4 lines):

```
new spelling closer : 28
old spelling closer : 15
exactly tied        : 17
mean |rel err| old  : 4.730e-15
mean |rel err| new  : 3.362e-15
```

The new spelling is **1.4x more accurate in float64**. The float64 values that moved
moved toward the truth.

### Method note

The first pass at the float64 comparison hand-replicated the pipeline in a probe and
reported the power-of-two split as *not* bit-exact. Adding the missing control —
does the hand replication reproduce the public API bit-for-bit? — showed it did not
(5.7e-14 on its own), so every verdict drawn from it was a measurement of the
instrument. **The numbers above are a cross-tree A/B**: one script, the public API
only, run against a pristine worktree at the pre-fix commit and against the fixed
tree, compared with `array_equal`. A reference that has not been checked is not a
reference, and a probe that re-implements the thing it is measuring is not an
instrument.

---

### Three ratchets that the fix tripped by succeeding

Running the `lint` list extracted from this branch's own
`.github/workflows/tests.yml` — with the PR template's `sed` commands rather than a
transcription — caught three failures that `ruff` and `pytest` do not see, all of
them the *fix* moving a counter the repo pins:

* `check_zero_hiding_clamps.py`: 96 → **95**. The retired site is
  `inv_qh = 1.0 / jnp.maximum(nion, 1e-30)`. A genuine retirement, not a hoist: the
  reciprocal existed only to be a divisor and is now a `−log10 Q_H` offset, so no
  denominator remains. **The clamp was not the defect but it was hiding one** —
  `jnp.maximum(inf, 1e-30)` is `inf`, so in float32 that guard fired on the wrong
  end of the range and handed back a plausible `0.0`.
* `check_representable_floors.py`: 46 → **45**, the same site.
* `check_numeric_guards.py`: one bucket moved from `floor 1e-30` to `floor -30.0` in
  `nebular_grid_precompute`. That is the *same* guard re-expressed in log space
  (−30 dex ≡ 1e-30 linear), not a new 30-orders-smaller floor; the matcher reads the
  literal, not the space it lives in, so the ledger carries a note saying so. Its
  own header documents a same-bucket-swap blind spot — this was a **cross**-bucket
  swap, and the guard caught it.

Both pins were lowered with a stated reason, per each tool's own instruction, never
raised. Regenerating the ledger also silently dropped a hand-written comment line
from it; that line is restored and the ledger now warns about the behavior.

### A guard that was waiting for this

`tests/regression/precision/test_no_raw_nion_read.py` is #1206's Tier B tracker: a
two-way gate over `derived["nion"]` reads that rejects new sites **and** flags
allow-list entries whose file no longer matches, so a migration cannot land without
the inventory noticing. `forward/sed_model.py` was on it as

> `"forward/sed_model.py": "Tier B item 3 — fast-line feeds the erg/s reconstruct (~1e41)"`

and the repair made that entry stale — the guard failed, by design, on the fix
succeeding. The entry is **removed**, not widened: Tier B item 3 has landed for that
file. `nebular_grid_precompute.py` keeps its entry, because `_nion_of_state` is
retained as a fallback for a state that publishes only the linear value.

## Guards

`_check_channel_scales` (#1495) is **untouched**, and so is the fast-grid
`allclose` guard. Neither was weakened to let this case through; both were failing
on genuinely non-finite inputs and both now see finite ones. That the failures were
loud is what kept this from ever being silently wrong, and it stays that way.

Converted from strict `xfail` to passing, in
`tests/regression/precision/test_float32_fitting_path_seams.py`:

* `test_the_discrete_line_catalog_operator_survives_float32` — now also asserts
  dtype (#1840), **non-zero** (#2100), and agreement with float64 to <1e-4 in norm.
* `test_the_feature_precomp_line_path_survives_float32` — the default
  `Fitter(approx="auto")` line fit; now also asserts the gradient dtype and that it
  is **not identically zero**.

Added:

* `test_the_cue_fast_nebular_grid_builds_and_reconstructs_in_float32` — defect 3,
  which had no `xfail` of its own because the measurement report recorded it in
  prose. Builds the grid at both precisions, asserts it attached (a config that
  never reaches the graph would pass vacuously), and checks the reconstructed
  fluxes are finite, non-zero and tracking float64.
* `test_the_mass_scale_power_of_two_split_is_exact_in_float64` — `array_equal` over
  a wide mass range, plus the premise itself: that `total_mass * L_sun` still
  overflows float32, so the test fails loudly if the repair ever becomes
  unnecessary rather than passing vacuously.

**Not touched:** `test_the_bare_unweighted_observable_gradient_is_nonzero_in_float32`
remains a strict `xfail`. That defect — the bare `sum(predict_photometry)` gradient
identically zero in float32 — is unrelated to the line channel, still open, and
still measured. It was not fixed here and its pin was not removed.

## What is still NOT fixed

* **The bare unweighted-observable gradient is still identically zero in float32**
  (PR #2100's open item). Untouched; still pinned.
* **`components/nebular/line_precompute.py` carries the same class of bug and was
  left alone.** `precompute_line_per_qh` does `apply_log10_scale(flux, ref) / nion`
  and `reconstruct_line_lums` does `nion * lpq` — both materialize ~1e40. It has
  **no caller in `src/`** (only its own test module), so it is not on any model path
  and no float32 fit reaches it. Fixing it would have been an unmeasured change to
  an unreachable surface; it is recorded here instead.
* **Spectroscopy remains an order of magnitude worse than photometry in float32**
  (Finding 1 of the measurement report). Nothing here touches it; the SNR ceiling
  stands as measured.
* **CUDA.** Every number in this report is CPU. The repairs are pure re-association
  with no reduction-order content, so a backend split is not expected — but PR
  #2104's cotangent qualification (bit-neutral on CPU, 1.4e-14 on CUDA for 9 of 13
  seams) is exactly the shape of thing that is not expected and happens anyway. The
  cross-tree A/B has not been re-run on CUDA and this report does not claim it.
* **The float32 line-channel *accuracy* numbers have not been re-taken.** Findings 2
  and 3 of the measurement report said "no fit"; they can now be measured, and the
  matrix of float32-vs-float64 posterior gradients for `lines_cue` and for the
  `auto` arms of `lines_meas` is empty. That is the natural follow-up and it is not
  done here.
* **Noise-convention sensitivity (Finding 4) is unchanged**, and so is the
  underflow it rests on: `sigma**2` ~9e-39 is below float32's smallest normal. A
  line channel's float32 verdict is still a verdict on its noise model too.

## Reproduce

```bash
# The guards, on the module that pinned the defects.
TENGRI_DISABLE_JAX_CACHE=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
    tests/regression/precision/test_float32_fitting_path_seams.py -q -n 0

# ...and the modules it extends, which must stay green. -n 2 --dist=loadfile, NOT
# -n 0: three precision modules in ONE process abort inside XLA's CPU compiler.
TENGRI_DISABLE_JAX_CACHE=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
    tests/regression/precision/ -q -n 2 --dist=loadfile

# The line-channel contract and LUT guards CLAUDE.md names for this surface.
TENGRI_DISABLE_JAX_CACHE=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
    tests/regression/bug/test_bug_1770_line_lut_survives_dust.py \
    tests/regression/bug/test_bug_1748_feature_precomp_effect.py \
    tests/components/nebular/ tests/contract/ -q -n 2 --dist=loadfile
```

The float64 cross-tree A/B is reproduced by running one script against two
worktrees — a pristine checkout of the pre-fix commit and the fixed branch — and
comparing the saved arrays with `array_equal`. The script is not committed: it is a
one-shot differential instrument, and committing it would invite it to be re-run
against a single tree, which is the failure mode described in the method note above.
