# Auditing physics parity against a reference code

A reproduction notebook that only shows agreement is under-tested. This is the
method for auditing tengri against a reference SED-fitting code, attributing
every residual to a proven mechanism, and separating genuine bugs from
deliberate convention differences.

Layout and rendering rules live in [`reproduction/CONTRACT.md`](../../reproduction/CONTRACT.md).
This document is about *method*. The Prospector audit is the worked example:
it produced four code fixes (union-grid filter quadrature, cloud-in-cell SSP age
weights, the SSP solar-luminosity contract, a template-loader recursion), one
opt-in parity toggle, and two "not a bug, a convention" findings.

## The method

### 1. Run the reference code's own engine

Never compare against a reimplementation of the reference. The driver wraps
whatever the reference actually executes: `python-fsps` for Prospector (the
engine it holds in `CSPSpecBasis.ssp`), the CIGALE modules, ProSpect through
`rpy2`. If the left panel is not the reference's real output, the comparison
tests nothing.

### 2. Compare end to end, not just per block

Per-block spectra miss whole classes of error. Drive the reference's real
predict/fit path (e.g. `SpecModel.predict`) and `tengri.predict_photometry`
through the *same* filter curves, with luminosity distance and every physical
constant pinned on both sides. Redshifting, distance dimming, mass-unit and
filter-convention bugs only surface here.

### 3. Decompose per code before comparing across codes

For each band, project each code's *own* spectrum through the filters yourself
and compare against what that code *reports*. An internally consistent code
returns ratio 1.000. A mismatch localizes a projection or quadrature bug with
no cross-code ambiguity — this is how the filter-grid quadrature bug was
isolated bit-exactly, before anyone had to argue about which code was right.

### 4. Arbitrate with a code-independent reference

When two independent engines disagree with tengri in the same direction, build a
dense pure-NumPy convolution of the *same* template arrays and converge it. The
reference decides. "Every other code is wrong" requires positive evidence: the
+1 % stellar offset was blamed on two upstream codes for months before a dense
reference showed the bug was ours.

### 5. Pin the convention axes

Most apparent disagreements are conventions, not bugs. Pin each explicitly and
state which side each notebook uses:

| Axis | How codes differ |
|---|---|
| Solar luminosity | FSPS uses 3.839e33 erg/s; IAU 2015 is 3.828e33 — a flat 0.29 % |
| IMF | Codes default differently (FSPS defaults to Kroupa); always pin it |
| IGM | tengri applies IGM by default; most codes do not |
| Filter convolution | Photon-counting `1/λ` (DSPS, FSPS) vs energy `1/λ²` (CIGALE) |
| Metallicity | Absolute `log10(Z)` vs solar-relative, and `Zsun` per SSP library |
| Dust energy balance | Lyman continuum excluded from dust heating (CIGALE, tengri) vs re-emitted (FSPS) |
| Nebular grid | Different photoionization inputs; report the ratio, do not force agreement |

A convention difference is not a defect, but an *undocumented* one is. Document
it, and where cross-code fits matter, expose an opt-in parity flag.

### 6. Attribute before acting

Close every residual with a number. Classify each as:

- **tengri bug** — fix it, with a regression test against an independent reference.
- **convention** — document it; consider a parity toggle.
- **reference-side or port artifact** — document it and move on.

Refuting a hypothesis is a result. Measuring that a suspected cause changes
nothing is worth recording, so the next person does not re-chase it.

## Fixing what you find

Work in an isolated worktree. Write the regression test first, against a
reference independent of the code under test — a golden value captured from the
implementation you are fixing proves only that it still does what it did.

When a fix changes physics, regenerate affected golden values and record *why*
in the test, next to the previous numbers.

When adding a parity toggle, wire it end to end and verify each layer: the build
grammar, `Parameters`, the component config, every runtime call site, any
precomputed lookup table that bakes the same choice, and the compile signature
(two models differing only by a flag must not share a compiled kernel). A toggle
that reaches only some layers is a silent no-op — verify by asserting the
prediction changes, not that the flag is stored.

Before claiming a test failure is pre-existing, reproduce it on a pristine
checkout of the base branch. Before claiming a fix works, measure it end to end
through `predict_photometry`, not through internal state.

## Recording the result

The per-section scalars printed by the notebook are the quantitative record.
When a section's headline is "these agree", print the number that shows it; when
a section carries a known offset, print that number too and name its cause. A
notebook that hides its residuals cannot be trusted with the ones it shows.
