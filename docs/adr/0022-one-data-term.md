# ADR-0022: One data term for the Gaussian likelihood

**Status**: Proposed
**Date**: 2026-11-10
**Related**: issue #3293; #3161, #3174; ADR-0010 (inference backend protocol); ADR-0021 (precompute providers)

## Context

The per-galaxy Gaussian data term is written in three places. The per-galaxy Fitter
builds it from private `Fitter` state in `inference/loss_functions.py`. The hierarchical
flat seam (`inference/_hierarchical_flat.py`) computes its own chi-squared over the
concatenated data, with no presence masks, no censoring and no noise model. The batched
spectroscopy path (`inference/batched_spectro.py`) has a third version. The marginal is
already shared through `observation/linear_marginal.py`; the duplicated part is the joint
calibration and emission-line design assembly.

Because the flat seam cannot reuse the Fitter's term, its likelihood differs from the
per-galaxy fit whenever a band is masked or censored. Any fix to one path has to be made
three times.

## Decision

1. `inference/data_term.py` is the one module that defines the Gaussian-family data term.
2. `DataTermOptions` fixes the structure of a term: data type, calibration marginalization,
   emission-line mode and prior, component routing and the line-model sigma flag. It is
   frozen and hashable.
3. `build_data_term(model, *, spec, data_args, options, likelihood)` returns a `DataTerm`.
   Its `log_prob(params, data_args)` is the positive log-likelihood, pure in its two
   arguments, and closes over structure only.
4. The traced bundle comes from `build_data_args`. Its key set is structure: a key is
   present exactly when its channel is configured. Its values are data.
5. `presence` (exact 0 or 1) and `data_mask` (censoring flags) are honoured whenever their
   key is present. No separate branch is needed for either.
6. Every marginal still goes through `observation/linear_marginal.py`. The term adds no
   marginalization math of its own.
7. The loss builders are adapters: they call `build_data_term` and negate. The loss sign
   convention stays where it was, so the minimizers see no change.
8. `DataTermContext` is a Fitter-free record with the attribute names the likelihood cohort
   reads. The cohort accepts it without a signature change.
9. The data term is not memoized on the Fitter in this change. No `Fitter` attribute moves,
   so the engine and fingerprint policies and the engine version are unchanged. A later
   memo would add one `exclude` row to each policy and never key the term by content.
10. The flat seam, the batched path and the calibration helper migrate in follow-up parts.
    The hierarchical and native VI copies are follow-up issues and out of scope here.

## Consequences

- One definition of the likelihood. A fix or a new channel is made once and reaches every
  path that composes the term.
- The hierarchical flat seam gains masks and censoring for population records that carry
  them. This changes results for masked data; populations without those keys are unchanged.
  The pull request that lands it names the addition.
- The threaded forward used by the term can change XLA fusion. Values on the flat seam
  agree with `Fitter` at a relative tolerance of 1e-10, not bit for bit with their earlier
  output.
- A new module with a wider surface than the loss builders it replaces. The emission-line
  resolution and the data-argument builder move out of `Fitter`, which is a net reduction
  in that class.
- Callers that imported private helpers from `loss_functions` keep working through
  re-exports and a thin shim. The shim is removed once the callers move.
- Trade-off: each loss builder call constructs a new term. The compiled engine cache
  keys are unchanged, so the cost is one Python-side construction per build.
