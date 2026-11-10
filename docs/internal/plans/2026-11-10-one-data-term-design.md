# One data term for the Gaussian likelihood (issue #3293)

Status: design, partly implemented. The module, the loss adapter and the contract tests
are in place; the flat seam, the batched path and the calibration helper follow.
Decision record: [ADR-0022](../../adr/0022-one-data-term.md).

## Problem

The per-galaxy Gaussian data term is written three times:

- `inference/loss_functions.py` builds it from a private `Fitter`, which is why the
  hierarchical flat seam cannot reuse it.
- `inference/_hierarchical_flat.py` computes its own chi-squared over the concatenated
  data and noise. It has no presence masks, no censoring and no noise model.
- `inference/batched_spectro.py` has a third version for the batched spectroscopy path.

The marginal is already shared. `observation/linear_marginal.py` is the one
linear-Gaussian marginal, and `batched_spectro.py` already calls the shared
`diag_gaussian_log_prob`. What is duplicated is the joint calibration and line design
assembly, which is written out again in `inference/likelihoods/marginalized.py`.

## Interface

`inference/data_term.py` holds one data term.

- `DataTermOptions(data_type, calibration_marginalize=False, cal_n_poly=3,
  cal_prior_sigma=1.0, eline_marginalize=None, eline_prior_type=None,
  use_components=False, line_model_sigma=True)` fixes the structure of a term. It is frozen
  and hashable.
- `build_data_args(model, data, noise, *, data_mask=None, presence=None,
  line_flux_data=None, runtime_redshift=None, params_override=None, line_model_sigma=True)`
  builds the traced bundle. The key set is structure and the values are data.
- `resolve_data_term_options(model, options)` takes the emission-line defaults from the
  model's spectroscopy configuration and returns the catalog arrays and fitted-amplitude
  priors that the Fitter assigns.
- `build_data_term(model, *, spec=None, data_args, options, likelihood=None)` returns a
  `DataTerm`. Its `log_prob(params, data_args)` is the positive log-likelihood
  `log p(d | params)`.
- `DataTermContext` is a Fitter-free record with the attribute names the likelihood
  cohort reads. `likelihood.py` accepts it without a signature change.

## Invariants

- `log_prob` is pure in `(params, data_args)`. It closes over structure only, so
  `jax.jit(term.log_prob)` compiles once per shape set. Calling it with new values in
  `data_args` moves the value and does not add a compile.
- `params` is the resolved physical dictionary: free and fixed values, with mirrors merged.
- The term is positive. The loss builders negate it.
- `presence` (exact 0 or 1) and `data_mask` (censoring flags) are honoured whenever the key
  is present in `data_args`. The keys are structure and the values are data.
- Every marginal goes through `observation/linear_marginal.py`.
- The term is not memoized on the Fitter in this change, so no cache-key policy row moves.

## Migration

1. Module and fitter delegation. The closure body moves from `loss_functions` into
   `build_data_term` with the sign flipped. The `Fitter` reads it needs are replaced by
   explicit inputs. `Fitter._build_data_args` and the emission-line initialization become
   delegates. `DataTermContext` carries the cohort into `_maybe_build_default_likelihood`.
2. Adapter. The three public loss builders negate `build_data_term`. A thin shim,
   `_build_data_neg_log_likelihood_fn`, keeps the keyword surface that the contract tests
   and `mass_profile` use. Its structural options are read with defaults so that a bare
   stand-in object still works.
3. Ledger. No `Fitter` attribute moves, so the engine and fingerprint policies are
   unchanged and the engine version does not bump.
4. Flat seam. A population record gains optional `presence` and `data_mask` keys.
   `FlatProblem.data_args` becomes `{"per_galaxy": ..., "shared": ...}`. The log-likelihood
   is `lax.map` of a checkpointed `log_prob`, summed over galaxies. The concatenated arrays
   stay in the body, and the LUT-bias warning call stays where the source-text test looks
   for it. A stochastic spec without the threaded inputs is refused by name, because the
   eager path drops the field-latent input.
5. Batched path. The preamble resolves the options. The plain and banded branches use
   `GaussianLikelihood(presence_key="presence")` and `MultivariateGaussianLikelihood` over
   the batch leaves. The joint calibration and line assembly moves into one helper in
   `observation/calibration.py`, used by both paths, and each path keeps its own
   normalization.
6. Documentation. The flat-seam, batched-spectroscopy and where-things-live pages describe
   the single term.

## Verification

- `tests/contract/test_data_term.py` checks the term against the loss builder for five
  configurations on the small synthetic grid: photometry, spectroscopy with emission-line
  marginalization and calibration, joint with a measured line-flux channel, a censored
  mask, and per-band presence. The comparison is `assert_array_equal`. The joint case
  adds a nebular backend because the loss cannot be built without one.
- A second test compiles `jax.jit(term.log_prob)` once and evaluates it on two value sets
  with the same shapes. The compile count stays at one and the value changes.
- A mutation that flips the sign inside `build_data_term` fails the bit-identity test.
- The existing contract tests for the cache keys, the unified loss builders and the
  banded covariance stay unchanged and green.

## Expected behaviour change

The flat seam gains masks and censoring when a population record carries them. Populations
without those keys are unaffected. The threaded forward used by the term can change the
XLA fusion, so a flat-seam value is not bit-identical to its earlier self. Parity with
`Fitter` at a relative tolerance of 1e-10 is the requirement, and the pull request should
state the addition.

## Out of scope

- The hierarchical path at `inference/hierarchical.py` and the native VI path at
  `inference/backends/vi/native.py` each have a copy of a similar term. They are a
  follow-up issue.
