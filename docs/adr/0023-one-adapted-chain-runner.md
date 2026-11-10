# ADR-0023: One adapted-chain runner for the MCMC families

**Status:** Accepted (2026-11-11)

**Date:** 2026-11-11

**Related:** issue #3292 (this ADR closes it); #2157, #2431, #3317, #3318; ADR-0010 (inference backend protocol), which this ADR completes.

## Context

Each MCMC family carried its own copies of the same three programs: a warmup scan, a sampling
scan, and a fused warmup-then-sampling scan. Five families (NUTS, HMC, dynamic HMC, GHMC and the
first-order samplers) gave thirteen such functions in `backends/mcmc/_shared.py`, the GHMC
MEADS ensemble scan among them. The dispatch that chose among them was spread over `fitter.py`
(`compile()` and the batched vmap path, each with its own string dispatch), the catalog lanes
in `catalog.py`, the hierarchical flat seam in `_hierarchical_flat.py`, and the per-family
backend modules.

The copies did not agree. `compile()` warmed programs that `run()` never executed, because the
dense-mass policy and the adaptation differed between the two. The flat seam repeated one
branch per family inside a single `if driver in (...)`. Only NUTS published `divergent_mask`;
HMC, dynamic HMC, GHMC and the flat seam reduced divergences to a count. Key choreography
(which key warms, which key starts the chain, which keys drive the steps) lived in the
callers, so a change in one caller could alter the draws of another path without any test
failing.

## Decision

1. `backends/mcmc/chain_program.py` is the one runner. A family is a frozen `ChainProgram`
   (`NUTS`, `HMC`, `DYNAMIC_HMC`, `GHMC`, `FIRST_ORDER`) with `init`, `step`, `collect` and
   `kernel(cfg)`. A warmup strategy is an `Adaptation` (`WINDOW_NUTS`, `PATHFINDER_NUTS`,
   `WINDOW_HMC(n_leapfrog, diag)`, `WINDOW_GHMC`, `MEADS`, `DUAL_AVERAGING`). The per-family
   statics are frozen configs (`NUTSConfig`, `NUTSSamplerConfig`, `HMCConfig`, `GHMCConfig`,
   `FirstOrderConfig`).
2. `bind_logdensity(ld2, data_args, wall)` is the only place where the walls attach. `ld2` and
   the configs are static, `data_args` is traced.
3. Three module-level jitted entries: `adapt_chain` (warmup; returns the parameters and the
   warmup divergence and gradient counts), `sample_chain` (sampling with fixed parameters) and
   `run_chain` (warmup then sampling in one program). Static positions are fixed in the
   decorators: `(0, 1, 4, 6, 7)` for `adapt_chain`, `(0, 3, 6)` for `sample_chain`, and
   `(0, 1, 6, 8, 9)` for `run_chain`.
4. The runner never splits keys. `warmup_key`, `init_key` and `chain_keys` are used as given.
   `init_key` is read only by families that start from a fresh state (`starts_from_warmup=False`:
   dynamic HMC and GHMC); the others pass `None`.
5. `run_chain` with an ensemble adaptation (MEADS) takes `chain_keys` of shape
   `(n_chains, n_iter, 2)`, starts one chain per leading index from the first `n_chains` warmed
   states, and vmaps the sampling body. Only GHMC may use the ensemble adaptation; `run_chain`
   refuses it for any other program by name, at trace time.
6. `divergence_record(divergent)` is the one reducer. It returns `n_divergent` and
   `divergent_mask`, or nothing for families without a divergence (first order).
7. Registry (`_backend_registry.py`, `_registration.py`): `BackendEntry` carries `chain_program`,
   the run-path adaptation rule (`adaptation_for(kwargs)`), and a compile plan. The run path and
   the compile path have separate adaptation rules. The compile plan reads the runner's own
   defaults for the dense-mass policy, so `compile()` warms the program `run()` executes.
8. Dispatch sites that now go through the registry: `Fitter.compile()`, the batched vmap path
   `_fit_batch_vmap_mcmc` (four copies of the sampling scan become one scan over
   `program.step` and `program.collect` inside the existing memoized kernel), the NUTS and HMC
   lanes of `catalog.py` (`run_chain`), and the flat seam (one `run_chain` branch for every
   driver in `flat_chain_drivers()`). The deleted shims were `_nuts_full_scan`, `_nuts_warmup_only`,
   `_nuts_chain_scan`, `_hmc_full_scan`, `_hmc_warmup_only`, `_hmc_chain_scan`,
   `_dynamic_hmc_full_scan`, `_dynamic_hmc_chain_scan`, `_ghmc_full_scan`, `_ghmc_chain_scan`,
   `_ghmc_meads_scan`, `_first_order_full_scan` and `_first_order_chain_scan`.
9. Named tolerances. (a) `divergent_mask` is added on HMC, dynamic HMC, GHMC and the flat seam,
   where the count was the only divergence output (#2431). The key is additive, and the count
   key is unchanged. (b) `compile()` warms what `run()` executes for NUTS, HMC, dynamic HMC and
   the first-order samplers. GHMC is the exception: `compile()` keeps the window pairing that the
   hierarchical lane runs (`WINDOW_GHMC`, diagonal mass). MEADS warming needs a chain count,
   and `compile()` does not take one, so a GHMC `compile()` does not warm the program that
   `run()` samples with.
10. Residual string dispatch, kept on purpose. (a) `Fitter.run()` routes `"auto"` and the legacy
    `"mcmc"` alias to a concrete method name before any program is chosen. (b) The batch
    adaptation in `_fit_batch_vmap_mcmc` picks its algorithm and kwargs by method name
    (`blackjax.hmc` for HMC and dynamic HMC, the NUTS window for NUTS and GHMC), and its sampling
    statics (`sample_cfg`) are chosen the same way, because the batch path builds its own
    adaptation kwargs. (c) The jit and blackjax routing in `fitter.py` (`method == "jit"`,
    `method == "blackjax"`) picks a backend, not a sampler family. Removing (b) needs the batch
    adaptation to move into a registry rule as well.
11. Two pre-existing defects found in review are left as issues and are not changed here.
    #3317: the Barker warm-up produces NaN step size and acceptance on the fixture of the parity
    tests, on both the old and the new code. #3318: the momentum scale of the window GHMC path.
    Neither affects the parity claim below, because both sides of each comparison share the
    defect.

## Parity evidence

- Draw-level A/B per family at fixed keys. `fitter.run(method, key=PRNGKey(0), n_warmup=20,
  n_samples=20)` on the small photometry fixture of `tests/contract/test_inference_cache_keys.py`,
  base (origin/main) against head, `np.array_equal` on every posterior array: NUTS, HMC,
  dynamic HMC, GHMC (MEADS default), mala and barker (cold and warm cache). All equal. The
  published Barker step size and acceptance are NaN on both sides (#3317), so they compare
  unequal by `np.array_equal`; they are NaN on both sides.
- Dispatch A/B: `fitter.run` for NUTS, HMC, dynamic HMC, GHMC and mala, and `fit_batch` on two
  galaxies for NUTS, HMC, dynamic HMC and GHMC, 56 arrays in all. All equal.
- Flat seam A/B on a real SSP grid, two galaxies, five bands, for NUTS, HMC, dynamic HMC and GHMC:
  `psd_sigma`, `psd_tau_myr`, divergence count and step size equal bit for bit.
- Inline raw-BlackJAX oracles in `tests/contract/test_chain_program.py`. Each family has an
  oracle written directly from `blackjax` (`build_kernel` with `lax.scan`, `window_adaptation`,
  `meads_adaptation`, `dual_averaging_adaptation`), and tests check `sample_chain`,
  `adapt_chain` and `run_chain` against it with `np.array_equal`. The oracles stay when the
  shims are deleted, so the deletion does not remove the reference.
- Lowering and cache tests: the lowered `run_chain` has the same `while` count as the raw
  NUTS program, and `sample_chain._cache_size()` does not move when a sampling-only static
  changes.
- Mutants, each caught by the named test: splitting `chain_keys[0]` inside the oracle
  (`sample_chain` and `run_chain` tests fail for NUTS, HMC, dynamic HMC, GHMC and mala);
  swapping the warmup and init keys for the fresh-start families (flat-seam A/B draws change);
  replacing the compile adaptation with Pathfinder (`test_compile_warms_the_programs_run_then_hits`
  fails); swapping the GHMC alpha and delta (MEADS parity fails).

## Consequences

- One definition per family of the warmup and sampling programs. A change to a family's kernel
  reaches every entry point through `chain_program.py`.
- The PRNG order of each family is set by its caller and is pinned by the parity tests, not by
  the scans.
- `compile()` warms the program `run()` executes for four families. GHMC remains a documented
  exception, and a GHMC `compile()` will not remove the first-call compile cost of `run()`.
- Divergence diagnostics on HMC, dynamic HMC, GHMC and the flat seam gain `divergent_mask`.
  Consumers that read only the count see no change.
- `run_chain` gained the `init_key` argument, and `ChainProgram.kernel` and `init` take the family
  config. Callers outside the repository that used the deleted scan names must move to the runner.
- The residual string dispatch in the list above is the next candidate. Moving the batch
  adaptation kwargs into the registry is the step that removes branch (b).
