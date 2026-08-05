# SPDX-License-Identifier: BSD-3-Clause
"""Hierarchical inference for population-level PSD recovery.

Shares PSD hyperparameters (σ_PSD, τ_PSD) across N galaxies while
each galaxy retains its own latent field ξ_i and physical parameters.

The total parameter vector is:
    Θ = {φ_shared, {ξ_i, θ_i}_{i=1}^N}

where φ_shared = {σ_PSD, τ_PSD} (or more generally, the PSD shape).

Usage:
    hfitter = PopulationFitter(model_template, galaxies)
    result = hfitter.run("vi", n_iterations=25)
    result.shared_params  # posterior on (σ_PSD, τ_PSD)
"""

from __future__ import annotations

import functools
import time
import warnings
from dataclasses import dataclass, field

__all__ = ["PopulationFitter", "PopulationPosterior"]

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference._hierarchical_flat import (
    FLAT_SAMPLERS,
    FLAT_UNSUPPORTED,
    run_flat_sampler,
)
from tengri.inference.fitter import resolve_method
from tengri.utils.transforms import to_bounded, to_unbounded

#: Acceptance below which a Ray Tracing chain is treated as not having sampled.
#:
#: Not a tuning knob — a separator between "mixed poorly" and "did not move".
#: The observed failure sits at 3.4e-10, orders below anything a working chain
#: produces, so the exact value is uncritical. 1e-4 leaves room for a genuinely
#: bad but non-degenerate run to come back and be judged on its own diagnostics
#: rather than refused here.
_DEGENERATE_ACCEPT_RATE: float = 1e-4


class DegenerateChainError(RuntimeError):
    """A sampler returned draws that never moved from their initialization.

    Distinct from a convergence warning. This is not a chain that mixed badly;
    it is a chain that did not sample. Raised rather than warned because the
    draws are indistinguishable from a successful fit by inspection — they
    carry the MAP solution, so they look like a plausible answer (#1530).
    """


def chain_is_degenerate(chain, accept_rate: float) -> bool:
    """Whether a finished chain failed to sample at all.

    Split out from the sampler so the decision can be tested against synthetic
    chains, without paying for a hierarchical fit to reach it. A guard whose
    only exercise is an end-to-end run tends to be tested in one direction
    (it fires) and not the other (it does not fire on a healthy chain).

    Parameters
    ----------
    chain : array_like, shape (n_samples, n_dim)
        Post-burn-in draws.
    accept_rate : float
        Mean acceptance probability over the same draws.

    Returns
    -------
    bool
        True when the chain never moved.

    Notes
    -----
    Both conditions are needed. ``accept_rate`` is an *expectation*, so a chain
    can carry a small mean acceptance and still contain one unique row; and a
    chain can contain near-duplicate rows while genuinely having moved, which
    the exact-uniqueness test correctly leaves alone.
    """
    n_unique = int(jnp.unique(jnp.asarray(chain), axis=0).shape[0])
    return n_unique <= 1 or float(accept_rate) < _DEGENERATE_ACCEPT_RATE


@dataclass
class PopulationPosterior:
    """Results from hierarchical PSD inference.

    This class holds the posterior distribution over shared PSD hyperparameters
    (σ_PSD, τ_PSD) inferred across a population of galaxies, along with optional
    per-galaxy individual posteriors.

    Parameters
    ----------
    shared_samples : dict
        Posterior samples for shared PSD params. Keys are param names (e.g.,
        'psd_sigma', 'psd_tau_myr'), values are arrays of shape (n_samples,).
    shared_params : dict
        Posterior mean of shared PSD params (computed from shared_samples).
    individual_samples : list of dict, optional
        Per-galaxy posterior samples. Each element is a dict with per-galaxy
        parameter names as keys. If None, individual posteriors are not stored
        (for memory efficiency).
    method : str
        Inference method used (e.g., "Hierarchical EVI (JIT)").
    wall_time_s : float
        Total wall-clock time for inference.
    diagnostics : dict
        Method-specific diagnostics (e.g., number of iterations, convergence info).

    Notes
    -----
    This dataclass is the return type for PopulationFitter.run(). Access
    population posteriors via `shared_samples` and `shared_params`, and
    per-galaxy posteriors via the `individual` property (returns a list of
    lightweight objects with `.samples` and `.params` attributes).

    Examples
    --------
    >>> result = fitter.run("vi", n_iterations=20)
    >>> result.summary()  # Median and 68% credible intervals
    >>> ax = result.plot_population(params=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"))
    """

    shared_samples: dict
    shared_params: dict
    individual_samples: list | None = None
    method: str = ""
    wall_time_s: float = 0.0
    diagnostics: dict = field(default_factory=dict)
    _model: object = field(default=None, repr=False)

    @functools.cached_property
    def properties(self):
        """The property catalog over the galaxy axis of a population fit.

        Contract §1 — **same names, more axes**. Each galaxy's properties are
        evaluated on its own parameter draws **merged with the shared
        hyperparameters**: in a hierarchical fit the per-galaxy block alone is
        not a complete parameter set, and evaluating it without the shared PSD
        block would silently answer a different question.

        Returns
        -------
        CatalogProperties
            ``[name]`` -> shape ``(n_galaxies, n_samples)``; ``ci(name)`` ->
            ``(n_galaxies, 3)``.

        Raises
        ------
        RuntimeError
            If the fit carries no per-galaxy samples, or no model reference.

        Examples
        --------
        >>> pop = pop_fitter.run(...)  # doctest: +SKIP
        >>> pop.properties["stellar_mass"].shape  # doctest: +SKIP
        (12, 400)
        """
        from tengri.inference.catalog_fitter import CatalogPosterior, CatalogProperties
        from tengri.inference.posterior import Posterior

        if self.individual_samples is None:
            raise RuntimeError(
                "This population fit carries no per-galaxy samples, so it has no "
                "per-galaxy properties. Shared hyperparameters are in "
                "`shared_samples` / `summary()`."
            )
        if self._model is None:
            raise RuntimeError(
                "No model reference on this PopulationPosterior — cannot compute "
                "properties. (It is populated automatically by PopulationFitter.run().)"
            )

        posts = [
            Posterior(
                # The per-galaxy block is NOT a complete parameter set on its own:
                # the shared PSD hyperparameters live in `shared_samples`. Merge
                # them, per-galaxy values winning on any key collision.
                samples={**self.shared_samples, **samp},
                params={},
                method=self.method,
                wall_time_s=0.0,
                diagnostics={},
                _model=self._model,
            )
            for samp in self.individual_samples
        ]
        return CatalogProperties(
            CatalogPosterior(posteriors=posts, method=self.method, n_galaxies=len(posts))
        )

    def summary(self) -> dict:
        """Median and 68% CI for shared PSD parameters.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary mapping parameter names to summary statistics. Each
            parameter has keys 'median', 'lo_68' (16th percentile), and
            'hi_68' (84th percentile).
        """
        result = {}
        for name, arr in self.shared_samples.items():
            vals = np.array(arr)
            result[name] = {
                "median": float(np.median(vals)),
                "lo_68": float(np.percentile(vals, 16)),
                "hi_68": float(np.percentile(vals, 84)),
            }
        return result

    def __repr__(self) -> str:
        n = next(iter(self.shared_samples.values())).shape[0]
        return (
            f"PopulationPosterior(method='{self.method}', "
            f"n_samples={n}, "
            f"wall_time={self.wall_time_s:.1f}s)"
        )

    @property
    def individual(self):
        """Per-galaxy posterior marginals as a list of lightweight objects.

        Parameters
        ----------
        None

        Returns
        -------
        list of SimpleNamespace
            Each element has ``.samples`` (dict) and ``.params`` (dict).
            Returns empty list if ``individual_samples`` is None.

        Notes
        -----
        Each per-galaxy posterior is marginalized over the shared PSD hyperparameters.
        The ``.params`` field contains the median of each per-galaxy parameter.
        """
        from types import SimpleNamespace

        if self.individual_samples is None:
            return []
        result = []
        for samp in self.individual_samples:
            params = {
                k: float(np.median(v)) if hasattr(v, "ndim") and v.ndim == 1 else v
                for k, v in samp.items()
            }
            result.append(SimpleNamespace(samples=samp, params=params))
        return result

    def population_diagnostics(
        self,
        exclude_prefixes: tuple[str, ...] = ("psd_xi",),
    ) -> dict:
        """Convergence diagnostics for the population fit.

        Computes split-Rhat and effective sample size for the shared PSD
        block, and (when ``individual_samples`` is present) aggregates
        per-galaxy diagnostics into population-level summaries.

        Parameters
        ----------
        exclude_prefixes : tuple of str, optional
            Parameter-name prefixes to skip when computing per-galaxy
            diagnostics. Default ``("psd_xi",)`` skips the GP latent
            fields, which carry one chain entry per grid point and
            inflate dict size without adding interpretable information.

        Returns
        -------
        dict
            Two-level structure::

                {
                    "shared": {
                        param_name: {"rhat": float, "ess": float},
                        ...
                    },
                    "per_galaxy": {                # only if individual_samples is set
                        param_name: {
                            "rhat_p50": float,    # median across galaxies
                            "rhat_p90": float,
                            "rhat_max": float,
                            "ess_p50": float,
                            "ess_min": float,
                            "n_galaxies": int,
                        },
                        ...
                    },
                }

            Use ``"per_galaxy"`` to spot a single galaxy whose chain has
            stalled — a high ``rhat_max`` with low ``rhat_p50`` is the
            signature.

        Notes
        -----
        Reuses :func:`tengri.analysis.diagnostics.rhat` and
        :func:`tengri.analysis.diagnostics.effective_sample_size`. Static
        (zero-variance) parameters are dropped silently.

        Examples
        --------
        >>> result = fitter.run("vi", n_iterations=20)
        >>> diag = result.population_diagnostics()
        >>> diag["shared"]["sfh_field_psd_sigma"]
        {'rhat': 1.012, 'ess': 318.4}
        """
        from tengri.analysis.diagnostics.autocorrelation import (
            effective_sample_size,
            rhat,
        )

        rhat_shared = rhat(self.shared_samples, exclude_prefixes=exclude_prefixes)
        ess_shared = effective_sample_size(self.shared_samples)
        out: dict = {
            "shared": {
                name: {
                    "rhat": float(rhat_shared.get(name, float("nan"))),
                    # effective_sample_size returns a dict-of-dicts; pull the scalar.
                    "ess": float(ess_shared.get(name, {}).get("ess", float("nan"))),
                }
                for name in rhat_shared
            }
        }

        if self.individual_samples is None or not self.individual_samples:
            return out

        # Aggregate per-galaxy diagnostics: for every parameter present in
        # at least one galaxy, collect that galaxy's rhat / ess and report
        # the median, 90th percentile, and (for rhat) max across galaxies.
        per_param_rhats: dict[str, list[float]] = {}
        per_param_esss: dict[str, list[float]] = {}
        for samp in self.individual_samples:
            r_i = rhat(samp, exclude_prefixes=exclude_prefixes)
            e_i = effective_sample_size(samp)
            for name, val in r_i.items():
                per_param_rhats.setdefault(name, []).append(float(val))
            for name, val in e_i.items():
                # effective_sample_size returns dict-of-dicts; pick the scalar.
                ess_scalar = val.get("ess", float("nan")) if isinstance(val, dict) else val
                per_param_esss.setdefault(name, []).append(float(ess_scalar))

        per_galaxy: dict = {}
        for name in per_param_rhats:
            r = np.array(per_param_rhats[name])
            e = np.array(per_param_esss.get(name, []))
            per_galaxy[name] = {
                "rhat_p50": float(np.median(r)) if r.size else float("nan"),
                "rhat_p90": float(np.percentile(r, 90)) if r.size else float("nan"),
                "rhat_max": float(np.max(r)) if r.size else float("nan"),
                "ess_p50": float(np.median(e)) if e.size else float("nan"),
                "ess_min": float(np.min(e)) if e.size else float("nan"),
                "n_galaxies": int(r.size),
            }
        out["per_galaxy"] = per_galaxy
        return out

    def plot_population(self, params=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"), ax=None):
        """Scatter plot of shared PSD parameter posteriors.

        Parameters
        ----------
        params : tuple of str
            Two parameter names for x and y axes.
        ax : matplotlib Axes, optional
            If None, creates a new figure.

        Returns
        -------
        matplotlib Axes
            The axes object with the scatter plot.

        Notes
        -----
        Plots posterior samples as a scatter cloud. Each point is one posterior sample.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        px, py = params
        if px in self.shared_samples and py in self.shared_samples:
            x = np.array(self.shared_samples[px])
            y = np.array(self.shared_samples[py])
            ax.scatter(x, y, s=8, alpha=0.4, color="C0", edgecolors="none")
            ax.set_xlabel(px)
            ax.set_ylabel(py)
            ax.set_title("Population posterior (shared PSD params)")
        else:
            ax.text(
                0.5,
                0.5,
                f"Parameters {px!r} or {py!r} not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return ax


class PopulationFitter:
    """Hierarchical inference for shared PSD parameters.

    Manages population-level inference via hierarchical VI, learning the shared
    PSD hyperparameters (σ_PSD, τ_PSD) across a population of galaxies while
    preserving per-galaxy latent fields and physical parameters.

    Parameters
    ----------
    model_factory : callable
        Function(psd_sigma, psd_tau_myr) → Model.
        Creates a model with the given PSD params. All other params
        (SFH, dust, etc.) come from the model's Parameters.
    galaxies : list of dict
        Each dict has 'flux_obs', 'noise', and optionally 'spec_obs',
        'spec_noise', 'wave_spec'.
    psd_sigma_prior : tuple
        (lo, hi) for uniform prior on σ_PSD.
    psd_tau_prior : tuple
        (lo, hi) for uniform prior on τ_PSD (Myr).
    data_type : str
        "photometry" or "spectroscopy".

    Notes
    -----
    Wraps all hierarchical inference methods (EVI, VI, MCMC) with automatic
    initialization via per-galaxy MAP estimation. The class builds a single
    flat parameter vector [φ_shared, ξ_1, θ_1, ..., ξ_N, θ_N] and optimizes
    it via the specified method.

    Attributes
    ----------
    n_galaxies : int
        Number of galaxies in the population.

    Examples
    --------
    >>> import jax
    >>> from tengri import PopulationFitter, SEDModel, Parameters, Uniform
    >>> # Define a factory that builds a model given shared PSD params
    >>> def model_factory(psd_sigma, psd_tau_myr):
    ...     spec = Parameters(
    ...         sfh_field_psd_sigma=Uniform(0.1, 4.0),
    ...         sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    ...     )
    ...     return SEDModel(spec, ssp_data)  # ssp_data loaded separately
    >>> # galaxies = [{'flux_obs': ..., 'noise': ...}, ...]
    >>> # pop = PopulationFitter(model_factory, galaxies)
    >>> # result = pop.run('vi', key=jax.random.PRNGKey(0))
    """

    def __init__(
        self,
        model_factory,
        galaxies,
        psd_sigma_prior=(0.1, 4.0),
        psd_tau_prior=(1.0, 300.0),
        data_type="photometry",
        *,
        _via_routing: bool = False,
    ):
        # ── Soft deprecation: prefer PopulationSEDModel + Fitter routing ──
        # Direct ``PopulationFitter(model_factory, galaxies, ...)`` keeps
        # working bit-for-bit; this nudges new callers to the canonical
        # surface. Routing through ``Fitter(forward, ...)`` (when
        # ``forward`` holds a :class:`PopulationSEDModel`) sets
        # ``_via_routing=True`` and silences the warning.
        if not _via_routing:
            import warnings

            warnings.warn(
                "PopulationFitter(model_factory, galaxies, ...) is deprecated "
                "and will be removed in tengri v1.0. The canonical entry point "
                "is the ForwardModel + PopulationSEDModel pattern: "
                "template = SEDModel.build(...); "
                "pop = PopulationSEDModel(sed=template, galaxies=galaxies, "
                "shared=('sfh_field_psd_sigma', 'sfh_field_psd_tau_myr')); "
                "forward = ForwardModel.build(population=pop, observation=obs); "
                "result = Fitter(forward).run('vi'). "
                "See issue #211.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.model_factory = model_factory
        self.galaxies = galaxies
        self.n_galaxies = len(galaxies)
        self.psd_sigma_bounds = psd_sigma_prior
        self.psd_tau_bounds = psd_tau_prior
        self.data_type = data_type

        # Create a template model to get spec info
        self._template = model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        self._spec = self._template.spec
        self._free_names = [
            n
            for n in self._spec.free_params
            if n not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr")
        ]

    def run(self, method="vi_nonlinear_fast", *, key=None, allow_unvalidated=False, **kwargs):
        """Run hierarchical inference.

        Parameters
        ----------
        method : str
            **NIFTy-backed (CorrelatedFieldMaker, native PSD learning)**

            - ``"vi_nonlinear_fast"`` — geoVI via NIFTy ``optimize_kl``
              (default).
            - ``"vi_nonlinear"`` — geoVI; same runner as fast, kept for API symmetry.
            - ``"vi_linear_fast"`` — MGVI via NIFTy ``optimize_kl``.
            - ``"vi_linear"`` — MGVI; same runner as fast, kept for API symmetry.

            **Pure-JAX (lax.while_loop, no NIFTy) — tier="broken"**

            - ``"native_vi_linear"`` — MGVI inside ``lax.while_loop``.
              3–4× faster than NIFTy MGVI on CPU; O(1) memory in N.
            - ``"native_vi_nonlinear"`` — geoVI inside ``lax.while_loop``.
              Comparable speed to NIFTy geoVI; prefers lower N (≤20).

            Both native backends are registered ``tier="broken"`` — they
            segfault on DPL/dense_basis photometry mocks (#231). They are
            substantially faster when they run, so they are kept and remain
            reachable via ``allow_unvalidated=True``, but validate per-problem
            before relying on one. They are also **not posterior-equivalent**
            to the NIFTy path: the fitted PSD timescale
            ``sfh_field_psd_tau_myr`` has been measured to differ by an order
            of magnitude between them (82 vs 6 Myr) — which is why reaching
            them has to be a deliberate act rather than a default.

            **MCMC**

            - ``"mcmc_raytrace"`` — Ray Tracing on flat vector.
            - ``"mcmc_ess"`` — **not** elliptical slice sampling here. ESS is a
              :class:`~tengri.inference.fitter.Fitter`-only method; on this class
              the name is an alias onto ``native_vi_linear``, so it is refused
              with the rest of the broken tier. Use ``mcmc_raytrace``, or run ESS
              per-galaxy through ``Fitter``.

            **Pure-JAX (lax.while_loop, no NIFTy) — tier="broken"**

            Faster on paper (3–4x NIFTy MGVI on CPU; O(1) memory in N) but
            registered ``tier="broken"``: ``[UNSTABLE]``, segfaults on
            DPL/dense_basis photometry mocks (#231). Both refuse to run without
            ``allow_unvalidated=True``.

            - ``"native_vi_linear"`` — MGVI inside ``lax.while_loop``.
            - ``"native_vi_nonlinear"`` — geoVI inside ``lax.while_loop``.

            .. note::
               ``native_vi_linear`` was the default from ``b7c4fa1e2`` until
               2026-07. It was chosen for speed *before* the segfault was
               validated (#231, 2026-05-22), and the tier change never
               propagated back to the signature — this method's own
               ``ValueError`` for an unknown method went on naming
               ``vi_nonlinear_fast`` "(default)" the whole time. There is no
               NUTS option here: ``mcmc_nuts`` is not in the hierarchical
               ``_method_map`` and raises.

        key : PRNGKey, optional
            Random key for reproducibility. If None, uses PRNGKey(0).
        allow_unvalidated : bool, optional
            Run a ``tier="broken"`` method anyway — for benchmarking or backend
            development, not for science. Default False.
        **kwargs
            Passed to the inference method.

        Returns
        -------
        PopulationPosterior
            Results object with shared_params, shared_samples, individual_samples,
            and diagnostics.

        Notes
        -----
        Unlike ``Fitter.run()``, ``PopulationFitter.run()`` does not support
        warm-start initialization via ``init_from`` because hierarchical
        inference methods (EVI, CFM-based geoVI, Ray Tracing) initialize
        per-galaxy parameters via MAP estimation automatically. The ``init_from``
        parameter is not meaningful in the hierarchical context.

        Automatic initialization: all methods initialize per-galaxy parameters
        via MAP estimation before starting the hierarchical inference. First call
        may be slow due to JIT compilation. Subsequent calls are fast.
        Approximate runtime: ~30 seconds for 10 galaxies on CPU (method-dependent).

        Compile-cost amortization across catalog sizes
        ----------------------------------------------
        Unlike :class:`~tengri.inference.catalog_fitter.CatalogFitter`,
        ``PopulationFitter`` does **not** expose an ``n_pad`` argument:
        the hierarchical population field couples all N galaxies, so
        padding with dummy galaxies would contribute spurious prior
        mass to the population hyperparameters (e.g. ``psd_sigma``,
        ``psd_tau_myr``) even with masked likelihoods. To amortize XLA
        compile cost across notebook restarts, slurm tasks, or sweeps
        over different catalog sizes, rely on the persistent
        compilation cache instead — see
        :func:`tengri.enable_persistent_cache` and
        ``docs/performance/compilation.md``.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Resolve old method names to canonical names, emitting deprecation warnings
        method = resolve_method(method, emit_warning=True)

        # There are no hierarchical-specific method overrides any more.
        #
        # There used to be one: `mcmc_ess -> native_vi_linear`, on the grounds
        # that "ESS is a Fitter-only method". It silently substituted a
        # DIFFERENT sampler — and after #231 a tier="broken" one — for the
        # method the caller named, with no warning and no entry in the result's
        # diagnostics. A user who asked for elliptical slice sampling got MGVI
        # and had no way to notice.
        #
        # The premise is now false: `mcmc_ess` runs hierarchically through the
        # flat seam like every other sampler. Silent substitution is never the
        # right repair for an unsupported method — either support it, or raise
        # and say so. `resolve_method` above still maps deprecated *spellings*
        # to canonical names, which is renaming, not substitution.

        # Gate on the name the caller actually asked for. This used to have to
        # run AFTER the override table, because `mcmc_ess` was rewritten to
        # `native_vi_linear` and gating the pre-override name would have let a
        # tier="broken" backend in through the alias. With the table gone there
        # is no alias left to sneak through, so the ordering constraint is gone
        # too -- but the gate is not. `resolve_method` above checks the *name*
        # only; it never consults the registry tier (#1394).
        #
        # This is the outer of two gates. `run_flat_sampler` applies
        # `check_usable` again on the flat path, which is deliberate
        # redundancy: the seam is reachable enough that neither gate should
        # depend on the other still being there.
        from tengri.inference._backend_registry import refuse_if_broken

        refuse_if_broken(method, allow_unvalidated=allow_unvalidated)

        # 6 canonical VI methods for PopulationFitter (no CFM in this table):
        #   vi_nonlinear / vi_nonlinear_fast → _run_geovi (standard NIFTy optimize_kl)
        #   vi_linear / vi_linear_fast       → _run_geovi with linear_resample mode
        #   native_vi_linear                 → _run_native_vi_linear (pure-JAX MGVI)
        #   native_vi_nonlinear              → _run_native_vi_nonlinear (pure-JAX geoVI)
        # vi_nonlinear_fast / vi_linear_fast use the same runner as their non-fast counterparts
        # because CFM is a different model (different parameter space) and must not be used as
        # a speed variant.  _run_geovi_cfm stays as a private method callable via "evi_nifty".
        _method_map = {
            "vi": ("geovi", None),
            "vi_nonlinear": ("geovi", None),
            "vi_nonlinear_fast": ("geovi", None),
            "vi_linear": ("geovi", "linear_resample"),
            "vi_linear_fast": ("geovi", "linear_resample"),
            "native_vi_linear": ("native_vi_linear", None),
            "native_vi_nonlinear": ("native_vi_nonlinear", None),
            "mcmc_raytrace": ("raytrace", None),
        }

        if method not in _method_map:
            if method == "evi_nifty":
                return self._run_geovi_cfm(key=key, sample_mode="evi", **kwargs)
            # Everything the flat seam can drive. The hierarchical posterior is
            # already a flat unconstrained vector with an iid N(0,1) prior (see
            # _hierarchical_flat), so a sampler being "hierarchical" is a
            # property of the problem, not of the sampler — there is nothing
            # left to special-case per backend.
            if method in FLAT_SAMPLERS:
                return run_flat_sampler(self, method, key=key, **kwargs)
            # Refuse the ones the seam knowingly cannot drive with a specific
            # reason rather than a generic "unknown method". A backend that is
            # absent because nobody wired it up and one that is absent because
            # its naive implementation returns biased samples deserve different
            # errors — the second is a warning to whoever tries to add it.
            if method in FLAT_UNSUPPORTED:
                raise NotImplementedError(
                    f"method={method!r} is not available for hierarchical fits. "
                    f"{FLAT_UNSUPPORTED[method]}"
                )
            # Derive the advertised list; never hand-write it. The literal this
            # replaced named 'vi_nonlinear_fast' as "(default)" for months after
            # b7c4fa1e2 moved the default off it (#1394).
            supported = sorted(set(_method_map) | set(FLAT_SAMPLERS) | {"evi_nifty"})
            raise ValueError(
                f"Unknown method: {method!r}. Supported "
                f"({len(supported)}): {', '.join(supported)}."
            )

        cfm_method, sample_mode = _method_map[method]
        if cfm_method == "geovi":
            extra = {"sample_mode": sample_mode} if sample_mode is not None else {}
            return self._run_geovi(key=key, **extra, **kwargs)
        elif cfm_method == "raytrace":
            return self._run_raytrace(key=key, **kwargs)
        elif cfm_method == "native_vi_linear":
            return self._run_native_vi_linear(key=key, **kwargs)
        elif cfm_method == "native_vi_nonlinear":
            return self._run_native_vi_nonlinear(key=key, **kwargs)
        else:
            raise ValueError(f"Unmapped method: {method!r}")

    def _run_native_vi_nonlinear(
        self,
        *,
        key,
        n_iterations=20,
        n_samples=3,
        n_posterior_samples=500,
        posterior_chunk_size=None,
        forward_chunk_size=1,
        memory_mode="low",
        kl_rtol=1e-2,
        n_seeds=3,
        verbose=True,
    ):
        """Fully JIT-compiled hierarchical native VI (nonlinear / geoVI).

        Pure-JAX equivalent of ``vi_nonlinear``: shared PSD parameters + per-galaxy
        latent vectors and physical params in a single flat array optimized via
        geoVI with nonlinear residual curving. Uses the same
        ``build_native_vi_nonlinear_engine`` factory as ``Fitter``.

        Parameters
        ----------
        n_iterations : int
            Maximum KL iterations. Auto-stops when converged.
        n_samples : int
            Samples per iteration (doubled by mirror_samples).
        n_posterior_samples : int
            Posterior samples drawn after convergence.
        forward_chunk_size : int
            Number of galaxies to evaluate in parallel per ``lax.map`` step (K).
            ``K=1`` (default) gives pure sequential lax.map, O(1) peak memory.
            ``K>1`` vmaps K galaxies per iteration for K-way parallelism.
            Must divide ``n_gal`` evenly after padding; all galaxies must have the
            same number of data points when ``K>1``.
        kl_rtol : float
            Relative KL tolerance for early stopping.
        n_seeds : int
            Number of random seeds. Best result (lowest H) is kept.
        verbose : bool
            Print progress.

        Notes
        -----
        Each key passed to ``draw_nonlinear_residuals_jit`` produces one
        mirrored pair (2 residuals), so ``n_posterior_samples // 2`` keys yield
        ``n_posterior_samples`` total draws.
        **JIT-compatible**: yes.
        """
        if n_samples > 12:
            warnings.warn(
                f"n_samples={n_samples} is unusually high for native_vi_nonlinear. "
                f"vmap over draw_linear_residual (which contains a while_loop) across "
                f"{n_samples} samples may cause XLA compilation issues. "
                f"Recommended: n_samples <= 6.",
                stacklevel=2,
            )

        from jax.flatten_util import ravel_pytree

        from tengri.inference.backends.vi.native import build_native_vi_nonlinear_engine

        n_gal = self.n_galaxies
        K = max(1, int(forward_chunk_size))
        # batch_size=K in lax.map handles non-divisible N internally — no padding needed.
        if K > 1:
            n_data_per_gal = len(self.galaxies[0]["flux_obs"])
            for g in self.galaxies[1:]:
                if len(g["flux_obs"]) != n_data_per_gal:
                    raise ValueError(
                        "forward_chunk_size > 1 requires all galaxies to have the same "
                        f"number of data points; got {n_data_per_gal} and {len(g['flux_obs'])}."
                    )
        else:
            n_data_per_gal = None
        spec = self._spec
        stochastic = spec.stochastic
        n_grid = spec.n_grid
        free_names = self._free_names
        sigma_lo, sigma_hi = self.psd_sigma_bounds
        tau_lo, tau_hi = self.psd_tau_bounds

        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds
        fixed_values = spec.get_fixed_values()

        # Precompute data
        all_data = jnp.concatenate([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise = jnp.concatenate([jnp.asarray(g["noise"]) for g in self.galaxies])

        # Build model once
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        data_type = self.data_type

        def _predict(params):
            """Predict data from parameters for single or batch mode."""
            if data_type == "photometry":
                return model.predict_photometry(params)
            return model.predict_spectrum(params)

        # --- Hierarchical signal_response (lax.map, O(1) memory in N_gal) ---
        def signal_response(p):
            """Compute predicted data from hierarchical parameters.

            Parameters
            ----------
            p : dict
                Hierarchical parameter dict with shared PSD + per-galaxy params.

            Returns
            -------
            ndarray, shape (n_data,)
                Predicted flux/spectrum for all galaxies concatenated.

            Notes
            -----
            **JIT-compatible**: yes.
            """
            psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

            def forward_one(ub_scalars, xi):
                """Evaluate forward model for one galaxy (Tier 4)."""
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val
                params["sfh_field_psd_sigma"] = psd_sigma
                params["sfh_field_psd_tau_myr"] = psd_tau
                if stochastic:
                    params["sfh_field_xi"] = xi
                params = spec.resolve_mirrors(params)
                return _predict(params)

            fwd = jax.checkpoint(forward_one) if memory_mode == "low" else forward_one
            # Unified path: lax.map(..., batch_size=K) handles K=1 (sequential) and
            # K>1 (vmap-K-then-scan) with no padding required for non-divisible N.
            if stochastic:
                predictions = jax.lax.map(
                    lambda args: fwd(args[0], args[1]),
                    (p["gal"], p["gal_xi"]),
                    batch_size=K,
                )
            else:
                predictions = jax.lax.map(
                    lambda ub: fwd(ub, None),
                    p["gal"],
                    batch_size=K,
                )
            return predictions.reshape(-1)

        # --- Build init structure ---
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        tau_mid = 0.5 * (tau_lo + tau_hi)

        init_template = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": {name: jnp.zeros(n_gal) for name in free_names},
        }
        if stochastic:
            init_template["gal_xi"] = jnp.zeros((n_gal, n_grid))

        _init_flat, unravel_fn = ravel_pytree(init_template)
        d_total = len(_init_flat)

        def flatten(d):
            """Flatten parameter tree to 1D vector (Tier 4)."""
            return ravel_pytree(d)[0]

        def unflatten(x):
            """Unflatten 1D vector to parameter tree (Tier 4)."""
            return unravel_fn(x)

        # --- Build shared native_vi_nonlinear backend ---
        run_native_vi_nonlinear_jit, draw_nonlinear_residuals_jit, hamiltonian = (
            build_native_vi_nonlinear_engine(
                signal_response, all_data, all_noise, flatten, unflatten
            )
        )

        if verbose:
            print(
                f"Hierarchical native_vi_nonlinear: {n_gal} galaxies, "
                f"D={d_total}, {n_iterations} max iterations, "
                f"{n_samples} samples/iter, {n_seeds} seeds"
            )
            print("  Compiling JIT engine...")

        t0 = time.time()

        # --- Initialize per-galaxy params via vectorized MAP ---
        from tengri import Fitter
        from tengri.inference.backends.map_dispatch import build_vectorized_map_solver

        init_keys = jax.random.split(key, n_gal + n_seeds + 2)

        if verbose:
            print("  Initializing per-galaxy params via vectorized MAP...")

        _template_gal = self.galaxies[0]
        _template_fitter = Fitter(
            model,
            _template_gal["flux_obs"],
            _template_gal["noise"],
            data_type=self.data_type,
        )
        map_solve_one = build_vectorized_map_solver(
            _template_fitter,
            n_steps=80,
            learning_rate=0.05,
        )

        all_flux_init = jnp.stack([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise_init = jnp.stack([jnp.asarray(g["noise"]) for g in self.galaxies])
        gal_keys = init_keys[:n_gal]

        all_init_unbounded = jax.lax.map(
            lambda args: map_solve_one(args[0], args[1], args[2]),
            (all_flux_init, all_noise_init, gal_keys),
            batch_size=K,
        )
        jax.block_until_ready(
            all_init_unbounded["psd_xi"] if stochastic else next(iter(all_init_unbounded.values()))
        )

        _gal_stacked = {
            name: all_init_unbounded.get(name, jnp.zeros(n_gal)) for name in free_names
        }
        map_init = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": _gal_stacked,
        }
        if stochastic:
            map_init["gal_xi"] = all_init_unbounded.get(
                "psd_xi",
                jnp.zeros((n_gal, n_grid)),
            )

        if verbose:
            print("  MAP init done. Running multi-seed optimization...")

        # --- Multi-seed optimization ---
        seed_keys = init_keys[n_gal:]
        best_flat = None
        best_loss = jnp.inf
        best_iters = 0

        for s in range(n_seeds):
            if s == 0:
                init_flat = flatten(map_init)
            else:
                perturb = 0.3 * jax.random.normal(seed_keys[s], shape=(d_total,))
                init_flat = flatten(map_init) + perturb

            opt_key = jax.random.fold_in(seed_keys[s], 999)
            converged_flat, n_iters = run_native_vi_nonlinear_jit(
                init_flat,
                opt_key,
                n_iter=n_iterations,
                n_samp=n_samples,
                rtol=kl_rtol,
            )
            n_iters = int(n_iters)

            loss = float(hamiltonian(converged_flat))

            if verbose and n_seeds > 1:
                print(f"    Seed {s + 1}/{n_seeds}: H={loss:.1f}, {n_iters} iters")

            if loss < best_loss:
                best_flat = converged_flat
                best_loss = loss
                best_iters = n_iters

        # --- Draw posterior samples ---
        # Each key produces one mirrored pair (2 residuals), so n // 2 keys
        # yields n total posterior samples.
        if verbose:
            print(f"  Drawing {n_posterior_samples} geoVI posterior samples...")

        draw_key = jax.random.fold_in(key, 12345)
        n_draw_keys = max(1, n_posterior_samples // 2)
        draw_keys = jax.random.split(draw_key, n_draw_keys)

        chunk = posterior_chunk_size if posterior_chunk_size else n_draw_keys
        chunk = min(int(chunk), int(n_draw_keys))
        if chunk >= n_draw_keys:
            residuals_flat = draw_nonlinear_residuals_jit(best_flat, draw_keys)
        else:
            residual_chunks = []
            for start in range(0, n_draw_keys, chunk):
                end = min(start + chunk, n_draw_keys)
                keys_chunk = draw_keys[start:end]
                pad = chunk - (end - start)
                if pad:
                    keys_chunk = jnp.concatenate([keys_chunk, draw_keys[:pad]])
                r = draw_nonlinear_residuals_jit(best_flat, keys_chunk)
                jax.block_until_ready(r)
                if pad:
                    r = r[: 2 * (end - start)]
                residual_chunks.append(r)
            residuals_flat = jnp.concatenate(residual_chunks, axis=0)

        residuals_flat = residuals_flat[:n_posterior_samples]
        wall_time = time.time() - t0

        # --- Extract posteriors (vectorized) ---
        converged_p = unflatten(best_flat)
        all_res_p = jax.vmap(unflatten)(residuals_flat)

        shared_samples = {
            "psd_sigma": to_bounded(
                converged_p["psd_sigma_u"] + all_res_p["psd_sigma_u"], sigma_lo, sigma_hi
            ),
            "psd_tau_myr": to_bounded(
                converged_p["psd_tau_u"] + all_res_p["psd_tau_u"], tau_lo, tau_hi
            ),
        }
        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        individual_samples = []
        for g in range(n_gal):
            gal_samples = {}
            for name in free_names:
                lo, hi = bounds[name]
                combined = converged_p["gal"][name][g] + all_res_p["gal"][name][:, g]
                gal_samples[name] = to_bounded(combined, lo, hi)
            if stochastic:
                gal_samples["psd_xi"] = converged_p["gal_xi"][g] + all_res_p["gal_xi"][:, g]
            individual_samples.append(gal_samples)

        if verbose:
            s = shared_params
            print(
                f"  Hierarchical native_vi_nonlinear complete in {wall_time:.1f}s, "
                f"{best_iters}/{n_iterations} iterations, "
                f"{n_posterior_samples} posterior samples"
            )
            print(f"  σ_PSD = {s['psd_sigma']:.2f}, τ_PSD = {s['psd_tau_myr']:.1f} Myr")

        return PopulationPosterior(
            _model=self._template,
            shared_samples=shared_samples,
            shared_params=shared_params,
            individual_samples=individual_samples,
            method="Hierarchical native_vi_nonlinear",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": best_iters,
                "n_iterations_max": n_iterations,
                "n_samples_posterior": n_posterior_samples,
                "n_seeds": n_seeds,
                "best_hamiltonian": float(best_loss),
                "D_total": d_total,
            },
        )

    def _run_native_vi_linear(
        self,
        *,
        key,
        n_iterations=20,
        n_samples=3,
        n_posterior_samples=500,
        posterior_chunk_size=None,
        forward_chunk_size=1,
        memory_mode="low",
        kl_rtol=1e-2,
        n_seeds=3,
        verbose=True,
    ):
        """Fully JIT-compiled hierarchical native VI (linear / MGVI).

        Pure-JAX equivalent of ``vi_linear``: shared PSD parameters + per-galaxy
        latent vectors and physical params in a single flat array optimized via
        Newton-CG. The entire loop runs inside ``jax.lax.while_loop`` — no NIFTy
        overhead, O(1) XLA graph size regardless of N_gal.

        Mirrors ``Fitter._run_vi_native_linear`` for the single-galaxy case.

        Parameters
        ----------
        n_iterations : int
            Maximum KL iterations. Auto-stops when converged.
        n_samples : int
            Samples per iteration (doubled by mirror_samples).
        n_posterior_samples : int
            Posterior samples drawn after convergence.
        forward_chunk_size : int
            Number of galaxies to evaluate in parallel per ``lax.map`` step (K).
            ``K=1`` (default) gives pure sequential lax.map, O(1) peak memory.
            ``K>1`` vmaps K galaxies per iteration for K-way parallelism.
            Must divide ``n_gal`` evenly after padding; all galaxies must have the
            same number of data points when ``K>1``.
        kl_rtol : float
            Relative KL tolerance for early stopping.
        n_seeds : int
            Number of random seeds. Best result (lowest H) is kept.
        verbose : bool
            Print progress.
        """

        from jax.flatten_util import ravel_pytree

        n_gal = self.n_galaxies
        K = max(1, int(forward_chunk_size))
        # batch_size=K in lax.map handles non-divisible N internally — no padding needed.
        if K > 1:
            n_data_per_gal = len(self.galaxies[0]["flux_obs"])
            for g in self.galaxies[1:]:
                if len(g["flux_obs"]) != n_data_per_gal:
                    raise ValueError(
                        "forward_chunk_size > 1 requires all galaxies to have the same "
                        f"number of data points; got {n_data_per_gal} and {len(g['flux_obs'])}."
                    )
        else:
            n_data_per_gal = None
        spec = self._spec
        stochastic = spec.stochastic
        n_grid = spec.n_grid
        free_names = self._free_names
        sigma_lo, sigma_hi = self.psd_sigma_bounds
        tau_lo, tau_hi = self.psd_tau_bounds

        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds
        fixed_values = spec.get_fixed_values()

        # Precompute data
        all_data = jnp.concatenate([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise = jnp.concatenate([jnp.asarray(g["noise"]) for g in self.galaxies])

        # Build model once
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        data_type = self.data_type

        def _predict(params):
            """Predict data from parameters for single or batch mode."""
            if data_type == "photometry":
                return model.predict_photometry(params)
            return model.predict_spectrum(params)

        # --- Hierarchical signal_response (lax.map, O(1) memory in N_gal) ---
        def signal_response(p):
            """Compute predicted data from hierarchical parameters.

            Parameters
            ----------
            p : dict
                Hierarchical parameter dict. Keys 'psd_sigma' and 'psd_tau' map to
                unbounded optimizer space internally; 'gal' contains per-galaxy unbounded
                params; 'gal_xi' present if stochastic field is enabled.

            Returns
            -------
            ndarray, shape (n_data,)
                Predicted flux/spectrum for all galaxies concatenated.

            Notes
            -----
            Internal method for hierarchical VI update. Not part of the public API.
            **JIT-compatible**: yes.
            """
            psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

            def forward_one(ub_scalars, xi):
                """Evaluate forward model for one galaxy with given unbounded params (Tier 4)."""
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val
                params["sfh_field_psd_sigma"] = psd_sigma
                params["sfh_field_psd_tau_myr"] = psd_tau
                if stochastic:
                    params["sfh_field_xi"] = xi
                params = spec.resolve_mirrors(params)
                return _predict(params)

            # memory_mode="low" wraps per-galaxy forward in jax.checkpoint so
            # the reverse-mode tape inside jvp/vjp does not materialize
            # activations for all n_gal galaxies simultaneously. Trades a
            # recomputation for a 2–3x peak-memory reduction during CG.
            #
            # lax.map (not vmap) keeps the XLA compiled artifact O(1) in N_gal.
            # vmap replicates the graph N times → XLA protobuf exceeds 2 GB for
            # complex SEDs with N ≥ 3.  lax.map compiles one galaxy and loops.
            fwd = jax.checkpoint(forward_one) if memory_mode == "low" else forward_one
            # Unified path: lax.map(..., batch_size=K) handles K=1 (sequential) and
            # K>1 (vmap-K-then-scan) with no padding required for non-divisible N.
            if stochastic:
                predictions = jax.lax.map(
                    lambda args: fwd(args[0], args[1]),
                    (p["gal"], p["gal_xi"]),
                    batch_size=K,
                )
            else:
                predictions = jax.lax.map(
                    lambda ub: fwd(ub, None),
                    p["gal"],
                    batch_size=K,
                )
            return predictions.reshape(-1)

        # --- Build init structure ---
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        tau_mid = 0.5 * (tau_lo + tau_hi)

        init_template = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": {name: jnp.zeros(n_gal) for name in free_names},
        }
        if stochastic:
            init_template["gal_xi"] = jnp.zeros((n_gal, n_grid))

        # Use ravel_pytree for flatten/unflatten
        _init_flat, unravel_fn = ravel_pytree(init_template)
        d_total = len(_init_flat)

        def flatten(d):
            """Flatten parameter tree to 1D vector (Tier 4)."""
            return ravel_pytree(d)[0]

        def unflatten(x):
            """Unflatten 1D vector to parameter tree (Tier 4)."""
            return unravel_fn(x)

        # --- Build shared native_vi_linear backend ---
        from tengri.inference.backends.vi.native import build_native_vi_linear_engine

        run_native_vi_linear_jit, draw_residuals_jit, hamiltonian = build_native_vi_linear_engine(
            signal_response, all_data, all_noise, flatten, unflatten
        )

        if verbose:
            print(
                f"Hierarchical vi_native_linear: {n_gal} galaxies, "
                f"D={d_total}, {n_iterations} max iterations, "
                f"{n_samples} samples/iter, {n_seeds} seeds"
            )
            print("  Compiling JIT engine...")

        t0 = time.time()

        # --- Initialize per-galaxy params via vectorized MAP ---
        # One JIT'd map_solve_one(flux, noise, key) compiled once, run for all
        # N galaxies via lax.map(batch_size=K). Compile time O(K), no Python
        # per-galaxy dispatch — replaces the prior O(N) Python loop.
        from tengri import Fitter
        from tengri.inference.backends.map_dispatch import build_vectorized_map_solver

        init_keys = jax.random.split(key, n_gal + n_seeds + 2)

        if verbose:
            print("  Initializing per-galaxy params via vectorized MAP...")

        # Template fitter: any galaxy's flux/noise will do — its model,
        # observation, and _data_args layout are reused inside map_solve_one.
        _template_gal = self.galaxies[0]
        _template_fitter = Fitter(
            model,
            _template_gal["flux_obs"],
            _template_gal["noise"],
            data_type=self.data_type,
        )
        map_solve_one = build_vectorized_map_solver(
            _template_fitter,
            n_steps=80,
            learning_rate=0.05,
        )

        # Stack per-galaxy data; lax.map streams them through map_solve_one.
        all_flux_init = jnp.stack([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise_init = jnp.stack([jnp.asarray(g["noise"]) for g in self.galaxies])
        gal_keys = init_keys[:n_gal]

        all_init_unbounded = jax.lax.map(
            lambda args: map_solve_one(args[0], args[1], args[2]),
            (all_flux_init, all_noise_init, gal_keys),
            batch_size=K,
        )
        jax.block_until_ready(
            all_init_unbounded["psd_xi"] if stochastic else next(iter(all_init_unbounded.values()))
        )

        _gal_stacked = {
            name: all_init_unbounded.get(name, jnp.zeros(n_gal)) for name in free_names
        }
        map_init = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": _gal_stacked,
        }
        if stochastic:
            map_init["gal_xi"] = all_init_unbounded.get(
                "psd_xi",
                jnp.zeros((n_gal, n_grid)),
            )

        if verbose:
            print("  MAP init done. Running multi-seed optimization...")

        # --- Multi-seed optimization ---
        seed_keys = init_keys[n_gal:]
        best_flat = None
        best_loss = jnp.inf
        best_iters = 0

        for s in range(n_seeds):
            if s == 0:
                init_flat = flatten(map_init)
            else:
                # Random perturbation of MAP init
                perturb = 0.3 * jax.random.normal(seed_keys[s], shape=(d_total,))
                init_flat = flatten(map_init) + perturb

            opt_key = jax.random.fold_in(seed_keys[s], 999)
            converged_flat, n_iters = run_native_vi_linear_jit(
                init_flat,
                opt_key,
                n_iter=n_iterations,
                n_samp=n_samples,
                rtol=kl_rtol,
            )
            n_iters = int(n_iters)

            # Evaluate Hamiltonian
            loss = float(hamiltonian(converged_flat))

            if verbose and n_seeds > 1:
                print(f"    Seed {s + 1}/{n_seeds}: H={loss:.1f}, {n_iters} iters")

            if loss < best_loss:
                best_flat = converged_flat
                best_loss = loss
                best_iters = n_iters

        # --- Draw posterior samples ---
        # Chunked draws: peak memory of draw_residuals is
        # O(chunk · d_total) instead of O(n_posterior_samples · d_total).
        # The JIT cache is shared across chunks of equal size, so the extra
        # dispatches are ~free. posterior_chunk_size=None preserves the
        # original fully-vmapped behavior.
        if verbose:
            print(f"  Drawing {n_posterior_samples} posterior samples...")

        draw_key = jax.random.fold_in(key, 12345)
        draw_keys = jax.random.split(draw_key, n_posterior_samples)

        chunk = posterior_chunk_size if posterior_chunk_size else n_posterior_samples
        chunk = min(int(chunk), int(n_posterior_samples))
        if chunk >= n_posterior_samples:
            residuals_flat = draw_residuals_jit(best_flat, draw_keys)
        else:
            residual_chunks = []
            for start in range(0, n_posterior_samples, chunk):
                end = min(start + chunk, n_posterior_samples)
                # Pad the final chunk to `chunk` so the JIT cache hits.
                keys_chunk = draw_keys[start:end]
                pad = chunk - (end - start)
                if pad:
                    keys_chunk = jnp.concatenate([keys_chunk, draw_keys[:pad]])
                r = draw_residuals_jit(best_flat, keys_chunk)
                jax.block_until_ready(r)
                if pad:
                    r = r[: end - start]
                residual_chunks.append(r)
            residuals_flat = jnp.concatenate(residual_chunks, axis=0)

        wall_time = time.time() - t0

        # --- Extract posteriors (vectorized) ---
        converged_p = unflatten(best_flat)
        all_res_p = jax.vmap(unflatten)(residuals_flat)

        shared_samples = {
            "psd_sigma": to_bounded(
                converged_p["psd_sigma_u"] + all_res_p["psd_sigma_u"], sigma_lo, sigma_hi
            ),
            "psd_tau_myr": to_bounded(
                converged_p["psd_tau_u"] + all_res_p["psd_tau_u"], tau_lo, tau_hi
            ),
        }
        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        individual_samples = []
        for g in range(n_gal):
            gal_samples = {}
            for name in free_names:
                lo, hi = bounds[name]
                combined = converged_p["gal"][name][g] + all_res_p["gal"][name][:, g]
                gal_samples[name] = to_bounded(combined, lo, hi)
            if stochastic:
                gal_samples["psd_xi"] = converged_p["gal_xi"][g] + all_res_p["gal_xi"][:, g]
            individual_samples.append(gal_samples)

        if verbose:
            s = shared_params
            print(
                f"  Hierarchical vi_native_linear complete in {wall_time:.1f}s, "
                f"{best_iters}/{n_iterations} iterations, "
                f"{n_posterior_samples} posterior samples"
            )
            print(f"  σ_PSD = {s['psd_sigma']:.2f}, τ_PSD = {s['psd_tau_myr']:.1f} Myr")

        return PopulationPosterior(
            _model=self._template,
            shared_samples=shared_samples,
            shared_params=shared_params,
            individual_samples=individual_samples,
            method="Hierarchical vi_native_linear",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": best_iters,
                "n_iterations_max": n_iterations,
                "n_samples_posterior": n_posterior_samples,
                "n_seeds": n_seeds,
                "best_hamiltonian": float(best_loss),
                "D_total": d_total,
            },
        )

    def _run_geovi_cfm(
        self,
        *,
        key,
        n_iterations=10,
        n_samples=3,
        n_posterior_samples=60,
        sample_mode="nonlinear_resample",
        vi_config=None,
        memory_mode="low",
        posterior_chunk_size=None,
        verbose=True,
    ):
        """Hierarchical geoVI using NIFTy's CorrelatedFieldMaker.

        This is the proper NIFTy approach: the PSD hyperparameters
        (fluctuation amplitude ≈ σ_PSD, spectral slope ≈ τ_PSD) are
        learned jointly inside the generative model, not as external
        flat parameters.

        Each galaxy gets its own ξ_i (white noise) but shares the
        PSD shape defined by the CorrelatedFieldMaker hyperparameters.
        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError("nifty8.re required: pip install nifty8[re]") from None

        from tengri.inference.vi_config import VIConfig, evi_sample_mode

        cfg = vi_config or VIConfig()

        n_gal = self.n_galaxies
        spec = self._spec
        n_grid = spec.n_grid
        free_names = self._free_names

        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds
        fixed_values = spec.get_fixed_values()

        # Pre-build model
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        data_type = self.data_type

        def _predict_cfm(params):
            """Predict data from parameters (CorrelatedFieldMaker variant)."""
            if data_type == "photometry":
                return model.predict_photometry(params)
            return model.predict_spectrum(params)

        if verbose:
            print(f"Hierarchical geoVI (CorrelatedFieldMaker): {n_gal} galaxies, n_grid={n_grid}")
            if model.has_fixedz_photometry_precompute:
                print("  Photometry precomputation: ACTIVE")

        t0 = time.time()

        # ── Build shared correlated field maker ───────────────
        # The CFM creates the generative model for the GP field.
        # PSD hyperparameters (fluctuations, slope) are SHARED across
        # all galaxies — this is the hierarchical coupling.
        cfm = jft.CorrelatedFieldMaker("psd_")
        cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-3, 1e-4))

        # Log-age grid spacing
        log_age_range = 10.14 - 6.0  # log10(yr)
        distance = log_age_range / n_grid

        # Fluctuations ~ σ_PSD: lognormal prior centered on 1.0
        # loglogavgslope ~ spectral index: DRW has slope -2 in log-log
        cfm.add_fluctuations(
            shape=(n_grid,),
            distances=(distance,),
            fluctuations=(1.0, 0.8),  # σ_PSD prior: lognormal(1.0, 0.8)
            loglogavgslope=(-2.0, 1.0),  # slope prior: N(-2, 1) — DRW = -2
            flexibility=(0.3, 0.2),  # small non-parametric correction
            asperity=None,
            prefix="shared_",
        )
        corr_field_template = cfm.finalize()

        # ── Build NIFTy domain ────────────────────────────────
        # Batched-leaf layout (see _run_geovi for rationale): one (n_gal,...)
        # array per per-galaxy param instead of N separate scalar leaves.
        # Collapses the NIFTy pytree from O(n_gal·n_free) leaves to O(n_free).
        domain = {}

        # Shared PSD hyperparameters (from CFM)
        for k, v in corr_field_template.domain.items():
            if k != "psd_xi":  # xi is per-galaxy
                domain[k] = v

        # Per-galaxy: batched xi + batched physical params
        domain["gal_xi"] = jft.ShapeWithDtype((n_gal, n_grid))
        for name in free_names:
            domain[f"gal_{name}"] = jft.ShapeWithDtype((n_gal,))

        # Precompute data
        all_data = []
        all_noise_inv = []
        for gal in self.galaxies:
            d = jnp.asarray(gal["flux_obs"])
            n = jnp.asarray(gal["noise"])
            all_data.append(d)
            all_noise_inv.append(1.0 / n**2)

        data_concat = jnp.concatenate(all_data)
        noise_inv_concat = jnp.concatenate(all_noise_inv)

        # ── Build signal response ─────────────────────────────
        def signal_response(primals):
            """Compute predicted data from NIFTy primals tree."""
            # Reconstruct the shared CFM primals (PSD hyperparams)
            cfm_primals = {}
            for k in corr_field_template.domain:
                if k != "psd_xi":
                    cfm_primals[k] = primals[k]

            # Batched per-galaxy primals are already (n_gal, ...) — no stacking.
            gal_xi = primals["gal_xi"]
            gal_ub = {name: primals[f"gal_{name}"] for name in free_names}

            # Single-galaxy forward (vmapped over galaxy axis)
            def forward_one(ub_scalars, xi):
                """Evaluate forward model for one galaxy."""
                # Generate the correlated field for this galaxy
                cfm_primals_i = dict(cfm_primals)
                cfm_primals_i["psd_xi"] = xi
                gp_field = corr_field_template(cfm_primals_i)

                # Build per-galaxy physical params
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val

                # CFM already applies sqrt(P) * xi, so pass the full
                # correlated field as the GP realization
                params["sfh_field_xi"] = gp_field
                params["sfh_field_psd_sigma"] = 1.0
                params["sfh_field_psd_tau_myr"] = 50.0
                params = spec.resolve_mirrors(params)
                return _predict_cfm(params)

            # lax.map keeps compiled graph O(1) in N_gal — see _run_vi_native_linear.
            fwd = jax.checkpoint(forward_one) if memory_mode == "low" else forward_one
            predictions = jax.lax.map(lambda args: fwd(args[0], args[1]), (gal_ub, gal_xi))
            return predictions.reshape(-1)

        signal_response_jit = jax.jit(signal_response)
        nifty_model = jft.Model(signal_response_jit, domain=domain)
        likelihood = jft.Gaussian(data_concat, noise_inv_concat).amend(nifty_model)

        # ── Initialize ────────────────────────────────────────
        init = {}
        # Shared PSD: start at prior means
        init_pos_template = jft.random_like(key, corr_field_template.domain)
        for k, v in init_pos_template.items():
            if k != "psd_xi":
                init[k] = jnp.zeros_like(v)  # start at prior mean

        # Per-galaxy: vectorized MAP initialization via lax.map(batch_size=1).
        # One JIT compile, then stream through all N galaxies — replaces the
        # prior O(N) Python dispatch loop. Stack into batched arrays to match
        # the batched-leaf domain layout.
        from tengri import Fitter
        from tengri.inference.backends.map_dispatch import build_vectorized_map_solver

        keys = jax.random.split(key, n_gal + 1)

        if verbose:
            print("  Initializing per-galaxy params via vectorized MAP...")

        _template_gal = self.galaxies[0]
        _template_fitter = Fitter(
            model,
            _template_gal["flux_obs"],
            _template_gal["noise"],
            data_type=self.data_type,
        )
        map_solve_one = build_vectorized_map_solver(
            _template_fitter,
            n_steps=80,
            learning_rate=0.05,
        )

        all_flux_init = jnp.stack([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise_init = jnp.stack([jnp.asarray(g["noise"]) for g in self.galaxies])
        gal_keys = keys[:n_gal]

        all_init_unbounded = jax.lax.map(
            lambda args: map_solve_one(args[0], args[1], args[2]),
            (all_flux_init, all_noise_init, gal_keys),
            batch_size=1,
        )
        jax.block_until_ready(
            all_init_unbounded.get("psd_xi", next(iter(all_init_unbounded.values())))
        )

        for name in free_names:
            init[f"gal_{name}"] = all_init_unbounded.get(name, jnp.zeros(n_gal))
        init["gal_xi"] = all_init_unbounded.get(
            "psd_xi",
            jnp.zeros((n_gal, n_grid)),
        )

        init_pos = jft.Vector(init)

        if verbose:
            n_total = sum(np.prod(v.shape) if hasattr(v, "shape") else 1 for v in init.values())
            print(f"  Total parameters: {n_total}")

        # ── Run optimize_kl ───────────────────────────────────
        import io
        import logging
        import sys
        import warnings

        warnings.filterwarnings("ignore")
        logging.getLogger("nifty8").setLevel(logging.ERROR)

        if verbose:
            print(
                f"  Running optimize_kl ({n_iterations} iterations, {n_samples} samples/iter)..."
            )

        # Resolve sample_mode
        if sample_mode == "evi":
            resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
        else:
            resolved_mode = sample_mode

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        key_opt = keys[-1]
        samples, _state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            key=key_opt,
            sample_mode=resolved_mode,
            residual_map=jax.vmap if cfg.use_vmap else "lmap",
            draw_linear_kwargs=cfg.draw_linear_kwargs,
            nonlinearly_update_kwargs=cfg.nonlinearly_update_kwargs,
            kl_kwargs=cfg.kl_kwargs,
            odir=None,
        )

        sys.stdout = old_stdout

        # ── Draw posterior samples ────────────────────────────
        if verbose:
            print(f"  Drawing {n_posterior_samples} posterior samples...")

        converged_pos = samples.pos
        key_draw = jax.random.fold_in(key, 999)

        all_sample_dicts = []
        for s in list(samples):
            sd = s.tree if hasattr(s, "tree") else dict(s)
            all_sample_dicts.append(sd)

        for _j in range(n_posterior_samples):
            key_draw, sub_key = jax.random.split(key_draw)
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 50},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                all_sample_dicts.append(combined)
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
                # TypeError: NIFTy API mismatch or dict() conversion failed
                # ValueError: invalid cg_kwargs configuration
                # AttributeError: missing .tree attribute
                # KeyError: position/sample tree key mismatch
                # RuntimeError: linear solver failed to converge
                break

        wall_time = time.time() - t0
        n_post = len(all_sample_dicts)

        # ── Extract shared PSD posteriors ─────────────────────
        # The CFM encodes fluctuations amplitude and spectral slope.
        # Extract them from the samples.
        shared_samples = {
            "psd_fluctuations": jnp.array(
                [float(d.get("psd_shared_fluctuations", jnp.nan)) for d in all_sample_dicts]
            ),
            "psd_loglogavgslope": jnp.array(
                [float(d.get("psd_shared_loglogavgslope", jnp.nan)) for d in all_sample_dicts]
            ),
        }

        # Also compute effective σ_PSD and τ_PSD from the CFM params
        # fluctuations ≈ exp(psd_shared_fluctuations) ≈ σ_PSD
        # loglogavgslope ≈ spectral slope (DRW = -2)
        shared_samples["psd_sigma_eff"] = jnp.exp(shared_samples["psd_fluctuations"])

        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            print(f"  Hierarchical geoVI (CFM) complete in {wall_time:.1f}s, {n_post} samples")
            print(
                f"  fluctuations = {shared_params.get('psd_fluctuations', '?'):.2f}, "
                f"slope = {shared_params.get('psd_loglogavgslope', '?'):.2f}"
            )

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            method="Hierarchical geoVI (CorrelatedFieldMaker)",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": n_iterations,
                "n_samples": n_post,
                "cfm_domain_keys": list(corr_field_template.domain.keys()),
            },
        )

    def _run_geovi(
        self,
        *,
        key,
        n_iterations=10,
        n_samples=3,
        n_posterior_samples=100,
        sample_mode="nonlinear_resample",
        vi_config=None,
        memory_mode="low",
        posterior_chunk_size=None,
        verbose=True,
    ):
        """Hierarchical geoVI via NIFTy.re.

        The joint model has:

        - 2 shared PSD params (unbounded)
        - N × (n_free + n_grid) per-galaxy params

        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError(
                "nifty8.re required for hierarchical geoVI: pip install nifty8[re]"
            ) from None

        from tengri.inference.vi_config import VIConfig, evi_sample_mode

        cfg = vi_config or VIConfig()

        n_gal = self.n_galaxies
        spec = self._spec
        stochastic = spec.stochastic
        n_grid = spec.n_grid
        free_names = self._free_names

        # Build bounds for per-galaxy free params
        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds

        fixed_values = spec.get_fixed_values()

        if verbose:
            n_per_gal = len(free_names) + (n_grid if stochastic else 0)
            n_total = 2 + n_gal * n_per_gal
            print(
                f"Hierarchical geoVI: {n_gal} galaxies, "
                f"{n_per_gal} params/galaxy + 2 shared = "
                f"{n_total} total parameters"
            )

        t0 = time.time()

        # ── Build NIFTy domain ────────────────────────────────
        # Batched-leaf layout: one (n_gal,)-shaped array per per-galaxy param
        # instead of N separate scalar leaves. Collapses the NIFTy pytree from
        # O(n_gal·n_free) leaves to O(n_free) — compile time, trace time, and
        # optimize_kl's per-sample pytree copies all drop by ~n_gal.
        domain = {}
        domain["psd_sigma_u"] = jft.ShapeWithDtype(())
        domain["psd_tau_u"] = jft.ShapeWithDtype(())
        for name in free_names:
            domain[f"gal_{name}"] = jft.ShapeWithDtype((n_gal,))
        if stochastic:
            domain["gal_psd_xi"] = jft.ShapeWithDtype((n_gal, n_grid))

        # ── Build signal response ─────────────────────────────
        galaxies = self.galaxies
        sigma_lo, sigma_hi = self.psd_sigma_bounds
        tau_lo, tau_hi = self.psd_tau_bounds
        data_type = self.data_type

        # Precompute data arrays
        all_data = []
        all_noise_inv = []
        for gal in galaxies:
            d = jnp.asarray(gal["flux_obs"])
            n = jnp.asarray(gal["noise"])
            all_data.append(d)
            all_noise_inv.append(1.0 / n**2)

        data_concat = jnp.concatenate(all_data)
        noise_inv_concat = jnp.concatenate(all_noise_inv)

        # Pre-build model once (PSD params will be overridden per-call)
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)

        # Verify precomputation is active
        if model.has_fixedz_photometry_precompute and verbose:
            print("  Photometry precomputation: ACTIVE (21.6x speedup)")
        elif verbose:
            print("  WARNING: Photometry precomputation NOT active")

        def _predict_single(params):
            """Single-galaxy forward model (for vmap)."""
            if data_type == "photometry":
                return model.predict_photometry(params)
            else:
                return model.predict_spectrum(params)

        def signal_response(primals):
            """Map hierarchical primals to stacked predictions for all galaxies."""
            # Shared PSD params (bounded)
            psd_sigma = to_bounded(primals["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(primals["psd_tau_u"], tau_lo, tau_hi)

            # Batched per-galaxy primals are already (n_gal, ...) — no stacking.
            gal_ub = {name: primals[f"gal_{name}"] for name in free_names}
            gal_xi = primals["gal_psd_xi"] if stochastic else jnp.zeros(n_gal)

            # Single-galaxy forward (vmapped over galaxy axis)
            def forward_one(ub_scalars, xi):
                """Run the forward model for one galaxy from unbounded to physical params."""
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val
                params["sfh_field_psd_sigma"] = psd_sigma
                params["sfh_field_psd_tau_myr"] = psd_tau
                if stochastic:
                    params["sfh_field_xi"] = xi
                params = spec.resolve_mirrors(params)
                return _predict_single(params)

            # lax.map keeps compiled graph O(1) in N_gal — see _run_vi_native_linear.
            fwd = jax.checkpoint(forward_one) if memory_mode == "low" else forward_one
            if stochastic:
                predictions = jax.lax.map(lambda args: fwd(args[0], args[1]), (gal_ub, gal_xi))
            else:
                predictions = jax.lax.map(lambda ub: fwd(ub, None), gal_ub)

            return predictions.reshape(-1)

        signal_response_jit = jax.jit(signal_response)
        nifty_model = jft.Model(signal_response_jit, domain=domain)

        # Gaussian likelihood
        likelihood = jft.Gaussian(data_concat, noise_inv_concat).amend(nifty_model)

        # ── Initialize ────────────────────────────────────────
        # Batched initialization: one (n_gal,) draw per param instead of a
        # Python loop over N galaxies. Same statistics as before (N(0, 0.1)
        # per galaxy), just vectorized.
        init = {}
        init["psd_sigma_u"] = jnp.array(0.0)
        init["psd_tau_u"] = jnp.array(0.0)

        keys = jax.random.split(key, len(free_names) + 2)
        for j, name in enumerate(free_names):
            init[f"gal_{name}"] = 0.1 * jax.random.normal(keys[j], shape=(n_gal,))
        if stochastic:
            init["gal_psd_xi"] = 0.1 * jax.random.normal(keys[-2], shape=(n_gal, n_grid))

        init_pos = jft.Vector(init)

        # ── Run optimize_kl ───────────────────────────────────
        key, opt_key = jax.random.split(keys[-1])

        import io
        import logging
        import sys
        import warnings

        warnings.filterwarnings("ignore")
        logging.getLogger("nifty8").setLevel(logging.ERROR)

        if verbose:
            print(f"  Running optimize_kl ({n_iterations} iterations)...")

        # Resolve sample_mode
        if sample_mode == "evi":
            resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
        else:
            resolved_mode = sample_mode

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        samples, _state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            key=opt_key,
            sample_mode=resolved_mode,
            residual_map=jax.vmap if cfg.use_vmap else "lmap",
            draw_linear_kwargs=cfg.draw_linear_kwargs,
            nonlinearly_update_kwargs=cfg.nonlinearly_update_kwargs,
            kl_kwargs=cfg.kl_kwargs,
            odir=None,
        )

        sys.stdout = old_stdout

        # ── Draw posterior samples ────────────────────────────
        if verbose:
            print(f"  Drawing {n_posterior_samples} posterior samples...")

        converged_pos = samples.pos
        key, draw_key = jax.random.split(key)

        all_sample_dicts = []

        # Include optimization samples
        for s in list(samples):
            sd = s.tree if hasattr(s, "tree") else dict(s)
            all_sample_dicts.append(sd)

        # Draw additional linear residual samples
        for _j in range(n_posterior_samples):
            draw_key, sub_key = jax.random.split(draw_key)
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 50},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                all_sample_dicts.append(combined)
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
                # TypeError: NIFTy API mismatch or dict() conversion failed
                # ValueError: invalid cg_kwargs configuration
                # AttributeError: missing .tree attribute
                # KeyError: position/sample tree key mismatch
                # RuntimeError: linear solver failed to converge
                break

        wall_time = time.time() - t0
        n_post = len(all_sample_dicts)

        # ── Extract shared PSD posteriors ─────────────────────
        shared_samples = {
            "psd_sigma": jnp.array(
                [float(to_bounded(d["psd_sigma_u"], sigma_lo, sigma_hi)) for d in all_sample_dicts]
            ),
            "psd_tau_myr": jnp.array(
                [float(to_bounded(d["psd_tau_u"], tau_lo, tau_hi)) for d in all_sample_dicts]
            ),
        }

        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            s = shared_params
            print(f"  Hierarchical geoVI complete in {wall_time:.1f}s, {n_post} samples")
            print(f"  σ_PSD = {s['psd_sigma']:.2f}, τ_PSD = {s['psd_tau_myr']:.1f} Myr")

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            method="Hierarchical geoVI",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": n_iterations,
                "n_samples": n_post,
            },
        )

    def _run_raytrace(
        self,
        *,
        key,
        n_burnin=200,
        n_steps=500,
        n_leapfrog_steps=10,
        step_size=None,
        memory_mode="low",
        posterior_chunk_size=None,
        allow_degenerate=False,
        verbose=True,
    ):
        """Hierarchical Ray Tracing (flat parameter vector).

        Flattens all shared + per-galaxy params into one vector and
        runs the Ray Tracing Sampler. Works for moderate N (~10-50 gal).
        """
        from tengri.inference._hierarchical_flat import build_flat_problem
        from tengri.inference.backends.mcmc.raytrace import sample_raytrace

        # ONE definition of the hierarchical posterior, shared with every other
        # sampler. This block used to build its own `init`, its own
        # `ravel_pytree`, and its own `log_prob` inline — ~135 lines that were
        # textually equivalent to `build_flat_problem` but structurally
        # independent, so nothing stopped the two from drifting apart and
        # quietly sampling different distributions.
        n_gal = self.n_galaxies
        prob = build_flat_problem(self, key=key, memory_mode=memory_mode, verbose=verbose)
        init_flat = prob.init_flat
        D = prob.n_dim
        log_prob = prob.log_prob
        keys = jax.random.split(key, n_gal + 2)

        if step_size is None:
            step_size = 0.005 if D > 100 else 0.01

        if verbose:
            print(
                f"Hierarchical Ray Tracing: {n_gal} galaxies, "
                f"{D} total parameters, "
                f"{n_burnin} burn-in + {n_steps} samples"
            )

        t0 = time.time()
        total_steps = n_burnin + n_steps

        key_rt = keys[-1]
        chain, _log_likelihood, accept_prob = sample_raytrace(
            key=key_rt,
            params_init=init_flat,
            log_prob_fn=log_prob,
            n_steps=total_steps,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=float(step_size),
        )

        wall_time = time.time() - t0
        chain = chain[n_burnin:]
        accept_prob_post = accept_prob[n_burnin:]

        # A chain that accepted nothing has not sampled (#1530). Its draws are
        # the initialization repeated n_steps times — and because that
        # initialization is a MAP solve, the numbers look *reasonable*. Measured
        # at D=516: acceptance 3.4e-10, all 500 draws collapsing to one point,
        # reporting sigma_PSD=2.05 beside MAP's 2.13. That reads as two
        # estimators agreeing; it is one number echoed back. Nothing raised, and
        # `tier="primary"` means `check_usable` does not gate this backend, so
        # using Ray Tracing to cross-check MAP was silently self-confirming.
        #
        # Tested on the realized chain, not on `accept_rate` alone: acceptance is
        # an expectation, so a chain can carry a small mean acceptance while
        # never actually having moved.
        _accept_rate = float(jnp.mean(accept_prob_post))
        _n_unique = int(jnp.unique(chain, axis=0).shape[0])
        if not allow_degenerate and chain_is_degenerate(chain, _accept_rate):
            raise DegenerateChainError(
                f"Ray Tracing accepted essentially nothing at D={D}: acceptance "
                f"{_accept_rate:.3g}, and the {int(chain.shape[0])} post-burn-in "
                f"draws collapse to {_n_unique} unique point(s). Those draws are "
                f"the MAP initialization repeated, not a posterior — and they "
                f"look plausible, which is exactly why returning them is unsafe.\n\n"
                f"step_size={float(step_size):.3g} is too large for this "
                f"dimension. Acceptance falls off a cliff rather than degrading "
                f"gently — measured at D=516, 3e-3 gives 53% and 4e-3 gives "
                f"zero — so the working value can sit under a factor of two "
                f"below where you are. Halve it and retry, keeping the largest "
                f"value that holds acceptance in roughly the 50-90% band; going "
                f"far smaller is not safer, it just buys 99% acceptance with "
                f"steps too short to explore.\n\n"
                f"Pass allow_degenerate=True to receive the chain regardless; "
                f"that is for debugging the sampler, not for inference."
            )

        # The same latent -> physical map every other sampler uses.
        shared_arr = jax.vmap(prob.extract_shared)(chain)  # (n_samples, 2)
        shared_samples = {
            "psd_sigma": shared_arr[:, 0],
            "psd_tau_myr": shared_arr[:, 1],
        }
        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            print(
                f"  Complete in {wall_time:.1f}s. Accept: {float(jnp.mean(accept_prob_post)):.1%}"
            )
            print(
                f"  σ_PSD = {shared_params['psd_sigma']:.2f}, "
                f"τ_PSD = {shared_params['psd_tau_myr']:.1f} Myr"
            )

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            method="Hierarchical Ray Tracing",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_burnin": n_burnin,
                "n_steps": n_steps,
                "n_samples": chain.shape[0],
                "accept_rate": _accept_rate,
                # Published so a caller can judge mixing without catching an
                # exception. `accept_rate` alone cannot distinguish "moved
                # rarely" from "never moved"; this can.
                "n_unique_draws": _n_unique,
                "step_size": float(step_size),
                "D_total": D,
            },
        )
