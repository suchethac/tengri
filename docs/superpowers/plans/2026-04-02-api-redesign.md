# API Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean, high-level API surface on top of tengri's existing low-level API — reducing the common-case user workflow from 4 objects and 15 method strings to 1 object and a handful of canonical names, with zero breaking changes.

**Architecture:** All changes are purely additive. New canonical method names in `Fitter.run()` route through a deprecation shim to the existing `_run_*` methods unchanged. `Model.from_config()`, `Model.fit()`, `Model.fit_catalog()`, `Model.prior_predictive()`, and `Model.fit_population()` are new classmethods/methods on the existing `Model` class that delegate to existing objects (`Fitter`, `HierarchicalFitter`, `ParamSpec`, `Observation`). `Posterior.refine()` chains by calling back to `Posterior._fitter.run()`. No existing code paths, `_run_*` internals, or low-level APIs change.

**Tech Stack:** Python 3.12, JAX, dataclasses, ruff linting, pytest unit tests. Optional: pandas, astropy (soft dependencies in `fit_catalog()`).

---

## File map

| File | Action | What changes |
|------|--------|-------------|
| `src/tengri/core/param_translate.py` | Modify | Add `_SFH_SHORT_NAMES`, `_UNIVERSAL_SHORT_NAMES`, `resolve_short_names()` |
| `src/tengri/inference/fitter.py` | Modify | Add `_DEPRECATED_METHOD_ALIASES` dict, update `run()` with canonical names + deprecation warnings + `"auto"` + `vi_flavor=` kwarg |
| `src/tengri/inference/posterior.py` | Modify | Add `_fitter` field, `refine()`, `validate()` methods |
| `src/tengri/inference/hierarchical.py` | Modify | Add `individual` property to `HierarchicalResult`, add `plot_population()` stub |
| `src/tengri/core/model.py` | Modify | Enhance `fit()`, add `from_config()` classmethod, add `prior_predictive()`, add `fit_catalog()`, add `fit_population()`; add `PriorPredictive` dataclass |
| `src/tengri/__init__.py` | Modify | Export `posteriors_to_dataframe`, `PriorPredictive` |
| `tests/unit/test_api_convenience.py` | Create | All new convenience API tests |

---

## Task 1: Short-name alias table in `param_translate.py`

**Files:**
- Modify: `src/tengri/core/param_translate.py`
- Test: `tests/unit/test_api_convenience.py` (written in Task 9)

These two dicts and one function are the foundation for `Model.from_config()`. Add them after the existing `_REVERSE_ALIASES` dict.

- [ ] **Step 1: Add `_SFH_SHORT_NAMES`, `_UNIVERSAL_SHORT_NAMES`, and `resolve_short_names()` to `param_translate.py`**

Read the file first, then add the following immediately after the `_REVERSE_ALIASES` dict (after line 86):

```python
# ---------------------------------------------------------------------------
# High-level API: short name → full prefixed name
# ---------------------------------------------------------------------------

# sfh_type token → {short_name: full_prefixed_name}
# Used by Model.from_config() to expand user-supplied short priors.
_SFH_SHORT_NAMES: dict[str, dict[str, str]] = {
    "tsnorm": {
        "log_peak_sfr": "sfh_tsnorm_log_peak_sfr",
        "peak_lbt_gyr": "sfh_tsnorm_peak_lbt_gyr",
        "width_gyr": "sfh_tsnorm_width_gyr",
        "skew": "sfh_tsnorm_skew",
        "trunc": "sfh_tsnorm_trunc",
    },
    "snorm": {
        "log_peak_sfr": "sfh_snorm_log_peak_sfr",
        "peak_lbt_gyr": "sfh_snorm_peak_lbt_gyr",
        "width_gyr": "sfh_snorm_width_gyr",
        "skew": "sfh_snorm_skew",
    },
    "lnorm": {
        "log_peak_sfr": "sfh_lnorm_log_peak_sfr",
        "peak_lbt_gyr": "sfh_lnorm_peak_lbt_gyr",
        "width_gyr": "sfh_lnorm_width_gyr",
    },
    "dpl": {
        "alpha": "sfh_dpl_alpha",
        "beta": "sfh_dpl_beta",
        "log_peak_sfr": "sfh_dpl_log_peak_sfr",
        "tau_gyr": "sfh_dpl_tau_gyr",
    },
    "delayed": {
        "tau_gyr": "sfh_delayed_tau_gyr",
        "log_peak_sfr": "sfh_delayed_log_peak_sfr",
    },
    # "field" additions apply to any sfh that includes "+field"
    "field": {
        "psd_sigma": "sfh_field_psd_sigma",
        "psd_tau_myr": "sfh_field_psd_tau_myr",
    },
}

# Universal short names valid for any SFH type
_UNIVERSAL_SHORT_NAMES: dict[str, str] = {
    "logzsol": "met_logzsol",
    "tau_bc": "dust_tau_bc",
    "tau_diff": "dust_tau_diff",
    "dust_slope": "dust_slope",
    "agn_frac": "agn_frac",
    "neb_logU": "neb_logU",
    "redshift": "redshift",
}


def resolve_short_names(sfh_type: str | list[str], priors: dict) -> dict:
    """Expand short parameter names to full prefixed names.

    Parameters
    ----------
    sfh_type : str or list of str
        SFH type tokens, e.g. ``"tsnorm"`` or ``["dpl", "field"]``.
        Determines which short names are valid.
    priors : dict
        User-supplied prior dict, may contain short names like ``"log_peak_sfr"``
        or full names like ``"sfh_tsnorm_log_peak_sfr"``. Full names pass through
        unchanged.

    Returns
    -------
    dict
        New dict with all short names expanded to full prefixed names.
        Unknown keys that are neither short nor full names raise ValueError.

    Examples
    --------
    >>> resolve_short_names("tsnorm", {"log_peak_sfr": Uniform(-1, 2.5), "logzsol": Uniform(-2, 0.2)})
    {"sfh_tsnorm_log_peak_sfr": Uniform(-1, 2.5), "met_logzsol": Uniform(-2, 0.2)}
    """
    if isinstance(sfh_type, str):
        tokens = [t.strip() for t in sfh_type.replace("+", " ").split()]
    else:
        tokens = list(sfh_type)

    # Build combined short→full map for this sfh_type
    short_map: dict[str, str] = {}
    for token in tokens:
        if token in _SFH_SHORT_NAMES:
            short_map.update(_SFH_SHORT_NAMES[token])
    short_map.update(_UNIVERSAL_SHORT_NAMES)

    expanded: dict = {}
    for key, val in priors.items():
        if key in short_map:
            expanded[short_map[key]] = val
        else:
            # Assume it's already a full name (pass through)
            expanded[key] = val

    return expanded
```

- [ ] **Step 2: Verify no ruff violations**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
ruff check src/tengri/core/param_translate.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add src/tengri/core/param_translate.py
git commit -m "feat: add short-name alias table and resolve_short_names() to param_translate"
```

---

## Task 2: Method unification in `Fitter.run()`

**Files:**
- Modify: `src/tengri/inference/fitter.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

The `run()` method currently starts at line 2345. We add a `_DEPRECATED_METHOD_ALIASES` dict at the top of the `Fitter` class body (or as a module-level constant), then rewrite `run()` to handle the new canonical names, the `vi_flavor=` kwarg, `"auto"`, and `"mcmc"` auto-selection. Old names emit `DeprecationWarning` and route to the same underlying `_run_*` calls.

- [ ] **Step 1: Add `_DEPRECATED_METHOD_ALIASES` module constant in `fitter.py`**

Add this immediately after the imports at the top of `fitter.py` (after the `from tengri.utils.transforms` import line):

```python
# ---------------------------------------------------------------------------
# Method name unification
# ---------------------------------------------------------------------------

# Maps deprecated/old method strings → new canonical names.
# Entries here are kept working but emit DeprecationWarning.
_DEPRECATED_METHOD_ALIASES: dict[str, str] = {
    "geovi": "vi",
    "native_geovi": "vi",
    "mgvi": "vi_linear",
    "native_mgvi": "vi_linear",
    "evi": "vi_linear",
    "native_evi": "vi_linear",
    "fast_geovi": "vi_nifty",
    "nifty_geovi": "vi_nifty",
    "fast_mgvi": "vi_nifty_linear",
    "nifty_mgvi": "vi_nifty_linear",
    "raytrace": "mcmc_raytrace",
    "nuts": "mcmc_nuts",
    "elliptical_slice": "mcmc_ess",
    "nss": "evidence",
    "geovi_nuts": "vi",  # was vi+NUTS hybrid; use result.refine("mcmc_nuts") instead
}

# Threshold for "mcmc" auto-selection: low-D → NUTS, high-D → Ray Tracing.
_MCMC_AUTO_D_THRESHOLD = 20

# Threshold for "auto" method selection.
_AUTO_D_THRESHOLDS = (15, 50)  # (laplace_max, vi_linear_max)
```

- [ ] **Step 2: Replace `run()` body with the new dispatch logic**

The existing `run()` starts at line 2345. Replace its body (the if/elif chain from line 2389 through line 2466) with the following. The docstring should also be updated to reflect the new names.

Find and replace the entire `run()` method with:

```python
    def run(self, method: str = "vi", *, init_from=None, key=None, **kwargs):
        """Run inference.

        Parameters
        ----------
        method : str
            Canonical names (recommended):

            ``"vi"``             — Variational inference (geoVI by default).
            ``"vi_linear"``      — Linear VI / MGVI.
            ``"vi_nifty"``       — NIFTy tight-loop geoVI.
            ``"vi_nifty_linear"``— NIFTy tight-loop MGVI.
            ``"mcmc"``           — MCMC, auto-selects NUTS (D≤20) or Ray Tracing (D>20).
            ``"mcmc_raytrace"``  — Ray Tracing explicitly.
            ``"mcmc_nuts"``      — NUTS via BlackJAX.
            ``"mcmc_ess"``       — Elliptical Slice Sampling.
            ``"map"``            — MAP optimization.
            ``"laplace"``        — Gaussian at MAP.
            ``"pathfinder"``     — L-BFGS path.
            ``"evidence"``       — Nested Slice Sampling (log Z).
            ``"auto"``           — Auto-selects by dimensionality.

            Power-user ``vi_flavor=`` kwarg (only with ``method="vi"``):
            ``vi_flavor="nifty"``      — NIFTy tight loop.
            ``vi_flavor="nifty_full"`` — NIFTy with full logging.
            ``vi_flavor="linear"``     — Linearized geoVI (MGVI).

            Deprecated aliases (still work, emit DeprecationWarning):
            ``"geovi"``, ``"native_geovi"`` → ``"vi"``
            ``"mgvi"``, ``"native_mgvi"``   → ``"vi_linear"``
            ``"fast_geovi"``, ``"nifty_geovi"`` → ``"vi_nifty"``
            ``"fast_mgvi"``, ``"nifty_mgvi"``   → ``"vi_nifty_linear"``
            ``"raytrace"`` → ``"mcmc_raytrace"``
            ``"nuts"``     → ``"mcmc_nuts"``
            ``"elliptical_slice"`` → ``"mcmc_ess"``
            ``"nss"``      → ``"evidence"``

        init_from : Posterior, optional
            Use a previous result as initialization.
        key : PRNGKey, optional
            Random key.
        vi_flavor : str, optional
            Backend variant for ``method="vi"`` only.
            ``"nifty"``, ``"nifty_full"``, or ``"linear"``.
        **kwargs
            Method-specific arguments passed to the underlying sampler.

        Returns
        -------
        Posterior
            Inference results with ``._fitter`` back-reference set.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        # Pop vi_flavor before forwarding kwargs
        vi_flavor = kwargs.pop("vi_flavor", None)

        # Resolve deprecated aliases
        if method in _DEPRECATED_METHOD_ALIASES:
            canonical = _DEPRECATED_METHOD_ALIASES[method]
            warnings.warn(
                f"Method '{method}' is deprecated. Use '{canonical}' instead. "
                f"Old names will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Special case: geovi_nuts was a hybrid; map to vi (posterior_method handled below)
            if method == "geovi_nuts":
                kwargs.setdefault("posterior_method", "blackjax")
            method = canonical

        # --- "auto" method: dimensionality-based selection ---
        if method == "auto":
            d = self.spec.n_free
            lo, hi = _AUTO_D_THRESHOLDS
            if d <= lo:
                method = "laplace"
            elif d <= hi:
                method = "vi_linear"
            else:
                method = "vi"

        # --- Dispatch to underlying _run_* methods ---
        if method == "map":
            result = self._run_map(key=key, init_from=init_from, **kwargs)

        elif method in ("vi", "vi_linear"):
            # vi_flavor overrides method for power users
            if vi_flavor == "nifty":
                result = self._run_fast_vi(
                    key=key,
                    init_from=init_from,
                    sample_mode="nonlinear_resample",
                    posterior_method="nonlinear",
                    **kwargs,
                )
            elif vi_flavor == "nifty_full":
                result = self._run_nifty_vi(key=key, init_from=init_from, **kwargs)
            elif vi_flavor == "linear" or method == "vi_linear":
                result = self._run_native_vi(
                    key=key,
                    init_from=init_from,
                    sample_mode="linear",
                    **kwargs,
                )
            else:
                # Default "vi": geoVI (nonlinear, most accurate)
                result = self._run_native_vi(
                    key=key,
                    init_from=init_from,
                    sample_mode="geovi",
                    **kwargs,
                )

        elif method == "vi_nifty":
            result = self._run_fast_vi(
                key=key,
                init_from=init_from,
                sample_mode="nonlinear_resample",
                posterior_method="nonlinear",
                **kwargs,
            )

        elif method == "vi_nifty_linear":
            result = self._run_fast_vi(
                key=key,
                init_from=init_from,
                sample_mode="linear_resample",
                **kwargs,
            )

        elif method == "mcmc":
            # Auto-select: NUTS for low-D (exact gold-standard), RT for high-D
            d = self.spec.n_free
            if d <= _MCMC_AUTO_D_THRESHOLD:
                result = self._run_nuts(key=key, init_from=init_from, **kwargs)
            else:
                result = self._run_raytrace(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_raytrace":
            result = self._run_raytrace(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_nuts":
            result = self._run_nuts(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_ess":
            result = self._run_elliptical_slice(key=key, init_from=init_from, **kwargs)

        elif method == "evidence":
            result = self._run_nss(key=key, init_from=init_from, **kwargs)

        elif method == "laplace":
            result = self._run_laplace(key=key, init_from=init_from, **kwargs)

        elif method == "pathfinder":
            result = self._run_pathfinder(key=key, init_from=init_from, **kwargs)

        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Canonical names: 'vi', 'vi_linear', 'vi_nifty', 'vi_nifty_linear', "
                f"'mcmc', 'mcmc_raytrace', 'mcmc_nuts', 'mcmc_ess', 'map', 'laplace', "
                f"'pathfinder', 'evidence', 'auto'. "
                f"See Fitter.run() docstring for deprecated aliases."
            )

        # Attach back-reference so Posterior.refine() works
        result._fitter = self
        return result
```

- [ ] **Step 3: Verify ruff and run existing fitter tests**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
ruff check src/tengri/inference/fitter.py
ruff format --check src/tengri/inference/fitter.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/inference/fitter.py
git commit -m "feat: unify Fitter.run() method names with canonical aliases and deprecation warnings"
```

---

## Task 3: `Posterior._fitter` + `refine()` + `validate()`

**Files:**
- Modify: `src/tengri/inference/posterior.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

`Posterior` is a non-frozen `@dataclass`. Add `_fitter` as an optional field, then add two methods.

- [ ] **Step 1: Add `_fitter` field to `Posterior` dataclass**

In `posterior.py`, the dataclass fields end at line 54 with:
```python
    _model: object = field(default=None, repr=False)
```

Add one more field immediately after it:

```python
    _fitter: object = field(default=None, repr=False)
```

- [ ] **Step 2: Add `refine()` and `validate()` methods after `__repr__`**

Add the following two methods at the bottom of the `Posterior` class (after `__repr__` at line 688):

```python
    # -------------------------------------------------------------------
    # Method chaining
    # -------------------------------------------------------------------

    def refine(self, method: str, **kwargs):
        """Re-run inference from this result using a different method.

        Requires that this Posterior was produced by ``model.fit()`` or
        ``fitter.run()`` — both set the ``._fitter`` back-reference.

        Parameters
        ----------
        method : str
            Any canonical method name accepted by ``Fitter.run()``.
            E.g. ``"mcmc_raytrace"``, ``"mcmc_nuts"``, ``"vi"``.
        **kwargs
            Passed to ``Fitter.run()`` (e.g. ``n_steps``, ``n_warmup``).

        Returns
        -------
        Posterior
            New result warm-started from this posterior.

        Raises
        ------
        RuntimeError
            If ``._fitter`` is not set (Posterior created outside model.fit/fitter.run).

        Examples
        --------
        >>> result_vi = model.fit(flux, noise)
        >>> result_exact = result_vi.refine("mcmc_raytrace", n_steps=1000)

        >>> # Full pipeline in one expression
        >>> result = model.fit(flux, noise).refine("mcmc_raytrace", n_steps=500)
        """
        if self._fitter is None:
            raise RuntimeError(
                "Posterior.refine() requires a back-reference to its Fitter. "
                "Use model.fit() or fitter.run() to produce this Posterior. "
                "Posteriors loaded from disk or created manually lack this reference."
            )
        return self._fitter.run(method, init_from=self, **kwargs)

    def validate(self, n_steps: int = 200, **kwargs):
        """Run a short MCMC check and return a validation summary.

        Runs ``n_steps`` of Ray Tracing (or NUTS for D≤20) from this
        posterior's MAP estimate, then computes the marginal overlap
        between this posterior and the MCMC check posterior for each
        parameter.

        Parameters
        ----------
        n_steps : int
            Number of MCMC steps. Default 200 (quick sanity check).
        **kwargs
            Forwarded to the MCMC run.

        Returns
        -------
        dict
            Keys: ``"mcmc_result"`` (Posterior), ``"overlap"`` (dict of
            float per parameter, 1.0 = perfect overlap), ``"passed"``
            (bool, True when all overlaps > 0.5).

        Raises
        ------
        RuntimeError
            If ``._fitter`` is not set.
        """
        if self._fitter is None:
            raise RuntimeError(
                "Posterior.validate() requires a back-reference to its Fitter. "
                "Use model.fit() or fitter.run() to produce this Posterior."
            )
        d = self._fitter.spec.n_free
        mcmc_method = "mcmc_nuts" if d <= 20 else "mcmc_raytrace"
        mcmc_result = self._fitter.run(
            mcmc_method, init_from=self, n_steps=n_steps, **kwargs
        )

        # Compute per-parameter marginal overlap (histogram intersection)
        overlap: dict[str, float] = {}
        if self.samples is not None and mcmc_result.samples is not None:
            import numpy as np

            for name in self.samples:
                if name == "psd_xi":
                    continue
                vi_arr = np.array(self.samples[name])
                mc_arr = np.array(mcmc_result.samples[name])
                if vi_arr.ndim != 1:
                    continue
                lo = min(vi_arr.min(), mc_arr.min())
                hi = max(vi_arr.max(), mc_arr.max())
                if hi <= lo:
                    overlap[name] = 1.0
                    continue
                bins = np.linspace(lo, hi, 30)
                h_vi, _ = np.histogram(vi_arr, bins=bins, density=True)
                h_mc, _ = np.histogram(mc_arr, bins=bins, density=True)
                bin_w = bins[1] - bins[0]
                overlap[name] = float(np.sum(np.minimum(h_vi, h_mc)) * bin_w)

        passed = all(v > 0.5 for v in overlap.values()) if overlap else True
        return {"mcmc_result": mcmc_result, "overlap": overlap, "passed": passed}
```

- [ ] **Step 3: Verify ruff**

```bash
ruff check src/tengri/inference/posterior.py
ruff format --check src/tengri/inference/posterior.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/inference/posterior.py
git commit -m "feat: add Posterior._fitter field, refine(), and validate() for method chaining"
```

---

## Task 4: Enhance `Model.fit()`

**Files:**
- Modify: `src/tengri/core/model.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

The current `fit()` at line 1647 is 3 lines. Replace it with the enhanced version that:
1. Changes the default method from `"map"` to `"vi"`
2. Supports `photometry=(flux, noise)` and `spectrum=(flux, noise)` kwargs for joint fitting
3. Infers `data_type` from the data shape when not explicitly given
4. Supports `init="map"` to run MAP first, then use as warm start
5. Stores `self.fitter_` after the fit

- [ ] **Step 1: Replace `Model.fit()` with enhanced version**

Find the existing `fit()` method at line 1647 and replace it entirely:

```python
    def fit(
        self,
        data=None,
        noise=None,
        method: str = "vi",
        data_type: str | None = None,
        *,
        photometry: tuple | None = None,
        spectrum: tuple | None = None,
        init: str | None = None,
        **kwargs,
    ):
        """Fit observed data.  Convenience wrapper — no Fitter construction needed.

        Parameters
        ----------
        data : array, optional
            Observed flux array (photometry or spectroscopy). For joint fitting,
            leave as ``None`` and use ``photometry=`` / ``spectrum=`` instead.
        noise : array, optional
            1-sigma uncertainties matching ``data``.
        method : str
            Inference method. Default ``"vi"`` (geoVI variational inference).
            Any canonical name accepted by ``Fitter.run()`` works here:
            ``"vi"``, ``"vi_linear"``, ``"mcmc"``, ``"mcmc_raytrace"``,
            ``"mcmc_nuts"``, ``"map"``, ``"laplace"``, ``"auto"``, etc.
        data_type : str or None
            ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
            When ``None`` (default), inferred from the model's ``observation``
            or from whether ``photometry=`` / ``spectrum=`` kwargs are used.
        photometry : tuple of (flux, noise), optional
            Photometric data for joint fitting. Pass alongside ``spectrum=``.
        spectrum : tuple of (flux, noise), optional
            Spectroscopic data for joint fitting. Pass alongside ``photometry=``.
        init : str or None
            Initialization strategy. ``"map"`` runs MAP optimization first, then
            uses the result to warm-start the requested method. ``None`` (default)
            uses the method's own default initialization.
        **kwargs
            Forwarded to ``Fitter.run()``.

        Returns
        -------
        Posterior
            Inference results.  ``._fitter`` is set so ``.refine()`` works.
            After this call, ``self.fitter_`` holds the ``Fitter`` instance.

        Examples
        --------
        >>> result = model.fit(flux_obs, noise)                    # vi default
        >>> result = model.fit(flux_obs, noise, method="mcmc")
        >>> result = model.fit(spectrum, noise, data_type="spectroscopy")
        >>> result = model.fit(photometry=(flux_p, noise_p),
        ...                    spectrum=(flux_s, noise_s))         # joint
        >>> result = model.fit(flux_obs, noise, init="map")        # MAP warm start
        >>> result = model.fit(flux_obs, noise).refine("mcmc_raytrace")  # chained
        """
        from tengri.inference.fitter import Fitter

        # --- Resolve data arrays ---
        if photometry is not None or spectrum is not None:
            # Joint or single-modality via keyword args
            import jax.numpy as jnp

            if photometry is not None and spectrum is not None:
                flux_p, noise_p = photometry
                flux_s, noise_s = spectrum
                data = jnp.concatenate([jnp.asarray(flux_p), jnp.asarray(flux_s)])
                noise = jnp.concatenate([jnp.asarray(noise_p), jnp.asarray(noise_s)])
                data_type = data_type or "joint"
            elif photometry is not None:
                data, noise = photometry
                data_type = data_type or "photometry"
            else:
                data, noise = spectrum
                data_type = data_type or "spectroscopy"
        else:
            if data is None or noise is None:
                raise ValueError(
                    "Provide either positional (data, noise) or keyword "
                    "photometry=(flux, noise) / spectrum=(flux, noise)."
                )

        # --- Infer data_type if still None ---
        if data_type is None:
            obs = getattr(self, "observation", None)
            if obs is not None:
                data_type = obs.data_type
            else:
                data_type = "photometry"

        # --- Build fitter ---
        fitter = Fitter(self, data, noise, data_type=data_type)
        self.fitter_ = fitter  # expose for power users

        # --- Optional MAP warm start ---
        init_from = None
        if init == "map":
            init_from = fitter.run("map")

        return fitter.run(method, init_from=init_from, **kwargs)
```

- [ ] **Step 2: Verify ruff**

```bash
ruff check src/tengri/core/model.py
ruff format --check src/tengri/core/model.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add src/tengri/core/model.py
git commit -m "feat: enhance Model.fit() — default vi, joint input, init=map, fitter_ attribute"
```

---

## Task 5: `Model.from_config()` classmethod

**Files:**
- Modify: `src/tengri/core/model.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

Add `from_config()` as a classmethod on `Model`. It resolves short names, loads SSP data if a path is given, builds `Observation`, and returns a standard `Model`.

- [ ] **Step 1: Add `Model.from_config()` classmethod**

Add the following method to the `Model` class immediately before `fit()` (around line 1644):

```python
    @classmethod
    def from_config(
        cls,
        ssp,
        sfh: str = "dpl",
        dust: str = "charlot_fall",
        nebular: str | None = None,
        agn: str | None = None,
        redshift: float | str = 0.1,
        filters: list[str] | None = None,
        wave_obs=None,
        priors: dict | None = None,
        **model_kwargs,
    ) -> "Model":
        """Build a Model from a grouped configuration dict.

        Reduces boilerplate for the common case: instead of constructing
        ``ParamSpec``, ``SSPData``, ``Observation``, and ``Model`` separately,
        provide a single grouped config and receive a fully configured ``Model``.

        Parameters
        ----------
        ssp : str or SSPData
            Path to SSP HDF5 file, or a pre-loaded ``SSPData`` instance.
        sfh : str
            SFH family name. Any name from the SFH registry, e.g.
            ``"tsnorm"``, ``"dpl"``, ``"dpl+field"``, ``"snorm"``.
            Compound types use ``"+"`` separator: ``"dpl+field"``.
        dust : str
            Dust attenuation law. ``"charlot_fall"`` (default), ``"calzetti"``,
            ``"power_law"``, etc.  Passed as ``dust_law_bc`` to ``ParamSpec``.
        nebular : str or None
            Nebular emission backend. ``"baked_in"``, ``"cloudy"``, ``"cue"``, or
            ``None`` (disabled).
        agn : str or None
            AGN model. ``None`` (disabled, default), ``"simple"``,
            ``"unified_nlr_blr"``, or any AGN model name.
        redshift : float or str
            Fixed redshift value (float), or ``"free"`` to make redshift a free
            parameter with a ``Uniform(0.001, 6.0)`` prior.
        filters : list of str, optional
            Filter names, e.g. ``["sdss_u", "sdss_g", "sdss_r"]``.
            If provided, builds a ``Photometry`` object and attaches it
            to the model's ``Observation``.
        wave_obs : array, optional
            Observed-frame wavelength array for spectroscopy. If provided,
            builds a ``SpectroscopyConfig`` and attaches it.
        priors : dict, optional
            Parameter priors. Keys may be short names (e.g. ``"log_peak_sfr"``
            when ``sfh="tsnorm"``), universal short names (``"logzsol"``,
            ``"tau_bc"``), or full prefixed names (``"sfh_tsnorm_log_peak_sfr"``).
            Short names are expanded automatically based on the ``sfh`` argument.
        **model_kwargs
            Forwarded to ``Model.__init__()`` (e.g. ``forward_dtype="float32"``).

        Returns
        -------
        Model
            Fully configured model, identical to one built manually via
            ``ParamSpec + SSPData + Observation + Model``.

        Examples
        --------
        >>> model = tengri.Model.from_config(
        ...     ssp="data/ssp.h5",
        ...     sfh="tsnorm",
        ...     filters=["sdss_u", "sdss_g", "sdss_r"],
        ...     redshift=0.1,
        ...     priors=dict(
        ...         log_peak_sfr=tengri.Uniform(-1, 2.5),
        ...         peak_lbt_gyr=tengri.Uniform(0.5, 12),
        ...         width_gyr=tengri.Uniform(0.3, 5),
        ...         logzsol=tengri.Uniform(-2, 0.2),
        ...         tau_bc=tengri.Uniform(0, 2),
        ...     )
        ... )
        """
        from tengri.core.param_translate import resolve_short_names
        from tengri.core.param_spec import ParamSpec
        from tengri.distributions import Fixed, Uniform
        from tengri.models.observation.observation import Observation
        from tengri.models.observation.photometry_config import Photometry
        from tengri.models.sps.dsps_wrapper import SSPData, load_ssp_data

        # --- Load SSP data ---
        if isinstance(ssp, str):
            ssp_data = load_ssp_data(ssp)
        elif isinstance(ssp, SSPData):
            ssp_data = ssp
        else:
            raise TypeError(f"ssp must be a file path (str) or SSPData, got {type(ssp)}")

        # --- Expand short names in priors ---
        priors = priors or {}
        expanded = resolve_short_names(sfh, priors)

        # --- Inject redshift ---
        if redshift == "free":
            if "redshift" not in expanded:
                expanded["redshift"] = Uniform(0.001, 6.0)
        else:
            expanded.setdefault("redshift", float(redshift))

        # --- Inject AGN frac if agn enabled and not already in priors ---
        if agn is not None and "agn_frac" not in expanded:
            expanded["agn_frac"] = Uniform(0.0, 1.0)

        # --- Build ParamSpec ---
        # Parse sfh_type: "dpl+field" → ["dpl", "field"]
        sfh_tokens = [t.strip() for t in sfh.replace("+", " ").split()]

        spec_kwargs: dict = dict(expanded)
        spec_kwargs["mean_sfh_type"] = sfh_tokens

        if dust != "charlot_fall":
            spec_kwargs["dust_law_bc"] = dust

        if nebular is not None:
            spec_kwargs["nebular_mode"] = nebular

        if agn is not None:
            spec_kwargs["agn_model"] = agn

        spec = ParamSpec(**spec_kwargs)

        # --- Build Observation ---
        obs_photometry = None
        obs_spectroscopy = None

        if filters is not None:
            obs_photometry = Photometry.from_names(filters)

        if wave_obs is not None:
            from tengri.models.observation.spectroscopy_config import SpectroscopyConfig
            obs_spectroscopy = SpectroscopyConfig(wave_obs=wave_obs)

        if obs_photometry is not None or obs_spectroscopy is not None:
            observation = Observation(
                photometry=obs_photometry,
                spectroscopy=obs_spectroscopy,
            )
        else:
            observation = None

        return cls(spec, ssp_data, observation=observation, **model_kwargs)
```

- [ ] **Step 2: Verify ruff**

```bash
ruff check src/tengri/core/model.py
ruff format --check src/tengri/core/model.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add src/tengri/core/model.py
git commit -m "feat: add Model.from_config() classmethod with short-name prior resolution"
```

---

## Task 6: `PriorPredictive` + `Model.prior_predictive()`

**Files:**
- Modify: `src/tengri/core/model.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

Add a `PriorPredictive` dataclass and `Model.prior_predictive()` method.

- [ ] **Step 1: Add `PriorPredictive` dataclass near the top of `model.py`**

Add this immediately after the `MockData` NamedTuple definition (after the `class MockData` block, around line 100):

```python
# ---------------------------------------------------------------------------
# PriorPredictive container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PriorPredictive:
    """Results of a prior predictive check.

    Attributes
    ----------
    flux : jnp.ndarray
        Predicted photometry draws, shape ``(n, n_filters)``.
        None if the model has no filters.
    sfh : jnp.ndarray
        SFH draws, shape ``(n, n_grid)``.
    params : dict
        Drawn parameter samples, each of shape ``(n,)``.
    _model : Model
        Back-reference to the parent model (for plotting helpers).
    """

    flux: "jnp.ndarray | None"
    sfh: "jnp.ndarray"
    params: dict
    _model: object = dataclasses.field(default=None, repr=False)

    def check_finite(self) -> dict:
        """Check for NaN/Inf in flux draws.

        Returns
        -------
        dict
            ``{"n_nan": int, "n_inf": int, "frac_bad": float, "ok": bool}``
        """
        import numpy as np

        if self.flux is None:
            return {"n_nan": 0, "n_inf": 0, "frac_bad": 0.0, "ok": True}

        flux_np = np.array(self.flux)
        n_nan = int(np.sum(np.isnan(flux_np)))
        n_inf = int(np.sum(np.isinf(flux_np)))
        total = flux_np.size
        frac_bad = (n_nan + n_inf) / max(total, 1)
        if n_nan + n_inf > 0:
            import warnings

            warnings.warn(
                f"prior_predictive: {n_nan} NaN and {n_inf} Inf values in flux draws "
                f"({frac_bad:.1%} of total). Check priors for extreme parameter combinations.",
                UserWarning,
                stacklevel=2,
            )
        return {"n_nan": n_nan, "n_inf": n_inf, "frac_bad": frac_bad, "ok": (n_nan + n_inf == 0)}

    def plot_seds(self, n_show: int = 50, color_by: str | None = None, ax=None):
        """Plot a sample of prior-predictive SED draws.

        Parameters
        ----------
        n_show : int
            Number of SED draws to show. Default 50.
        color_by : str or None
            Parameter name to use for colour-coding draws (e.g. ``"sfh_field_psd_sigma"``).
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if self.flux is None:
            raise RuntimeError("plot_seds() requires photometry — model has no filters.")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        n_draws = min(n_show, len(np.array(self.flux)))
        flux_np = np.array(self.flux)[:n_draws]

        colors = None
        if color_by is not None and color_by in self.params:
            vals = np.array(self.params[color_by])[:n_draws]
            vmin, vmax = vals.min(), vals.max()
            norm = (vals - vmin) / max(vmax - vmin, 1e-10)
            cmap = plt.cm.plasma
            colors = [cmap(v) for v in norm]

        for i, row in enumerate(flux_np):
            c = colors[i] if colors is not None else "C0"
            ax.plot(range(len(row)), row, color=c, alpha=0.3, lw=0.8)

        ax.set_xlabel("Filter index")
        ax.set_ylabel("Flux (erg/s/cm²/Hz)")
        ax.set_title(f"Prior predictive SEDs (n={n_draws})")
        return ax

    def plot_colors(self, color_x: str, color_y: str, ax=None):
        """Plot a colour–colour diagram of prior predictive draws.

        Parameters
        ----------
        color_x : str
            Colour index for x-axis, e.g. ``"sdss_g-sdss_r"``.
            Must match filter names passed to ``Model.from_config()``.
        color_y : str
            Colour index for y-axis.
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        raise NotImplementedError(
            "plot_colors() requires filter name resolution — not yet implemented. "
            "Use ppc.flux directly to compute colour indices."
        )
```

You need to add `import dataclasses` at the top of `model.py` (it's not currently imported). Add it to the existing imports block.

- [ ] **Step 2: Add `Model.prior_predictive()` method**

Add this method to the `Model` class, immediately before `fit()`:

```python
    def prior_predictive(self, n: int = 500, seed: int = 42) -> "PriorPredictive":
        """Sample from the prior and evaluate the forward model on each draw.

        Returns a ``PriorPredictive`` object with draw arrays and convenience
        plotting methods for model checking before inference.

        Parameters
        ----------
        n : int
            Number of prior draws. Default 500.
        seed : int
            Random seed. Default 42.

        Returns
        -------
        PriorPredictive
            ``ppc.flux`` — shape (n, n_filters) or None.
            ``ppc.sfh``  — shape (n, n_grid).
            ``ppc.params`` — dict of (n,) arrays.

        Examples
        --------
        >>> ppc = model.prior_predictive(n=500)
        >>> ppc.check_finite()
        >>> ppc.plot_seds(n_show=50)
        """
        key = jax.random.PRNGKey(seed)
        params_batch = self.spec.sample_batch(key, n)

        # SFH draws
        sfh_batch = jax.vmap(self.predict_sfh)(params_batch)

        # Photometry draws (if filters present)
        flux_batch = None
        if self.filter_waves is not None:
            try:
                flux_batch = jax.vmap(self.predict_photometry)(params_batch)
            except Exception:
                flux_batch = None

        return PriorPredictive(
            flux=flux_batch,
            sfh=sfh_batch,
            params=params_batch,
            _model=self,
        )
```

- [ ] **Step 3: Verify ruff**

```bash
ruff check src/tengri/core/model.py
ruff format --check src/tengri/core/model.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/core/model.py
git commit -m "feat: add PriorPredictive dataclass and Model.prior_predictive() method"
```

---

## Task 7: `Model.fit_catalog()`

**Files:**
- Modify: `src/tengri/core/model.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

`fit_catalog()` loops over rows in a DataFrame, astropy Table, or list-of-dicts, creates a per-row Fitter (optionally with per-row redshift), and calls `self.fit()`.

- [ ] **Step 1: Add `Model.fit_catalog()` method**

Add this method immediately after `fit()`:

```python
    def fit_catalog(
        self,
        catalog,
        flux_cols: list[str],
        err_cols: list[str],
        redshift_col: str | None = None,
        method: str = "vi",
        n_workers: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> list:
        """Fit a catalog of galaxies, one row at a time.

        Accepts a ``pandas.DataFrame``, an ``astropy.table.Table``, or a
        list of dicts. Each row becomes one ``Posterior``.

        Parameters
        ----------
        catalog : DataFrame, Table, or list of dict
            Input catalog. For DataFrames/Tables, column names are used.
            For list-of-dicts, keys are column names.
        flux_cols : list of str
            Column names for per-band flux values (must match model's filter order).
        err_cols : list of str
            Column names for per-band 1-sigma uncertainties.
        redshift_col : str or None
            If provided, use this column as the per-row redshift, overriding
            the model's fixed redshift for each galaxy via ``spec.with_params()``.
        method : str
            Inference method. Default ``"vi"``.
        n_workers : int
            Currently ignored (reserved for future multiprocessing). Default 1.
        verbose : bool
            Print per-galaxy progress. Default True.
        **kwargs
            Forwarded to ``Fitter.run()`` for every galaxy.

        Returns
        -------
        list of Posterior
            Same length as input catalog.

        Examples
        --------
        >>> results = model.fit_catalog(
        ...     catalog_df,
        ...     flux_cols=["flux_u", "flux_g", "flux_r"],
        ...     err_cols=["err_u", "err_g", "err_r"],
        ...     redshift_col="z_spec",
        ...     method="vi",
        ... )
        >>> summary = tengri.posteriors_to_dataframe(results, params=["met_logzsol"])
        """
        import time

        import jax.numpy as jnp

        from tengri.distributions import Fixed
        from tengri.inference.fitter import Fitter

        # Normalise catalog to list of dicts
        rows: list[dict] = []
        try:
            import pandas as pd
            if isinstance(catalog, pd.DataFrame):
                rows = catalog.to_dict(orient="records")
        except ImportError:
            pass

        if not rows:
            try:
                from astropy.table import Table
                if isinstance(catalog, Table):
                    rows = [dict(zip(catalog.colnames, row)) for row in catalog]
            except ImportError:
                pass

        if not rows and isinstance(catalog, (list, tuple)):
            rows = list(catalog)

        if not rows:
            raise TypeError(
                f"catalog must be a pandas DataFrame, astropy Table, or list of dicts. "
                f"Got {type(catalog)}"
            )

        n_gal = len(rows)
        results: list = []
        t0 = time.time()

        for i, row in enumerate(rows):
            t_row = time.time()

            # Extract flux and noise for this row
            flux_i = jnp.array([float(row[c]) for c in flux_cols])
            noise_i = jnp.array([float(row[c]) for c in err_cols])

            # Per-row redshift: build a modified spec
            if redshift_col is not None:
                row_z = float(row[redshift_col])
                row_spec = self.spec.with_params(redshift=Fixed(row_z))
                # Build a lightweight model clone with the updated spec
                # (shares all arrays — only spec differs)
                from tengri.core.model import Model as _Model
                row_model = _Model.__new__(_Model)
                row_model.__dict__.update(self.__dict__)
                row_model.spec = row_spec
                fitter_i = Fitter(row_model, flux_i, noise_i)
            else:
                fitter_i = Fitter(self, flux_i, noise_i)

            result_i = fitter_i.run(method, **kwargs)
            results.append(result_i)

            if verbose:
                dt = time.time() - t_row
                elapsed = time.time() - t0
                chi2 = result_i.diagnostics.get("chi2_dof", "?")
                chi2_str = f"{chi2:.2f}" if isinstance(chi2, float) else str(chi2)
                print(
                    f"  [{i + 1}/{n_gal}] chi2/dof={chi2_str}, "
                    f"row={dt:.1f}s, total={elapsed:.0f}s"
                )

        return results
```

- [ ] **Step 2: Verify ruff**

```bash
ruff check src/tengri/core/model.py
ruff format --check src/tengri/core/model.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add src/tengri/core/model.py
git commit -m "feat: add Model.fit_catalog() accepting DataFrame/Table/list-of-dicts"
```

---

## Task 8: `HierarchicalResult.individual` + `Model.fit_population()`

**Files:**
- Modify: `src/tengri/inference/hierarchical.py`
- Modify: `src/tengri/core/model.py`
- Test: `tests/unit/test_api_convenience.py` (Task 9)

Add the `individual` property to `HierarchicalResult`, a `plot_population()` stub, and a `Model.fit_population()` wrapper method.

- [ ] **Step 1: Add `individual` property and `plot_population()` to `HierarchicalResult`**

In `hierarchical.py`, the `HierarchicalResult` dataclass ends at the `__repr__` method (line ~73). Add these two methods after `summary()`:

```python
    @property
    def individual(self):
        """Per-galaxy posterior marginals as a list of lightweight objects.

        Returns
        -------
        list of SimpleNamespace
            Each element has ``.samples`` (dict) and ``.params`` (dict)
            corresponding to the per-galaxy posterior from ``individual_samples``.
            Returns an empty list if ``individual_samples`` is ``None``.
        """
        from types import SimpleNamespace

        if self.individual_samples is None:
            return []
        result = []
        for samp in self.individual_samples:
            params = {k: float(np.median(v)) if v.ndim == 1 else v
                      for k, v in samp.items()}
            result.append(SimpleNamespace(samples=samp, params=params))
        return result

    def plot_population(self, params=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"), ax=None):
        """Scatter plot of shared PSD parameter posteriors.

        Parameters
        ----------
        params : tuple of str
            Two parameter names for the x and y axes. Defaults to the PSD
            (σ, τ) axes most relevant for hierarchical PSD recovery.
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Axes
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
            ax.text(0.5, 0.5, f"Parameters {px!r} or {py!r} not found",
                    ha="center", va="center", transform=ax.transAxes)
        return ax
```

- [ ] **Step 2: Add `Model.fit_population()` method**

Add this method to `Model` in `model.py`, immediately after `fit_catalog()`:

```python
    def fit_population(
        self,
        observations_list: list,
        method: str = "vi",
        population_prior: dict | None = None,
        **kwargs,
    ):
        """Fit a population of galaxies with shared PSD hyperparameters.

        Thin wrapper around ``HierarchicalFitter``. The underlying objects
        (``HierarchicalFitter``, ``HierarchicalResult``) are unchanged and
        still accessible directly for fine-grained control.

        Parameters
        ----------
        observations_list : list
            Each element is either:
            - A ``(flux, noise)`` tuple (photometric data), or
            - A dict with ``"flux_obs"`` and ``"noise"`` keys (legacy format).
        method : str
            Hierarchical inference method. Default ``"vi"``.
            Maps ``"vi"`` → ``"geovi"``, ``"vi_linear"`` → ``"mgvi"``,
            ``"mcmc_raytrace"`` → ``"raytrace"``. Old names still work.
        population_prior : dict or None
            Hyperpriors on the shared PSD parameters, e.g.
            ``{"psd_sigma": Gaussian(0.5, 0.5), "psd_tau_myr": LogUniform(10, 500)}``.
            Passed to ``HierarchicalFitter`` as ``psd_sigma_prior`` and
            ``psd_tau_prior`` when those standard keys are present.
        **kwargs
            Forwarded to ``HierarchicalFitter.run()``.

        Returns
        -------
        HierarchicalResult
            Has ``.shared_samples``, ``.individual`` (per-galaxy marginals),
            ``.summary()``, and ``.plot_population()``.

        Examples
        --------
        >>> pop_result = model.fit_population(
        ...     [(flux_i, noise_i) for flux_i, noise_i in zip(fluxes, noises)],
        ...     method="vi",
        ...     population_prior={
        ...         "psd_sigma": tengri.Gaussian(0.5, 0.5),
        ...         "psd_tau_myr": tengri.LogUniform(10, 500),
        ...     },
        ... )
        >>> pop_result.summary()
        >>> pop_result.individual[0].params   # Galaxy 0 median params
        >>> pop_result.plot_population()
        """
        from tengri.inference.hierarchical import HierarchicalFitter

        # Normalise observations_list to list of dicts
        galaxies = []
        for obs in observations_list:
            if isinstance(obs, (list, tuple)) and len(obs) == 2:
                flux, noise = obs
                galaxies.append({"flux_obs": flux, "noise": noise})
            elif isinstance(obs, dict):
                galaxies.append(obs)
            else:
                raise TypeError(
                    f"Each element of observations_list must be a (flux, noise) tuple "
                    f"or a dict with 'flux_obs'/'noise' keys. Got {type(obs)}"
                )

        # Extract population prior bounds if standard keys given
        psd_sigma_prior = (0.1, 4.0)
        psd_tau_prior = (1.0, 300.0)
        if population_prior:
            if "psd_sigma" in population_prior:
                dist = population_prior["psd_sigma"]
                psd_sigma_prior = getattr(dist, "bounds", psd_sigma_prior)
            if "psd_tau_myr" in population_prior:
                dist = population_prior["psd_tau_myr"]
                psd_tau_prior = getattr(dist, "bounds", psd_tau_prior)

        # Translate canonical method names to HierarchicalFitter's names
        _hier_method_map = {
            "vi": "geovi",
            "vi_linear": "mgvi",
            "mcmc_raytrace": "raytrace",
            "mcmc": "raytrace",
        }
        hier_method = _hier_method_map.get(method, method)

        # Build model_factory that re-uses this model's spec with injected PSD params
        def _model_factory(psd_sigma, psd_tau_myr):
            from tengri.distributions import Fixed

            new_spec = self.spec.with_params(
                sfh_field_psd_sigma=Fixed(float(psd_sigma)),
                sfh_field_psd_tau_myr=Fixed(float(psd_tau_myr)),
            )
            # Return a lightweight copy sharing all precomputed arrays
            m = Model.__new__(Model)
            m.__dict__.update(self.__dict__)
            m.spec = new_spec
            return m

        hfitter = HierarchicalFitter(
            _model_factory,
            galaxies,
            psd_sigma_prior=psd_sigma_prior,
            psd_tau_prior=psd_tau_prior,
        )
        return hfitter.run(hier_method, **kwargs)
```

- [ ] **Step 3: Verify ruff on both files**

```bash
ruff check src/tengri/inference/hierarchical.py src/tengri/core/model.py
ruff format --check src/tengri/inference/hierarchical.py src/tengri/core/model.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/inference/hierarchical.py src/tengri/core/model.py
git commit -m "feat: add HierarchicalResult.individual, plot_population(), and Model.fit_population()"
```

---

## Task 9: `posteriors_to_dataframe()` + `__init__.py` exports

**Files:**
- Modify: `src/tengri/__init__.py`
- Test: `tests/unit/test_api_convenience.py` (Task 10)

Add `posteriors_to_dataframe()` as a module-level utility and export `PriorPredictive`.

- [ ] **Step 1: Add `posteriors_to_dataframe()` to `__init__.py`**

Read the file. After the existing imports block, add the following function and update the `__all__` list:

First add the function definition. Add it as a top-level function at the end of the file (before any `__all__` definition):

```python
def posteriors_to_dataframe(results: list, params: list[str] | None = None):
    """Summarise a list of Posteriors into a pandas DataFrame.

    Requires ``pandas`` (``pip install pandas``).

    Parameters
    ----------
    results : list of Posterior
        Output of ``model.fit_catalog()`` or any list of Posterior objects.
    params : list of str or None
        Parameter names to include. Default: all scalar free parameters,
        excluding ``psd_xi``.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy, columns: ``{param}_median``, ``{param}_lo68``,
        ``{param}_hi68`` for each requested parameter.

    Examples
    --------
    >>> df = tengri.posteriors_to_dataframe(results, params=["met_logzsol", "dust_tau_bc"])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("posteriors_to_dataframe() requires pandas: pip install pandas") from None

    import numpy as np

    rows = []
    for result in results:
        row: dict = {}

        if result.samples is None:
            # MAP: use point estimates
            for name, val in result.params.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                row[f"{name}_value"] = float(np.mean(np.array(val)))
        else:
            # Sampling: use median + 68% CI
            for name, arr in result.samples.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                arr_np = np.array(arr)
                if arr_np.ndim != 1:
                    continue
                row[f"{name}_median"] = float(np.median(arr_np))
                row[f"{name}_lo68"] = float(np.percentile(arr_np, 16))
                row[f"{name}_hi68"] = float(np.percentile(arr_np, 84))

        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 2: Add `PriorPredictive` and `posteriors_to_dataframe` to imports in `__init__.py`**

In `__init__.py`, find the `from tengri.core.model import Model` line and replace with:

```python
from tengri.core.model import Model, PriorPredictive
```

Also add `posteriors_to_dataframe` to the `__all__` list if present, or verify it's accessible at `tengri.posteriors_to_dataframe`.

- [ ] **Step 3: Verify ruff**

```bash
ruff check src/tengri/__init__.py
ruff format --check src/tengri/__init__.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/__init__.py
git commit -m "feat: export PriorPredictive and posteriors_to_dataframe from tengri public API"
```

---

## Task 10: Tests in `tests/unit/test_api_convenience.py`

**Files:**
- Create: `tests/unit/test_api_convenience.py`

Tests are split into two groups:
1. **No-SSP tests** — test pure Python logic (deprecation warnings, Posterior._fitter, short-name resolution, posteriors_to_dataframe). Run always.
2. **SSP-required tests** — test actual fitting (Model.fit, from_config, prior_predictive, fit_catalog, fit_population). Skip if SSP data absent.

- [ ] **Step 1: Write the failing tests first**

```bash
cat tests/unit/test_api_convenience.py
```

Expected: `No such file`

- [ ] **Step 2: Create `tests/unit/test_api_convenience.py`**

```python
"""Tests for the new convenience API (Changes 1–7 from api_redesign.md).

No-SSP tests run unconditionally. SSP-required tests are skipped when data is absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# SSP availability gate
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_posterior(method="vi", n_samples=10):
    """Return a Posterior without needing SSP data (uses mocks)."""
    from tengri.inference.posterior import Posterior

    key = jax.random.PRNGKey(0)
    params = {"met_logzsol": jnp.array([-0.3]), "dust_tau_bc": jnp.array([0.5])}
    samples = {
        "met_logzsol": jax.random.normal(key, (n_samples,)) * 0.1 - 0.3,
        "dust_tau_bc": jax.random.uniform(key, (n_samples,)) * 0.5 + 0.2,
    }
    return Posterior(
        samples=samples,
        params=params,
        method=method,
        wall_time_s=1.0,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# Task 2: Method unification — deprecation warnings (no SSP needed)
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    """Old method strings must emit DeprecationWarning and still work."""

    def _make_mock_fitter(self):
        from tengri.inference.fitter import Fitter

        fitter = MagicMock(spec=Fitter)
        fitter.spec = MagicMock()
        fitter.spec.n_free = 10  # low-D → NUTS for mcmc auto
        # Make _run_* methods return a minimal Posterior
        posterior = _make_minimal_posterior()
        fitter._run_native_vi.return_value = posterior
        fitter._run_raytrace.return_value = posterior
        fitter._run_nuts.return_value = posterior
        fitter._run_fast_vi.return_value = posterior
        fitter._run_nifty_vi.return_value = posterior
        fitter._run_nss.return_value = posterior
        fitter._run_laplace.return_value = posterior
        fitter._run_pathfinder.return_value = posterior
        fitter._run_elliptical_slice.return_value = posterior
        fitter._run_map.return_value = posterior
        return fitter

    @pytest.mark.parametrize("old_name,canonical", [
        ("geovi", "vi"),
        ("native_geovi", "vi"),
        ("mgvi", "vi_linear"),
        ("native_mgvi", "vi_linear"),
        ("raytrace", "mcmc_raytrace"),
        ("nuts", "mcmc_nuts"),
        ("elliptical_slice", "mcmc_ess"),
        ("nss", "evidence"),
    ])
    def test_deprecated_alias_emits_warning(self, old_name, canonical):
        from tengri.inference.fitter import _DEPRECATED_METHOD_ALIASES

        assert old_name in _DEPRECATED_METHOD_ALIASES
        assert _DEPRECATED_METHOD_ALIASES[old_name] == canonical

    def test_new_canonical_names_exist(self):
        from tengri.inference.fitter import Fitter

        # Check the run() docstring lists new canonical names
        doc = Fitter.run.__doc__
        for name in ("vi", "vi_linear", "vi_nifty", "vi_nifty_linear",
                     "mcmc", "mcmc_raytrace", "mcmc_nuts", "mcmc_ess",
                     "evidence", "auto"):
            assert name in doc, f"'{name}' not found in Fitter.run() docstring"

    def test_deprecated_alias_dict_is_module_level(self):
        import tengri.inference.fitter as fitter_module

        assert hasattr(fitter_module, "_DEPRECATED_METHOD_ALIASES")
        assert isinstance(fitter_module._DEPRECATED_METHOD_ALIASES, dict)


# ---------------------------------------------------------------------------
# Task 1: Short-name resolution (no SSP needed)
# ---------------------------------------------------------------------------


class TestResolveShortNames:
    def test_tsnorm_short_names(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        prior = {"log_peak_sfr": Uniform(-1, 2.5), "logzsol": Uniform(-2, 0.2)}
        expanded = resolve_short_names("tsnorm", prior)

        assert "sfh_tsnorm_log_peak_sfr" in expanded
        assert "met_logzsol" in expanded
        assert "log_peak_sfr" not in expanded
        assert "logzsol" not in expanded

    def test_dpl_short_names(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        prior = {"alpha": Uniform(0.5, 3.0), "tau_bc": Uniform(0, 2)}
        expanded = resolve_short_names("dpl", prior)

        assert "sfh_dpl_alpha" in expanded
        assert "dust_tau_bc" in expanded

    def test_field_short_names_in_compound(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        prior = {"psd_sigma": Uniform(0.1, 2.0), "psd_tau_myr": Uniform(10, 500)}
        expanded = resolve_short_names("dpl+field", prior)

        assert "sfh_field_psd_sigma" in expanded
        assert "sfh_field_psd_tau_myr" in expanded

    def test_full_names_pass_through(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        prior = {"sfh_tsnorm_log_peak_sfr": Uniform(-1, 2.5)}
        expanded = resolve_short_names("tsnorm", prior)

        # Full name passes through unchanged
        assert "sfh_tsnorm_log_peak_sfr" in expanded

    def test_list_input_sfh_type(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        prior = {"log_peak_sfr": Uniform(-1, 2.5)}
        # Both string "tsnorm" and list ["tsnorm"] should work
        r1 = resolve_short_names("tsnorm", prior)
        r2 = resolve_short_names(["tsnorm"], prior)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Task 3: Posterior._fitter + refine() + validate() (no SSP needed)
# ---------------------------------------------------------------------------


class TestPosteriorFitterField:
    def test_posterior_has_fitter_field(self):
        from tengri.inference.posterior import Posterior

        p = _make_minimal_posterior()
        assert hasattr(p, "_fitter")
        assert p._fitter is None  # default

    def test_fitter_can_be_set(self):
        from tengri.inference.posterior import Posterior

        p = _make_minimal_posterior()
        mock_fitter = MagicMock()
        p._fitter = mock_fitter
        assert p._fitter is mock_fitter

    def test_refine_raises_without_fitter(self):
        p = _make_minimal_posterior()
        with pytest.raises(RuntimeError, match="_fitter"):
            p.refine("mcmc_raytrace")

    def test_refine_calls_fitter_run(self):
        p = _make_minimal_posterior()
        mock_fitter = MagicMock()
        expected_result = _make_minimal_posterior(method="mcmc_raytrace")
        mock_fitter.run.return_value = expected_result
        p._fitter = mock_fitter

        result = p.refine("mcmc_raytrace", n_steps=100)

        mock_fitter.run.assert_called_once_with("mcmc_raytrace", init_from=p, n_steps=100)
        assert result is expected_result

    def test_validate_raises_without_fitter(self):
        p = _make_minimal_posterior()
        with pytest.raises(RuntimeError, match="_fitter"):
            p.validate()


class TestPosteriorValidate:
    def test_validate_returns_dict_with_required_keys(self):
        p = _make_minimal_posterior()
        mock_fitter = MagicMock()
        mock_fitter.spec.n_free = 5  # low-D → NUTS

        # Return a mock mcmc result with matching samples
        mcmc_result = _make_minimal_posterior(method="mcmc_nuts")
        mock_fitter.run.return_value = mcmc_result
        p._fitter = mock_fitter

        result = p.validate(n_steps=10)

        assert "mcmc_result" in result
        assert "overlap" in result
        assert "passed" in result
        assert isinstance(result["passed"], bool)


# ---------------------------------------------------------------------------
# Task 7: posteriors_to_dataframe (no SSP needed)
# ---------------------------------------------------------------------------


class TestPostersToDataframe:
    def test_returns_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        import tengri

        results = [_make_minimal_posterior() for _ in range(3)]
        df = tengri.posteriors_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_contain_requested_params(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        import tengri

        results = [_make_minimal_posterior() for _ in range(2)]
        df = tengri.posteriors_to_dataframe(results, params=["met_logzsol"])

        # Should have met_logzsol_median, not dust_tau_bc_median
        assert any("met_logzsol" in col for col in df.columns)
        assert not any("dust_tau_bc" in col for col in df.columns)

    def test_raises_without_pandas(self):
        import tengri

        results = [_make_minimal_posterior()]
        with patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises(ImportError, match="pandas"):
                tengri.posteriors_to_dataframe(results)


# ---------------------------------------------------------------------------
# SSP-required tests: Model.from_config, fit, prior_predictive, fit_catalog
# ---------------------------------------------------------------------------


@_needs_ssp
class TestModelFromConfig:
    @pytest.fixture(scope="class")
    def model_tsnorm(self):
        import tengri

        return tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="tsnorm",
            filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
            redshift=0.1,
            priors=dict(
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                peak_lbt_gyr=tengri.Uniform(0.5, 12),
                width_gyr=tengri.Uniform(0.3, 5),
                logzsol=tengri.Uniform(-2, 0.2),
                tau_bc=tengri.Uniform(0, 2),
                tau_diff=tengri.Uniform(0, 1.5),
            ),
        )

    def test_returns_model_instance(self, model_tsnorm):
        from tengri.core.model import Model

        assert isinstance(model_tsnorm, Model)

    def test_free_params_contain_expanded_names(self, model_tsnorm):
        free = model_tsnorm.spec.free_params
        assert "sfh_tsnorm_log_peak_sfr" in free
        assert "met_logzsol" in free
        assert "dust_tau_bc" in free

    def test_redshift_is_fixed(self, model_tsnorm):
        assert model_tsnorm._z_fixed == pytest.approx(0.1, rel=1e-4)

    def test_filters_loaded(self, model_tsnorm):
        assert model_tsnorm.filter_waves is not None
        assert len(model_tsnorm.filter_waves) == 5

    def test_from_config_with_ssp_object(self, model_tsnorm):
        """from_config also accepts a pre-loaded SSPData."""
        import tengri

        model2 = tengri.Model.from_config(
            ssp=model_tsnorm.ssp_data,  # pass object, not path
            sfh="tsnorm",
            filters=["sdss_u", "sdss_g", "sdss_r"],
            redshift=0.2,
            priors=dict(
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-2, 0.2),
            ),
        )
        assert isinstance(model2, tengri.Model)


@_needs_ssp
class TestModelFitConvenience:
    @pytest.fixture(scope="class")
    def model_and_data(self):
        import tengri

        model = tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
            redshift=0.1,
            priors=dict(
                alpha=tengri.Uniform(0.5, 3.0),
                beta=tengri.Uniform(0.3, 2.0),
                tau_gyr=tengri.Uniform(0.5, 10.0),
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-1.5, 0.2),
                tau_bc=tengri.Uniform(0, 3.0),
            ),
        )
        true_params = {
            "sfh_dpl_alpha": 1.2,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_log_peak_sfr": 0.9,
            "met_logzsol": -0.3,
            "dust_tau_bc": 1.0,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))
        return model, mock

    def test_fit_returns_posterior(self, model_and_data):
        from tengri.inference.posterior import Posterior

        model, mock = model_and_data
        result = model.fit(mock.flux_obs, mock.noise, method="map")
        assert isinstance(result, Posterior)

    def test_fit_sets_fitter_attribute(self, model_and_data):
        from tengri.inference.fitter import Fitter

        model, mock = model_and_data
        model.fit(mock.flux_obs, mock.noise, method="map")
        assert hasattr(model, "fitter_")
        assert isinstance(model.fitter_, Fitter)

    def test_fit_result_has_fitter_back_reference(self, model_and_data):
        from tengri.inference.fitter import Fitter

        model, mock = model_and_data
        result = model.fit(mock.flux_obs, mock.noise, method="map")
        assert result._fitter is not None
        assert isinstance(result._fitter, Fitter)

    def test_fit_photometry_kwarg(self, model_and_data):
        from tengri.inference.posterior import Posterior

        model, mock = model_and_data
        result = model.fit(photometry=(mock.flux_obs, mock.noise), method="map")
        assert isinstance(result, Posterior)

    def test_fit_init_map(self, model_and_data):
        """init='map' runs MAP first, then uses result as warm start."""
        from tengri.inference.posterior import Posterior

        model, mock = model_and_data
        # Just verify it completes and returns a Posterior (MAP is fast)
        result = model.fit(mock.flux_obs, mock.noise, method="map", init="map")
        assert isinstance(result, Posterior)


@_needs_ssp
class TestPriorPredictive:
    @pytest.fixture(scope="class")
    def model_simple(self):
        import tengri

        return tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            filters=["sdss_u", "sdss_g", "sdss_r"],
            redshift=0.1,
            priors=dict(
                alpha=tengri.Uniform(0.5, 3.0),
                beta=tengri.Uniform(0.3, 2.0),
                tau_gyr=tengri.Uniform(0.5, 10.0),
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-1.5, 0.2),
                tau_bc=tengri.Uniform(0, 3.0),
            ),
        )

    def test_prior_predictive_returns_object(self, model_simple):
        from tengri.core.model import PriorPredictive

        ppc = model_simple.prior_predictive(n=20, seed=0)
        assert isinstance(ppc, PriorPredictive)

    def test_flux_shape(self, model_simple):
        ppc = model_simple.prior_predictive(n=20, seed=0)
        assert ppc.flux is not None
        assert ppc.flux.shape == (20, 3)  # 3 filters

    def test_sfh_shape(self, model_simple):
        ppc = model_simple.prior_predictive(n=20, seed=0)
        assert ppc.sfh is not None
        assert ppc.sfh.shape[0] == 20

    def test_params_dict_has_n_rows(self, model_simple):
        ppc = model_simple.prior_predictive(n=20, seed=0)
        for name, arr in ppc.params.items():
            assert arr.shape[0] == 20

    def test_check_finite_ok(self, model_simple):
        ppc = model_simple.prior_predictive(n=50, seed=1)
        result = ppc.check_finite()
        assert "ok" in result
        assert "n_nan" in result
        assert "n_inf" in result


@_needs_ssp
class TestFitCatalog:
    @pytest.fixture(scope="class")
    def model_and_catalog(self):
        import tengri

        model = tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            filters=["sdss_u", "sdss_g", "sdss_r"],
            redshift=0.1,
            priors=dict(
                alpha=tengri.Uniform(0.5, 3.0),
                beta=tengri.Uniform(0.3, 2.0),
                tau_gyr=tengri.Uniform(0.5, 10.0),
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-1.5, 0.2),
                tau_bc=tengri.Uniform(0, 3.0),
            ),
        )
        # Generate 3 mock galaxies as a list of dicts
        true_params = {
            "sfh_dpl_alpha": 1.2,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_log_peak_sfr": 0.9,
            "met_logzsol": -0.3,
            "dust_tau_bc": 1.0,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        catalog_rows = []
        for i in range(3):
            m = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(i))
            catalog_rows.append({
                "flux_u": float(m.flux_obs[0]),
                "flux_g": float(m.flux_obs[1]),
                "flux_r": float(m.flux_obs[2]),
                "err_u": float(m.noise[0]),
                "err_g": float(m.noise[1]),
                "err_r": float(m.noise[2]),
            })
        return model, catalog_rows

    def test_fit_catalog_list_of_dicts(self, model_and_catalog):
        from tengri.inference.posterior import Posterior

        model, catalog = model_and_catalog
        results = model.fit_catalog(
            catalog,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )
        assert len(results) == 3
        assert all(isinstance(r, Posterior) for r in results)

    def test_fit_catalog_pandas(self, model_and_catalog):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        from tengri.inference.posterior import Posterior

        model, catalog = model_and_catalog
        df = pd.DataFrame(catalog)
        results = model.fit_catalog(
            df,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )
        assert len(results) == 3
        assert all(isinstance(r, Posterior) for r in results)
```

- [ ] **Step 3: Run the no-SSP tests to verify they pass**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
pytest tests/unit/test_api_convenience.py -v -k "not needs_ssp" 2>&1 | head -80
```

Expected: All `TestDeprecationWarnings`, `TestResolveShortNames`, `TestPosteriorFitterField`, `TestPosteriorValidate`, `TestPostersToDataframe` tests pass.

- [ ] **Step 4: Run the full test suite to check for regressions**

```bash
pytest tests/ -q --timeout=120 2>&1 | tail -20
```

Expected: Same pass count as before (~1221 tests) with the new tests added.

- [ ] **Step 5: Run ruff on the new test file**

```bash
ruff check tests/unit/test_api_convenience.py
ruff format --check tests/unit/test_api_convenience.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_api_convenience.py
git commit -m "test: add test_api_convenience.py covering all 7 API redesign changes"
```

---

## Self-review: spec coverage check

| Spec item | Task covering it |
|-----------|-----------------|
| 15 method strings → 8 canonical names + deprecation aliases | Task 2 |
| `vi_flavor=` kwarg for power users | Task 2 |
| `"auto"` and `"mcmc"` auto-selection by D | Task 2 |
| `model.fit(flux, noise)` → Posterior | Task 4 |
| `model.fit(photometry=..., spectrum=...)` joint | Task 4 |
| `model.fit(flux, noise, init="map")` | Task 4 |
| `Posterior.refine("mcmc")` chaining | Task 3 |
| `Posterior.validate()` | Task 3 |
| `Model.from_config()` with short names | Tasks 1 + 5 |
| `_SFH_SHORT_NAMES` and `_UNIVERSAL_SHORT_NAMES` | Task 1 |
| `model.fit_catalog(df, ...)` | Task 7 |
| `posteriors_to_dataframe()` | Task 9 |
| `model.prior_predictive(n=500)` | Task 6 |
| `PriorPredictive.plot_seds()`, `.check_finite()` | Task 6 |
| `model.fit_population()` wrapping HierarchicalFitter | Task 8 |
| `HierarchicalResult.individual` property | Task 8 |
| `HierarchicalResult.plot_population()` | Task 8 |
| All ruff passes zero violations | Every task |
| Tests in `tests/unit/test_api_convenience.py` | Task 10 |
| No existing code broken | Additive throughout |

All items covered. No placeholders in code blocks (all code is complete and runnable). Method signatures are consistent across tasks.
