# SPDX-License-Identifier: BSD-3-Clause
"""Prior-sampling helper for parametric SFH families.

Lets a user draw a fan of SFH realizations from a registered model's default
priors with one call::

    import jax
    from tengri.components.stellar.sfh import sample_sfh_prior

    age_grid, curves = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=20)
    # age_grid: (n_age,) lookback time [yr]
    # curves:   (n, n_age) SFR(t_lookback) [Msun/yr]

Designed for didactic notebooks (plot-the-prior cells) and quick visual
sanity checks. Inference pipelines should keep using ``Parameters`` +
``Fitter`` rather than this helper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, resolve_sfh
from tengri.parameters.priors import Distribution

__all__ = ["DEFAULT_AGE_GRID_YR", "sample_sfh_prior"]


DEFAULT_AGE_GRID_YR: jnp.ndarray = 10.0 ** make_log_age_grid(n_grid=256)


def _spec_for(name: str):
    entry = SFH_REGISTRY.get(name)
    if entry is None:
        valid = sorted(SFH_REGISTRY.keys())
        raise KeyError(f"Unknown SFH model '{name}'. Registered: {valid}")
    # SFHRegistryEntry forwards attribute access to the underlying SFHModelSpec.
    return entry


def _draw_internal_kwargs(
    family: str | list[str],
    key: jax.Array,
    n: int,
    overrides: dict[str, Distribution],
) -> tuple[list[str], dict[str, jnp.ndarray]]:
    """Draw `n` samples for every fittable param of `family`, return internal kwargs.

    Returns
    -------
    family_list: list[str]
        Normalized list of model names (input as list, even if scalar str).
    internal_kwargs: dict[str, ndarray of shape (n,)]
        Dict keyed by the *internal* parameter name expected by the SFH closure.
    """
    family_list = [family] if isinstance(family, str) else list(family)

    if any(name in ("field",) for name in family_list):
        # Field modulator depends on a latent xi vector + GP machinery; not in v1 scope.
        raise NotImplementedError(
            "sample_sfh_prior does not yet support the 'field' (GP) modulator. "
            "Use a smooth family (e.g. 'tsnorm', 'dpl') for the prior fan; the "
            "stochastic-field demo lives in notebooks/02_sed_anatomy.py."
        )

    internal_kwargs: dict[str, jnp.ndarray] = {}

    for fname in family_list:
        spec = _spec_for(fname)
        for public_name, paramdef in spec.params.items():
            dist = overrides.get(public_name, paramdef.default)
            keys = jax.random.split(key, n + 1)
            key = keys[0]
            draws = jnp.stack([dist.sample(k) for k in keys[1:]])
            internal_name, scale, offset = spec.internal_param_map[public_name]
            internal_kwargs[internal_name] = draws * scale + offset

    return family_list, internal_kwargs


def sample_sfh_prior(
    family: str | list[str],
    key: jax.Array,
    n: int = 20,
    age_grid_yr: jnp.ndarray | None = None,
    **prior_overrides: Distribution,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw `n` SFH realizations from a registered family's default prior.

    Parameters
    ----------
    family: str or list[str]
        Registered SFH model name (e.g. ``"tsnorm"``, ``"dpl"``) or a
        composed list (e.g. ``["tsnorm", "burst"]``). The ``"field"``
        modulator is intentionally not supported here: see notes.
    key: jax.Array
        JAX PRNG key.
    n: int, optional
        Number of prior draws. Default 20.
    age_grid_yr: array_like, shape (n_age,), optional
        Lookback time grid [yr]. Defaults to a 256-point log grid spanning
        1 Myr to ~13.8 Gyr (matches :func:`make_log_age_grid`).
    **prior_overrides: Distribution
        Per-parameter prior overrides keyed by the *public* parameter name
        (e.g. ``sfh_dpl_alpha=Uniform(0.5, 2.0)``). Anything not overridden
        uses the registry default from :data:`SFH_REGISTRY`.

    Returns
    -------
    age_grid_yr: ndarray, shape (n_age,)
        Lookback time grid [yr].
    sfr_curves: ndarray, shape (n, n_age)
        SFR(t_lookback) [Msun/yr] for each prior draw.

    Raises
    ------
    KeyError
        If `family` is not a registered SFH model.
    NotImplementedError
        If `family` includes the ``"field"`` GP modulator.

    Notes
    -----
    **JIT-compatible**: yes for the inner evaluation; the outer Python loop
    over draws is intentional (one-shot helper, not a hot path).

    The returned curves are not mass-normalized: they are raw SFR(t) at the
    sampled parameter point. Use :math:`\\int \\mathrm{SFR}\\,\\mathrm{d}t`
    to recover total stellar mass formed if needed.

    Examples
    --------
    >>> import jax
    >>> from tengri.components.stellar.sfh import sample_sfh_prior
    >>> age, curves = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=5)
    >>> curves.shape
    (5, 256)

    Override one prior::

    >>> from tengri import Uniform
    >>> age, curves = sample_sfh_prior(
    ...     "dpl",
    ...     jax.random.PRNGKey(0),
    ...     n=5,
    ...     sfh_dpl_alpha=Uniform(0.5, 1.5),
    ... )

    See Also
    --------
    tengri.components.stellar.sfh.registry.resolve_sfh: the underlying
        model resolver used by the fitter.
    """
    if age_grid_yr is None:
        age_grid_yr = DEFAULT_AGE_GRID_YR

    composed_fn, _, _, settings = resolve_sfh(family)

    family_list, internal_kwargs = _draw_internal_kwargs(family, key, n, prior_overrides)
    # `settings` carries non-fittable knobs (e.g. sfh_field_ngrid). For the
    # families we currently support, it is empty and can be ignored.
    del family_list, settings

    def _one(kw_i: dict) -> jnp.ndarray:
        return composed_fn(age_grid_yr, **kw_i)

    # vmap over the leading "draws" axis of every internal kwarg.
    sfr_curves = jax.vmap(_one)(internal_kwargs)

    return age_grid_yr, sfr_curves
