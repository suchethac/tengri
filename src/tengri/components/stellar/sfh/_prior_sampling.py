# SPDX-License-Identifier: BSD-3-Clause
"""Prior-sampling helper for SFH families, with or without the GP field.

Lets a user draw a fan of SFH realizations from a registered model's default
priors with one call::

    import jax
    from tengri.components.stellar.sfh import sample_sfh_prior

    age_grid, curves = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=20)
    # age_grid: (n_age,) lookback time [yr]
    # curves:   (n, n_age) SFR(t_lookback) [Msun/yr]

The draws are public-name parameter dicts, evaluated by the same
:func:`~tengri.components.stellar.component.evaluate_sfh` the forward model uses,
so a prior fan and an exact forward cannot read a family differently (#3505).

Designed for didactic notebooks (plot-the-prior cells) and quick visual
sanity checks. Inference pipelines should keep using ``Parameters`` +
``Fitter`` rather than this helper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.sfh.nonparametric import DEFAULT_N_BINS, make_agebins_from_zred
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.parameters.priors import Distribution
from tengri.utils.grid import DEFAULT_LOG_AGE_MAX, DEFAULT_LOG_AGE_MIN, DEFAULT_N_GRID
from tengri.utils.host_array import host_array

__all__ = ["DEFAULT_AGE_GRID_YR", "sample_sfh_prior"]


# Host-side twin of ``10.0 ** make_log_age_grid(n_grid=DEFAULT_N_GRID)`` (#2271):
# a module-scope jnp array is a device buffer whose dtype is fixed at import,
# which a float64-less backend cannot even allocate. A numpy grid is converted at
# the op, in the working dtype. Pinned equal to the helper by test_sfh_prior_sampling.
DEFAULT_AGE_GRID_YR: np.ndarray = host_array(
    10.0 ** np.linspace(DEFAULT_LOG_AGE_MIN, DEFAULT_LOG_AGE_MAX, DEFAULT_N_GRID)
)

#: Draws evaluated together by :func:`sample_sfh_prior`; bounds its peak memory.
_DRAW_CHUNK = 64

#: The GP-field modulator's registry key: a modulator, not a family.
_FIELD = "field"


def _spec_for(name: str):
    entry = SFH_REGISTRY.get(name)
    if entry is None:
        valid = sorted(SFH_REGISTRY.keys())
        raise KeyError(f"Unknown SFH model '{name}'. Registered: {valid}")
    # SFHRegistryEntry forwards attribute access to the underlying SFHModelSpec.
    return entry


def _bin_edges_for(family: str | list[str], redshift: float | None):
    """Age-bin edges the forward model uses at ``redshift``, or None for the default.

    The binned families are built on the ``make_agebins_from_zred`` ladder, whose
    last edge is age(z), as the forward model builds it. ``None`` and ``0`` keep
    the default ladder, which already ends inside age(0) (#3455).
    """
    from tengri.parameters.groups import _Z_SCALED_DEFAULT_BIN_LADDER_FAMILIES

    members = [family] if isinstance(family, str) else list(family)
    if not (_Z_SCALED_DEFAULT_BIN_LADDER_FAMILIES & set(members)):
        return None
    if not redshift:
        return None
    return make_agebins_from_zred(zred=redshift, n_bins=DEFAULT_N_BINS)


def _draw_publics(
    names: list[str],
    key: jax.Array,
    n: int,
    overrides: dict[str, Distribution],
) -> tuple[jax.Array, dict[str, jnp.ndarray]]:
    """Draw ``n`` values of every declared parameter of ``names``, keyed by public name.

    The key sequence is the one the helper has always used, so the draws of an
    existing family are unchanged.
    """
    publics: dict[str, jnp.ndarray] = {}
    for name in names:
        for public_name, paramdef in _spec_for(name).params.items():
            dist = overrides.get(public_name, paramdef.default)
            keys = jax.random.split(key, n + 1)
            key = keys[0]
            publics[public_name] = jnp.stack([dist.sample(k) for k in keys[1:]])
    return key, publics


def _prior_draws(
    family_list: list[str],
    key: jax.Array,
    n: int,
    overrides: dict[str, Distribution],
    *,
    n_grid: int,
) -> dict[str, jnp.ndarray]:
    """Public-name parameter dict with a leading ``n`` axis, from the priors.

    A field family adds the modulator's PSD parameters and the latent
    ``sfh_field_xi ~ N(0, 1)`` of shape ``(n, n_grid)``.
    """
    families = [f for f in family_list if f != _FIELD]
    key, publics = _draw_publics(families, key, n, overrides)
    # A timing is either its Gyr value or its *_frac, never both (the resolver
    # refuses both). A frac is converted unless its Gyr twin is overridden; when
    # it is converted the frac wins, as the Gyr twin's draw was always overwritten.
    for name in families:
        for public_name, paramdef in _spec_for(name).params.items():
            if paramdef.gyr_twin is None:
                continue
            if paramdef.gyr_twin in overrides:
                publics.pop(public_name, None)
            else:
                publics.pop(paramdef.gyr_twin, None)
    if _FIELD in family_list:
        key, field_publics = _draw_publics([_FIELD], key, n, overrides)
        publics.update(field_publics)
        key, xi_key = jax.random.split(key)
        publics["sfh_field_xi"] = jax.random.normal(xi_key, (n, n_grid))
    return publics


def sample_sfh_prior(
    family: str | list[str],
    key: jax.Array,
    n: int = 20,
    age_grid_yr: jnp.ndarray | None = None,
    *,
    redshift: float | None = None,
    **prior_overrides: Distribution,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw `n` SFH realizations from a registered family's default prior.

    Parameters
    ----------
    family : str or list[str]
        Registered SFH model name (e.g. ``"tsnorm"``, ``"dpl"``) or a
        composed list (e.g. ``["tsnorm", "burst"]``). Add ``"field"`` to the
        list for the GP modulator: each draw then carries its own latent field
        ``sfh_field_xi ~ N(0, 1)`` and the PSD priors of the modulator.
    key : jax.Array
        JAX PRNG key.
    n : int, optional
        Number of prior draws. Default 20.
    age_grid_yr : array_like, shape (n_age,), optional
        Lookback time grid [yr]. Defaults to a 256-point log grid spanning
        1 Myr to ~13.8 Gyr (matches :func:`make_log_age_grid`).
    redshift : float, optional
        Redshift at which the draws are evaluated. Required for the
        age-anchored families (``dense_basis``, ``dense_basis_pure``, ``psb``,
        and composites containing one), whose time axis is set by the age of
        the universe at the galaxy's redshift; there is no default, because
        no single age is right for every use. The binned families
        (``continuity``, ``dirichlet``, ...) take their age-bin ladder from
        the same redshift, so no bin lies beyond age(z) (#3455). Ignored by
        the other families.
    **prior_overrides : Distribution
        Per-parameter prior overrides keyed by the *public* parameter name
        (e.g. ``sfh_dpl_alpha=Uniform(0.5, 2.0)``). Anything not overridden
        uses the registry default from :data:`SFH_REGISTRY`.

    Returns
    -------
    age_grid_yr : ndarray, shape (n_age,)
        Lookback time grid [yr].
    sfr_curves : ndarray, shape (n, n_age)
        SFR(t_lookback) [Msun/yr] for each prior draw. With the field, the
        modulated history: the field is defined on its own log grid, so keep the
        default grid for an exact evaluation (another grid is interpolated).

    Raises
    ------
    KeyError
        If `family` is not a registered SFH model.
    ValueError
        If `family` is age-anchored and `redshift` is not given, or if it names
        only the ``"field"`` modulator with no SFH to modulate.

    Notes
    -----
    **JIT-compatible**: yes for the inner evaluation; the outer draws are a
    one-shot helper, not a hot path.

    For an age-anchored family the age of the universe at `redshift` comes
    from :func:`~tengri.components.stellar.component.age_universe_kwargs`, the
    rule the forward model uses. The default grid reaches 13.8 Gyr, so at
    ``redshift > 0`` it extends past age(z); the history there is zero for
    ``dense_basis`` and ``dense_basis_pure`` (no star formation before the
    Big Bang). For ``psb`` the age parameter is drawn from its prior
    independently of the redshift, so a draw can place star formation at
    lookback times beyond age(z). This helper evaluates the SFH without the
    age-of-universe truncation that the stellar component applies to the
    model's history.

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
    tengri.components.stellar.component.evaluate_sfh : the one SFH evaluation the
        draws go through.
    """
    from tengri.components.stellar.component import (
        StellarSEDComponentConfig,
        age_universe_kwargs,
        evaluate_sfh,
    )

    if age_grid_yr is None:
        age_grid_yr = DEFAULT_AGE_GRID_YR
    family_list = [family] if isinstance(family, str) else list(family)
    families = [f for f in family_list if f != _FIELD]
    if not families:
        raise ValueError("sample_sfh_prior needs an SFH family besides the 'field' modulator.")
    for name in family_list:
        _spec_for(name)

    if redshift is None and any(age_universe_kwargs(f, 0.0) for f in families):
        raise ValueError(
            f"sample_sfh_prior({family!r}) needs redshift=: the family's time axis is "
            "anchored to the age of the universe at the galaxy's redshift."
        )
    z = 0.0 if redshift is None else float(redshift)

    n_grid = DEFAULT_N_GRID
    config = StellarSEDComponentConfig(
        sfh_model=families[0] if len(families) == 1 else families,
        field=_FIELD in family_list,
        n_grid=n_grid,
        sfh_bin_edges_gyr=_bin_edges_for(families, redshift),
    )
    publics = _prior_draws(family_list, key, n, prior_overrides, n_grid=n_grid)
    redshift_value = jnp.asarray(z)

    def _one(params_i: dict) -> jnp.ndarray:
        params_i = {**params_i, "redshift": redshift_value}
        return evaluate_sfh(config, params_i, jnp.asarray(age_grid_yr), n_grid).sfr_on_ssp

    # Chunked vmap: each draw's evaluation state (the SSP-age integrand, the field
    # latent and its history) is held for ``_DRAW_CHUNK`` draws at a time, not for all ``n``.
    sfr_curves = jax.lax.map(_one, publics, batch_size=_DRAW_CHUNK)
    return age_grid_yr, sfr_curves
