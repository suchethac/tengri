# SPDX-License-Identifier: BSD-3-Clause
"""Fractional star-formation timing parameters (#2748).

A timing parameter that anchors a star-formation onset or age to the age of
the universe at the galaxy's redshift is sampled as a fraction ``frac`` of
``age(z)`` in (0, 1], so its support can never reach past the Big Bang at any
redshift. :func:`timing_gyr` converts it back to absolute Gyr for the SFH
kernels, which are unchanged.
"""

from __future__ import annotations

import functools


@functools.cache
def timing_frac_params() -> dict[str, str]:
    """Fractional timing parameter -> the absolute-Gyr parameter it replaces (#2748).

    Derived from the SFH registry: every ParamDef whose ``gyr_twin`` is set. The
    Gyr name keeps its ``internal_param_map`` entry and its role as the Fixed
    shim; the translate post-pass writes ``frac * age(z)`` under its internal
    name when the fraction is sampled.

    Returns
    -------
    dict of str to str
        ``frac public name -> Gyr public name``.
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    return {
        name: pdef.gyr_twin
        for spec in SFH_REGISTRY.values()
        for name, pdef in spec.params.items()
        if pdef.gyr_twin is not None
    }


def timing_gyr(*, frac=None, fixed_gyr=None, age_universe_gyr):
    """Absolute timing [Gyr]: ``fixed_gyr`` if given, else ``frac * age(z)``.

    Parameters
    ----------
    frac : float or array_like, optional
        Fraction of the age of the universe at the galaxy's redshift, in
        (0, 1]. May be traced.
    fixed_gyr : float, optional
        An absolute Gyr value (the Fixed shim). Takes precedence over ``frac``.
    age_universe_gyr : float or array_like
        Age of the universe at the galaxy's redshift [Gyr]. May be traced.

    Returns
    -------
    float or array_like
        The timing [Gyr].

    Notes
    -----
    **JIT-compatible**: yes; a pure product, no Python branch on traced values.
    """
    if fixed_gyr is not None:
        return fixed_gyr
    return frac * age_universe_gyr


#: Provenance tags of a key the user named explicitly (not a wildcard or default).
_USER_NAMED = ("user_fixed", "user_prior", "user_free")


@functools.cache
def timing_frac_defaults() -> dict[str, float]:
    """Registry default of every ``*_frac`` timing, as a fraction of age(z) (#2748)."""
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    return {
        name: float(pdef.default.default)
        for spec in SFH_REGISTRY.values()
        for name, pdef in spec.params.items()
        if pdef.gyr_twin is not None
    }


def settle_timing_pairs(resolved, provenance, *, redshift_fixed=None):
    """Keep exactly one of each ``*_frac`` / ``*_gyr`` timing pair (#2748).

    Decided at build time from provenance. A Gyr name the user fixed keeps the
    absolute Gyr value and drops its fraction. Any other case keeps the fraction
    (default or wildcard-freed) and drops the Gyr name. Naming both, or giving
    the Gyr name a free prior, is a :class:`ConfigError`.

    Parameters
    ----------
    resolved : dict
        Resolved ``public_name -> Distribution``; updated in place.
    provenance : dict
        ``public_name -> provenance tag``; updated in place.
    redshift_fixed : float or None
        The build redshift when it is Fixed, else None. A Fixed Gyr shim above
        age(z) is refused at build.

    Returns
    -------
    frozenset of str
        The public names dropped from ``resolved``, which the built spec must
        also leave out.

    Raises
    ------
    ConfigError
        If both names are user-given, if a Gyr name carries a free prior, or if
        a Fixed Gyr shim exceeds the age of the universe at a Fixed redshift.
    """
    from tengri.config.exceptions import ConfigError

    dropped = set()
    for frac_name, gyr_name in timing_frac_params().items():
        if frac_name not in resolved and gyr_name not in resolved:
            continue
        gyr_user = gyr_name in resolved and provenance.get(gyr_name) in _USER_NAMED
        frac_user = frac_name in resolved and provenance.get(frac_name) in _USER_NAMED
        if gyr_user and frac_user:
            raise ConfigError(
                f"give one of {gyr_name!r} or {frac_name!r}, not both: both are named"
            )
        if not gyr_user:
            if frac_name in resolved and gyr_name in resolved:
                resolved.pop(gyr_name, None)
                provenance.pop(gyr_name, None)
                dropped.add(gyr_name)
            continue
        gyr_dist = resolved[gyr_name]
        if not gyr_dist.is_fixed:
            check_free_gyr_alias(gyr_name, gyr_dist)
        if redshift_fixed is not None:
            _refuse_gyr_above_age(gyr_name, float(gyr_dist.value), redshift_fixed)
        resolved.pop(frac_name, None)
        provenance.pop(frac_name, None)
        dropped.add(frac_name)
    return frozenset(dropped)


def _gyr_to_frac() -> dict[str, str]:
    """Absolute-Gyr timing alias -> the fraction that replaces it (#2748)."""
    return {gyr: frac for frac, gyr in timing_frac_params().items()}


def check_free_gyr_alias(name, value):
    """Refuse a free request on an old ``*_gyr`` timing alias (#2748).

    The Gyr names are Fixed-only shims: a free onset is sampled as its
    ``*_frac`` of age(z). ``name`` is checked against the ``gyr_twin`` table, so
    a non-alias name passes through untouched.

    Parameters
    ----------
    name : str
        Public parameter name, flat or grouped spelling resolved to it.
    value : object
        The requested value: ``FREE``, a Distribution, or any other input.

    Raises
    ------
    ConfigError
        If ``name`` is a Gyr alias and ``value`` is ``FREE`` or a
        non-Fixed Distribution. The message names the fraction and converts
        the Gyr prior to fractions of the age of the universe at z = 0.
    """
    from tengri.config.exceptions import ConfigError
    from tengri.parameters.priors import Distribution, Uniform
    from tengri.parameters.sentinels import FREE

    frac_name = _gyr_to_frac().get(name)
    if frac_name is None:
        return
    is_free = value is FREE or (isinstance(value, Distribution) and not value.is_fixed)
    if not is_free:
        return
    from tengri.components.stellar.sfh.registry import _AGE_FRAC_PRIORS, _age_frac

    if isinstance(value, Uniform):
        lo_gyr, hi_gyr = value.bounds
        converted = (
            f"Uniform({lo_gyr:g}, {hi_gyr:g}) Gyr at z = 0 is "
            f"Uniform({_age_frac(lo_gyr):.4g}, {_age_frac(hi_gyr):.4g})"
        )
    else:
        lo, hi = _AGE_FRAC_PRIORS[frac_name].bounds
        converted = f"the free prior of {frac_name} is Uniform({lo:.4g}, {hi:.4g})"
    raise ConfigError(
        f"{name} is now a Fixed-only alias: free the fraction {frac_name} instead "
        f"(age = frac * age_of_universe(z)); {converted} (#2748)."
    )


def timing_names_to_exclude(user_names, valid_names):
    """Timing names a flat ``Parameters`` must leave out of its spec (#2748).

    The same one-name rule as :func:`settle_timing_pairs`, for a spec built
    without ``parse_groups``: a named Gyr keeps its Gyr and drops the fraction;
    otherwise the fraction is kept and the Gyr dropped.

    Parameters
    ----------
    user_names : collection of str
        Names the caller supplied.
    valid_names : collection of str
        Names the spec would otherwise declare.

    Returns
    -------
    frozenset of str
        The names to leave out.

    Raises
    ------
    ConfigError
        If both names of a pair are supplied.
    """
    from tengri.config.exceptions import ConfigError

    user = set(user_names)
    valid = set(valid_names)
    excluded = set()
    for frac_name, gyr_name in timing_frac_params().items():
        if frac_name not in valid or gyr_name not in valid:
            continue
        if frac_name in user and gyr_name in user:
            raise ConfigError(
                f"give one of {gyr_name!r} or {frac_name!r}, not both: both are named"
            )
        excluded.add(frac_name if gyr_name in user else gyr_name)
    return frozenset(excluded)


def _refuse_gyr_above_age(gyr_name, gyr, z):
    """Refuse a Fixed Gyr onset later than the age of the universe at a Fixed z (#2748)."""
    from tengri.config.exceptions import ConfigError
    from tengri.cosmology import age_at_z

    age = float(age_at_z(z))
    if gyr > age:
        raise ConfigError(
            f"{gyr_name!r}={gyr} Gyr is later than the age of the universe at "
            f"z={z:g} ({age:.3f} Gyr); the onset cannot precede the Big Bang (#2748)."
        )


def resolve_timing_fracs(internal, params, param_map, *, redshift):
    """Write the absolute Gyr of every present ``*_frac`` timing (#2748).

    A fraction in ``params`` (sampled or at its default) sets the internal Gyr
    key to ``frac * age(z)``, scaled by the Gyr entry's own scale. A Gyr name
    alone in ``params`` is the absolute shim and passes through as the caller
    filled it. The fraction's own internal entries are always removed.

    Parameters
    ----------
    internal : dict
        Internal parameter dict, updated in place.
    params : dict
        Public parameter dict holding the fraction or the Gyr name, not both.
    param_map : dict
        ``public_name -> (internal_name, scale, offset)``.
    redshift : float or array_like
        Redshift of this evaluation [dimensionless]. May be traced.

    Raises
    ------
    ConfigError
        If ``redshift`` is None while a fraction is present.

    Notes
    -----
    **JIT-compatible**: yes, provided ``redshift`` is traced or concrete. The
    key tests are Python-level and static.
    """
    from tengri.config.exceptions import ConfigError
    from tengri.cosmology import age_at_z

    defaults = timing_frac_defaults()
    for frac_name, gyr_name in timing_frac_params().items():
        if frac_name not in param_map or gyr_name not in param_map:
            continue
        frac_int = param_map[frac_name][0]
        gyr_int, gyr_scale, _ = param_map[gyr_name]
        internal.pop(frac_int, None)
        internal.pop(frac_name, None)
        if gyr_name in params:
            if frac_name in params:
                raise ConfigError(f"give one of {gyr_name!r} or {frac_name!r}, not both")
            continue
        if redshift is None:
            raise ConfigError(
                f"{frac_name!r} needs the build redshift to convert to Gyr, and "
                "'redshift' is absent from the parameters"
            )
        frac = params[frac_name] if frac_name in params else defaults[frac_name]
        gyr = timing_gyr(frac=frac, age_universe_gyr=age_at_z(redshift))
        internal[gyr_int] = gyr * gyr_scale
        if gyr_name in internal:
            # Composite SFHs also carry the public name (see translate); a
            # plain SFH does not, and its kernel rejects unknown keyword names.
            internal[gyr_name] = internal[gyr_int]


def route_timing_fracs(full_params, sampled):
    """Drop the Gyr name of every timing whose fraction is present (#2748).

    The build keeps exactly one of each pair, so this only matters for a
    caller that merges both. A fraction wins, as it is the default parameter.

    Parameters
    ----------
    full_params : dict
        Merged parameter dict (Fixed values and sampled values together).
    sampled : dict
        The sampled (non-Fixed) parameters alone.

    Returns
    -------
    dict
        A new dict; ``full_params`` is not modified.

    Raises
    ------
    ConfigError
        If a fraction and its Gyr name are both sampled.
    """
    from tengri.config.exceptions import ConfigError

    out = dict(full_params)
    for frac_name, gyr_name in timing_frac_params().items():
        if frac_name in sampled and gyr_name in sampled:
            raise ConfigError(
                f"give one of {gyr_name!r} or {frac_name!r}, not both: both are sampled"
            )
        if frac_name in out:
            out.pop(gyr_name, None)
    return out
