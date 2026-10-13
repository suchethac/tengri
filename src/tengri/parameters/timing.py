# SPDX-License-Identifier: BSD-3-Clause
"""Fractional star-formation timing parameters (#2748).

A timing parameter that anchors a star-formation onset or age to the age of
the universe at the galaxy's redshift is sampled as a fraction ``frac`` of
``age(z)`` in (0, 1], so its support can never reach past the Big Bang at any
redshift. :func:`timing_gyr` converts it back to absolute Gyr for the SFH
kernels, which are unchanged.
"""

from __future__ import annotations

#: Fractional timing parameter -> the absolute-Gyr parameter it replaces. The
#: Gyr name keeps its ``internal_param_map`` entry and its role as the Fixed
#: shim; the translate post-pass writes ``frac * age(z)`` under its internal
#: name when the fraction is sampled.
TIMING_FRAC_PARAMS: dict[str, str] = {
    "sfh_dpl_age_frac": "sfh_dpl_age_gyr",
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


def resolve_timing_fracs(internal, params, param_map, *, redshift):
    """Replace sampled ``*_frac`` timings with Gyr values, in place (#2748).

    For each :data:`TIMING_FRAC_PARAMS` entry present in ``param_map``, the
    fraction's own internal entries are removed. If the fraction was sampled
    (present in ``params``), the Gyr internal name is written as
    ``frac * age(z)`` at ``redshift``, scaled by the Gyr entry's own scale.
    Otherwise the Gyr entry is left as the caller filled it.

    Parameters
    ----------
    internal : dict
        Internal parameter dict, updated in place.
    params : dict
        Public parameter dict; a fraction is sampled iff its name is a key.
    param_map : dict
        ``public_name -> (internal_name, scale, offset)``.
    redshift : float or array_like
        Redshift of this evaluation [dimensionless]. May be traced.

    Raises
    ------
    ConfigError
        If ``redshift`` is None.

    Notes
    -----
    **JIT-compatible**: yes, provided ``redshift`` is traced or concrete. The
    key tests are Python-level and static. The both-sampled refusal lives in
    :func:`route_timing_fracs` and in the translate wrapper, which see the
    sampled parameters alone.
    """
    from tengri.config.exceptions import ConfigError
    from tengri.cosmology import age_at_z

    for frac_name, gyr_name in TIMING_FRAC_PARAMS.items():
        if frac_name not in param_map or gyr_name not in param_map:
            continue
        frac_int = param_map[frac_name][0]
        gyr_int, gyr_scale, _ = param_map[gyr_name]
        internal.pop(frac_int, None)
        internal.pop(frac_name, None)
        if frac_name not in params:
            continue
        if redshift is None:
            raise ConfigError(
                f"{frac_name!r} needs the build redshift to convert to Gyr, and "
                "'redshift' is absent from the parameters"
            )
        gyr = timing_gyr(frac=params[frac_name], age_universe_gyr=age_at_z(redshift))
        internal[gyr_int] = gyr * gyr_scale
        if gyr_name in internal:
            # Composite SFHs also carry the public name (see translate); a
            # plain SFH does not, and its kernel rejects unknown keyword names.
            internal[gyr_name] = internal[gyr_int]


def route_timing_fracs(full_params, sampled):
    """Route fixed and sampled ``*_frac`` timings before the SFH is evaluated (#2748).

    ``predict_state`` merges every Fixed value into the parameter dict, so a
    fraction's presence there does not show that it was sampled. This keeps
    the two cases apart: a sampled fraction (a key of ``sampled``) replaces the
    absolute Gyr value of the same timing, and an unsampled fraction is
    removed so the Gyr value stands.

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
        If a fraction and its Gyr name are both sampled, or if a Fixed fraction
        differs from its default of 1 (not yet supported).
    """
    from tengri.config.exceptions import ConfigError

    out = dict(full_params)
    for frac_name, gyr_name in TIMING_FRAC_PARAMS.items():
        if frac_name in sampled:
            if gyr_name in sampled:
                raise ConfigError(
                    f"give one of {gyr_name!r} or {frac_name!r}, not both: both are sampled"
                )
            out.pop(gyr_name, None)
            continue
        # Only a concrete Fixed value is refused: a traced fraction arriving
        # through a JIT ``fixed_values`` override must not be concretized here.
        value = out.get(frac_name)
        if isinstance(value, (int, float)) and float(value) != 1.0:
            raise ConfigError(
                f"{frac_name!r} is Fixed({value!r}); a Fixed fraction is not "
                "supported yet, give a Uniform prior or set the absolute Gyr name"
            )
        out.pop(frac_name, None)
    return out
