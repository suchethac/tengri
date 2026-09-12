# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation key vocabulary and normalization.

This module is the single home of the dust attenuation grammar vocabulary:
SCREENS ('bc', 'diff', 'neb'), OVERRIDE_STEMS (the four law parameter stems),
SCREEN_CHOICES/SCREEN_SOURCES (the per-source screen-selector grammar, #2234),
and utilities to convert between spellings and validate them.

The grammar accepts two spellings for each parameter:
- Short form (stem): 'tau_bc', 'slope_diff', 'Rv_neb'
- Full form (registry name): 'dust_tau_bc', 'dust_slope_diff', 'dust_Rv_neb'

They are equivalent and normalize to one spelling early in parse_groups (Pass 0d),
before the structural translator, completeness check, and all later passes see
the dict. This central vocabulary prevents the three hand-lists (structural keys,
_resolve_value locals, and grammar normalization) from drifting apart.

Tests pin this vocabulary against TWO_COMPONENT_OVERRIDE_KEYS.

Separately, SCREEN_SOURCES ('nebular', 'shock', 'agn') names the emission
sources that pick a dust screen independently of the stars, and
:func:`resolve_screen_choices` is the one validator both the grammar
(``dust_attenuation={'nebular_screen': ...}``) and the flat surface
(``Parameters(dust_nebular_screen=...)``) call, so a request refused on one
surface is refused on the other for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping

DUST_PREFIX = "dust_"
SCREENS: tuple[str, ...] = ("bc", "diff", "neb")
SCREEN_LABEL = {"bc": "birth-cloud", "diff": "diffuse-ISM", "neb": "nebular birth-cloud"}
# The only home of this list (beyond tests); every other hand-list derives here.
OVERRIDE_STEMS: tuple[str, ...] = ("slope", "bump_strength", "delta", "Rv")

# ── Per-source dust-screen choice (#2234 replacement) ──────────────────
#
# Each non-stellar emission source (nebular continuum + line catalog, shock,
# AGN) picks which of the two-component screens attenuates it: the transmission
# the youngest stars get (``"birth_cloud"``: f_obscuration floor +
# exp(-(tau_bc*k_bc + tau_diff*k_diff))), the transmission the oldest stars get
# (``"diffuse"``: same floor, tau_diff*k_diff only), or no attenuation at all
# (``"none"``, resolved once at build time, never re-evaluated at predict).
#
# ``SCREEN_SOURCES`` and ``screen_keys()`` are the single home of "which
# sources have a screen choice" and "what the grammar/flat key is called for
# each": every other list (``_GROUP_STRUCTURAL_KEYS``, the flat-kwarg pops in
# ``Parameters._init_dust_config``, the grammar round-trip table) derives from
# them instead of hand-listing the three names again.
SCREEN_CHOICES: tuple[str, ...] = ("birth_cloud", "diffuse", "none")
# The ``'off'`` synonym for ``'none'`` is NOT spelled here: the off-switch
# vocabulary has exactly one home, ``tengri.parameters.groups._normalize_off_switch``
# (the helper every off-switch group's ``type`` goes through), and both
# surfaces -- ``_translate_dust_attenuation`` for the grammar and
# ``Parameters._init_dust_config`` for the flat kwargs -- pass each raw
# selector value through it BEFORE calling :func:`resolve_screen_choices`, so
# this module only ever sees the canonical spelling.
#: The three sources with a configurable screen. Order matches the tuple
#: ``screen_keys()`` returns and the per-source default in
#: :data:`_SCREEN_DEFAULTS`.
SCREEN_SOURCES: tuple[str, ...] = ("nebular", "shock", "agn")
#: Per-source default, read by both the grammar translator and the flat-kwarg
#: resolver so the two surfaces cannot drift. ``agn`` defaults to (and today
#: must stay) ``"none"``: the AGN component runs after dust in the pipeline
#: and carries its own polar-dust screen (see ``resolve_screen_choices``).
_SCREEN_DEFAULTS: dict[str, str] = {
    "nebular": "birth_cloud",
    "shock": "diffuse",
    "agn": "none",
}


def screen_keys() -> tuple[str, ...]:
    """Grammar/flat key name for each entry in :data:`SCREEN_SOURCES`.

    Returns
    -------
    tuple of str
        ``("nebular_screen", "shock_screen", "agn_screen")``. The flat-kwarg
        spelling prepends :data:`DUST_PREFIX` (``"dust_nebular_screen"``, ...);
        the grammar spelling is exactly this.
    """
    return tuple(f"{source}_screen" for source in SCREEN_SOURCES)


def normalize_screen_choice(value: str, *, key: str) -> str:
    """Check one dust-screen selector against :data:`SCREEN_CHOICES`.

    Parameters
    ----------
    value : str
        The choice as the calling surface hands it: one of
        :data:`SCREEN_CHOICES`. The ``'off'`` spelling of ``'none'`` is
        already canonicalized by then (``groups._normalize_off_switch``, the
        one home of the off-switch vocabulary, runs at the raw read on both
        surfaces), so it is neither accepted nor named here.
    key : str
        The key name to name in the error message (e.g. ``"nebular_screen"``
        or ``"dust_shock_screen"``), so the raised error points at exactly
        what the caller wrote.

    Returns
    -------
    str
        One of :data:`SCREEN_CHOICES`.

    Raises
    ------
    ParameterError
        If ``value`` is not a recognized choice. Names ``key`` and the three
        choices.
    """
    from tengri.config.exceptions import ParameterError

    if value not in SCREEN_CHOICES:
        raise ParameterError(
            f"{key}={value!r} is not a valid dust-screen choice. Choose one of {SCREEN_CHOICES!r}."
        )
    return value


def resolve_screen_choices(raw: Mapping[str, object], *, dust_model: str, surface: str) -> dict:
    """Validate the nebular_screen / shock_screen / agn_screen triad (#2234).

    THE single validator for both surfaces (mirrors :func:`validate_shape_requests`):
    the grammar (``tengri.parameters.groups._translate_dust_attenuation``) and the
    flat ``Parameters(...)`` constructor (:meth:`Parameters._init_dust_config`) both
    call this, so a request refused on one surface is refused on the other with the
    same reasoning.

    Sources absent from ``raw`` (value ``None``, i.e. the caller did not name that
    key) are left OUT of the returned dict entirely -- filling in the per-source
    default (:data:`_SCREEN_DEFAULTS`) is the caller's job. This function only
    tightens the story for a choice somebody actually asked for; it never invents a
    value and then rejects it as though the user had written it (that would make
    every dust-free/single-screen/WG00 model unbuildable, since a default sits in
    ``raw`` on every call otherwise).

    Parameters
    ----------
    raw : mapping
        ``{source: value_or_None}`` for zero or more of :data:`SCREEN_SOURCES`.
        A source not present in ``raw`` is treated the same as ``None``.
    dust_model : str
        The resolved dust attenuation model: ``"two_component"``,
        ``"single_component"``, ``"wg00"``, or ``"off"``.
    surface : str
        ``"grammar"`` (bare key, e.g. ``"nebular_screen"``) or ``"flat"``
        (prefixed key, e.g. ``"dust_nebular_screen"``) -- selects the spelling
        used in error messages.

    Returns
    -------
    dict
        ``{source: resolved_choice}``, one of :data:`SCREEN_CHOICES`, for
        every source the caller actually named. Sources not named are absent
        from the result.

    Raises
    ------
    ParameterError
        On an unrecognized value (see :func:`normalize_screen_choice`); on
        ``agn_screen`` resolving to anything but ``"none"`` (today the AGN
        component runs after dust and carries its own polar-dust screen, so
        galaxy screening of AGN light is not yet wired); on a two-screen-only
        key given ANY explicit value when ``dust_model in ("wg00", "off")``
        (there are no birth-cloud/diffuse screens to choose between); on a
        value other than ``"none"`` or the source's own default when
        ``dust_model == "single_component"`` (one screen, so there is no
        birth-cloud/diffuse distinction to select).
    """
    resolved: dict[str, str] = {}
    for source in SCREEN_SOURCES:
        if raw.get(source) is None:
            continue
        default = _SCREEN_DEFAULTS[source]
        grammar_key = f"{source}_screen"
        display_key = grammar_key if surface == "grammar" else f"{DUST_PREFIX}{grammar_key}"
        choice = normalize_screen_choice(raw[source], key=display_key)

        from tengri.config.exceptions import ParameterError

        if source == "agn" and choice != "none":
            raise ParameterError(
                f"{display_key}={raw[source]!r}: galaxy screening of AGN light lands "
                f"in the next change; today the AGN component runs after dust and "
                f"carries its own polar-dust screen. Leave {display_key!r} unset, or "
                f"set it to 'none'."
            )
        if dust_model in ("wg00", "off"):
            raise ParameterError(
                f"{display_key!r} is a two-component-only dust_attenuation key (it "
                f"chooses between the birth-cloud and diffuse-ISM screens), and "
                f"dust_model={dust_model!r} has no such screens, so writing it here "
                f"would be silently ignored. Drop {display_key!r}, or select the "
                f"two-component dust model."
            )
        if dust_model == "single_component" and choice not in ("none", default):
            raise ParameterError(
                f"{display_key}={raw[source]!r} is not valid with a single-screen "
                f"dust model: single screen, no birth-cloud/diffuse distinction "
                f"exists. Use 'none' (skip attenuating this source), or drop "
                f"{display_key!r} to keep the default ({default!r})."
            )
        resolved[source] = choice
    return resolved


def short_to_full(stem: str) -> str:
    """Convert a short stem (e.g., 'slope', 'tau_bc') to its full registry name.

    Returns the stem unchanged if it already starts with 'dust_', otherwise
    prepends the 'dust_' prefix.

    Parameters
    ----------
    stem : str
        The short form of a dust parameter (e.g., 'tau_bc', 'slope').

    Returns
    -------
    str
        The full registry name (e.g., 'dust_tau_bc', 'dust_slope').
    """
    if stem.startswith(DUST_PREFIX):
        return stem
    return DUST_PREFIX + stem


def full_to_short(name: str) -> str:
    """Convert a full registry name (e.g., 'dust_tau_bc') to its short stem.

    Strips a leading 'dust_' prefix if present; otherwise returns the name
    unchanged.

    Parameters
    ----------
    name : str
        The full registry name or a plain stem.

    Returns
    -------
    str
        The short stem (e.g., 'tau_bc', 'slope').
    """
    if name.startswith(DUST_PREFIX):
        return name[len(DUST_PREFIX) :]
    return name


def split_screen_suffix(key: str) -> tuple[str, str | None]:
    """Detect and strip a screen suffix from a per-screen key.

    For a key like 'slope_bc', detects that 'bc' is a valid screen suffix
    and returns ('slope', 'bc'). If no valid suffix is present, returns
    (key, None).

    Parameters
    ----------
    key : str
        A potentially per-screen key (e.g., 'slope_bc', 'tau', 'unknown_xyz').

    Returns
    -------
    tuple of (str, str or None)
        (stem, screen) where stem is the key without the suffix, and screen
        is one of SCREENS or None if no suffix was found.
    """
    for screen in SCREENS:
        suffix = "_" + screen
        if key.endswith(suffix) and len(key) > len(suffix):
            return (key[: -len(suffix)], screen)
    return (key, None)


def per_screen_keys() -> frozenset[str]:
    """Generate all valid per-screen keys (cartesian product of OVERRIDE_STEMS × SCREENS).

    Returns
    -------
    frozenset of str
        Keys like 'slope_bc', 'slope_diff', 'slope_neb', 'bump_strength_bc', etc.
    """
    return frozenset(f"{stem}_{screen}" for stem in OVERRIDE_STEMS for screen in SCREENS)


def normalize_dust_group_keys(group, accepted) -> dict:
    """Normalize dust_attenuation group keys to a single spelling.

    If `group` is not a dict, returns it unchanged. Otherwise, returns a NEW
    dict with dust_* keys stripped to their short form IF the short form is
    in the accepted set AND the key is not already in its accepted full form.

    For example, 'dust_tau_bc' becomes 'tau_bc' if 'tau_bc' is in accepted.
    'dust_curve' stays as-is if it is not in accepted (unknown key, error caught later).

    Raises ValueError if both a key and its dust_* variant are present
    (e.g., both 'tau_bc' and 'dust_tau_bc').

    Parameters
    ----------
    group : any
        The dust_attenuation dict or another value.
    accepted : set or frozenset of str
        The set of accepted key names (stems and structural keys).

    Returns
    -------
    dict
        A new dict with normalized keys, or the original value unchanged if not a dict.

    Raises
    ------
    ValueError
        If both a key and its dust_* variant are present in the dict.
    """
    if not isinstance(group, dict):
        return group

    result = {}
    for key, value in group.items():
        if isinstance(key, str) and key.startswith(DUST_PREFIX):
            # This key has a dust_ prefix.
            short = full_to_short(key)
            if short in accepted:
                # The short form is accepted.
                # Check for collision: is the short form already present?
                if short in result:
                    raise ValueError(
                        f"dust_attenuation names both {key!r} and {short!r}, "
                        f"two spellings of one key; keep one."
                    )
                result[short] = value
            else:
                # The short form is not accepted (unknown key).
                # Keep the full form as-is; error will be caught later.
                result[key] = value
        else:
            # Key does not start with dust_. It can still collide with a
            # dust_-prefixed spelling normalized earlier in the iteration.
            if key in result:
                raise ValueError(
                    f"dust_attenuation names both {DUST_PREFIX + key!r} and {key!r}, "
                    f"two spellings of one key; keep one."
                )
            result[key] = value

    return result


def validate_shape_requests(requests, laws, *, surface: str) -> None:
    """Validate per-screen shape parameter requests against declared laws.

    Checks that each requested parameter is read by the law(s) on its screen.
    Raises ParameterError with messages appropriate for the surface (grammar
    vs. flat Parameters).

    Parameters
    ----------
    requests : iterable of (full_name, screen_or_None)
        Each item is a (full_name, screen) pair where full_name is the registry
        name (e.g., 'dust_slope') and screen is one of SCREENS or None.
    laws : dict mapping screen to law name
        Keys are 'bc', 'diff', 'neb'; values are law names (str) or None.
        For screen 'neb', the law falls back to 'bc' if 'neb' is None.
    surface : str
        Either 'grammar' (for grammar error messages) or 'flat' (for
        flat Parameters(...) messages).

    Raises
    ------
    ParameterError
        If a requested parameter is not read by the relevant law(s).
    """
    from tengri.components.dust.laws._registry import law_kwarg_names
    from tengri.config.exceptions import ParameterError

    for full_name, screen in requests:
        # Validate screen name
        if screen is not None and screen not in SCREENS:
            raise ParameterError(
                f"{surface}: {full_name!r} names screen {screen!r}; "
                f"the screens are 'bc', 'diff' and 'neb'."
            )

        if screen is None:
            # Shared parameter: must be read by at least one law in play.
            screens_in_play = [s for s in SCREENS if laws.get(s) is not None]
            # Resolve 'neb' to 'bc' fallback
            neb_law = laws.get("neb") or laws.get("bc")
            if neb_law is None:
                screens_in_play = [s for s in SCREENS if s != "neb" and laws.get(s) is not None]
            else:
                if "neb" not in screens_in_play and neb_law is not None:
                    screens_in_play = [s for s in SCREENS if laws.get(s) is not None]

            distinct_laws = {laws.get(s) for s in ["bc", "diff"] if laws.get(s) is not None}
            if neb_law is not None:
                distinct_laws.add(neb_law)

            # Check if any law reads this parameter
            reads_it = any(full_name in law_kwarg_names(law) for law in distinct_laws if law)

            if not reads_it:
                # Build description and accepted list
                desc_parts = []
                for s in ["bc", "diff", "neb"]:
                    law = laws.get(s)
                    if s == "neb" and law is None:
                        law = laws.get("bc")
                    if law is not None:
                        desc_parts.append(f"{s}={law!r}")
                desc = ", ".join(desc_parts) if desc_parts else "no laws"

                accepts_set = set()
                for law in distinct_laws:
                    if law:
                        accepts_set.update(law_kwarg_names(law))
                accepts_set.discard("wavelength")
                accepts_set.discard("redshift")
                accepts = ", ".join(sorted(accepts_set)) if accepts_set else "(none)"

                raise ParameterError(
                    f"{full_name!r} is read by none of the attenuation laws in play "
                    f"({desc}), so writing it would be silently ignored: the parameter "
                    f"belongs to another attenuation law. Drop it, or select a law that "
                    f"reads it. Laws in play accept: {accepts}."
                )
        else:
            # Per-screen parameter
            law = laws.get(screen)
            if screen == "neb" and law is None:
                law = laws.get("bc")

            if law is None:
                # No law selected for this screen, skip validation
                continue

            if full_name not in law_kwarg_names(law):
                # Build error message based on surface type
                short_name = full_to_short(full_name)
                if surface == "grammar":
                    # Grammar surface: use short form in error message
                    reads = law_kwarg_names(law)
                    accepted = sorted(
                        f"{full_to_short(kw)}_{screen}"
                        for kw in reads
                        if full_to_short(kw) in OVERRIDE_STEMS
                    )
                    accepts = ", ".join(accepted) if accepted else "no per-screen keys at all"
                    key_name = f"{short_name}_{screen}"
                    raise ParameterError(
                        f"{key_name!r} is not read by the 'dust_attenuation' "
                        f"{SCREEN_LABEL[screen]} law {law!r}, so writing it here would be "
                        f"silently ignored: the key belongs to another attenuation law. "
                        f"Law {law!r} accepts: {accepts}. Drop the key, or select a law "
                        f"that reads it."
                    )
                else:
                    # Flat surface: use full form in error message
                    reads = law_kwarg_names(law)
                    accepts = ", ".join(sorted(reads)) if reads else "(none)"
                    raise ParameterError(
                        f"dust_law_overrides[{screen!r}][{full_name!r}] is not read by "
                        f"the selected law {law!r}, so writing it would be silently ignored: "
                        f"the parameter belongs to another attenuation law. Drop it, or "
                        f"select a law that reads it. Law {law!r} accepts: {accepts}."
                    )
