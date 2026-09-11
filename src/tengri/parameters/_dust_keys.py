# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation key vocabulary and normalization.

This module is the single home of the dust attenuation grammar vocabulary:
SCREENS ('bc', 'diff', 'neb'), OVERRIDE_STEMS (the four law parameter stems),
and utilities to convert between spellings and validate them.

The grammar accepts two spellings for each parameter:
- Short form (stem): 'tau_bc', 'slope_diff', 'Rv_neb'
- Full form (registry name): 'dust_tau_bc', 'dust_slope_diff', 'dust_Rv_neb'

They are equivalent and normalize to one spelling early in parse_groups (Pass 0d),
before the structural translator, completeness check, and all later passes see
the dict. This central vocabulary prevents the three hand-lists (structural keys,
_resolve_value locals, and grammar normalization) from drifting apart.

Tests pin this vocabulary against TWO_COMPONENT_OVERRIDE_KEYS.
"""

from __future__ import annotations

DUST_PREFIX = "dust_"
SCREENS: tuple[str, ...] = ("bc", "diff", "neb")
SCREEN_LABEL = {"bc": "birth-cloud", "diff": "diffuse-ISM", "neb": "nebular birth-cloud"}
# The only home of this list (beyond tests); every other hand-list derives here.
OVERRIDE_STEMS: tuple[str, ...] = ("slope", "bump_strength", "delta", "Rv")


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
