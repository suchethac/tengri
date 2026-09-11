# SPDX-License-Identifier: BSD-3-Clause
"""Single owner of the (dust_law_bc, dust_law_diff) resolution rule.

Two public surfaces each need to turn a caller's law-selection kwargs into a
concrete ``(dust_law_bc, dust_law_diff)`` pair for a given ``dust_model``:
:class:`~tengri.parameters.parameters.Parameters`'s flat-kwarg constructor
(``_init_dust_config``) and ``SEDModel.from_config`` /
``build_model_from_config`` (a single ``dust=``/``dust_attenuation_law=``
string that must name both screens explicitly). Before this module the two
resolutions were separate: ``_init_dust_config`` silently discarded
``dust_law_diff`` on ``dust_model="single_component"`` (#2224), and
``build_model_from_config`` set only ``dust_law_bc``, relying on
``Parameters``'s own two-component inheritance to fill in the diffuse screen
by accident of a default it never chose on purpose (#2021).

Pure string logic with no ``tengri`` imports, so both call sites -- one in
``parameters/parameters.py``, one in ``forward/convenience.py`` -- share
this module without a layering cycle, and the rule is testable without
constructing a ``Parameters`` or loading any SSP data.

The two-/off/wg00 inheritance rule below is unchanged from PR #1989 and the
publicly documented contract (``CHANGELOG.md``, "The low-level
``Parameters(dust_law_bc=...)`` kwargs path is unchanged and still inherits
``dust_law_diff`` from ``dust_law_bc``."); only ``single_component``'s rule
changed, from silent discard to an explicit refusal.
"""

from __future__ import annotations

#: from_config's model-level dust name -> the law-registry key it selects.
#: "charlot_fall" is the shipped ``[from_config] dust`` default (the classic
#: Charlot & Fall 2000 two-screen model) but is not itself a registered
#: attenuation law -- ``power_law`` is the law-registry anchor for that curve
#: (``src/tengri/components/dust/laws/_registry.py``). Every other name
#: from_config accepts (``"calzetti"``, ``"kl04"``, ...) already is a
#: law-registry key and passes through unchanged.
FROM_CONFIG_DUST_ALIASES: dict[str, str] = {"charlot_fall": "power_law"}


def resolve_from_config_dust_law(dust: str) -> str:
    """Resolve ``from_config``'s single dust string to a law-registry key.

    Parameters
    ----------
    dust : str
        The ``dust=`` / ``dust_attenuation_law=`` value from
        ``SEDModel.from_config`` / ``build_model_from_config``, e.g.
        ``"charlot_fall"``, ``"calzetti"``.

    Returns
    -------
    str
        A law-registry key, e.g. ``"power_law"``, ``"calzetti"``.

    Notes
    -----
    **JIT-compatible**: no (construction-time Python).
    """
    return FROM_CONFIG_DUST_ALIASES.get(dust, dust)


def resolve_dust_screen_laws(
    dust_model: str, law_bc: str | None, law_diff: str | None
) -> tuple[str, str]:
    """Resolve the birth-cloud / diffuse-ISM attenuation law pair.

    The single owner of the ``(dust_law_bc, dust_law_diff)`` pair for a given
    ``dust_model``, shared by ``Parameters._init_dust_config`` (flat-kwarg
    construction) and ``build_model_from_config`` (the ``from_config`` law
    resolution, #2021) so the two public surfaces cannot disagree about what
    a given triple of inputs means.

    On ``dust_model="single_component"`` there is exactly one attenuation
    screen, so ``dust_law_diff`` has nothing to select: a lone
    ``dust_law_diff`` used to be silently discarded and a disagreeing pair
    used to silently keep ``dust_law_bc``, building physics nobody asked for
    (#2224). Both shapes now raise, naming the offending kwarg(s) and the
    fix. An *equal* pair is accepted without complaint -- it is exactly what
    the grammar path (``_translate_dust_attenuation``) already writes for
    ``single_component``, which stores its single ``law`` on both
    ``result["dust_law_bc"]`` and ``result["dust_law_diff"]`` before this
    function ever sees it.

    On ``dust_model`` in ``("two_component", "wg00", "off")`` the rule is
    unchanged from PR #1989 (see ``CHANGELOG.md``): either law given alone
    inherits into the other, both given pass through unmodified, and neither
    given defaults to ``"power_law"`` for both screens.

    Parameters
    ----------
    dust_model : str
        One of ``"single_component"``, ``"two_component"``, ``"wg00"``,
        ``"off"``. Already validated by the caller.
    law_bc : str or None
        Explicit birth-cloud law, or ``None`` if not given.
    law_diff : str or None
        Explicit diffuse-ISM law, or ``None`` if not given.

    Returns
    -------
    tuple of (str, str)
        ``(dust_law_bc, dust_law_diff)``, both concrete law-registry keys.

    Raises
    ------
    ValueError
        On ``single_component``, if ``dust_law_diff`` is given without a
        matching ``dust_law_bc``, or if both are given and disagree.

    Notes
    -----
    **JIT-compatible**: no (construction-time Python).

    References
    ----------
    #2224 : ``dust_model="single_component"`` silently discarded
        ``dust_law_diff``.
    #1989 : ``two_component``/``wg00``/``off`` inheritance rule (unchanged
        here; preserved for backward compatibility).
    """
    if dust_model == "single_component":
        return _resolve_single_component(law_bc, law_diff)
    return _resolve_two_screen(law_bc, law_diff)


def _resolve_single_component(law_bc: str | None, law_diff: str | None) -> tuple[str, str]:
    """Resolve the single-screen pair; see :func:`resolve_dust_screen_laws`."""
    if law_bc is None and law_diff is None:
        return "power_law", "power_law"
    if law_diff is None:
        return law_bc, law_bc
    if law_bc is None:
        raise ValueError(
            f"dust_model='single_component' has a single attenuation screen, so "
            f"only dust_law_bc is applied -- dust_law_diff={law_diff!r} would be "
            f"silently discarded. Set dust_law_bc={law_diff!r} instead of "
            f"dust_law_diff, or drop dust_law_diff."
        )
    if law_bc != law_diff:
        raise ValueError(
            f"dust_model='single_component' has a single attenuation screen, so "
            f"dust_law_bc={law_bc!r} and dust_law_diff={law_diff!r} cannot both "
            f"apply -- only one screen exists to satisfy them. Set "
            f"dust_law_bc={law_diff!r} instead of dust_law_diff, or drop "
            f"dust_law_diff."
        )
    return law_bc, law_diff


def _resolve_two_screen(law_bc: str | None, law_diff: str | None) -> tuple[str, str]:
    """Resolve the two-/off/wg00-screen pair (#1989, unchanged)."""
    if law_bc is None and law_diff is None:
        return "power_law", "power_law"
    if law_bc is None:
        return law_diff, law_diff
    if law_diff is None:
        return law_bc, law_bc
    return law_bc, law_diff
