# SPDX-License-Identifier: BSD-3-Clause
"""Runtime registry of X-ray emission models.

Mirrors ``SFH_REGISTRY`` (``components/stellar/sfh/registry.py``) and
``AGN_MODELS`` (``components/agn/unified.py``): a flat dict keyed by
variant name with metadata fields that :func:`tengri.list_xray_models`
introspects. Adding a new X-ray model = one ``register_xray_model``
call below; ``_VALID_XRAY_TYPES`` derives from this registry per
ADR-0005 / ADR-0008 (single source of truth).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class XRayRegistryEntry:
    """Registry entry for an X-ray emission model.

    Attributes
    ----------
    callable : Callable or None
        The X-ray model function. ``None`` for the ``'none'`` toggle.
    citation : str
        Academic citation. Empty string for the disable-toggle.
    status : str
        ``"production"`` / ``"experimental"`` / ``"demo"`` / ``"deprecated"``.
    short_doc : str
        One-line description.
    """

    callable: Callable | None
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn = object.__getattribute__(self, "callable")
        if fn is None:
            raise TypeError("The 'none' X-ray model has no callable.")
        return fn(*args, **kwargs)


XRAY_MODELS: dict[str, XRayRegistryEntry] = {}

#: ``xray={'type': ...}`` variants that carry an AGN corona term (as opposed
#: to X-ray-binary-only host emission). Determined by reading
#: ``components/xray/xray.py`` / ``xray_model.py`` / ``agn_xray_model.py``:
#:
#: - ``"simple"`` / ``"yang20"`` (alias): ``xray_total`` = HMXB + LMXB + hot
#:   gas + an alpha_ox(L_2500) corona (Just+2007 / Yang+2020).
#: - ``"lopez24"``: ``xray_total_lopez24`` = the same XRBs + hot gas, corona
#:   via alpha_IRX(L_12um) (Lopez+2024) instead of alpha_ox.
#: - ``"xray_aird"`` (a ``SEDModelComponent``, not in :data:`XRAY_MODELS`
#:   itself but reachable through the same ``xray={'type': ...}`` grammar key
#:   -- see ``parameters/groups.py::_valid_xray_types``): also calls
#:   ``xray_total``, so it is the same HMXB+LMXB+hot-gas+corona mix as
#:   ``"simple"``.
#: - ``"agn_xray_corona"`` (a ``SEDModelComponent``): corona-only, no XRBs.
#:
#: There is currently **no registered host-XRB-only variant** (one that never
#: adds a corona term): every active ``xray`` selection other than ``"none"``
#: includes a corona whose amplitude is driven by the AGN's published
#: ``L_2500_intrinsic`` / ``L_12um`` / ``L_agn_bol`` (see
#: ``components/xray/component.py::emitter_inputs``), so it fires whenever an
#: AGN disc is active regardless of which of these four names is selected.
AGN_CORONA_XRAY_VARIANTS: frozenset[str] = frozenset(
    {"simple", "yang20", "lopez24", "xray_aird", "agn_xray_corona"}
)


def register_xray_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable[[Callable | None], Callable | None]:
    """Register an X-ray emission model in :data:`XRAY_MODELS`.

    Can be used either as a decorator on the model callable or invoked
    directly with ``callable=None`` for the disable-toggle entry::

        @register_xray_model("simple", citation=..., short_doc=...)
        def xray_total(...): ...

        register_xray_model("none", short_doc="Disable")(None)
    """

    def decorator(fn: Callable | None) -> Callable | None:
        XRAY_MODELS[name] = XRayRegistryEntry(
            callable=fn, citation=citation, status=status, short_doc=short_doc
        )
        return fn

    return decorator


def check_disc_xray_double_count(disc_type: str | None, xray_type: str | None) -> None:
    """Refuse an AGN-corona ``xray`` variant on a disc that already has one.

    Some AGN disc templates (``kubota_done``, ``kd18_agnfitter``,
    ``kd18_agnfitter_warmindex``) already carry a hot corona baked into the
    tabulated/analytic SED. Composing one of them with an ``xray`` group
    selection that *also* adds an AGN corona
    (:data:`AGN_CORONA_XRAY_VARIANTS`) would add that corona a second time --
    tengri's own X-CIGALE-derived corona (alpha_ox or alpha_IRX) layered on
    top of the disc's own, roughly +51% over 0.5-10 keV for the Kubota & Done
    corona (its own corona fraction, ``agn_f_hard``, is O(0.5) of the total
    coronal power).

    Parameters
    ----------
    disc_type : str or None
        The selected ``agn.disc`` block name (``result["agn_disc_block"]``),
        or ``None`` if no composable AGN disc is configured.
    xray_type : str or None
        The selected ``xray`` group type (``result["xray_model"]``), or
        ``None`` if the ``xray`` group is absent/disabled.

    Raises
    ------
    ValueError
        If ``disc_type`` carries its own corona (see
        ``disc_emits_xray``) and
        ``xray_type`` is one of :data:`AGN_CORONA_XRAY_VARIANTS`.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only, called once
    from ``parameters/groups.py::_translate_structural`` after both the
    ``agn`` and ``xray`` groups have been translated.
    """
    if not disc_type or not xray_type or xray_type not in AGN_CORONA_XRAY_VARIANTS:
        return
    from tengri.components.agn.blocks._protocol import disc_emits_xray

    if not disc_emits_xray(disc_type):
        return
    raise ValueError(
        f"disc {disc_type!r} already carries a hot corona; xray={{'type': "
        f"'{xray_type}'}} would add a second α_ox corona (+51% over "
        "0.5-10 keV). Set xray={'type': 'none'} or choose a disc without "
        "intrinsic X-rays."
    )
