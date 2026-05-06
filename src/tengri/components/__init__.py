"""Forward model components: stellar, dust, nebular, AGN, IGM, radio, X-ray.

The ``sfh`` and ``sps`` subpackages were folded into ``stellar`` in
Phase II-2.1. The old dotted names remain importable as deprecation
shims (firing :class:`DeprecationWarning` on first use); they are not
preloaded here so that ``import tengri.components`` is warning-clean.
"""

# Import canonical submodules eagerly so users can write
# ``tengri.components.dust`` etc. without an explicit submodule import.
from tengri.components import agn, dust, igm, nebular, radio, stellar, xray

__all__ = [
    "agn",
    "dust",
    "igm",
    "nebular",
    "radio",
    "stellar",
    "xray",
]


def _get_registered_components() -> list:
    """Lazily discover all registered SEDComponent classes.

    Deferred until after module initialization to avoid circular imports.
    Called by :func:`tengri.parameters.translate._build_param_map` to
    auto-derive identity parameter mappings.

    Returns
    -------
    list
        SEDComponent classes available in the current build.
    """
    # Import component classes directly from their modules to avoid
    # circular dependencies and ensure they are available even if not
    # re-exported from the submodule __init__.py.
    from tengri.components.agn.component import AGNSEDComponent
    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.nebular.component import NebularSEDComponent
    from tengri.components.radio.component import RadioSEDComponent
    from tengri.components.stellar.component import StellarSEDComponent
    from tengri.components.xray.component import XRaySEDComponent

    return [
        StellarSEDComponent,
        DustAttenuationSEDComponent,
        NebularSEDComponent,
        AGNSEDComponent,
        IGMSEDComponent,
        RadioSEDComponent,
        XRaySEDComponent,
    ]
