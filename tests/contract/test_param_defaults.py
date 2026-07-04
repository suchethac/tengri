# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: every declared component parameter carries an in-bounds default.

Foundation laid in #478 (``Distribution.default`` kwarg + ``ParameterDefaultMissingError``).
This test enforces the migration is complete — every ``Uniform`` / ``Gaussian`` /
``LogUniform`` / ``LogNormal`` / ``StudentT`` declaration that ships in a registered
``SEDModelComponent`` must carry a physically-motivated ``default=`` so the
``parse_groups`` FIXED-fallback path can never collapse to an arbitrary
``unstandardize(0.0)`` midpoint again.

``Fixed`` declarations satisfy the contract automatically (``Fixed.default`` returns
``self.value``). New declarations must either be ``Fixed(value)`` or carry an
explicit ``default=``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def _force_import_all_components() -> None:
    """Import every SEDModelComponent module so the registry is fully populated.

    The auto-import via ``import tengri`` doesn't reach every component because
    some live under conditional / lazy paths. This list mirrors what we need
    the contract to cover.
    """
    import tengri

    # Non-auto-imported components — list grows when new components ship.
    import tengri.components.dust.draine2021_pah_ir
    import tengri.components.dust.schreiber2016_ir
    import tengri.components.nebular.shock_model
    import tengri.components.radio.radio_dpl_model
    import tengri.components.spatial.exponential
    import tengri.components.spatial.flat_slab
    import tengri.components.spatial.sersic
    import tengri.components.xray.agn_xray_model  # noqa: F401


def test_every_declared_param_has_in_bounds_default():
    """Every registered SEDModelComponent declaration carries a usable default.

    The bug surfaced in #477 (Cue ``gas_logn`` defaulting to the prior midpoint
    rather than the CIGALE-faithful ``n_H=100`` value) was a symptom of an
    implicit fallback in ``parse_groups``. Foundation #478 added ``default=``
    on ``Distribution``; this test pins the migration: every declaration must
    set it (or be ``Fixed``).

    Raises a single AssertionError listing every offending (component, param)
    pair so a future declaration that misses a default fails loudly with a
    clear migration TODO rather than silently falling through to the midpoint.
    """
    _force_import_all_components()

    from tengri.components.sed_model_component import _REGISTRY
    from tengri.parameters.priors import Fixed

    offenders: list[str] = []
    for comp_name, cls in sorted(_REGISTRY.items()):
        comp = cls()
        for decl in comp.declared_parameters():
            prior = decl.prior
            if isinstance(prior, Fixed):
                continue  # Fixed.default returns self.value automatically
            if prior.default is None:
                offenders.append(f"  {comp_name}.{decl.name}  ({type(prior).__name__})")
                continue
            lo, hi = prior.bounds
            if lo is None or hi is None:
                continue
            if not (lo <= float(prior.default) <= hi):
                offenders.append(
                    f"  {comp_name}.{decl.name}  default={prior.default} "
                    f"outside bounds [{lo}, {hi}]"
                )

    if offenders:
        raise AssertionError(
            f"{len(offenders)} component parameter(s) missing in-bounds defaults:\n"
            + "\n".join(offenders)
            + "\n\nEach declaration must either be ``Fixed(value)`` or carry "
            "``default=<physical_value>``. See docs/dev/parameter-defaults.md (or "
            "the #478 commit message) for rationale."
        )
