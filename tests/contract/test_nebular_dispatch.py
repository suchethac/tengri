# SPDX-License-Identifier: BSD-3-Clause
"""Single-dispatch + default-hygiene contract for the nebular component (#845).

`NebularSEDComponent` now dispatches through the `_REGISTRY` seam (like the
other main-chain components). It is a construction-only convergence: the
per-backend `apply()` physics is unchanged.

Registering it surfaced a default-hygiene gap that the global
`test_param_defaults` cannot see — that test only instantiates the DEFAULT
component (nebular default backend = ``baked_in``, which declares zero params),
so the cue/cloudy/shock backends' free params escaped the check. This module
closes that hole: every photoionization/shock backend's declared params must
carry an explicit `default=` (or be `Fixed`), so `'all_params': FIXED` never silently
falls back to the prior midpoint (#477/#478).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

_BACKENDS = ("baked_in", "cloudy_grid", "cb19", "mappings", "cue", "shock")


class _MockSSP:
    ssp_wave = None


def test_nebular_dispatches_via_registry():
    """build_components(nebular_backend=...) builds the _REGISTRY nebular class."""
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import build_components

    comps = build_components(
        ssp_data=_MockSSP(),
        dust_model="two_component",
        dust_emission_model=None,
        use_dust=False,
        nebular_backend="baked_in",
        agn_model=None,
        use_radio=False,
        use_xray=False,
        use_igm=False,
    )
    neb = [c for c in comps if type(c) is _REGISTRY["nebular"]]
    assert len(neb) == 1 and type(neb[0]).__name__ == "NebularSEDComponent"


def test_seam_construction_is_bit_identical_to_direct():
    """Physics-equivalence gate: seam-built == directly-built (frozen-dataclass eq)."""
    from tengri.components.nebular.component import NebularSEDComponent, NebularSEDComponentConfig
    from tengri.forward.component_factory import _resolve_registry_component

    cfg = NebularSEDComponentConfig(backend="baked_in")
    assert _resolve_registry_component(
        "nebular", "nebular", config=cfg, backend=None
    ) == NebularSEDComponent(config=cfg, backend=None)


def test_no_legacy_class_in_component_factory():
    import tengri.forward.component_factory as cf

    assert not hasattr(cf, "NebularSEDComponent"), (
        "component_factory must resolve nebular via _REGISTRY, not import the class (#845)"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_every_backend_param_has_default(backend):
    """Every free param of every nebular backend carries an in-bounds default.

    Guards the #845 default-hygiene fix (ionspec = 1-Myr solar-Z BPASS fit;
    physical defaults for neb_logU/gas_*/shock_velocity). ``test_param_defaults``
    only checks the default ``baked_in`` instance, so this per-backend sweep is
    the real guard against a future undefaulted nebular free param.
    """
    from tengri.components.nebular.component import NebularSEDComponent, NebularSEDComponentConfig
    from tengri.parameters.priors import Fixed

    comp = NebularSEDComponent(config=NebularSEDComponentConfig(backend=backend))
    offenders = []
    for decl in comp.declared_parameters():
        prior = decl.prior
        if isinstance(prior, Fixed):
            continue
        if prior.default is None:
            offenders.append(f"{decl.name} (no default)")
            continue
        lo, hi = prior.bounds
        if lo is not None and hi is not None and not (lo <= float(prior.default) <= hi):
            offenders.append(f"{decl.name} default={prior.default} outside [{lo}, {hi}]")
    assert not offenders, f"backend {backend!r} has undefaulted params: {offenders}"


def test_cue_ionspec_gas_single_source_of_truth():
    """#887: the Cue ionizing-spectrum + gas-extra params are declared ONCE
    (in ``components/nebular/_params.py``) and consumed by BOTH construction
    paths — ``declared_parameters()`` (SEDComponent / grammar) and the
    flat-builder bucket (``_resolve_lazy_bucket``). Guard that the two paths
    report byte-identical priors, defaults, and bounds so they cannot drift
    (previously the bucket carried ``None`` priors while the component
    re-declared the same params inline with ``Uniform`` defaults)."""
    from tengri.components.nebular._params import (
        CUE_GAS_EXTRA_PARAMS,
        CUE_IONSPEC_PARAMS,
    )
    from tengri.components.nebular.component import (
        NebularSEDComponent,
        NebularSEDComponentConfig,
    )
    from tengri.parameters._builders import _resolve_lazy_bucket

    canonical = {d.name: d for d in (*CUE_IONSPEC_PARAMS, *CUE_GAS_EXTRA_PARAMS)}
    assert len(canonical) == 10  # 7 ionspec + 3 gas

    # (a) declared_parameters() returns the canonical declarations verbatim.
    comp = NebularSEDComponent(config=NebularSEDComponentConfig(backend="cue"))
    declared = {d.name: d for d in comp.declared_parameters()}
    for name, cdecl in canonical.items():
        assert declared[name] is cdecl, f"{name}: declared_parameters is not the canonical decl"

    # (b) the flat-builder bucket derives from the SAME canonical declarations:
    #     its resolved prior/default/bounds match (no None-vs-Uniform drift).
    bucket = {
        **_resolve_lazy_bucket("_CUE_IONSPEC_PARAMS"),
        **_resolve_lazy_bucket("_CUE_GAS_EXTRA_PARAMS"),
    }
    for name, cdecl in canonical.items():
        _desc, _check, _err, prior = bucket[name]
        assert prior is cdecl.prior, (
            f"{name}: bucket prior diverged from the canonical declaration"
        )
        assert prior is not None and prior.default is not None, (
            f"{name}: bucket path lost the physical default (the #887 None-fallback bug)"
        )
