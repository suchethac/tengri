# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: ``dust.emission``'s ``'*'`` must free parameters the backend reads.

``dust={'emission': {'type': X, 'all_params': FREE}}`` freed the same seven parameters
whichever engine ``X`` named, and for two engines not one of the seven was a
parameter that engine reads (#1482).

Each backend *does* declare its own parameters correctly — every one is a
:class:`SEDModelComponent` and ``Dale2014IRSEDComponent`` declares exactly
``alpha_dale`` and ``frac_agn``. The wildcard does not consult them. ``'all_params': FREE``
means "use the registry default", so it can only free a parameter whose registry
default is a *distribution*; the 15 dust-emission parameters defaulting to
``Fixed`` scalars stay pinned. Which 7 of the 22 carry distribution defaults has
nothing to do with the selected engine.

``_check_wildcard_freed_something`` guards precisely this failure, but it was
scoped to the **group**, raising only when *zero* of the group's 22 parameters
were freed. Those same 7 always free, so ``any(freed)`` was always true and the
guard could never fire for ``dust.emission`` — including when zero of the
selected backend's parameters were freed. A guard that fails open.

``_narrow_outcome_to_selected_component`` restricts the outcome to the selected
component's ``declared_parameters()`` before the check, so the guard now fires:

Composed with the three-outcome predicate from #1474, the narrowing makes every
verdict count the selected engine's **own** declared parameters:

======================  ==================  =========================
``emission.type``       declares            ``'all_params': FREE`` outcome
======================  ==================  =========================
``dale2014``            2, both unfreeable  **raises** ("0 of 2")
``casey2012``           3, all unfreeable   **raises** ("0 of 3")
``themis``              4, 2 freeable       **warns** ("2 of 4")
``draine_li2014``       4, 2 freeable       **warns**
``draine_li2007``       3, 1 freeable       **warns** ("1 of 3")
``modified_blackbody``  3, 1 freeable       **warns**
``astrodust``           1, freeable         silent — all of its own freed
======================  ==================  =========================

Without the narrowing those counts are all "7 of 22", diluted by parameters the
selected engine never reads; without #1474's predicate only the two zero cases
are caught at all.

**The expansion now resolves per backend too.** Narrowing only the *check* left
the freed *set* backend-blind: ``themis`` received seven parameters and read two,
so five (``dust_alpha_dl14``, ``dust_epsilon_mbb``, ``dust_f_cold``,
``dust_f_pah``, ``dust_lgU``) stayed free no-op dimensions — loud since #1474,
but still explored by every sampler. ``parse_groups`` now scopes the sub-block
wildcard to the selected engine's declared parameters, the same block scoping the
AGN and radio sub-blocks have always used: a parameter outside the set resolves
to ``wildcard_fixed_inactive`` and stays declared-but-``Fixed``. Under
``schreiber2016`` that is measured to be the difference between six inert free
dimensions and none, with its own ``dust_T`` — worth 94% of the dust IR SED —
correctly reported as the pinned one.

Two engines keep the unscoped behavior by design, because an empty ``_priors``
does not distinguish them: ``pah_drude`` is genuinely parameter-free, while
``energy_balance_split`` reads six parameters declared in
``components/dust/_params.py`` rather than on its class. Narrowing the latter to
the empty set would pin all six — the failure this scoping exists to prevent — so
``_declared_param_names`` returns ``None`` and the wildcard is left alone.

The bands span 1500 A - 500 um so dust IR emission has somewhere to land — an
optical-only filter set would report every dust-emission parameter as inert for
the honest reason that its emission falls outside the bandpasses, which is filter
coverage rather than a wiring defect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.config.exceptions import ParameterError
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve
from tests._data_skip import PAHSPEC_EMISSION_TYPES, requires_pahspec

pytestmark = pytest.mark.contract

#: Backends this suite refuses to measure, each with the reason it cannot
#: discriminate #1482 -- not a convenience list.
#:
#: Empty, and the entry that used to be here is why the distinction matters.
#: ``draine2021_pah`` was excluded unconditionally on the strength of a
#: measurement -- it contributed ``3.706636e+42`` to the rest-frame 10-500 um
#: flux, bit-identical to ``emission={'type': 'none'}``, against ``1.84e+49``
#: for ``dl07`` -- read as a property of the component. It is a property of the
#: **machine**: that number comes from a checkout with no PAHspec grid, where
#: the component warns and contributes nothing by design (#1278). With the grid
#: present it emits, and its ``dust_lgU`` moves the photometry like any other
#: engine's parameter. An environment-dependent zero belongs behind
#: :data:`~tests._data_skip.requires_pahspec`, which reports the absence as a
#: skip, and not in a list whose contract is "cannot discriminate #1482 for a
#: reason no download will change".
_UNDISCRIMINATING: dict[str, str] = {}


def _emission_types() -> tuple[str, ...]:
    """Every non-deprecated IR backend, derived from the live menu.

    Hardcoding this list is what hid the bug it was written to catch. The
    frozen tuple named seven backends and so covered five of the thirteen
    production engines. The eight it missed included ``pah_drude`` and
    ``dh02_ce01``, which were freeing **twenty** parameters and reading
    **one** -- the exact defect of #1482, in the suite that owns #1482,
    unmeasured because the parametrization never reached them. ``dh02_ce01``
    was registered later still and inherited the same gap.

    **Experimental backends are included**, and that is not incidental: the
    frozen tuple already carried two of them (``themis``, ``astrodust``), so
    filtering to ``production`` would quietly *drop* coverage while appearing
    to broaden it. It would also miss ``energy_balance_split``, which is
    experimental and measures **20 freed, 6 read, 14 inert** -- the largest
    surviving instance of this bug. A wildcard that hands a sampler fourteen
    flat directions is the same defect whatever the maturity label says.

    Only ``deprecated`` is excluded, plus anything named in
    :data:`_UNDISCRIMINATING` with a measured reason, plus the **building
    blocks** — types the menu lists because they compose into custom models but
    which ``SEDModel.build`` refuses as a model's only dust emitter. Every test
    here builds one model per name with a ``dust_emission`` wildcard, so a name
    the builder refuses cannot be measured through this file at all; it is
    excluded by the same derivation the builder uses rather than by name.
    ``pah_drude`` is the one such name today, and it leaves
    :data:`DECLARES_NOTHING` non-empty behind it (``dh02_ce01``), so the
    declares-nothing branch keeps a subject.

    A backend whose template grid is not on this machine is **not** excluded
    here: it is parametrized and skipped, by a mark on
    :data:`EMISSION_TYPE_PARAMS` rather than by absence from this tuple, so the
    derivations that read this tuple keep seeing it.

    So a backend added tomorrow is measured on the day it ships.
    """
    from tengri import registry
    from tengri.parameters.groups import _standalone_dust_emission_types

    standalone = _standalone_dust_emission_types()
    return tuple(
        sorted(
            e["name"]
            for e in registry.list_dust_emission_models()
            if e.get("name")
            and e["name"] in standalone
            and e.get("status", "production") != "deprecated"
            and e["name"] not in _UNDISCRIMINATING
            and e["name"] != "none"
        )
    )


EMISSION_TYPES = _emission_types()

#: :data:`EMISSION_TYPES` as parametrization arguments, with the two PAHspec
#: spellings marked to skip where their grid is absent.
#:
#: The gate is on the *measurement*, never on the derivation: a name dropped
#: from :data:`EMISSION_TYPES` would also vanish from :func:`_nothing_freeable`
#: and :func:`_declares_nothing`, which read it, and this file's whole argument
#: is that a hardcoded parametrization is what hid #1482 in the first place.
#: Skipped, the name still shows in the report; excluded, it does not.
EMISSION_TYPE_PARAMS = tuple(
    pytest.param(name, marks=(requires_pahspec,) if name in PAHSPEC_EMISSION_TYPES else ())
    for name in EMISSION_TYPES
)


def _nothing_freeable() -> dict[str, set[str]]:
    """Backends whose every declared parameter still resolves to a ``Fixed`` scalar.

    ``'all_params': FREE`` can free none of these, so the guard must refuse the build.

    Derived rather than listed. #887 is giving parameters their declared
    ``free_prior`` a subsystem at a time, and each one moves a backend off this
    list -- a hardcoded set would keep asserting that a guard fires after the
    fix that stops it firing. (That is not hypothetical: this set previously
    named ``dale2014``, and declaring ``dust_alpha_dale`` -- the parameter
    #1482's own title calls unreachable -- turned the assertion red.)
    """
    from tengri.parameters.groups import _declared_param_names
    from tengri.parameters.registry import registry

    reg = registry()
    out: dict[str, set[str]] = {}
    for etype in EMISSION_TYPES:
        declared = _declared_param_names(etype)
        if not declared:
            continue
        if all(getattr(reg.get(n), "free_prior", None) is None for n in declared):
            out[etype] = set(declared)
    return out


#: Maps type -> the declared names the error is required to name. Empty is the
#: #887 goal state, not a failure: it means every shipped IR backend has at
#: least one parameter the wildcard can genuinely free.
NOTHING_FREEABLE = _nothing_freeable()


def _declares_nothing() -> frozenset[str]:
    """Backends that state they read no parameters at all.

    Distinct from :data:`NOTHING_FREEABLE`, which is "declares parameters, but
    every one is Fixed-by-default". These declare none: ``dh02_ce01`` is a fixed
    template pair, so ``'all_params': FREE`` correctly frees nothing and "at
    least one live parameter" is the wrong question rather than a failed one.

    ``pah_drude``, the other pure template shape, is no longer reachable from
    here: it is a building block the builder refuses, so
    :func:`_emission_types` drops it before this derivation runs. Its half of
    the same guarantee is pinned in
    ``tests/regression/bug/test_parameter_free_emission_wildcard.py``, which
    checks the narrowing mechanism itself rather than a built model.

    Both were invisible until :func:`_emission_types` started deriving the
    parametrization -- and both were freeing **twenty** parameters and reading
    one, because an empty declaration could not be told from an absent one. The
    marker that fixed that (``declares_no_parameters``) is the same one read
    here, so this set cannot drift from the narrowing it describes.
    """
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES, _REGISTRY

    return frozenset(
        etype
        for etype in EMISSION_TYPES
        if getattr(
            _REGISTRY.get(_EMISSION_TYPE_ALIASES.get(etype, etype)),
            "declares_no_parameters",
            False,
        )
    )


#: Backends for which "frees at least one live parameter" is meaningless.
DECLARES_NOTHING = _declares_nothing()

#: Dust attenuation deep enough that L_absorbed is large and IR emission is a
#: real term rather than a rounding error.
_TAU_BC, _TAU_DIFF = 2.0, 1.5


@pytest.fixture(scope="module")
def panchromatic_obs():
    """Six top-hats from 1500 A to 500 um — UV through far-IR."""

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 5000.0, 2.0e4, 2.4e5, 1.0e6, 5.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, obs, emission_type, wildcard=FREE):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": _TAU_BC,
            "tau_diff": _TAU_DIFF,
        },
        dust_emission={"type": emission_type, "all_params": wildcard},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _live_params(model):
    """Freed dust parameters that change ``predict_photometry``.

    Substitutes each parameter's value from other prior draws, so every trial
    value is inside the parameter's own support and no bound can be violated.
    """
    freed = [p for p in model.spec.free_params if p.startswith("dust_")]
    draws = [dict(model.spec.sample(jax.random.PRNGKey(k))) for k in (0, 3, 11)]
    base_params = draws[0]
    base = np.asarray(model.predict_photometry(base_params))

    live = []
    for name in freed:
        for alt in draws[1:]:
            if name not in alt or np.allclose(
                np.asarray(alt[name]), np.asarray(base_params[name])
            ):
                continue
            got = np.asarray(model.predict_photometry({**base_params, name: alt[name]}))
            if not np.allclose(got, base, rtol=1e-9, atol=0.0):
                live.append(name)
                break
    return freed, live


@pytest.mark.parametrize("emission_type", EMISSION_TYPE_PARAMS)
def test_wildcard_frees_at_least_one_live_parameter(
    synthetic_ssp_wide, panchromatic_obs, emission_type
):
    """``'all_params': FREE`` must free something the selected engine actually reads."""
    if emission_type in NOTHING_FREEABLE:
        pytest.skip(f"{emission_type} has nothing freeable — covered by the raise test")
    if emission_type in DECLARES_NOTHING:
        # Not a failure: freeing nothing is the correct outcome for an engine
        # that reads nothing, and is asserted positively by
        # test_no_freed_parameter_is_inert, which stays enabled for these.
        pytest.skip(f"{emission_type} declares no parameters — freeing none is correct")

    model = _build(synthetic_ssp_wide, panchromatic_obs, emission_type)
    freed, live = _live_params(model)

    assert live, (
        f"dust.emission {{'type': '{emission_type}', 'all_params': FREE}} freed {freed} "
        f"and not one of them moves predict_photometry — every freed dimension "
        f"is a silent no-op a sampler would explore for free (#1482)"
    )


@pytest.mark.parametrize("emission_type", sorted(NOTHING_FREEABLE))
def test_wildcard_refuses_when_the_backend_has_nothing_freeable(
    synthetic_ssp_wide, panchromatic_obs, emission_type
):
    """A backend whose every parameter is Fixed-by-default must refuse, not pretend.

    Before #1482 this built happily and handed back seven free dimensions from
    *other* engines, so a sampler reported a confident posterior over parameters
    that cannot move the likelihood by one ULP.
    """
    with pytest.raises(ParameterError) as excinfo:
        _build(synthetic_ssp_wide, panchromatic_obs, emission_type)

    message = str(excinfo.value)
    declared = NOTHING_FREEABLE[emission_type]

    # The error must name the engine's OWN parameters — naming the 22-name group
    # is what made the old group-scoped guard useless.
    for name in declared:
        assert name in message, (
            f"the guard fired but never named {name}, which is what the caller "
            f"needs to free: {message}"
        )
    assert f"0 of {len(declared)}" in message, (
        f"the guard is still counting the whole group rather than "
        f"{emission_type}'s {len(declared)} declared parameters: {message}"
    )


def test_the_freed_set_depends_on_the_backend(synthetic_ssp_wide, panchromatic_obs):
    """The expansion resolves per backend, not once for the whole sub-block (#1482).

    The inversion of the old ``test_expansion_is_still_backend_blind``. While the
    expansion ignored ``emission.type``, these four engines received a byte-identical
    seven-name set; each must now receive its own.
    """
    freed_sets = {
        etype: tuple(
            p
            for p in _build(synthetic_ssp_wide, panchromatic_obs, etype).spec.free_params
            if p.startswith("dust_")
        )
        for etype in ("themis", "draine_li2007", "modified_blackbody", "astrodust")
    }

    assert len(set(freed_sets.values())) > 1, (
        "dust.emission '*' expands to one backend-blind set again — the engine's "
        f"own declarations are not being consulted (#1482). Freed sets: {freed_sets}"
    )


@pytest.mark.parametrize("emission_type", EMISSION_TYPE_PARAMS)
def test_no_freed_parameter_is_inert(synthetic_ssp_wide, panchromatic_obs, emission_type):
    """Every parameter ``'all_params': FREE`` hands the sampler must move the prediction.

    Stronger than :func:`test_wildcard_frees_at_least_one_live_parameter`, which
    only requires *one* live parameter and so passed throughout the era when
    ``themis`` was handed seven and read two. A free dimension the selected engine
    never reads is a flat direction: the sampler explores it at full cost, the
    posterior comes back matching the prior, and nothing in the fit says why.
    """
    if emission_type in NOTHING_FREEABLE:
        pytest.skip(f"{emission_type} has nothing freeable — covered by the raise test")

    model = _build(synthetic_ssp_wide, panchromatic_obs, emission_type)
    freed, live = _live_params(model)
    inert = sorted(set(freed) - set(live))

    assert not inert, (
        f"dust.emission {{'type': '{emission_type}', 'all_params': FREE}} freed {inert}, "
        f"which do not move predict_photometry — they belong to other engines "
        f"(freed={freed}, live={live}) (#1482)"
    )


def test_dale2014_reads_alpha_dale_not_alpha(synthetic_ssp_wide, panchromatic_obs):
    """The knob that works is the one the wildcard cannot reach (#1482).

    Without this, a fix could satisfy the sweep above by freeing *any* live
    parameter while leaving Dale+2014's slope unreachable.
    """
    # Built with the wildcard Fixed(DEFAULT), since 'all_params': FREE now correctly refuses here.
    model = _build(synthetic_ssp_wide, panchromatic_obs, "dale2014", wildcard=Fixed(DEFAULT))
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    def photometry_at(name, value):
        return np.asarray(model.predict_photometry({**params, name: jnp.asarray(value)}))

    # THEMIS's slope: accepted, validated, and ignored by Dale+2014.
    assert np.allclose(photometry_at("dust_alpha", 1.0625), photometry_at("dust_alpha", 2.5)), (
        "dust_alpha now moves a dale2014 model — the parameter split changed"
    )

    # Dale+2014's own slope: wired correctly, but Fixed-by-default so the
    # wildcard cannot free it. Only an explicit prior reaches it.
    spread = np.max(
        np.abs(
            photometry_at("dust_alpha_dale", 1.0625) / photometry_at("dust_alpha_dale", 2.5) - 1.0
        )
    )
    assert spread > 1e-3, f"dust_alpha_dale no longer drives dale2014 (rel change {spread:.2e})"


def test_an_explicit_prior_reaches_the_parameter_the_wildcard_cannot(
    synthetic_ssp_wide, panchromatic_obs
):
    """The remedy the error message recommends must actually work.

    The guard tells the caller to "pass explicit priors instead, e.g.
    dust={'alpha_dale': Uniform(lo, hi)}". Advice that does not work is worse
    than no advice.
    """
    from tengri import Uniform

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=panchromatic_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": _TAU_BC,
            "tau_diff": _TAU_DIFF,
        },
        dust_emission={"type": "dale2014", "alpha_dale": Uniform(1.0625, 4.0)},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )

    assert "dust_alpha_dale" in model.spec.free_params, (
        "the explicit prior the guard recommends did not free dust_alpha_dale"
    )
    _freed, live = _live_params(model)
    assert "dust_alpha_dale" in live, "dust_alpha_dale is free but does not move the photometry"


def test_dust_emission_actually_contributes(synthetic_ssp_wide, panchromatic_obs):
    """Anti-vacuity: if IR emission were negligible, no parameter could be live.

    The sweeps above would then pass or fail for reasons unrelated to the
    wildcard. Pin that the far-IR bands are genuinely emission-dominated.
    """
    with_emission = _build(
        synthetic_ssp_wide, panchromatic_obs, "dale2014", wildcard=Fixed(DEFAULT)
    )
    without = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=panchromatic_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": _TAU_BC,
            "tau_diff": _TAU_DIFF,
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    params = dict(with_emission.spec.sample(jax.random.PRNGKey(0)))
    on = np.asarray(with_emission.predict_photometry(params))
    off = np.asarray(without.predict_photometry({}))

    # 100 um (band index 4) is stellar-negligible and emission-dominated.
    assert on[4] > 1e3 * off[4], (
        f"far-IR band is not emission-dominated (on={on[4]:.3e}, off={off[4]:.3e}); "
        "the wildcard sweep cannot detect a dropped dust-emission parameter"
    )
