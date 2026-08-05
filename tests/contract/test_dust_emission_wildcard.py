# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: ``dust.emission``'s ``'*'`` must free parameters the backend reads.

``dust={'emission': {'type': X, '*': FREE}}`` freed the same seven parameters
whichever engine ``X`` named, and for two engines not one of the seven was a
parameter that engine reads (#1482).

Each backend *does* declare its own parameters correctly — every one is a
:class:`SEDModelComponent` and ``Dale2014IRSEDComponent`` declares exactly
``alpha_dale`` and ``frac_agn``. The wildcard does not consult them. ``'*': FREE``
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
``emission.type``       declares            ``'*': FREE`` outcome
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

from tengri import FIXED, FREE, Fixed, SEDModel
from tengri.config.exceptions import ParameterError
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

#: Backends reachable from the build grammar without an external template file.
#: ``draine2021_pah`` is deliberately absent: it also shows zero live parameters,
#: but that is confounded with #1278 (it silently emits nothing when its template
#: is missing), so it cannot discriminate this bug.
EMISSION_TYPES = (
    "dale2014",
    "themis",
    "draine_li2007",
    "draine_li2014",
    "modified_blackbody",
    "casey2012",
    "astrodust",
)

#: Backends every one of whose declared parameters defaults to a ``Fixed`` scalar,
#: so ``'*': FREE`` can free none of them and the guard must refuse the build.
#: Maps type -> the declared names the error is required to name.
NOTHING_FREEABLE = {
    "dale2014": {"dust_alpha_dale", "dust_frac_agn"},
    "casey2012": {"dust_T", "dust_beta_ir", "dust_alpha_mir"},
}

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
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": _TAU_BC,
            "tau_diff": _TAU_DIFF,
            "emission": {"type": emission_type, "*": wildcard},
        },
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


@pytest.mark.parametrize("emission_type", EMISSION_TYPES)
def test_wildcard_frees_at_least_one_live_parameter(
    synthetic_ssp_wide, panchromatic_obs, emission_type
):
    """``'*': FREE`` must free something the selected engine actually reads."""
    if emission_type in NOTHING_FREEABLE:
        pytest.skip(f"{emission_type} has nothing freeable — covered by the raise test")

    model = _build(synthetic_ssp_wide, panchromatic_obs, emission_type)
    freed, live = _live_params(model)

    assert live, (
        f"dust.emission {{'type': '{emission_type}', '*': FREE}} freed {freed} "
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


@pytest.mark.parametrize("emission_type", EMISSION_TYPES)
def test_no_freed_parameter_is_inert(synthetic_ssp_wide, panchromatic_obs, emission_type):
    """Every parameter ``'*': FREE`` hands the sampler must move the prediction.

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
        f"dust.emission {{'type': '{emission_type}', '*': FREE}} freed {inert}, "
        f"which do not move predict_photometry — they belong to other engines "
        f"(freed={freed}, live={live}) (#1482)"
    )


def test_dale2014_reads_alpha_dale_not_alpha(synthetic_ssp_wide, panchromatic_obs):
    """The knob that works is the one the wildcard cannot reach (#1482).

    Without this, a fix could satisfy the sweep above by freeing *any* live
    parameter while leaving Dale+2014's slope unreachable.
    """
    # Built with the wildcard FIXED, since '*': FREE now correctly refuses here.
    model = _build(synthetic_ssp_wide, panchromatic_obs, "dale2014", wildcard=FIXED)
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
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": _TAU_BC,
            "tau_diff": _TAU_DIFF,
            "emission": {"type": "dale2014", "alpha_dale": Uniform(1.0625, 4.0)},
        },
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
    with_emission = _build(synthetic_ssp_wide, panchromatic_obs, "dale2014", wildcard=FIXED)
    without = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=panchromatic_obs,
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
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
