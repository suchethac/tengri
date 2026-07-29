# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: ``dust.emission``'s ``'*'`` must free parameters the backend reads.

``dust={'emission': {'type': X, '*': FREE}}`` expands ``'*'`` against
``_DUST_EMISSION_PARAM_NAMES`` (``parameters/groups.py``) — a static union over
*every* dust-emission backend, computed with no reference to ``X``. So the same
seven names are freed whichever engine is selected, and for two engines not one
of them is a parameter that engine reads (#1482):

===================  =============  ==============
``emission.type``    freed by ``*``  actually live
===================  =============  ==============
``dale2014``         7              **0**
``casey2012``        7              **0**
``draine_li2007``    7              1
``themis``           7              2
===================  =============  ==============

Nothing warns, because every freed name is a real declared parameter with real
bounds — just one belonging to a different engine. A sampler explores up to
seven dimensions that cannot move the likelihood by one ULP, while the knob that
*would* have worked stays at its default: Dale+2014 reads ``dust_alpha_dale``
(varying it moves the photometry 9x), and ``dust_alpha_dale`` is not in the
freed set. ``dust_alpha`` is THEMIS's slope.

This pins the invariant rather than the instances: **whatever** ``'*'`` expands
to, at least one freed parameter must move the observable. Adding a backend
whose parameters the bucket does not cover fails here on the day it lands.

The bands span 1500 A - 500 um so dust IR emission has somewhere to land — an
optical-only filter set would report every dust-emission parameter as inert for
the honest reason that its emission falls outside the bandpasses, which is
filter coverage rather than a wiring defect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel
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

#: Backends for which the wildcard currently frees *no* live parameter (#1482).
#: ``strict=True`` so the fix cannot land without deleting these entries.
ZERO_LIVE_PENDING_1482 = {"dale2014", "casey2012"}

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


def _build(ssp, obs, emission_type):
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
            "emission": {"type": emission_type, "*": FREE},
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
    if emission_type in ZERO_LIVE_PENDING_1482:
        pytest.xfail(f"#1482: dust.emission '*' frees no live parameter for {emission_type}")

    model = _build(synthetic_ssp_wide, panchromatic_obs, emission_type)
    freed, live = _live_params(model)

    assert live, (
        f"dust.emission {{'type': '{emission_type}', '*': FREE}} freed {freed} "
        f"and not one of them moves predict_photometry — every freed dimension "
        f"is a silent no-op a sampler would explore for free (#1482)"
    )


def test_the_wildcard_expansion_ignores_the_selected_backend(synthetic_ssp_wide, panchromatic_obs):
    """Pin the root cause: the freed set does not depend on ``emission.type``.

    Deleting this test is the signal that #1482 is fixed — once ``'*'`` resolves
    per backend the freed sets must differ, and this assertion inverts.
    """
    freed_sets = {
        etype: tuple(_build(synthetic_ssp_wide, panchromatic_obs, etype).spec.free_params)
        for etype in ("dale2014", "themis", "casey2012")
    }
    distinct = set(freed_sets.values())

    assert len(distinct) == 1, (
        "dust.emission '*' now expands per backend — #1482 is fixed. Remove this "
        f"test and the ZERO_LIVE_PENDING_1482 xfails. Freed sets: {freed_sets}"
    )


def test_dale2014_reads_alpha_dale_not_alpha(synthetic_ssp_wide, panchromatic_obs):
    """The knob that works is the one the wildcard omits (#1482).

    Without this, a fix could satisfy the sweep above by freeing *any* live
    parameter while leaving Dale+2014's slope unreachable.
    """
    model = _build(synthetic_ssp_wide, panchromatic_obs, "dale2014")
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    def photometry_at(name, value):
        return np.asarray(model.predict_photometry({**params, name: jnp.asarray(value)}))

    # THEMIS's slope: accepted, validated, and ignored by Dale+2014.
    assert np.allclose(photometry_at("dust_alpha", 1.0625), photometry_at("dust_alpha", 2.5)), (
        "dust_alpha now moves a dale2014 model — the parameter split changed"
    )

    # Dale+2014's own slope: wired correctly, but never freed by the wildcard.
    spread = np.max(
        np.abs(
            photometry_at("dust_alpha_dale", 1.0625) / photometry_at("dust_alpha_dale", 2.5) - 1.0
        )
    )
    assert spread > 1e-3, f"dust_alpha_dale no longer drives dale2014 (rel change {spread:.2e})"
    assert "dust_alpha_dale" not in model.spec.free_params, (
        "dust_alpha_dale is now freed by the wildcard — #1482 is fixed, update this test"
    )


def test_dust_emission_actually_contributes(synthetic_ssp_wide, panchromatic_obs):
    """Anti-vacuity: if IR emission were negligible, no parameter could be live.

    The sweep above would then pass or fail for reasons unrelated to the
    wildcard. Pin that the far-IR bands are genuinely emission-dominated.
    """
    with_emission = _build(synthetic_ssp_wide, panchromatic_obs, "dale2014")
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
