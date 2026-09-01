# SPDX-License-Identifier: BSD-3-Clause
"""``dust.emission {'all_params': FREE}`` freed 14 parameters ``energy_balance_split`` never reads.

The last surviving instance of #1482, and the one its own fix could not reach.

#1482 narrowed the ``dust.emission`` wildcard to the selected engine's declared
parameters, which works for every engine that declares its priors on the class.
``energy_balance_split`` does not: its warm/cold knobs live in
``components/dust/_params.py``, because ``dust_eta_balance`` and the
energy-balance bookkeeping are shared with the attenuator and re-declaring them
on the class would raise a duplicate declaration. Its class-level ``_priors`` is
therefore empty -- indistinguishable, to ``_declared_param_names``, from
``pah_drude``, which is empty because it genuinely reads nothing.

Faced with that ambiguity ``_declared_param_names`` returned ``None`` and left
the sub-block unnarrowed, so the wildcard fell back to the static union of every
IR engine's parameters. Measured on ``origin/main``:

    freed 20, live 6, inert 14

The fourteen were ``dust_qpah``, ``dust_umin``, ``dust_gamma_dl``,
``dust_alpha_dale``, ``dust_alpha_dl14``, ``dust_alpha_mir``, ``dust_f_pah``,
``dust_qhac``, ``dust_lgU``, ``dust_T``, ``dust_alpha``, ``dust_beta_ir``,
``dust_epsilon_mbb`` and ``dust_log_ssfr`` -- knobs of Draine & Li, Dale,
THEMIS and MBB engines that are not built in this model at all. A sampler
explores every one at full cost, each posterior returns matching its prior, and
nothing in the fit says why.

The fix is the marker ``reads_parameters``, the complement of the
``declares_no_parameters`` that resolved the ``pah_drude`` side. An empty
``_priors`` is three questions, and the component is now asked which one it is
rather than inferred from.

Why this went unmeasured for so long: ``EMISSION_TYPES`` in
``tests/contract/test_dust_emission_wildcard.py`` was a hardcoded seven-name
tuple covering five of the thirteen production engines, and
``energy_balance_split`` is ``experimental`` besides. It is now derived from the
live menu, which is what surfaced this.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Knobs the engine's ``predict`` reads. ``dust_eta_balance`` is deliberately
#: not among them: it is the attenuator's, applied to ``L_ir`` upstream (this
#: component is called with ``eta_balance=1.0``), so it moves the prediction
#: only by scaling the input. ``dust_L_agn_ir`` is read but declares no
#: ``free_prior`` by design, so the wildcard cannot free it.
_ENGINE_KNOBS = frozenset(
    {
        "dust_T_warm",
        "dust_T_cold",
        "dust_f_cold",
        "dust_beta_warm",
        "dust_beta_cold",
        "dust_L_agn_ir",
    }
)

#: A sample of the foreign knobs that used to come back freed. Naming them
#: makes a regression legible: the failure message says *whose* parameters
#: leaked in, not merely that the count changed.
_FOREIGN = ("dust_qpah", "dust_umin", "dust_alpha_dale", "dust_lgU", "dust_epsilon_mbb")


@pytest.fixture(scope="module")
def panchromatic_ssp() -> SSPData:
    """SSP spanning 100 A - 1 cm, so far-IR engines have something to emit into."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1500)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def panchromatic_obs() -> Observation:
    """Six top-hats, 1500 A to 500 um -- UV through far-IR."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 5000.0, 2.0e4, 2.4e5, 1.0e6, 5.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp: SSPData, obs: Observation) -> SEDModel:
    """Deep attenuation, so L_absorbed is large and the IR term is real."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 2.0,
                "tau_diff": 1.5,
            },
            dust_emission={"type": "energy_balance_split", "all_params": FREE},
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )


def _freed_and_live(model: SEDModel) -> tuple[list[str], list[str]]:
    """Freed dust parameters, and which of them move ``predict_photometry``.

    Substitutes each parameter's value from other prior draws, so every trial
    value is inside that parameter's own support and no bound is violated.
    """
    freed = [p for p in model.spec.free_params if p.startswith("dust_")]
    draws = [dict(model.spec.sample(jax.random.PRNGKey(k))) for k in (0, 3, 11)]
    base_params = draws[0]
    base = np.asarray(model.predict_photometry(base_params))

    live: list[str] = []
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


def test_wildcard_frees_no_foreign_engine_parameter(panchromatic_ssp, panchromatic_obs):
    """The regression: 14 knobs of engines this model does not build were freed."""
    freed = set(_freed_and_live(_build(panchromatic_ssp, panchromatic_obs))[0])

    leaked = sorted(f for f in _FOREIGN if f in freed)
    assert not leaked, (
        f"dust.emission {{'type': 'energy_balance_split', 'all_params': FREE}} freed {leaked}, "
        "which belong to the Draine & Li / Dale / MBB engines this model never "
        "builds. The sub-block fell back to the static union of every IR engine's "
        "parameters because an empty class-level _priors could not be told from "
        "'declares nothing' (#1482)."
    )


def test_every_freed_parameter_moves_the_prediction(panchromatic_ssp, panchromatic_obs):
    """No freed dimension may be flat -- the property #1482 is really about.

    Stated over the freed set rather than against a fixed count, so that
    declaring a new prior for one of these knobs (#887 is doing this a
    subsystem at a time) does not turn this red for the wrong reason.
    """
    freed, live = _freed_and_live(_build(panchromatic_ssp, panchromatic_obs))
    inert = sorted(set(freed) - set(live))

    assert not inert, (
        f"{len(inert)} freed parameters do not move predict_photometry: {inert}. "
        f"(freed={sorted(freed)}, live={sorted(live)}) A sampler explores each at "
        "full cost and returns a posterior matching the prior (#1482)."
    )


def test_the_engines_own_knobs_are_still_freed(panchromatic_ssp, panchromatic_obs):
    """Narrowing must not overshoot into pinning the engine's real parameters.

    The opposite failure to the one above, and the reason ``_declared_param_names``
    returned ``None`` rather than an empty frozenset in the first place: answering
    "declares nothing" for an engine that declares elsewhere would pin every knob
    it reads, silently making ``'all_params': FREE`` a no-op.
    """
    freed = set(_freed_and_live(_build(panchromatic_ssp, panchromatic_obs))[0])

    # dust_L_agn_ir is excluded: it declares no free_prior by design, so no
    # wildcard can free it. The rest must all be present.
    expected = _ENGINE_KNOBS - {"dust_L_agn_ir"}
    missing = sorted(expected - freed)
    assert not missing, (
        f"'all_params': FREE no longer frees {missing}, which energy_balance_split's predict "
        "reads. The narrowing has overshot from 'scoped' into 'pinned'."
    )


def test_narrowing_does_not_orphan_eta_balance(panchromatic_ssp, panchromatic_obs):
    """Narrowing must not leave a live parameter freed by *no* wildcard.

    ``dust_eta_balance`` is the trap in this fix, and it caught the first
    attempt. ``predict`` never reads it -- it is applied to ``L_ir`` upstream by
    the attenuator, and this component runs with ``eta_balance=1.0`` -- so on
    physical grounds it reads as the attenuator's knob and the first
    ``reads_parameters`` left it out.

    But the grammar partitions it into ``dust.emission``, not ``dust``::

        dust={'eta_balance': ...}
        ValueError: 'eta_balance' is a 'dust.emission' parameter, not a 'dust'
        one, so writing it here would be silently ignored.

    So excluding it from the sub-block's declared set left it freed by nothing:
    ``dust={'all_params': FREE}`` does not reach it either (measured: that wildcard frees
    ``dust_tau_bc`` and ``dust_tau_diff``, and no eta_balance). Orphaning a live
    parameter is the mirror image of the bug being fixed -- inertness tests stay
    green, because a parameter that is never freed is never inert.
    """
    freed = set(_freed_and_live(_build(panchromatic_ssp, panchromatic_obs))[0])
    assert "dust_eta_balance" in freed, (
        "dust.emission {'all_params': FREE} no longer frees dust_eta_balance. The grammar "
        "assigns it to dust.emission, and dust={'all_params': FREE} does not reach it, so "
        "it is now freed by no wildcard at all despite moving the prediction."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group_wildcard = SEDModel.build(
            ssp_data=panchromatic_ssp,
            observation=panchromatic_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": FREE,
            },
            dust_emission={"type": "energy_balance_split", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )
    # Pins the partition this test's reasoning rests on: eta_balance is not a
    # dust-group parameter, so dust.emission is the only wildcard that can free
    # it. If that ever changes, this assertion should be revisited rather than
    # the one above deleted.
    assert "dust_eta_balance" not in set(group_wildcard.spec.free_params), (
        "dust {'all_params': FREE} now reaches dust_eta_balance, so it is no longer "
        "exclusively a dust.emission parameter -- re-derive which wildcard owns it."
    )
