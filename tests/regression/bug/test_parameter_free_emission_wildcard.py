# SPDX-License-Identifier: BSD-3-Clause
"""A parameter-free IR engine must not free 19 parameters it never reads (#1482).

#1482 was "``dust.emission 'all_params': FREE`` frees the same seven parameters for every
backend". Its fix narrows the sub-block wildcard to the selected engine's own
declared priors, and works for eleven of thirteen production engines.

Two were left unnarrowed **by design**, and the reason is documented on
``_declared_param_names``: an empty ``_priors`` cannot distinguish
``pah_drude``, which is genuinely parameter-free, from ``energy_balance_split``,
which reads six real knobs declared in ``components/dust/_params.py`` rather
than on its class. Returning an empty set for both would pin all six of the
latter's parameters -- the failure the narrowing exists to prevent -- so the
function returns ``None`` and the wildcard is left alone.

Left alone means the static union ``_DUST_EMISSION_PARAM_NAMES``, which is
#1482's original root cause. Measured under #1482's own fixture:

===============  ======  ======  =======
engine           freed   live    inert
===============  ======  ======  =======
``pah_drude``    20      1       **19**
``dh02_ce01``    20      1       **19**
eleven others    --      all     0
===============  ======  ======  =======

``dh02_ce01`` is not an old case: it was registered recently and inherited the
unscoped wildcard, so the residual grows as engines are added.

The fix separates the two cases by asking rather than inferring:
``declares_no_parameters = True`` says "genuinely none", and only then does
``_declared_param_names`` narrow to the empty set.

``energy_balance_split`` answers the same question from the other side. It
originally kept ``None`` -- correct for this bug, but it left *its* wildcard
unnarrowed too, and a later census measured that engine at **20 freed, 6 read,
14 inert**: the same defect, in the engine this fix deliberately did not touch.
It now carries ``reads_parameters``, naming the knobs it reads. So an empty
``_priors`` is three questions and each gets its own answer, with ``None``
reserved for a component that has stated nothing.

Note on ``dust_eta_balance``, the one live parameter in the table above: it is
the engine-agnostic energy-balance relaxation (``L_IR = eta * L_absorbed``),
declared in ``_params.py`` rather than on any engine. Every engine that
declares its own priors *already* excludes it from this wildcard -- ``dale2014``
frees only ``dust_alpha_dale`` -- so narrowing these two makes them consistent
with the other eleven rather than taking a live knob away from them.

An earlier revision of this note added "and freeable through the ``dust``
group". That was wrong, and it is corrected here rather than deleted because
the mistake is instructive: the grammar partitions ``eta_balance`` into
**dust.emission**, so ``dust={'eta_balance': ...}`` raises *"'eta_balance' is a
'dust.emission' parameter, not a 'dust' one"* and ``dust={'all_params': FREE}`` does not
reach it either (measured: that wildcard frees ``dust_tau_bc`` and
``dust_tau_diff`` only). For a parameter-free engine that is harmless -- there
is no engine to un-anchor. For ``energy_balance_split`` it is not, which is why
that engine's ``reads_parameters`` includes ``dust_eta_balance`` even though its
``predict`` never reads it: leaving it out would have made a live parameter
freeable by no wildcard at all.

Since #2187, the parameter-free engines refuse the ``'all_params': FREE`` wildcard
at build time with ``ParameterError("covers no parameters")``, superseding the earlier
check that verified no inert dimension entered the fit. The question has moved from
"frees nothing, satisfies the promise trivially" to "refuses the zero-coverage wildcard
at the build step", so the inert-parameter analysis no longer applies.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.config import ParameterError
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Engines documented as declaring no free parameters of their own.
_PARAMETER_FREE_ENGINES = ("pah_drude", "dh02_ce01")


@pytest.fixture(scope="module")
def ir_ssp() -> SSPData:
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
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def ir_obs() -> Observation:
    """1500 A - 500 um, so dust IR emission has somewhere to land."""

    def _tophat(center: float, frac: float = 0.25, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    centers = (1500.0, 3500.0, 6200.0, 2.4e5, 5.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ir_ssp, ir_obs, engine: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ir_ssp,
            observation=ir_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(2.0),
                "tau_diff": Fixed(1.5),
            },
            dust_emission={"type": engine, "all_params": FREE},
            redshift=Fixed(0.5),
        )


@pytest.mark.parametrize("engine", _PARAMETER_FREE_ENGINES)
def test_parameter_free_engine_wildcard_is_refused(engine, ir_ssp, ir_obs):
    """The zero-coverage wildcard must be refused at build time.

    Both engines take the same path even though #2179 separately refuses
    ``pah_drude`` as a model's only dust emitter: the wildcard's covered-0
    ``ParameterError`` fires during ``parse_groups``, before the only-emitter
    check is reached (measured), so the parametrization deliberately stays
    over the full pair rather than the standalone-selectable subset.

    Before #2187, the empty narrowing (from the ``declares_no_parameters`` marker
    added during #1482) made ``dust_emission={'type': engine, 'all_params': FREE}``
    build successfully with the wildcard freeing nothing — indistinguishable at the
    call site from a working wildcard. Since #2187, the zero-coverage wildcard is
    explicitly refused at build with ``ParameterError("covers no parameters")``. The
    check has moved from silent success to explicit refusal at the build step.
    """
    with pytest.raises(ParameterError, match=r"covers no parameters"):
        _build(ir_ssp, ir_obs, engine)


@pytest.mark.parametrize("engine", _PARAMETER_FREE_ENGINES)
def test_parameter_free_engine_is_declared_not_inferred(engine):
    """The narrowing must come from a declaration, not from an empty ``_priors``.

    ``energy_balance_split`` also has empty ``_priors`` while reading six real
    parameters, so inferring "declares nothing" from "declares nothing on the
    class" would pin those six. The marker is what separates them, and this
    asserts the separation still exists rather than trusting it.
    """
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES, _REGISTRY
    from tengri.parameters.groups import _declared_param_names

    cls = _REGISTRY.get(_EMISSION_TYPE_ALIASES.get(engine, engine))
    assert cls is not None, f"{engine} is not a registered component"
    assert getattr(cls, "declares_no_parameters", False) is True, (
        f"{engine} must declare `declares_no_parameters = True`; without it an empty "
        "_priors is indistinguishable from energy_balance_split's six knobs."
    )
    assert _declared_param_names(engine) == frozenset()

    split = _REGISTRY.get("energy_balance_split")
    assert split is not None, "probe setup failed: energy_balance_split was not registered"
    assert getattr(split, "declares_no_parameters", False) is False, (
        "energy_balance_split reads six parameters declared in "
        "components/dust/_params.py; marking it parameter-free would pin all six."
    )
    # It answers the same question from the other side, via
    # ``reads_parameters``. What matters is that the two engines get
    # *different* answers -- this one must never be narrowed to the empty
    # set, which is what would pin every knob it reads.
    split_declared = _declared_param_names("energy_balance_split")
    assert split_declared, (
        "energy_balance_split narrowed to an empty/None declared set. Empty "
        "would pin all six knobs it reads; the marker that distinguishes it "
        f"from {engine} has stopped working. Got: {split_declared!r}"
    )
    assert split_declared != _declared_param_names(engine), (
        f"energy_balance_split and {engine} now narrow to the same set, so the "
        "empty-_priors ambiguity is no longer being resolved."
    )
    assert "dust_T_warm" in split_declared, (
        "energy_balance_split's declared set no longer contains dust_T_warm, "
        f"a knob its predict reads. Got: {sorted(split_declared)}"
    )
