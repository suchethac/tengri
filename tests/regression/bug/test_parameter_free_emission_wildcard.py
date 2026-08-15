# SPDX-License-Identifier: BSD-3-Clause
"""A parameter-free IR engine must not free 19 parameters it never reads (#1482).

#1482 was "``dust.emission '*': FREE`` frees the same seven parameters for every
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
``_declared_param_names`` narrow to the empty set. ``energy_balance_split``
does not carry the marker and keeps ``None``.

Note on ``dust_eta_balance``, the one live parameter in the table above: it is
the engine-agnostic energy-balance relaxation (``L_IR = eta * L_absorbed``),
declared in ``_params.py`` rather than on any engine. Every engine that
declares its own priors *already* excludes it from this wildcard -- ``dale2014``
frees only ``dust_alpha_dale`` -- so narrowing these two makes them consistent
with the other eleven rather than taking a live knob away from them. It remains
settable explicitly, and freeable through the ``dust`` group.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Engines documented as declaring no free parameters of their own.
_PARAMETER_FREE_ENGINES = ("pah_drude", "dh02_ce01")

_INERT_TOL = 1e-9


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
            sfh={"type": "dpl", "all_params": FIXED},
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "tau_bc": Fixed(2.0),
                "tau_diff": Fixed(1.5),
                "emission": {"type": engine, "all_params": FREE},
            },
            redshift=Fixed(0.5),
        )


def _sed(model, params) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(params).rest_sed())


@pytest.mark.parametrize("engine", _PARAMETER_FREE_ENGINES)
def test_parameter_free_engine_frees_no_inert_parameter(engine, ir_ssp, ir_obs):
    """Every parameter the wildcard frees must move the SED.

    Asserted as "no inert freed parameter" rather than "frees nothing", because
    the promise #1482 is about is that a freed dimension is one a fit can use.
    An engine that declares nothing and frees nothing satisfies it trivially;
    one that frees 19 no-op dimensions does not.
    """
    model = _build(ir_ssp, ir_obs, engine)
    freed = sorted(p for p in model.spec.free_params if p.startswith("dust_"))

    base = dict(model.spec.sample(jax.random.PRNGKey(0)))
    sed0 = _sed(model, base)
    denom = np.where(np.abs(sed0) > 0, np.abs(sed0), 1.0)

    inert = []
    for name in freed:
        lo, hi = model.spec.get_distribution(name).bounds
        worst = 0.0
        for value in np.linspace(float(lo), float(hi), 5):
            worst = max(
                worst,
                float(
                    np.max(np.abs(_sed(model, {**base, name: np.float64(value)}) - sed0) / denom)
                ),
            )
        if worst <= _INERT_TOL:
            inert.append(name)

    assert not inert, (
        f"{engine}: 'all_params': FREE freed {len(inert)} parameter(s) that cannot move "
        f"the SED anywhere in their own declared support: {inert}. A sampler explores "
        "every one of them and reports a confident posterior over dimensions the "
        "likelihood cannot see (#1482)."
    )


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
    if split is not None:
        assert getattr(split, "declares_no_parameters", False) is False, (
            "energy_balance_split reads six parameters declared in "
            "components/dust/_params.py; marking it parameter-free would pin all six."
        )
        assert _declared_param_names("energy_balance_split") is None
