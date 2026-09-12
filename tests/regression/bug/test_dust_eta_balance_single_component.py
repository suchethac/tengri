# SPDX-License-Identifier: BSD-3-Clause
"""``dust_eta_balance`` was a silently dead parameter on the single-screen path.

Declared ``src/tengri/components/dust/_params.py`` ("L_IR = eta * L_absorbed")
and wired in ``two_component.py`` (:class:`~tengri.components.dust.two_component.
DustSEDComponent`, the birth-cloud + diffuse-ISM screen), ``dust_eta_balance``
was never read by ``component.py``'s :class:`~tengri.components.dust.component.
DustAttenuationSEDComponent` (the single-screen ``dust_attenuation={'type':
'single_component', ...}`` path) -- grepping ``params.get("dust_eta_balance"``
in that file returned zero hits. ``apply()`` published ``L_ir = L_absorbed``
unconditionally, so freeing ``dust_eta_balance`` on this path had no effect on
the SED at all: a declared free parameter whose posterior always equals its
prior.

Separately, the ``dust_emission`` group's wildcard (``_wildcard_scopes`` in
``parameters/groups.py``) narrows ``'all_params': FREE`` to the SELECTED IR
engine's own declared parameters (#1482). ``dust_eta_balance`` is not declared
on any engine's class (it lives in ``components/dust/_params.py``, applied by
the ATTENUATOR, not by any emission engine's ``predict``), so every engine
except ``energy_balance_split`` (which states it via its own
``reads_parameters`` marker) silently excluded it: ``dust_emission={'type':
'schreiber2018', 'all_params': FREE}`` froze ``dust_eta_balance`` at
``Fixed(1.0)`` even though it moves the prediction once wired above -- the
same "live parameter freed by no wildcard" failure ``energy_balance_split``'s
marker was written to prevent, left open for every other engine.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.observation.photometry import FilterCurve
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _obs() -> Observation:
    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, *, eta_balance, dust_emission_type: str = "schreiber2018") -> SEDModel:
    """Single-component (screen-only) Calzetti attenuation + a dust IR engine."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(),
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_v": 0.5,
            },
            dust_emission={
                "type": dust_emission_type,
                "eta_balance": eta_balance,
                "other_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )


def _params(model: SEDModel) -> dict:
    return {**model.spec.get_fixed_values()}


def _l_ir_integral(model: SEDModel, params: dict) -> float:
    """abs(integral sed_dust_ir dnu) [arbitrary flux units, consistent across calls]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = model.predict_state(params)
    sed_dust_ir = np.asarray(state.derived["sed_dust_ir"])
    wave = np.asarray(state.wave)
    nu = C_AA / wave
    return abs(float(np.trapezoid(sed_dust_ir, nu)))


@pytest.mark.parametrize("dust_emission_type", ["schreiber2018", "dl07", "modified_blackbody"])
def test_eta_halves_the_emitted_ir_on_single_component_path(
    synthetic_ssp_wide, dust_emission_type
):
    """eta=0.5 vs eta=1.0 halves integral(sed_dust_ir) on the single-screen path.

    Before the fix this ratio was 1.0 for every ``dust_emission_type`` --
    ``eta`` changed the free-parameter's declared VALUE but never reached
    ``L_ir``, so the SED (and this integral) was identical regardless of eta.
    """
    m_full = _build(
        synthetic_ssp_wide, eta_balance=Fixed(1.0), dust_emission_type=dust_emission_type
    )
    m_half = _build(
        synthetic_ssp_wide, eta_balance=Fixed(0.5), dust_emission_type=dust_emission_type
    )

    e_full = _l_ir_integral(m_full, _params(m_full))
    e_half = _l_ir_integral(m_half, _params(m_half))

    assert e_full > 0.0
    ratio = e_full / e_half
    assert ratio == pytest.approx(2.0, rel=1e-6), (
        f"[{dust_emission_type}] eta=1.0/eta=0.5 emitted-IR ratio = {ratio:.6f} (expected 2.0). "
        "dust_eta_balance is not reaching L_ir on the single-component attenuation path."
    )


@pytest.mark.parametrize("dust_emission_type", ["schreiber2018", "dl07", "modified_blackbody"])
def test_eta_default_reproduces_l_ir_equal_l_absorbed(synthetic_ssp_wide, dust_emission_type):
    """Default eta=1.0 keeps L_ir == L_absorbed exactly (strict energy balance).

    This is the "must not change any default SED" guard: at eta=1.0 the eta
    multiply is a no-op (log10(1.0) == 0), so wiring it in must reproduce
    today's L_ir bit-for-bit.
    """
    model = _build(
        synthetic_ssp_wide, eta_balance=Fixed(DEFAULT), dust_emission_type=dust_emission_type
    )
    assert "dust_eta_balance" not in model.spec.free_params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = model.predict_state(_params(model))
    np.testing.assert_allclose(
        float(state.derived["L_ir"]),
        float(state.derived["L_absorbed"]),
        rtol=1e-12,
        err_msg=(
            f"[{dust_emission_type}] default dust_eta_balance=1.0 no longer reproduces "
            "L_ir == L_absorbed on the single-component path."
        ),
    )


def test_grad_wrt_eta_balance_is_nonzero(synthetic_ssp_wide):
    """jax.grad(sum(predict_photometry), wrt dust_eta_balance) != 0.

    A dead parameter has an exactly-zero gradient everywhere; a live one that
    scales L_ir linearly must not.
    """
    model = _build(synthetic_ssp_wide, eta_balance=Fixed(DEFAULT))
    base_params = _params(model)

    def loss(eta):
        p = {**base_params, "dust_eta_balance": eta}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.sum(model.predict_photometry(p))

    grad = jax.grad(loss)(1.0)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0, "grad(predict_photometry, dust_eta_balance) == 0: eta is inert."


@pytest.mark.parametrize("dust_emission_type", ["schreiber2018", "dl07", "modified_blackbody"])
def test_dust_emission_wildcard_frees_eta_balance(synthetic_ssp_wide, dust_emission_type):
    """``dust_emission={'all_params': FREE}`` must free dust_eta_balance.

    Before the groups.py fix, every engine except ``energy_balance_split``
    excluded it from the narrowed wildcard scope (#1482's own narrowing,
    applied one entry too literally): the parameter is not declared on any
    emission engine's class, so ``_declared_param_names(engine)`` never
    contained it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=_obs(),
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_v": 0.5,
            },
            dust_emission={"type": dust_emission_type, "all_params": FREE},
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )
    assert "dust_eta_balance" in model.spec.free_params, (
        f"[{dust_emission_type}] dust_emission={{'all_params': FREE}} does not free "
        "dust_eta_balance -- the wildcard scope still narrows to the selected engine's "
        "own declared parameters only, orphaning this live cross-engine knob."
    )
