# SPDX-License-Identifier: BSD-3-Clause
r"""Dust IR emission must be exactly linear in ``L_ir`` — or say so (#1206).

Every dust emission model normalizes a template shape to the absorbed
luminosity: ``sed = (L_ir / integral) * shape``. That makes ``sed_dust_ir``
exactly proportional to ``L_ir``, which is what lets the float32 work factor
``L_ir`` out of the template and re-apply it in log space — ``L_ir`` is ~2.4e43
and therefore ``inf`` in pure float32, while ``log_L_ir`` is finite.

The proportionality is an *assumption* about every registered emission model,
and it is not free: ``energy_balance_split`` computes

.. math:: L_{\rm IR}^{\rm tot} = \eta L_{\rm stellar} + L_{\rm AGN,IR}

which is **affine, not linear** — doubling :math:`\eta` does not double the
output once :math:`L_{\rm AGN,IR}` is comparable to the stellar term. It looks
linear at default settings only because ``dust_L_agn_ir`` defaults to 0.

So this file pins the invariant per model rather than assuming it globally. A
new emission model that is not proportional to ``L_ir`` fails here, loudly,
instead of silently returning wrong fluxes once the log-domain migration lands.

``dust_eta_balance`` multiplies ``L_ir`` directly (``L_ir = eta * L_absorbed``),
so it is the cleanest end-to-end handle on the scaling — it exercises the real
wiring rather than poking a component's internals.
"""

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.builders.dust import emission as emission_builders

pytestmark = pytest.mark.contract

#: Models whose output is NOT proportional to L_ir, with the reason. The
#: log-domain migration must handle these explicitly (combining the two terms
#: with ``log10_add``) rather than factoring a single scale out.
AFFINE_MODELS = {
    "energy_balance_split": "L_ir_total = eta * L_stellar + L_agn_ir (additive AGN term)",
}

#: Models whose non-proportionality is a different shape than AFFINE_MODELS's
#: additive term: the template SHAPE itself is looked up as a function of
#: L_ir (a grid lookup), not merely scaled by it. BOSA (Boquien & Salim 2021)
#: interpolates on a (log L_TIR, log sSFR) grid keyed by log10(L_TIR/Lsun), so
#: ``sed_dust_ir(2*L) != 2*sed_dust_ir(L)`` in general (#2272: before the
#: fix, ``factors_l_ir=True`` hid this by always evaluating the shape at the
#: unit luminosity; a second, compounding defect -- the axis lookup used
#: erg/s directly with no Lsun conversion -- would otherwise have saturated
#: any astrophysically realistic probe to the grid's ceiling node regardless,
#: masking the effect even with ``factors_l_ir=False``. Both are fixed.)
#:
#: The eta=1/eta=2 probe in ``test_sed_dust_ir_is_proportional_to_l_ir``
#: actually DOES see this now: at the realistic stellar mass that probe
#: builds, doubling eta moves ``log10(L_ir/Lsun)`` from ~9.81 to ~10.11 --
#: both comfortably inside the grid's interior, at different interpolation
#: nodes -- for a measured ~8.6% normalized-shape difference, nine orders of
#: magnitude past the ``deviation < 1e-10`` threshold. bosa is listed here
#: (and additionally exercised by
#: :func:`test_shape_nonlinear_models_really_are_nonlinear`, at two more
#: widely-separated L_ir values chosen directly on the grid's own axis) so
#: the ledger records the property as a stated, understood exemption rather
#: than a bare skip, and the dedicated proof pins the effect independently of
#: this file's SFH/eta machinery.
SHAPE_NONLINEAR_MODELS = {
    "bosa": "template SHAPE is looked up by log10(L_TIR/Lsun) on a (log L_TIR, "
    "log sSFR) grid, so it is a genuine function of L_ir rather than merely scaled by it",
}


def _is_standalone(model_name: str) -> bool:
    """Whether ``SEDModel.build`` accepts this type as a model's only emitter."""
    from tengri.parameters.groups import _standalone_dust_emission_types

    return model_name in _standalone_dust_emission_types()


def _flat_model(ssp, model_name, eta, **emission_extra):
    """The same model through the flat ``Parameters`` expert escape hatch.

    ``SEDModel.build`` refuses a *building block* — a backend that scales a
    template by ``L_ir`` without renormalizing to it, ``pah_drude`` being the
    one shipped — as a model's only dust emitter. The proportionality this file
    pins is a property of the **component**, not of the grammar, and it is what
    ``EmissionComponent.factors_l_ir`` rests on for every backend including that
    one, so the sweep must still reach it. The flat form is how, and it is
    deliberate rather than a workaround: composing a custom model out of pieces
    is exactly what that form is for.
    """
    from tengri import Parameters

    kwargs = {
        "mean_sfh_type": "delayed",
        "sfh_delayed_tau_gyr": Fixed(1.0),
        "sfh_delayed_age_gyr": Fixed(5.0),
        "sfh_delayed_log_total_mass": Fixed(10.0),
        "met_logzsol": Fixed(0.0),
        "dust_tau_bc": Fixed(1.0),
        "dust_tau_diff": Fixed(0.7),
        "dust_emission": model_name,
        "dust_eta_balance": Fixed(eta),
        "redshift": Fixed(0.0),
    }
    kwargs.update({f"dust_{k}": v for k, v in emission_extra.items()})
    return SEDModel(Parameters(**kwargs), ssp)


def _model(ssp, model_name, eta, **emission_extra):
    """Two-component dust with one emission model at a given eta_balance."""
    if not _is_standalone(model_name):
        return _flat_model(ssp, model_name, eta, **emission_extra)
    emission = {"type": model_name, "all_params": Fixed(DEFAULT), "eta_balance": Fixed(eta)}
    emission.update(emission_extra)
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.7),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission=emission,
        redshift=Fixed(0.0),
    )


def _sed_dust_ir(ssp, model_name, eta, **extra):
    derived = _model(ssp, model_name, eta, **extra).predict_state({}).derived
    if "sed_dust_ir" not in derived:
        pytest.skip(f"{model_name} publishes no sed_dust_ir (see #1278)")
    return np.asarray(derived["sed_dust_ir"], dtype=np.float64)


@pytest.mark.parametrize("model_name", sorted(emission_builders.available()))
def test_sed_dust_ir_is_proportional_to_l_ir(synthetic_ssp_wide, model_name):
    """Doubling ``L_ir`` must exactly double ``sed_dust_ir``.

    This is the invariant the float32 log-domain migration rests on. Models
    listed in :data:`AFFINE_MODELS` are expected to violate it and are checked
    for the opposite.
    """
    try:
        s1 = _sed_dust_ir(synthetic_ssp_wide, model_name, 1.0)
        s2 = _sed_dust_ir(synthetic_ssp_wide, model_name, 2.0)
    except pytest.skip.Exception:
        raise
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"{model_name} not constructible here: {type(exc).__name__}: {exc}")

    nonzero = np.abs(s1) > 0
    if not nonzero.any():
        pytest.skip(f"{model_name} emits nothing at eta_balance=1")

    ratio = s2[nonzero] / s1[nonzero]
    deviation = float(max(abs(ratio.min() - 2.0), abs(ratio.max() - 2.0)))

    if model_name in AFFINE_MODELS:
        pytest.skip(f"{model_name} is affine by construction: {AFFINE_MODELS[model_name]}")
    if model_name in SHAPE_NONLINEAR_MODELS:
        pytest.skip(
            f"{model_name} is not proportional to L_ir: {SHAPE_NONLINEAR_MODELS[model_name]} "
            "(see test_shape_nonlinear_models_really_are_nonlinear for the dedicated, "
            "independent proof at two more widely-separated L_ir values)"
        )
    assert deviation < 1e-10, (
        f"{model_name} is not proportional to L_ir (ratio range "
        f"[{ratio.min()!r}, {ratio.max()!r}], deviation {deviation:.3e}). "
        "The float32 log-domain migration factors L_ir out of the template and "
        "re-applies it in log space, which is only valid for a proportional "
        "model. Either make it proportional or add it to AFFINE_MODELS and "
        "handle it explicitly."
    )


@pytest.mark.parametrize("model_name", sorted(AFFINE_MODELS))
def test_affine_models_really_are_affine(synthetic_ssp_wide, model_name):
    """The affine exemption must be earned, not asserted.

    An entry in :data:`AFFINE_MODELS` that has since become proportional would
    silently keep its exemption, so prove the non-proportionality here: with the
    additive term comparable to the stellar term, the ratio must depart from 2.
    """
    stellar_l_ir = float(
        np.asarray(_model(synthetic_ssp_wide, model_name, 1.0).predict_state({}).derived["L_ir"])
    )
    assert stellar_l_ir > 0.0, "setup: expected a positive absorbed luminosity"

    # Additive term equal to the stellar term: L_tot = eta*L + L, so doubling
    # eta takes the ratio from 2 to (2L + L)/(L + L) = 1.5, not 2.
    s1 = _sed_dust_ir(synthetic_ssp_wide, model_name, 1.0, L_agn_ir=Fixed(stellar_l_ir))
    s2 = _sed_dust_ir(synthetic_ssp_wide, model_name, 2.0, L_agn_ir=Fixed(stellar_l_ir))
    nonzero = np.abs(s1) > 0
    ratio = s2[nonzero] / s1[nonzero]

    assert abs(float(ratio.max()) - 2.0) > 1e-6, (
        f"{model_name} is listed in AFFINE_MODELS but behaved proportionally "
        f"(ratio {ratio.max()!r}); if it is now linear, remove the exemption"
    )
    # The exact affine prediction, which also pins that L_agn_ir is applied.
    np.testing.assert_allclose(ratio, 1.5, rtol=1e-9)


@pytest.mark.parametrize("model_name", sorted(SHAPE_NONLINEAR_MODELS))
def test_shape_nonlinear_models_really_are_nonlinear(model_name):
    """The SHAPE_NONLINEAR_MODELS exemption is earned, not asserted (#2272).

    Unlike :func:`test_affine_models_really_are_affine`'s eta-doubling probe,
    an eta probe at a realistic stellar mass cannot demonstrate bosa's
    non-proportionality (see the comment on :data:`SHAPE_NONLINEAR_MODELS`):
    both probe luminosities clip to the same grid-ceiling row. Call the
    component directly at two erg/s ``L_ir`` values that land, after the
    Lsun conversion (#2272 part 2), inside the grid's own ``log_ltir_grid``
    axis instead, exactly as
    ``tests/regression/bug/test_bug_2272_bosa_shape_follows_l_ir.py`` does
    for the dedicated regression pin.
    """
    from tengri.components.dust.emission.templates.bosa import BosaIRSEDComponent
    from tengri.protocols.component import ForwardState
    from tengri.utils.sed_quantities import LOG10_L_SUN

    component = BosaIRSEDComponent()
    grid = component.load()
    if grid is None:
        pytest.skip("BOSA template grid not on disk")

    log_ltir_grid = np.asarray(grid["log_ltir_grid"])
    span = float(log_ltir_grid.max() - log_ltir_grid.min())
    lo_axis = float(log_ltir_grid.min() + 0.25 * span)
    hi_axis = float(log_ltir_grid.min() + 0.75 * span)
    # The grid's axis is log10(L_TIR/Lsun); L_ir is erg/s, so add LOG10_L_SUN
    # to land the erg/s value at the intended axis percentile.
    l_lo = 10.0 ** (lo_axis + LOG10_L_SUN)
    l_hi = 10.0 ** (hi_axis + LOG10_L_SUN)

    import jax.numpy as jnp

    wave = jnp.geomspace(3.0e3, 3.0e8, 3000)
    params = {"dust_log_ssfr": jnp.asarray(-10.0)}

    def _sed(l_ir):
        state = ForwardState(
            wave=wave,
            derived={"L_ir": jnp.asarray(l_ir), "log_L_ir": jnp.asarray(np.log10(l_ir))},
        )
        return np.asarray(component.apply(state, params).derived["sed_dust_ir"], dtype=np.float64)

    sed_lo = _sed(l_lo)
    sed_hi = _sed(l_hi)
    norm_lo = sed_lo / l_lo
    norm_hi = sed_hi / l_hi
    scale = np.maximum(np.abs(norm_lo), np.abs(norm_hi))
    mask = scale > 1.0e-3 * scale.max()
    max_rel_diff = float(np.max(np.abs(norm_hi[mask] - norm_lo[mask]) / scale[mask]))

    assert max_rel_diff > 0.05, (
        f"{model_name} is listed in SHAPE_NONLINEAR_MODELS but its normalized shape barely "
        f"moved between L_ir={l_lo:.3e} and L_ir={l_hi:.3e} (max relative difference "
        f"{max_rel_diff:.3e}); if it is now proportional, remove the exemption"
    )


def test_every_advertised_emission_model_can_be_evaluated(synthetic_ssp_wide):
    """The builders menu must not name a model the forward pass rejects.

    The linearity sweep above is parametrized off the same menu and skips
    whatever raises, so a menu entry that does not resolve costs a silent hole
    in the sweep rather than a failure. This asserts the two registries agree.

    Evaluation, not construction: a build-only check proves nothing about
    whether the forward pass runs.

    ``_model`` routes each name through the surface that is supposed to accept
    it — ``SEDModel.build`` for a model, the flat ``Parameters`` form for a
    building block such as ``pah_drude``, which the builder refuses on purpose.
    That keeps this a check on *evaluability* rather than a restatement of
    which names the grammar happens to allow; the refusal itself is pinned in
    ``tests/contract/test_valid_dust_types_parity.py``.

    This carried an ``xfail(strict=True)`` exempting ``dh02_ce01``, which
    ``emission_builders.available()`` advertised while ``predict_state``
    raised ``ValueError: ... not found in registry``. The ratchet fired on
    2026-08-16: the marker went XPASS after #1807 landed on main, with the
    companion check confirming the name was still advertised — so it was
    registered rather than quietly dropped from the menu, which is the outcome
    the exemption was written to wait for. Marker and exemption both removed;
    the sweep now covers every advertised name with no holes.
    """
    broken = {}
    for name in sorted(emission_builders.available()):
        try:
            _model(synthetic_ssp_wide, name, 1.0).predict_state({})
        except (ValueError, KeyError) as exc:
            broken[name] = f"{type(exc).__name__}: {exc}"
    assert not broken, f"advertised but not evaluable: {sorted(broken)}"


def test_the_menu_is_not_empty():
    """The sweep above is vacuous if the menu it reads is empty.

    This replaces ``test_the_unevaluable_list_still_describes_reality``, which
    existed only to say *which way* the exemption above had been resolved. With
    the exemption gone that question is answered, but the sweep is parametrized
    off ``available()`` and would pass trivially on an empty menu, so the one
    thing still worth pinning is that the menu has entries.
    """
    advertised = set(emission_builders.available())
    assert len(advertised) >= 5, (
        f"the dust-emission builders menu advertises only {sorted(advertised)}; "
        f"the sweep above is parametrized off it and proves nothing if it is empty"
    )
