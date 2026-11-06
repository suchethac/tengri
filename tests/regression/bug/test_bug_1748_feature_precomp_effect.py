# SPDX-License-Identifier: BSD-3-Clause
r"""``FeaturePrecomp`` must not be attached to a model it cannot help (#1748).

Since #1281 the fast nebular grid may serve photometry only when nothing downstream
reads ``sed_nebular``, because serving it requires zeroing the continuum. A dust
component reads it unless it can take the nebular from the grid, which needs the
stellar energy-balance LUT. Until c4f897d43 (2026-11-06) that LUT was gated on an IR
consumer, so **a dusty model with no dust emission was disarmed**: its pair cost the
same FLOPs as WavePrecomp alone (1.00x, bit-identical compiled FLOPs), while the advice,
two resolvers and ``CLAUDE.md`` still advertised a ~21x line speedup (#1748).

A dusty model with no IR now keeps the grid when ``FeaturePrecomp`` is configured. That
is the change the #1673 check was for: the grid's band photometry against
``approx=None`` over tau_bc and tau_diff across the prior, young and old SFH, logU and
logZ, at z 0.1 and 2, with and without dale2014, has a worst band error of 1.14e-3
(des_r, z 0.1, tau edges at 2.0), against 3e-4 at the prior draw. The eligible dale2014
model has the same worst case to about 1 %. The owner accepted up to about 1.2e-3 on
2026-11-06; the grid-edge bound is pinned in ``tests/contract``.

This is not a regression to undo. On the pre-#1281 tree the fast pair's photometry for
a dusty model differed from exact by **0.41 %** against 0.0115 % for a dust-free
control — the shortcut bought a biased answer, and a constant forward bias enters the
gradient multiplied by SNR (#1671). The grid's current worst band error is under a
third of that. What the guard now pins is the contract: a dusty no-IR model engages
the grid, and a model that cannot engage stays exact-equal to WavePrecomp.

**Why FLOPs and not wall clock.** A timing guard on this would be folded away: XLA
already proved it can compile the "exact" arm down to the fast one and make a
wall-clock guard pass on the cost it was checking (#1696). FLOPs read off
``compile().cost_analysis()`` are a property of the compiled graph, so exact equality
is unambiguous evidence that a config never reached it.

**The dust-free control is not optional.** Without it, "the ratio is 1.00x" is equally
consistent with "the LUT is inert" and "this fixture has nothing to tabulate", and the
second reading would let the defect through.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug

_PARAMS_BASE = {"sfh_delayed_log_total_mass": 10.0}


def _build(ssp, approx, *, dust_attenuation: bool):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    dust_group = (
        {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        }
        if dust_attenuation
        # Omitting `dust_attenuation=` still builds a DustSEDComponent, which would make the
        # control identical to the treatment in the one respect under test.
        else {"type": "none"}
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation=dust_group,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=approx,
    )


def _grad_flops(model):
    """FLOPs of the compiled photometry gradient, from the HLO cost analysis."""
    free = list(model.spec.free_params)
    defaults = {k: 10.0 if "log_total_mass" in k else 0.5 for k in free}

    def loss(v):
        params = dict(defaults)
        params["sfh_delayed_log_total_mass"] = v
        return jnp.sum(model.predict_photometry(params))

    return int(jax.jit(jax.grad(loss)).lower(jnp.asarray(10.0)).compile().cost_analysis()["flops"])


def test_setup_the_control_is_really_dust_free(ssp_data_fsps):
    """Guard the guard: if both arms carry dust, every ratio below is meaningless."""
    from tengri.forward.sed_model import _nebular_continuum_consumers

    dusty = _build(ssp_data_fsps, None, dust_attenuation=True)
    clean = _build(ssp_data_fsps, None, dust_attenuation=False)

    dusty_consumers = _nebular_continuum_consumers(dusty._build_component_chain())
    clean_consumers = _nebular_continuum_consumers(clean._build_component_chain())

    assert dusty_consumers, "the dusty arm has no sed_nebular consumer — it is not dusty"
    assert not clean_consumers, (
        f"the control still consumes sed_nebular ({[type(c).__name__ for c in clean_consumers]}), "
        "so it is not a control. Omitting dust= builds a dust component; pass "
        "dust={'type': 'none'}."
    )


def test_feature_precomp_still_pays_on_a_dust_free_model(ssp_data_fsps):
    """The lever must still work where it is legitimately available.

    Asserted before the dusty case, because a guard that only checks "no effect on
    dusty models" would pass if ``FeaturePrecomp`` were broken everywhere.
    """
    from tengri import FeaturePrecomp, WavePrecomp

    wave = _grad_flops(_build(ssp_data_fsps, (WavePrecomp(),), dust_attenuation=False))
    pair = _grad_flops(
        _build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp()), dust_attenuation=False)
    )
    assert pair < wave / 5.0, (
        f"FeaturePrecomp buys only {wave / pair:.2f}x on a dust-free model "
        f"({wave:,} -> {pair:,} gradient FLOPs). It should be an order of magnitude; "
        "the fast nebular grid is not engaging even where it is allowed to."
    )


def test_feature_precomp_pays_on_a_dusty_model_without_ir(ssp_data_fsps):
    """A dusty model with no IR emission engages the grid (c4f897d43, #1748, #1673).

    The grid's accuracy on this model is pinned in ``tests/contract``; this guard pins
    the speed contract: the pair's gradient must cost less than a third of WavePrecomp
    alone. Measured on the same fixture: see the commit message.
    """
    from tengri import FeaturePrecomp, WavePrecomp

    wave = _grad_flops(_build(ssp_data_fsps, (WavePrecomp(),), dust_attenuation=True))
    pair = _grad_flops(
        _build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp()), dust_attenuation=True)
    )
    assert pair < wave / 3.0, (
        f"FeaturePrecomp buys only {wave / pair:.2f}x on a dusty no-IR model "
        f"({wave:,} -> {pair:,} gradient FLOPs). The grid should engage here; check "
        "_nebular_grid_will_serve_dust and the LUT gate in sed_model.py (#1748)."
    )


def test_feature_precomp_stays_exact_on_a_dusty_model_that_cannot_engage(ssp_data_fsps):
    """A free attenuation-curve shape keeps the pair exactly equal to WavePrecomp (#1748).

    ``f_obscuration`` is free, so the stellar energy-balance LUT is refused and the
    dust must read ``sed_nebular``. The grid then serves nothing, and the compiled
    gradient is bit-identical to the WavePrecomp-only model, FLOP for FLOP.
    """
    from tengri import (
        DEFAULT,
        FeaturePrecomp,
        Fixed,
        Observation,
        Photometry,
        SEDModel,
        Uniform,
        WavePrecomp,
    )

    def build(approx):
        return SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=Observation(
                photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
            ),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                "tau_gyr": 1.0,
                "age_gyr": 5.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Uniform(0.0, 1.5),
                "tau_bc": 0.0,
                "f_obscuration": Uniform(0.0, 0.5),
            },
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            approx=approx,
        )

    wave = _grad_flops(build((WavePrecomp(),)))
    pair = _grad_flops(build((WavePrecomp(), FeaturePrecomp())))
    assert pair == wave, (
        f"FeaturePrecomp changed the compiled gradient on a dusty model that cannot "
        f"engage the grid ({wave:,} -> {pair:,}). It must stay exact-equal."
    )


def test_the_resolver_does_not_attach_a_config_that_cannot_engage(ssp_data_fsps):
    """No fit may pay a second compiled kernel for a no-op config.

    ``compile_signature()`` includes the approx state, so appending an inert
    ``FeaturePrecomp`` costs a distinct compiled kernel while changing no FLOPs.
    """
    import jax.numpy as jnp_

    from tengri.inference.fitter import Fitter, fast_nebular_can_engage

    dusty = _build(ssp_data_fsps, None, dust_attenuation=True)
    assert not fast_nebular_can_engage(dusty)

    n_bands = 3
    fitter = Fitter(dusty, jnp_.ones(n_bands), jnp_.ones(n_bands))
    # "auto" is the default policy every fit surface resolves through.
    resolved = fitter._resolve_fit_approx(dusty, "auto")

    state = getattr(resolved, "approx", None)
    attached = state is not None and getattr(state, "feature_precomp", False)
    assert not attached, (
        "the resolver attached FeaturePrecomp to a dusty model, where it is measured "
        "bit-identical in compiled FLOPs — a separate compiled kernel for no effect "
        "(#1748)."
    )
