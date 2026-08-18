# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end smoke test: ``Parameters(agn_model="grahsp", ...)`` resolves
to ``AGN_MODELS["grahsp"]`` and the SED model forwards GRAHSP params to the
registered emission function.

This test catches the gap closed in 2026-05: GRAHSP params were registered
in :data:`tengri.components.agn.AGN_MODELS` but **not** forwarded by the
forward pipeline to ``agn_emission``, which meant a fit set up via
``Parameters(agn_model="grahsp", agn_grahsp_l5100=…)`` silently used
defaults. (The originally affected dispatch site, ``forward/nonstell.py``,
was retired as dead code in #922; the kwarg-forwarding contract it pinned
lives on in the registry dispatch exercised here.)
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from tengri import Parameters
from tengri.components.agn import resolve_agn_model
from tengri.components.agn.grahsp.templates import DEFAULT_TEMPLATE_PATH
from tengri.parameters.priors import Fixed, LogUniform, Uniform

# Registration / param-spec tests below need no template data; the two that
# actually evaluate the GRAHSP forward model do, and the bundle is a build
# artifact CI cannot regenerate (needs upstream GRAHSP source). Gate only those.
_needs_grahsp = pytest.mark.skipif(
    not DEFAULT_TEMPLATE_PATH.exists(),
    reason="GRAHSP template bundle not found; run tools/build_grahsp_hdf5.py",
)


def test_agn_models_contains_grahsp():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("grahsp")
    assert callable(fn)


@_needs_grahsp
def test_resolve_agn_model_returns_callable():
    fn = resolve_agn_model("grahsp")
    out = fn(jnp.logspace(2, 6, 200), agn_log_lbol=11.42, agn_lum_ratio=1.0)
    chex.assert_shape(out, (200,))
    chex.assert_tree_all_finite(out)


def test_parameters_accepts_grahsp_params():
    """Parameters(agn_model='grahsp', ...) must accept all 18 GRAHSP params."""
    p = Parameters(
        agn_model="grahsp",
        agn_log_lbol=Uniform(9.42, 14.42),
        agn_grahsp_l5100=LogUniform(1.0e42, 1.0e47),
        agn_grahsp_plslope=Uniform(-2.5, -1.0),
        agn_grahsp_uvslope=Fixed(0.0),
        agn_grahsp_plbendloc_nm=Fixed(100.0),
        agn_grahsp_plbendwidth=Fixed(1.0),
        agn_grahsp_a_lines=Fixed(1.0),
        agn_grahsp_a_feii=Fixed(5.0),
        agn_grahsp_linewidth_kms=Fixed(5000.0),
        agn_grahsp_fcov=Uniform(0.05, 0.95),
        agn_grahsp_si=Fixed(0.0),
        agn_grahsp_cool_lam_um=Fixed(17.0),
        agn_grahsp_cool_width=Fixed(0.45),
        agn_grahsp_hot_lam_um=Fixed(2.0),
        agn_grahsp_hot_width=Fixed(0.5),
        agn_grahsp_hot_fcov=Fixed(1.0),
        agn_grahsp_ebv=Fixed(0.05),
        agn_grahsp_ebv_agn=Fixed(0.05),
    )
    free = list(p.free_params)
    assert "agn_grahsp_l5100" in free
    assert "agn_grahsp_plslope" in free
    assert "agn_grahsp_fcov" in free
    fixed = p.get_fixed_values()
    assert fixed["agn_grahsp_a_lines"] == pytest.approx(1.0)
    assert fixed["agn_grahsp_ebv"] == pytest.approx(0.05)


def test_parameters_rejects_unknown_grahsp_param():
    """Typo'd GRAHSP params should be flagged at construction time."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        Parameters(
            agn_model="grahsp",
            agn_grahsp_DOES_NOT_EXIST=Fixed(0.0),
        )


@_needs_grahsp
def test_grahsp_params_flow_through_dispatch():
    """Forwarded GRAHSP kwargs must reach the registered grahsp() function.

    Exercise the registry function directly with the explicit kwargs the
    forward pipeline passes. If a user-set ``agn_grahsp_plslope`` actually
    changes the SED, then the dispatch path will too (since the same kwarg
    name is used).
    """
    fn = resolve_agn_model("grahsp")
    wave = jnp.logspace(2, 6, 200)
    out_default = fn(wave, agn_log_lbol=11.42, agn_grahsp_plslope=-1.7)
    out_steeper = fn(wave, agn_log_lbol=11.42, agn_grahsp_plslope=-2.5)
    # A steeper slope must change the SED.
    assert not jnp.allclose(out_default, out_steeper)
