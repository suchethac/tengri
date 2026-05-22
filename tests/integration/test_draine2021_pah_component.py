# SPDX-License-Identifier: BSD-3-Clause
"""SEDComponent tests for Draine+2021 PAHspec emission template.

The PAHspec model is exposed as the ``template="draine2021_pah"``
branch of :class:`DustEmissionSEDComponent` (see
``src/tengri/components/dust/emission_component.py``).  This file
exercises the SEDComponent contract — declared_parameters,
precompute, apply, JIT compatibility, gradient propagation through
``dust_lgU``, energy balance with ``L_ir``.

Uses the smoke fixture under ``tests/fixtures/pahspec_smoke.h5``
(mMMP non-slab, all 15 x 3 x 3 cells filled).
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pahspec_smoke.h5"


@pytest.fixture(scope="module")
def component():
    pytest.importorskip("h5py")
    from tengri.components.dust.emission_component import (
        DustEmissionSEDComponent,
        DustEmissionSEDComponentConfig,
    )

    cfg = DustEmissionSEDComponentConfig(
        template="draine2021_pah",
        pahspec_starlight="mMMP",
        pahspec_ionization="st",
        pahspec_size_distribution="std",
        pahspec_slab=False,
        pahspec_template_path=str(FIXTURE),
    )
    return DustEmissionSEDComponent(config=cfg)


@pytest.fixture(scope="module")
def precomputed(component):
    wave_aa = jnp.asarray(np.geomspace(1.0e4, 1.0e7, 800))
    state = component.precompute(ssp_data=None, wave_grid=wave_aa)
    return state, wave_aa


def test_declared_parameters_includes_lgU(component):
    decls = component.declared_parameters()
    names = [d.name for d in decls]
    assert names == ["dust_lgU"]


def test_parameter_prefix_is_dust(component):
    assert component.parameter_prefix == "dust_"


def test_modified_blackbody_default_branch_unchanged():
    """The default template path must keep declaring (dust_T, dust_beta_ir)
    so that existing pipelines don't break."""
    from tengri.components.dust.emission_component import DustEmissionSEDComponent

    decls = DustEmissionSEDComponent().declared_parameters()
    names = {d.name for d in decls}
    assert names == {"dust_T", "dust_beta_ir"}


def test_precompute_returns_state_with_grid(precomputed):
    state, wave_aa = precomputed
    chex.assert_shape(state.pahspec_lgU_grid, (15,))
    chex.assert_shape(state.pahspec_lnu_template, (15, wave_aa.size))
    chex.assert_shape(state.pahspec_norm_per_lgU, (15,))
    assert (np.asarray(state.pahspec_norm_per_lgU) > 0).all()


def test_apply_adds_pah_emission(component, precomputed):
    from tengri.protocols.component import ForwardState

    state_pre, wave_aa = precomputed
    L_ir = 1.0e44  # erg/s
    pipeline = ForwardState(
        wave=wave_aa,
        sed_intrinsic=None,
        derived={"L_ir": L_ir},
    )
    params = {"dust_lgU": 1.0, "redshift": 0.0}
    state_post = component.apply(pipeline, params, precomputed=state_pre)

    sed = np.asarray(state_post.sed_intrinsic)
    chex.assert_shape(sed, (wave_aa.size,))
    assert np.isfinite(sed).all()
    assert (sed >= 0).all()
    assert sed.sum() > 0


def test_energy_balance(component, precomputed):
    """The integrated L_nu over nu must equal L_ir within tolerance."""
    from tengri.protocols.component import ForwardState

    state_pre, wave_aa = precomputed
    L_ir = 3.0e44
    pipeline = ForwardState(
        wave=wave_aa,
        sed_intrinsic=None,
        derived={"L_ir": L_ir},
    )
    params = {"dust_lgU": 2.0, "redshift": 0.0}
    state_post = component.apply(pipeline, params, precomputed=state_pre)

    L_nu = np.asarray(state_post.derived["sed_dust_ir"])
    c_aa_per_s = 2.99792458e18
    nu = c_aa_per_s / np.asarray(wave_aa)
    order = np.argsort(nu)
    integral = np.trapezoid(L_nu[order], nu[order])
    np.testing.assert_allclose(integral, L_ir, rtol=0.02)


def test_apply_jit_compiles(component, precomputed):
    from tengri.protocols.component import ForwardState

    state_pre, wave_aa = precomputed

    @jax.jit
    def _run(L_ir, lgU):
        pipeline = ForwardState(
            wave=wave_aa,
            sed_intrinsic=None,
            derived={"L_ir": L_ir},
        )
        out = component.apply(
            pipeline,
            {"dust_lgU": lgU, "redshift": 0.0},
            precomputed=state_pre,
        )
        return jnp.sum(out.sed_intrinsic)

    val = float(_run(jnp.asarray(1.0e44), jnp.asarray(1.0)))
    assert np.isfinite(val) and val > 0


def test_gradient_through_lgU(component, precomputed):
    from tengri.protocols.component import ForwardState

    state_pre, wave_aa = precomputed

    def _loss(lgU):
        pipeline = ForwardState(
            wave=wave_aa,
            sed_intrinsic=None,
            derived={"L_ir": jnp.asarray(1.0e44)},
        )
        out = component.apply(
            pipeline,
            {"dust_lgU": lgU, "redshift": 0.0},
            precomputed=state_pre,
        )
        return jnp.sum(out.sed_intrinsic)

    grad = float(jax.grad(_loss)(jnp.asarray(1.5)))
    assert np.isfinite(grad)
    assert grad != 0.0


def test_missing_template_raises_with_build_instructions(tmp_path):
    """If the HDF5 file does not exist, precompute() must raise
    FileNotFoundError with the exact build-script invocation in the
    message body.  No analytic fallback is permitted."""
    from tengri.components.dust.emission_component import (
        DustEmissionSEDComponent,
        DustEmissionSEDComponentConfig,
    )

    cfg = DustEmissionSEDComponentConfig(
        template="draine2021_pah",
        pahspec_template_path=str(tmp_path / "does_not_exist.h5"),
    )
    comp = DustEmissionSEDComponent(config=cfg)
    wave_aa = jnp.geomspace(1.0e4, 1.0e7, 100)
    with pytest.raises(FileNotFoundError) as ei:
        comp.precompute(wave_grid=wave_aa)
    msg = str(ei.value)
    assert "does_not_exist.h5" in msg
    assert "build_pahspec_hdf5.py" in msg
    assert "--download" in msg


def test_zero_L_ir_zero_emission(component, precomputed):
    from tengri.protocols.component import ForwardState

    state_pre, wave_aa = precomputed
    pipeline = ForwardState(
        wave=wave_aa,
        sed_intrinsic=None,
        derived={"L_ir": 0.0},
    )
    params = {"dust_lgU": 1.0, "redshift": 0.0}
    state_post = component.apply(pipeline, params, precomputed=state_pre)
    sed = np.asarray(state_post.sed_intrinsic)
    np.testing.assert_array_equal(sed, np.zeros_like(sed))


def test_unsupported_template_raises():
    from tengri.components.dust.emission_component import (
        DustEmissionSEDComponent,
        DustEmissionSEDComponentConfig,
    )

    cfg = DustEmissionSEDComponentConfig(template="not_a_real_template")
    comp = DustEmissionSEDComponent(config=cfg)
    with pytest.raises(ValueError, match="unsupported template"):
        comp.declared_parameters()


def test_pahspec_starlight_auto_resolves_to_named_template():
    """Setting ``pahspec_starlight="auto"`` plus the auto fields must
    resolve to a real PAHspec starlight name at precompute time."""
    from tengri.components.dust.emission_component import (
        DustEmissionSEDComponent,
        DustEmissionSEDComponentConfig,
    )

    cfg = DustEmissionSEDComponentConfig(
        template="draine2021_pah",
        pahspec_starlight="auto",
        pahspec_auto_age_myr=10.0,
        pahspec_auto_log_z_solar=0.0,
        pahspec_auto_sps_family="BC03",
        pahspec_template_path=str(FIXTURE),
    )
    comp = DustEmissionSEDComponent(config=cfg)
    # _resolve_pahspec_starlight must not require disk I/O.
    assert comp._resolve_pahspec_starlight() == "BC03_Z0.02_10Myr"


def test_pahspec_starlight_auto_missing_age_raises():
    from tengri.components.dust.emission_component import (
        DustEmissionSEDComponent,
        DustEmissionSEDComponentConfig,
    )

    cfg = DustEmissionSEDComponentConfig(
        template="draine2021_pah",
        pahspec_starlight="auto",
        # Missing pahspec_auto_age_myr.
        pahspec_auto_log_z_solar=0.0,
        pahspec_template_path=str(FIXTURE),
    )
    comp = DustEmissionSEDComponent(config=cfg)
    with pytest.raises(ValueError, match="pahspec_auto_age_myr"):
        comp._resolve_pahspec_starlight()


def test_pahspec_starlight_auto_passthrough_to_precompute():
    """The auto-resolved starlight must drive precompute slicing.
    For our smoke fixture (mMMP only), only auto-selection that
    resolves to mMMP should succeed; all others raise via
    select_pahspec_axes."""
    from tengri.components.dust.emission_component import (
        DustEmissionSEDComponent,
        DustEmissionSEDComponentConfig,
    )

    # FSPS family with old age -> bulge fallback OR mMMP fallback;
    # neither matches our smoke fixture (which has only mMMP).
    cfg = DustEmissionSEDComponentConfig(
        template="draine2021_pah",
        pahspec_starlight="auto",
        pahspec_auto_age_myr=5000.0,
        pahspec_auto_log_z_solar=0.0,
        pahspec_auto_sps_family="FSPS",
        pahspec_template_path=str(FIXTURE),
    )
    comp = DustEmissionSEDComponent(config=cfg)
    resolved = comp._resolve_pahspec_starlight()
    # FSPS+5Gyr falls back to a non-SSP ambient choice.
    assert resolved in {"mMMP", "m31bulge"}
    # If it resolves to mMMP, the smoke fixture supports it.
    if resolved == "mMMP":
        wave_aa = jnp.geomspace(1.0e4, 1.0e7, 100)
        state = comp.precompute(wave_grid=wave_aa)
        chex.assert_shape(state.pahspec_lgU_grid, (15,))
