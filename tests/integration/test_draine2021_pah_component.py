# SPDX-License-Identifier: BSD-3-Clause
"""SEDModelComponent tests for the Draine+2021 PAHspec emission port.

The PAHspec model is authored as
:class:`~tengri.components.dust.draine2021_pah_ir.Draine2021PAHIRSEDComponent`
(dispatched via ``dust={'emission': {'type': 'draine2021_pah'}}``). This file
exercises the port contract — declared_parameters, load, predict, JIT
compatibility, gradient propagation through ``dust_lgU``, energy balance with
``L_ir`` — plus the ``starlight="auto"`` nearest-neighbor selection.

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
    from tengri.components.dust.draine2021_pah_ir import (
        Draine2021PAHIRConfig,
        Draine2021PAHIRSEDComponent,
    )

    cfg = Draine2021PAHIRConfig(
        starlight="mMMP",
        ionization="st",
        size_distribution="std",
        slab=False,
        template_path=str(FIXTURE),
    )
    return Draine2021PAHIRSEDComponent(config=cfg)


@pytest.fixture(scope="module")
def loaded(component):
    wave_aa = jnp.asarray(np.geomspace(1.0e4, 1.0e7, 800))
    component.data = component.load(wave_aa)
    return component.data, wave_aa


def _predict_sed(component, wave_aa, lgU, L_ir):
    sed_out, _ = component.predict(
        {"lgU": lgU}, jnp.zeros_like(wave_aa), wave_aa, L_ir=jnp.asarray(L_ir)
    )
    return sed_out


def test_declared_parameters_includes_lgU(component):
    decls = component.declared_parameters()
    names = [d.name for d in decls]
    assert names == ["dust_lgU"]


def test_parameter_prefix_is_dust(component):
    assert component.parameter_prefix == "dust_"


def test_load_returns_grid(loaded):
    data, wave_aa = loaded
    chex.assert_shape(data["lgU_grid"], (15,))
    chex.assert_shape(data["lnu_template"], (15, wave_aa.size))
    chex.assert_shape(data["norm_per_lgU"], (15,))
    assert (np.asarray(data["norm_per_lgU"]) > 0).all()


def test_predict_adds_pah_emission(component, loaded):
    _data, wave_aa = loaded
    sed = np.asarray(_predict_sed(component, wave_aa, 1.0, 1.0e44))
    chex.assert_shape(sed, (wave_aa.size,))
    assert np.isfinite(sed).all()
    assert (sed >= 0).all()
    assert sed.sum() > 0


def test_energy_balance(component, loaded):
    """The integrated L_nu over nu must equal L_ir within tolerance."""
    _data, wave_aa = loaded
    L_ir = 3.0e44
    sed = np.asarray(_predict_sed(component, wave_aa, 2.0, L_ir))
    c_aa_per_s = 2.99792458e18
    nu = c_aa_per_s / np.asarray(wave_aa)
    order = np.argsort(nu)
    integral = np.trapezoid(sed[order], nu[order])
    np.testing.assert_allclose(integral, L_ir, rtol=0.02)


def test_predict_jit_compiles(component, loaded):
    _data, wave_aa = loaded

    @jax.jit
    def _run(L_ir, lgU):
        sed_out, _ = component.predict({"lgU": lgU}, jnp.zeros_like(wave_aa), wave_aa, L_ir=L_ir)
        return jnp.sum(sed_out)

    val = float(_run(jnp.asarray(1.0e44), jnp.asarray(1.0)))
    assert np.isfinite(val) and val > 0


def test_gradient_through_lgU(component, loaded):
    _data, wave_aa = loaded

    def _loss(lgU):
        sed_out, _ = component.predict(
            {"lgU": lgU}, jnp.zeros_like(wave_aa), wave_aa, L_ir=jnp.asarray(1.0e44)
        )
        return jnp.sum(sed_out)

    grad = float(jax.grad(_loss)(jnp.asarray(1.5)))
    assert np.isfinite(grad)
    assert grad != 0.0


def test_zero_L_ir_zero_emission(component, loaded):
    _data, wave_aa = loaded
    sed = np.asarray(_predict_sed(component, wave_aa, 1.0, 0.0))
    np.testing.assert_array_equal(sed, np.zeros_like(sed))


def test_pahspec_starlight_auto_resolves_to_named_template():
    """Setting ``starlight="auto"`` plus the auto fields must resolve to a real
    PAHspec starlight name (no disk I/O required)."""
    from tengri.components.dust.draine2021_pah_ir import (
        Draine2021PAHIRConfig,
        Draine2021PAHIRSEDComponent,
    )

    cfg = Draine2021PAHIRConfig(
        starlight="auto",
        auto_age_myr=10.0,
        auto_log_z_solar=0.0,
        auto_sps_family="BC03",
        template_path=str(FIXTURE),
    )
    comp = Draine2021PAHIRSEDComponent(config=cfg)
    assert comp._resolve_starlight() == "BC03_Z0.02_10Myr"


def test_pahspec_starlight_auto_missing_age_raises():
    from tengri.components.dust.draine2021_pah_ir import (
        Draine2021PAHIRConfig,
        Draine2021PAHIRSEDComponent,
    )

    cfg = Draine2021PAHIRConfig(
        starlight="auto",
        # Missing auto_age_myr.
        auto_log_z_solar=0.0,
        template_path=str(FIXTURE),
    )
    comp = Draine2021PAHIRSEDComponent(config=cfg)
    with pytest.raises(ValueError, match="auto_age_myr"):
        comp._resolve_starlight()


def test_pahspec_starlight_auto_passthrough_to_load():
    """The auto-resolved starlight must drive load() slicing. For the smoke
    fixture (mMMP only), auto-selection that resolves to mMMP loads a grid."""
    from tengri.components.dust.draine2021_pah_ir import (
        Draine2021PAHIRConfig,
        Draine2021PAHIRSEDComponent,
    )

    cfg = Draine2021PAHIRConfig(
        starlight="auto",
        auto_age_myr=5000.0,
        auto_log_z_solar=0.0,
        auto_sps_family="FSPS",
        template_path=str(FIXTURE),
    )
    comp = Draine2021PAHIRSEDComponent(config=cfg)
    resolved = comp._resolve_starlight()
    # FSPS+5Gyr falls back to a non-SSP ambient choice.
    assert resolved in {"mMMP", "m31bulge"}
    if resolved == "mMMP":
        wave_aa = jnp.geomspace(1.0e4, 1.0e7, 100)
        data = comp.load(wave_aa)
        chex.assert_shape(data["lgU_grid"], (15,))
