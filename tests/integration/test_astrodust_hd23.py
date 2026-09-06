# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Hensley & Draine 2023 Astrodust+PAH emission component.

Exercises :class:`~tengri.components.dust.emission.templates.astrodust.AstrodustIRSEDComponent`
(the faithful native ``lgU`` interpolation of the published grid; #871) against
the real HDF5 grid built from the Harvard Dataverse FITS file, plus the shared
loader in :mod:`tengri.components.dust.astrodust_hd23`.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

HDF5 = Path("data/astrodust_templates.h5")


@pytest.fixture(scope="module")
def fixture_path():
    if not HDF5.is_file():
        pytest.skip(
            f"Astrodust HDF5 not built at {HDF5}; run "
            f"`python scripts/build_astrodust_hdf5.py --download`"
        )
    return str(HDF5)


@pytest.fixture(scope="module")
def component(fixture_path):
    from tengri.components.dust.emission.templates.astrodust import (
        AstrodustIRConfig,
        AstrodustIRSEDComponent,
    )

    cfg = AstrodustIRConfig(
        component="total",
        spinning_dust=False,
        template_path=fixture_path,
    )
    return AstrodustIRSEDComponent(config=cfg)


@pytest.fixture(scope="module")
def loaded(component):
    wave_aa = jnp.asarray(np.geomspace(1.0e3, 3.0e8, 1500))
    component.data = component.load(wave_aa)
    return component.data, wave_aa


# ── shared loader (tengri.components.dust.astrodust_hd23) ──────────────


def test_loader_axes_match_published_grid(fixture_path):
    """Grid dimensions match the (91 lgU x 1000 wave x 3 components)
    layout published in HDU 7 of the HD23 FITS file."""
    from tengri.components.dust.astrodust_hd23 import (
        load_astrodust_hd23_or_raise,
    )

    tpl = load_astrodust_hd23_or_raise(fixture_path)
    chex.assert_shape(tpl.lgU, (91,))
    chex.assert_shape(tpl.wavelength_um, (1000,))
    chex.assert_shape(tpl.L_nu_total, (91, 1000))
    chex.assert_shape(tpl.L_nu_astrodust, (91, 1000))
    chex.assert_shape(tpl.L_nu_pah, (91, 1000))
    np.testing.assert_allclose(np.asarray(tpl.lgU)[0], -3.0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(tpl.lgU)[-1], 6.0, atol=1e-3)
    assert float(tpl.wavelength_um[0]) < 1.0  # extends to UV (~0.1 μm)
    assert float(tpl.wavelength_um[-1]) >= 1e4  # extends to mm wavelengths


def test_total_equals_astrodust_plus_pah(fixture_path):
    """Per the FITS file design, the 'total' column = astrodust + PAH."""
    from tengri.components.dust.astrodust_hd23 import (
        load_astrodust_hd23_or_raise,
    )

    tpl = load_astrodust_hd23_or_raise(fixture_path)
    total = np.asarray(tpl.L_nu_total)
    summed = np.asarray(tpl.L_nu_astrodust) + np.asarray(tpl.L_nu_pah)
    # atol covers float32 denormals at the UV/optical end of the
    # wavelength grid where dust emission is essentially zero.
    np.testing.assert_allclose(total, summed, rtol=2e-3, atol=1e-40)


def test_dust_mass_constants(fixture_path):
    """The published M_Ad/M_H and M_PAH/M_H constants are stored on
    the loader output for downstream conversions."""
    from tengri.components.dust.astrodust_hd23 import (
        load_astrodust_hd23_or_raise,
    )

    tpl = load_astrodust_hd23_or_raise(fixture_path)
    assert tpl.M_Ad_over_M_H == pytest.approx(0.00642, rel=1e-4)
    assert tpl.M_PAH_over_M_H == pytest.approx(0.000659, rel=1e-3)


# ── AstrodustIRSEDComponent ───────────────────────────────────────────


def test_declared_parameter_is_lgU_only(component):
    decls = component.declared_parameters()
    names = [d.name for d in decls]
    assert names == ["dust_lgU"]


def test_load_returns_grid(loaded):
    data, wave_aa = loaded
    chex.assert_shape(data["lgU_grid"], (91,))
    chex.assert_shape(data["lnu_template"], (91, wave_aa.size))
    # Normalization is computed in predict(), not load(), to ensure exact
    # energy balance on the evaluation grid after both thermal and spinning
    # dust components are assembled (issue #XXXX).
    assert "norm_per_lgU" not in data
    # No spinning dust requested -> zeros.
    np.testing.assert_array_equal(
        np.asarray(data["lnu_spinning"]),
        np.zeros(wave_aa.size),
    )


def test_apply_energy_balance(component, loaded):
    """int L_nu d nu == L_ir within trapezoid discretization tolerance."""
    from tengri.protocols.component import ForwardState

    _data, wave_aa = loaded
    L_ir = 5.0e44
    pipeline = ForwardState(wave=wave_aa, sed_intrinsic=None, derived={"L_ir": L_ir})
    out = component.apply(pipeline, {"dust_lgU": 0.2, "redshift": 0.0})
    L_nu = np.asarray(out.derived["sed_dust_ir"])
    nu = 2.99792458e18 / np.asarray(wave_aa)
    order = np.argsort(nu)
    integral = np.trapezoid(L_nu[order], nu[order])
    np.testing.assert_allclose(integral, L_ir, rtol=0.02)


def test_apply_jit_and_grad(component, loaded):
    from tengri.protocols.component import ForwardState

    _data, wave_aa = loaded

    @jax.jit
    def _run(L_ir, lgU):
        pipeline = ForwardState(wave=wave_aa, sed_intrinsic=None, derived={"L_ir": L_ir})
        out = component.apply(pipeline, {"dust_lgU": lgU, "redshift": 0.0})
        return jnp.sum(out.sed_intrinsic)

    val = float(_run(jnp.asarray(1.0e44), jnp.asarray(0.0)))
    assert np.isfinite(val) and val > 0
    grad = float(jax.grad(_run, argnums=1)(jnp.asarray(1.0e44), jnp.asarray(0.5)))
    assert np.isfinite(grad)
    assert np.any(grad != 0.0), (
        "`grad` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_pah_only_component(fixture_path):
    """Selecting ``component="pah"`` returns only the PAH contribution;
    both renormalize to the same L_ir, so the test checks finiteness +
    positivity (shape-only)."""
    from tengri.components.dust.emission.templates.astrodust import (
        AstrodustIRConfig,
        AstrodustIRSEDComponent,
    )
    from tengri.protocols.component import ForwardState

    wave_aa = jnp.asarray(np.geomspace(1.0e4, 1.0e7, 800))
    L_ir = 1.0e44
    pipeline = ForwardState(wave=wave_aa, sed_intrinsic=None, derived={"L_ir": L_ir})

    a = AstrodustIRSEDComponent(
        config=AstrodustIRConfig(component="total", template_path=fixture_path)
    )
    b = AstrodustIRSEDComponent(
        config=AstrodustIRConfig(component="pah", template_path=fixture_path)
    )
    a.data = a.load(wave_aa)
    b.data = b.load(wave_aa)
    sed_total = np.asarray(a.apply(pipeline, {"dust_lgU": 0.2, "redshift": 0.0}).sed_intrinsic)
    sed_pah = np.asarray(b.apply(pipeline, {"dust_lgU": 0.2, "redshift": 0.0}).sed_intrinsic)
    assert np.isfinite(sed_pah).all() and (sed_pah >= 0).all()
    assert sed_pah.sum() > 0
    assert sed_total.sum() > 0


def test_spinning_dust_inclusion_changes_microwave(fixture_path):
    """Including spinning dust must change the SED at microwave
    wavelengths but leave the IR substantially unchanged."""
    from tengri.components.dust.emission.templates.astrodust import (
        AstrodustIRConfig,
        AstrodustIRSEDComponent,
    )
    from tengri.protocols.component import ForwardState

    wave_aa = jnp.asarray(np.geomspace(1.0e3, 3.0e8, 1500))
    L_ir = 1.0e44
    pipeline = ForwardState(wave=wave_aa, sed_intrinsic=None, derived={"L_ir": L_ir})

    no_spd = AstrodustIRSEDComponent(
        config=AstrodustIRConfig(spinning_dust=False, template_path=fixture_path),
    )
    yes_spd = AstrodustIRSEDComponent(
        config=AstrodustIRConfig(spinning_dust=True, template_path=fixture_path),
    )
    no_spd.data = no_spd.load(wave_aa)
    yes_spd.data = yes_spd.load(wave_aa)
    sed_no = np.asarray(no_spd.apply(pipeline, {"dust_lgU": 0.2, "redshift": 0.0}).sed_intrinsic)
    sed_yes = np.asarray(yes_spd.apply(pipeline, {"dust_lgU": 0.2, "redshift": 0.0}).sed_intrinsic)

    # Spinning dust peaks ~1 cm wavelength = 1e8 Å.
    wave_aa_np = np.asarray(wave_aa)
    microwave_mask = (wave_aa_np > 5.0e7) & (wave_aa_np < 2.0e8)
    ir_mask = (wave_aa_np > 5.0e4) & (wave_aa_np < 5.0e6)
    assert np.any(sed_yes[microwave_mask] > sed_no[microwave_mask])
    np.testing.assert_allclose(sed_yes[ir_mask], sed_no[ir_mask], rtol=1e-3)


def test_missing_template_raises_with_build_instructions(tmp_path):
    """Astrodust carries no analytic fallback: a missing grid must raise
    FileNotFoundError with the build-script invocation, not silently emit
    zeros."""
    from tengri.components.dust.emission.templates.astrodust import (
        AstrodustIRConfig,
        AstrodustIRSEDComponent,
    )

    cfg = AstrodustIRConfig(template_path=str(tmp_path / "nope.h5"))
    comp = AstrodustIRSEDComponent(config=cfg)
    wave_aa = jnp.geomspace(1.0e4, 1.0e7, 100)
    with pytest.raises(FileNotFoundError) as ei:
        comp.load(wave_aa)
    msg = str(ei.value)
    assert "nope.h5" in msg
    assert "build_astrodust_hdf5.py" in msg
    assert "10.7910/DVN/3B6E6S" in msg
