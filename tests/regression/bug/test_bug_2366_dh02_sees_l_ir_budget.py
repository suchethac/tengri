# SPDX-License-Identifier: BSD-3-Clause
"""DH02_CE01's template SHAPE must track the fitted L_ir, not just its normalization (#2366).

The Dale & Helou (2002) / Chary & Elbaz (2001) template library is indexed by
``log10(L_TIR/L_sun)`` and the whole point of that family is that the SED shape
correlates with luminosity — warmer, broader SEDs at higher L_IR. tengri's
component declares no free parameters (``declares_no_parameters=True``) and
inherits ``factors_l_ir=True`` from the base class, so ``apply()`` evaluates
the closure at a unit-luminosity placeholder ``L_ir = 1``, retrieving the
``log10(1) = 0`` template regardless of the actual galaxy's L_IR budget, then
only rescales the amplitude afterwards. The shape DH02 reports therefore never
moves with the fitted luminosity, though the model's entire purpose is to let
it move with L_IR.

Fix shape:
- Set ``factors_l_ir=False`` in ``DH02CE01IRSEDComponent``, exactly as
  ``BosaIRSEDComponent`` does.
- Declare ``optional_inputs={'log_L_ir': 'dex'}`` on the component class.
- The closure ``dh02_ce01_tabulated`` loses its hardcoded ``dust_log_lir=10.0``
  default.
- Instead it receives ``log_L_ir`` from ``predict()``, converts erg/s to L_sun
  (same as BOSA's precedent, #2272/#2273), and uses that for the lookup.
- The axis conversion happens in ``create_dh02_ce01`` in
  ``emission_templates.py``.

This file defines one regression test, called at astrophysically realistic erg/s
scales: build a ``DH02CE01IRSEDComponent`` model on the tracked wNE SSP, at two
``dust_log_L_ir`` values chosen (after the Lsun conversion) to land at the 25th/75th
percentile of the DH02 grid's own ``irlum_axis`` (~1e10.3/~1e11.3 L_sun, corresponding
to ~1e43.3/~1e44.3 erg/s), and assert the NORMALIZED dust-emission SED shapes differ
materially (DH02 templates are L_IR-dependent by construction: warmer at higher L_IR —
the test checks peak-wavelength movement, not just any difference).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.dust.emission.templates.dh02_ce01 import DH02CE01IRSEDComponent
from tengri.protocols.component import ForwardState
from tengri.utils.sed_quantities import LOG10_L_SUN

pytestmark = pytest.mark.regression_bug

_C_AA_PER_S = 2.99792458e18
_WAVE = jnp.geomspace(3.0e3, 3.0e8, 3000)
_PARAMS = {}  # DH02CE01 declares no parameters


@pytest.fixture(scope="module")
def dh02_grid():
    """The packaged DH02_CE01 template grid, or a skip if unavailable.

    ``data/dh02_ce01_grid.h5`` is a tracked repository asset (explicit
    ``!data/dh02_ce01_grid.h5`` negation in ``.gitignore``, alongside the
    other template libraries), not a synthesized CI stand-in, so this should
    not skip in CI -- the guard exists for an exotic checkout only.
    """
    comp = DH02CE01IRSEDComponent()
    try:
        grid = comp.load()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"DH02_CE01 template grid not on disk: {exc}")
    if grid is None:
        pytest.skip("DH02_CE01 template grid not found by find_data_str")
    return grid


@pytest.fixture(scope="module")
def two_l_ir_values(dh02_grid):
    """Two erg/s ``L_ir`` values landing at the 25th/75th percentile of the
    grid's own ``irlum_axis`` axis, AFTER the closure's Lsun conversion.

    The grid's axis is ``log10(L_TIR/Lsun)`` (spanning 8.3–14.3, Dale & Helou
    2002); ``L_ir`` is erg/s (the tengri-wide contract), so the erg/s value
    that lands a given axis target is ``10 ** (target + LOG10_L_SUN)`` -- these
    come out at ~1e43.3/~1e44.3 erg/s, astrophysically realistic IR luminosities.
    Chosen at the 25th/75th percentile of the grid's own span so the pair is
    robust to a future grid re-generation with different axis bounds, and far
    enough apart (>1 dex on the axis) to force materially different interpolation
    nodes and therefore materially different shapes (warmer vs cooler templates).
    """
    irlum_axis = np.asarray(dh02_grid["irlum_axis"])
    span = float(irlum_axis.max() - irlum_axis.min())
    lo_axis = float(irlum_axis.min() + 0.25 * span)
    hi_axis = float(irlum_axis.min() + 0.75 * span)
    assert hi_axis - lo_axis > 1.0, "grid too narrow to force a materially different shape"
    return 10.0 ** (lo_axis + LOG10_L_SUN), 10.0 ** (hi_axis + LOG10_L_SUN)


def _sed_dust_ir(component: DH02CE01IRSEDComponent, l_ir: float) -> np.ndarray:
    """Run ``DH02CE01IRSEDComponent.apply()`` at a hand-set ``L_ir``, return ``sed_dust_ir``."""
    state = ForwardState(
        wave=_WAVE,
        sed_intrinsic=None,
        derived={"L_ir": jnp.asarray(l_ir), "log_L_ir": jnp.asarray(np.log10(l_ir))},
    )
    out = component.apply(state, _PARAMS)
    return np.asarray(out.derived["sed_dust_ir"], dtype=np.float64)


def _integral_over_l_ir(sed: np.ndarray, l_ir: float) -> float:
    """``int(sed_nu dnu) / L_ir`` -- mirrors ``test_dust_emission_exact_energy_balance.py``."""
    nu = _C_AA_PER_S / np.asarray(_WAVE, dtype=np.float64)
    # nu descends with wavelength (wave is ascending), so the signed
    # trapezoid over it is negative; negate to get the physical (positive)
    # integral, exactly as components/dust/emission_templates.py does.
    return float(-np.trapezoid(sed, nu)) / l_ir


def _peak_wavelength(sed: np.ndarray, wave: np.ndarray) -> float:
    """Wavelength at the peak of the SED."""
    idx_max = np.argmax(sed)
    return float(wave[idx_max])


def _max_relative_shape_difference(norm_lo: np.ndarray, norm_hi: np.ndarray) -> float:
    """Largest relative difference between two normalized (``sed / L_ir``) shapes."""
    scale = np.maximum(np.abs(norm_lo), np.abs(norm_hi))
    mask = scale > 1.0e-3 * scale.max()
    return float(np.max(np.abs(norm_hi[mask] - norm_lo[mask]) / scale[mask]))


@pytest.fixture(scope="module")
def ssp():
    """The packaged default SSP grid, or a skip if unavailable."""
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _build_dh02_model(ssp_data, dust_log_l_ir: float | None = None):
    """A minimal model with dh02_ce01 dust emission."""
    import tengri

    kwargs = {
        "ssp_data": ssp_data,
        "sfh": {
            "type": "tsnorm",
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "dust_attenuation": {
            "type": "two_component",
            "law": "power_law",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "tau_diff": 0.5,
            "tau_bc": 0.5,
        },
        "dust_emission": {"type": "dh02_ce01", "all_params": tengri.Fixed(tengri.DEFAULT)},
        "redshift": tengri.Fixed(0.1),
    }
    if dust_log_l_ir is not None:
        kwargs["dust_emission"]["dust_log_L_ir"] = tengri.Fixed(dust_log_l_ir)
    return tengri.SEDModel.build(**kwargs)


def test_dh02_shape_follows_the_real_l_ir(two_l_ir_values):
    """DH02_CE01's normalized shape must differ materially at two well-separated L_ir values.

    #2366: before the fix, ``apply()`` always evaluated ``predict()`` at
    ``L_ir = 1`` (``factors_l_ir=True``, inherited default never overridden),
    so the shape was pinned to the ``log10(1) = 0`` template (clipped to the
    grid's minimum, 8.3, which selects the hottest, broadest template —
    wrong for most realistic galaxies). This is the distinguishing-content test:
    it is RED before the fix (max relative difference ~1e-15, fp roundoff --
    shapes bit-identical, both pegging the grid's minimum node) and GREEN with
    both the ``factors_l_ir=False`` and the Lsun-conversion fix parts in place.

    DH02 templates are L_IR-dependent by construction: hotter (peak at shorter
    wavelength), broader SEDs at higher L_IR (Dale & Helou 2002, Fig. 1).
    This test verifies the peak wavelength moves appropriately — the
    distinguishing content that cannot be satisfied by a shape that is pinned
    to a single grid node regardless of the real budget.
    """
    l_lo, l_hi = two_l_ir_values
    component = DH02CE01IRSEDComponent()

    sed_lo = _sed_dust_ir(component, l_lo)
    sed_hi = _sed_dust_ir(component, l_hi)
    norm_lo = sed_lo / l_lo
    norm_hi = sed_hi / l_hi

    # DH02 shape difference: warmer (shorter peak wavelength) at higher L_IR.
    wave_arr = np.asarray(_WAVE, dtype=np.float64)
    peak_lo = _peak_wavelength(norm_lo, wave_arr)
    peak_hi = _peak_wavelength(norm_hi, wave_arr)

    # Higher-L_IR template should be warmer (peak at shorter wavelength).
    assert peak_hi < peak_lo, (
        f"Expected peak_wavelength to decrease with higher L_ir (warmer template), "
        f"but got peak_lo={peak_lo:.1f} at L_ir={l_lo:.3e}, "
        f"peak_hi={peak_hi:.1f} at L_ir={l_hi:.3e}"
    )

    # Also check overall shape difference.
    max_rel_diff = _max_relative_shape_difference(norm_lo, norm_hi)
    assert max_rel_diff > 0.05, (
        f"DH02_CE01's normalized sed_dust_ir shape barely changed between L_ir={l_lo:.3e} "
        f"and L_ir={l_hi:.3e} (max relative difference {max_rel_diff:.3e}); expected the "
        "template-shape lookup to move with the real luminosity (#2366)."
    )

    # The scale half: normalization still integrates to the budget exactly,
    # at BOTH luminosities -- the shape moves, the energy balance does not.
    for label, sed, l_ir in (("lo", sed_lo, l_lo), ("hi", sed_hi, l_hi)):
        ratio = _integral_over_l_ir(sed, l_ir)
        assert abs(ratio - 1.0) < 1.0e-6, (
            f"{label}: integral(sed_dust_ir)/L_ir = {ratio:.8f}, expected ~1.0"
        )


def test_dh02_energy_balance_through_apply(ssp):
    """Verify energy balance through SEDModel.build path; guards normalization.

    The test builds a full SEDModel (triggering apply() and predict_state()),
    extracting sed_dust_ir and L_ir from the forward state. It verifies that
    the dust SED integrates to the budget (integral/L_ir ≈ 1.0). The
    factors_l_ir=False flag declares non-linearity to the framework; the
    closure normalizes in erg/s regardless, so numeric results are independent
    of the flag. This test guards the integration contract, not the flag.
    """
    import jax

    import tengri

    model = _build_dh02_model(ssp)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)

    l_ir = float(np.asarray(state.derived["L_ir"]))
    sed_dust_ir = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    wave = np.asarray(state.wave, dtype=np.float64)

    # Energy balance: integral of sed_dust_ir over frequency must equal L_ir.
    nu = _C_AA_PER_S / wave
    integral = float(-np.trapezoid(sed_dust_ir, nu))
    ratio = integral / l_ir

    assert abs(ratio - 1.0) < 1.0e-2, (
        f"Energy balance violated: integral(sed_dust_ir)/L_ir = {ratio:.8f}, "
        f"expected ~1.0. With factors_l_ir=True mutation this becomes ~{l_ir:.3e}"
        f" (amplitude double-counted)."
    )
