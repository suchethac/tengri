# SPDX-License-Identifier: BSD-3-Clause
"""BOSA's template SHAPE must track the fitted L_ir, not just its normalization (#2272).

``EmissionComponent.factors_l_ir`` (``components/dust/emission/_component_base.py``)
is an ``apply()``-level speed shortcut: for a model whose SED is *linear* in
``L_ir``, ``apply()`` may evaluate ``predict()`` once at the unit luminosity
``L_ir = 1`` and re-apply the true scale afterwards in log space (float32
safety, #1206). BOSA (Boquien & Salim 2021) is not that kind of model: it
parametrizes its template library by ``(log L_TIR, log sSFR)``, so the
emission *shape* -- which row of the grid gets interpolated -- is itself a
function of the absorbed luminosity. ``BosaIRSEDComponent`` never overrode
the base class's ``factors_l_ir = True`` default (unlike
``energy_balance_split``, which does), so ``apply()`` always pinned BOSA's
``predict()`` call at ``L_ir = 1``: the shape selected by
``log10(1) = 0`` (clipped to the grid's lowest node), then only the
*amplitude* rescaled to the real budget afterwards. The shape BOSA reports
therefore never moved with the fitted luminosity, though the model exists
specifically to let it move.

Fix, part 1: ``BosaIRSEDComponent.factors_l_ir = False`` (mirroring
``energy_balance_split``), so ``apply()`` hands ``predict()`` the real
``L_ir`` and the grid lookup sees the real budget.

Fix, part 2 (the other half of the same disease): the packaged grid's axis
is ``log10(L_TIR / Lsun)`` (Boquien & Salim 2021 convention -- see
``scripts/build_bosa_hdf5.py``), while ``L_ir``/``log_L_ir`` arrive in erg/s
(the tengri-wide SED contract). Without converting, ANY astrophysically
realistic ``L_ir`` (~1e42-1e45 erg/s, i.e. dex 42-45) numerically saturates
the grid's ceiling node (12.5) regardless of the real budget -- part 1 alone
stops ``apply()`` from pinning the lookup at a fixed unit-luminosity
placeholder, but does not make the lookup land on the *right* node for a real
galaxy. ``bosa_emission`` (``components/dust/emission_templates.py``) now
subtracts ``LOG10_L_SUN`` (``tengri.utils.sed_quantities``, the #2273
``dust_log_L_ir`` precedent) from the erg/s log budget for the AXIS LOOKUP
only; the normalization stays in erg/s throughout, so the delivered SED still
integrates to the erg/s budget exactly.

This file has two regression tests, both now at astrophysically realistic
erg/s scales (a consequence of part 2: an ``L_ir``/``log_L_ir`` this file
hands to ``apply()`` or ``predict_state`` is always interpreted as erg/s and
converted internally, so there is no longer a "direct grid-axis, deliberately
not realistic" regime to probe separately). The first calls
``BosaIRSEDComponent.apply()`` directly with two hand-built ``ForwardState``
objects at two erg/s ``L_ir`` values chosen to land, after the Lsun
conversion, at the 25th/75th percentile of the packaged grid's own
``log_ltir_grid`` axis (~1e43/~1e45 erg/s) -- a fast, SSP-free unit test of
exactly the ``apply()`` + closure wiring both fix parts touch. The second
reproduces the issue's literal scenario through the full ``SEDModel.build``
+ ``predict_state`` path at two realistic total stellar masses, self-calibrated
from the same grid percentiles via the mass -> L_ir scaling.

Measured (this repro): direct-axis test -- with the Lsun conversion reverted
(``LOG10_L_SUN`` patched to 0), the two normalized shapes differ by ~5.1e-15
(fp roundoff -- bit-identical, matching the issue's "bit-identical to 1e-16"
observation, both erg/s values saturating the grid's ceiling node); with the
conversion (current code), up to ~45.5%. Realistic-scale test -- reverted,
~2.9e-14 (bit-identical); current code, ~45.5%.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.dust.emission.templates.bosa import BosaIRSEDComponent
from tengri.protocols.component import ForwardState
from tengri.utils.sed_quantities import LOG10_L_SUN

pytestmark = pytest.mark.regression_bug

_C_AA_PER_S = 2.99792458e18
_WAVE = jnp.geomspace(3.0e3, 3.0e8, 3000)
_PARAMS = {"dust_log_ssfr": jnp.asarray(-10.0)}


@pytest.fixture(scope="module")
def bosa_grid():
    """The packaged BOSA template grid, or a skip if it is unavailable.

    ``data/bosa_templates.h5`` is a tracked repository asset (explicit
    ``!data/bosa_templates.h5`` negation in ``.gitignore``, alongside the
    other template libraries), not a synthesized CI stand-in, so this should
    not skip in CI -- the guard exists for an exotic checkout only.
    """
    comp = BosaIRSEDComponent()
    try:
        grid = comp.load()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"BOSA template grid not on disk: {exc}")
    if grid is None:
        pytest.skip("BOSA template grid not found by find_data_str")
    return grid


@pytest.fixture(scope="module")
def two_l_ir_values(bosa_grid):
    """Two erg/s ``L_ir`` values landing at the 25th/75th percentile of the
    grid's own ``log_ltir_grid`` axis, AFTER the closure's Lsun conversion.

    The grid's axis is ``log10(L_TIR/Lsun)`` (#2272 part 2); ``L_ir`` is
    erg/s (the tengri-wide contract), so the erg/s value that lands a given
    axis target is ``10 ** (target + LOG10_L_SUN)`` -- these come out at
    ~1e43/~1e45 erg/s, astrophysically realistic IR luminosities, not
    arbitrary numbers. Chosen at the 25th/75th percentile of the grid's own
    span so the pair is robust to a future grid re-generation with different
    axis bounds, and far enough apart (>1 dex on the axis) to force
    materially different interpolation nodes.
    """
    log_ltir_grid = np.asarray(bosa_grid["log_ltir_grid"])
    span = float(log_ltir_grid.max() - log_ltir_grid.min())
    lo_axis = float(log_ltir_grid.min() + 0.25 * span)
    hi_axis = float(log_ltir_grid.min() + 0.75 * span)
    assert hi_axis - lo_axis > 1.0, "grid too narrow to force a materially different shape"
    return 10.0 ** (lo_axis + LOG10_L_SUN), 10.0 ** (hi_axis + LOG10_L_SUN)


def _sed_dust_ir(component: BosaIRSEDComponent, l_ir: float) -> np.ndarray:
    """Run ``BosaIRSEDComponent.apply()`` at a hand-set ``L_ir``, return ``sed_dust_ir``."""
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


def _max_relative_shape_difference(norm_lo: np.ndarray, norm_hi: np.ndarray) -> float:
    """Largest relative difference between two normalized (``sed / L_ir``) shapes."""
    scale = np.maximum(np.abs(norm_lo), np.abs(norm_hi))
    mask = scale > 1.0e-3 * scale.max()
    return float(np.max(np.abs(norm_hi[mask] - norm_lo[mask]) / scale[mask]))


def test_bosa_shape_follows_the_real_l_ir(two_l_ir_values):
    """BOSA's normalized shape must differ materially at two well-separated L_ir values.

    #2272, both parts: before part 1, ``apply()`` always evaluated
    ``predict()`` at ``L_ir = 1`` (``factors_l_ir=True``, the inherited
    default, never overridden for BOSA), so the shape was pinned to the
    unit-luminosity template row regardless of the real budget. Part 1 alone
    is not sufficient at these (astrophysically realistic, ~1e43-1e45 erg/s)
    values: without part 2's Lsun conversion, both still saturate the grid's
    ceiling node identically. This is the distinguishing-content test: it is
    RED with the Lsun conversion reverted (max relative difference ~5e-15,
    fp roundoff) and GREEN with both fix parts in place.
    """
    l_lo, l_hi = two_l_ir_values
    component = BosaIRSEDComponent()

    sed_lo = _sed_dust_ir(component, l_lo)
    sed_hi = _sed_dust_ir(component, l_hi)
    norm_lo = sed_lo / l_lo
    norm_hi = sed_hi / l_hi

    max_rel_diff = _max_relative_shape_difference(norm_lo, norm_hi)
    assert max_rel_diff > 0.05, (
        f"BOSA's normalized sed_dust_ir shape barely changed between L_ir={l_lo:.3e} "
        f"and L_ir={l_hi:.3e} (max relative difference {max_rel_diff:.3e}); expected the "
        "template-shape lookup to move with the real luminosity (#2272)."
    )

    # The scale half: normalization still integrates to the budget exactly,
    # at BOTH luminosities -- the shape moves, the energy balance does not.
    for label, sed, l_ir in (("lo", sed_lo, l_lo), ("hi", sed_hi, l_hi)):
        ratio = _integral_over_l_ir(sed, l_ir)
        assert abs(ratio - 1.0) < 1.0e-6, (
            f"{label}: integral(sed_dust_ir)/L_ir = {ratio:.8f}, expected ~1.0"
        )
    # This energy-balance check doubles as the guard against a future
    # accidental revert of ``factors_l_ir`` to True: with the ``log_L_ir``
    # plumbing this fix also adds (float32 safety, see bosa.py), reverting
    # just the flag does not reproduce the old "shape pinned" symptom -- it
    # double-applies the luminosity scale instead (measured: ratio off by
    # exactly L_ir, i.e. ~3e11 for the ``l_hi`` case here), which this
    # assertion catches immediately.


@pytest.fixture(scope="module")
def ssp():
    """The packaged default SSP grid, or a skip if it is unavailable.

    ``tengri.load_ssp()`` resolves to
    ``data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5``, a tracked
    repository asset committed on purpose so CI needs no network (#1146) --
    this should not skip in CI.
    """
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _build_bosa_model(ssp_data, log_total_mass: float, tau: float = 1.0):
    """A minimal star-forming + two-component-attenuated + BOSA model."""
    return tengri.SEDModel.build(
        ssp_data,
        sfh={
            "type": "tsnorm",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_total_mass": tengri.Fixed(log_total_mass),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "tau_diff": tau,
            "tau_bc": tau,
        },
        dust_emission={"type": "bosa", "all_params": tengri.Fixed(tengri.DEFAULT)},
        redshift=tengri.Fixed(0.0),
    )


def _l_ir_and_sed(model) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the built model's forward pass, returning (L_ir, sed_dust_ir, wave)."""
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)
    l_ir = float(np.asarray(state.derived["L_ir"]))
    sed = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    wave = np.asarray(state.wave, dtype=np.float64)
    return l_ir, sed, wave


@pytest.fixture(scope="module")
def two_realistic_total_masses(ssp, bosa_grid):
    """Two ``sfh_*_log_total_mass`` values landing at the grid's 25th/75th percentile.

    ``log_total_mass`` scales ``L_ir`` 1:1 in log space at fixed attenuation
    (mass is a pure normalization knob), so one reference build calibrates
    the mass -> log10(L_TIR/Lsun) mapping for this SSP/attenuation choice,
    and the two target masses are solved from it -- self-calibrating rather
    than a magic-number pair tied to one specific SSP file.
    """
    log_ltir_grid = np.asarray(bosa_grid["log_ltir_grid"])
    span = float(log_ltir_grid.max() - log_ltir_grid.min())
    target_lo = float(log_ltir_grid.min() + 0.25 * span)
    target_hi = float(log_ltir_grid.min() + 0.75 * span)

    ref_log_mass = 10.0
    ref_l_ir, _, _ = _l_ir_and_sed(_build_bosa_model(ssp, ref_log_mass))
    ref_log_ratio = np.log10(ref_l_ir) - LOG10_L_SUN

    mass_lo = ref_log_mass + (target_lo - ref_log_ratio)
    mass_hi = ref_log_mass + (target_hi - ref_log_ratio)
    return float(mass_lo), float(mass_hi)


def test_bosa_shape_follows_the_real_l_ir_at_astrophysical_scales(ssp, two_realistic_total_masses):
    """The issue's literal reproduction: two realistic-mass builds, shapes must differ.

    #2272 part 2: before converting the erg/s budget to the grid's own
    Lsun-relative axis, this test is RED even with ``factors_l_ir=False`` --
    both masses' erg/s ``L_ir`` (~1e42-1e45) saturate the grid's ceiling node
    identically (measured ~2.9e-14, bit-identical), reproducing the issue's
    "normalized sed_dust_ir shape bit-identical ... across a 1000x L_ir
    range" at masses picked to actually land inside the grid's Lsun-relative
    span once converted.
    """
    log_mass_lo, log_mass_hi = two_realistic_total_masses

    l_lo, sed_lo, wave_lo = _l_ir_and_sed(_build_bosa_model(ssp, log_mass_lo))
    l_hi, sed_hi, wave_hi = _l_ir_and_sed(_build_bosa_model(ssp, log_mass_hi))
    np.testing.assert_allclose(wave_lo, wave_hi)

    norm_lo = sed_lo / l_lo
    norm_hi = sed_hi / l_hi
    max_rel_diff = _max_relative_shape_difference(norm_lo, norm_hi)
    assert max_rel_diff > 0.05, (
        f"BOSA's normalized sed_dust_ir shape barely changed between two realistic-mass "
        f"builds (L_ir={l_lo:.3e} vs {l_hi:.3e} erg/s; max relative difference "
        f"{max_rel_diff:.3e}); expected the Lsun-converted grid lookup to land on "
        "different interpolation nodes at astrophysical scales (#2272)."
    )

    nu = _C_AA_PER_S / wave_lo
    for label, sed, l_ir in (("lo", sed_lo, l_lo), ("hi", sed_hi, l_hi)):
        ratio = float(-np.trapezoid(sed, nu)) / l_ir
        assert abs(ratio - 1.0) < 1.0e-6, (
            f"{label}: integral(sed_dust_ir)/L_ir = {ratio:.8f}, expected ~1.0"
        )
