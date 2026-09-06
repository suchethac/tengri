# SPDX-License-Identifier: BSD-3-Clause
r"""Dust IR re-emission must survive pure float32 (#1206 item 6).

``sed_dust_ir`` is the last thing in the forward model that is ``inf`` under
pure float32, and it poisons everything downstream: it is summed into
``sed_intrinsic``, so the total SED and every photometric band go non-finite
with it.

The cause is not the emitted SED's own magnitude — it is ~4.4e30 erg/s/Hz,
comfortably inside float32. It is the *input*: every emission model normalizes
its template to the absorbed luminosity ``L_ir``, which is ~2.4e43 and so
``inf``. ``log_L_ir`` has been published and finite since the energy-balance
work; the emission side simply never read it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

#: Rescales the fixture's per-Msun flux to a real grid's regime. See
#: ``test_float32_boundary_inventory`` for why the raw fixture cannot be used.
_SSP_FLUX_SCALE = 1.0e-17

#: Template-based emission models, all of which normalize a tabulated shape to
#: ``L_ir``. Factoring the luminosity out of that normalization makes every one
#: of them float32-clean (measured max relative error 1.9e-6 to 1.6e-2).
TEMPLATE_MODELS = (
    "dale2014",
    "dale2014_cigale",
    "dl07",
    "dl14",
    "draine_li2007",
    "draine_li2014",
    "bosa",
    "themis",
    "schreiber2018",
    "astrodust",
    "pah_drude",
    # Registered as a component in #1777 — before that it resolved only on the
    # legacy loader path, so the completeness guard below could not see it and
    # this file's claim silently excluded it. (#1738's registry emit census
    # reached the same gap independently, from the menu side.) Measured here,
    # not assumed: float64 peak 3.1452e+30 (float32-representable), sed_dust_ir
    # fully finite in pure float32, peak-relative float32-vs-float64 error
    # 2.09e-04 — the same order as dale2014's 9.03e-05 and far inside the
    # 2e-2 bound.
    "dh02_ce01",
    # Analytic Planck closures, float32-clean since the nu**3 intermediate was
    # removed from planck_bnu (#1206).
    "mbb",
    "modified_blackbody",
    "casey2012",
    "schreiber2016",
    # Affine model — proportional to the *total* budget L_ir + dust_L_agn_ir.
    # Float32-clean since the budget itself moved to log space (log10_add of the
    # two terms) so neither ~1e43 erg/s term is materialized linearly (#1206).
    "energy_balance_split",
)

#: Subset that is strictly energy-balanced, i.e. normalizes its template so the
#: frequency integral equals L_ir. pah_drude is excluded: it scales a PAH
#: template by L_ir without renormalizing to it, so it re-emits only a fraction.
ENERGY_BALANCED_MODELS = tuple(m for m in TEMPLATE_MODELS if m != "pah_drude")

#: Models still not float32-capable, with the measured reason. Empty: every
#: emission model this file *can* measure is now float32-clean.
#: ``energy_balance_split`` was the last holdout (affine in ``L_ir``); it moved
#: to :data:`TEMPLATE_MODELS` once its two-term budget was assembled in log
#: space (#1206).
NOT_YET_FLOAT32: dict[str, str] = {}

#: Names in :data:`TEMPLATE_MODELS` that are legacy spellings resolved through
#: the loader cache rather than components of their own, so they never appear in
#: the forward registry. Measuring them is still worthwhile — they are what a
#: user typing the old name gets — but they must not be mistaken for missing
#: registrations by the completeness guard below.
LEGACY_ALIASES = frozenset({"dl07", "dl14", "mbb"})


def _physical_ssp(ssp):
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * _SSP_FLUX_SCALE,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _model(ssp, emission_type):
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
        dust_emission={"type": emission_type, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.5),
    )


@pytest.mark.parametrize("emission_type", TEMPLATE_MODELS)
def test_dust_ir_sed_is_finite_in_pure_float32(synthetic_ssp_wide, emission_type):
    """``sed_dust_ir`` must be finite in float32 and match float64.

    The float64 pass doubles as the setup assertion: the emitted SED has to be
    inside float32 range to begin with, or a finite float32 result would prove
    nothing about the ``L_ir`` seam.
    """
    ssp = _physical_ssp(synthetic_ssp_wide)
    ref = np.asarray(_model(ssp, emission_type).predict_state({}).derived["sed_dust_ir"])
    ref = ref.astype(np.float64)
    peak = float(np.abs(ref).max())
    assert np.all(np.isfinite(ref)), "setup: float64 dust IR is not finite"
    assert np.any(ref != 0.0), (
        "`ref` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert 0.0 < peak < 3.4e38, f"setup: emitted SED {peak:.3e} is not float32-representable"

    with jax.enable_x64(False):
        state32 = _model(ssp, emission_type).predict_state({})
        got = np.asarray(state32.derived["sed_dust_ir"])
        total = np.asarray(state32.sed_intrinsic)
        assert got.dtype == jnp.float32  # precondition: genuinely pure float32

    finite_fraction = float(np.isfinite(got).mean())
    assert finite_fraction == 1.0, (
        f"{emission_type}: only {finite_fraction:.2%} of sed_dust_ir is finite in "
        "pure float32 — the emission model is still normalizing to the linear "
        "L_ir (~2.4e43, inf in float32) instead of log_L_ir"
    )
    # Compare against the SED's own peak, not per-element. An SED spans many
    # decades, so a far-wing bin carrying 1e-12 of the peak can differ by 100%
    # relative while being physically irrelevant; peak-normalized error is what
    # a flux measurement actually sees. 2% covers astrodust, the loosest of the
    # family (measured 1.58e-2); the rest sit near 1e-6. This is float32
    # template interpolation, not the L_ir seam.
    peak_relative_error = float(np.abs(got.astype(np.float64) - ref).max() / peak)
    assert peak_relative_error < 2.0e-2, (
        f"{emission_type}: float32 dust IR departs from float64 by "
        f"{peak_relative_error:.3e} of the SED peak"
    )

    assert total is not None, "probe setup failed: total SED was not published"
    assert np.all(np.isfinite(total)), (
        f"{emission_type}: dust IR poisoned the total SED in float32"
    )
    assert np.any(total != 0.0), (
        "`total` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


@pytest.mark.parametrize("emission_type", ENERGY_BALANCED_MODELS)
def test_dust_ir_reradiates_the_absorbed_luminosity(synthetic_ssp_wide, emission_type):
    r"""``\int sed_dust_ir d\nu`` must equal the absorbed ``L_ir``.

    The absolute anchor for the whole factoring scheme. Every other assertion
    here compares float32 against float64 of the *same* code, so a change that
    is wrong in both precisions equally — dropping the re-scale after
    evaluating at unit luminosity, say — passes them all while returning an SED
    ~43 decades too faint. Verified: with the re-scale removed, the float32
    tests still pass and only this one fails.

    Energy conservation is not self-referential: it ties the emitted SED to a
    luminosity computed by an entirely different part of the pipeline.
    """
    ssp = _physical_ssp(synthetic_ssp_wide)
    state = _model(ssp, emission_type).predict_state({})

    wave = np.asarray(state.wave, dtype=np.float64)
    sed = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    l_ir = float(np.asarray(state.derived["L_ir"]))
    assert l_ir > 0.0, "setup: expected a positive absorbed luminosity"

    nu = C_AA / wave
    emitted = abs(np.trapezoid(sed, nu))
    np.testing.assert_allclose(
        emitted,
        l_ir,
        rtol=5e-2,
        err_msg=(
            f"{emission_type}: re-emitted {emitted:.6e} erg/s against an absorbed "
            f"{l_ir:.6e} erg/s — energy balance is broken by a factor "
            f"{emitted / l_ir:.3e}"
        ),
    )


def test_known_non_float32_models_stay_documented(synthetic_ssp_wide):
    """The not-yet-float32 list must not go stale in either direction.

    A model here that becomes clean should be promoted to
    :data:`TEMPLATE_MODELS`; leaving it listed would understate what pure
    float32 delivers. This mirrors the two-way discipline of
    ``test_float32_boundary_inventory``.

    Not parametrized on purpose. It used to be, and with
    :data:`NOT_YET_FLOAT32` now empty pytest collected an empty parameter set
    and emitted ``SKIPPED [1] got empty parameter set`` — a guard reported as a
    skip reads as "ran, nothing to say" when it means "did not run". Looping
    inside the test keeps it visible. When the dict is empty this body asserts
    nothing, which is precisely why
    :func:`test_emission_inventory_is_complete` exists and is the load-bearing
    guard: it is the one that cannot go quiet.
    """
    ssp = _physical_ssp(synthetic_ssp_wide)
    for emission_type, reason in sorted(NOT_YET_FLOAT32.items()):
        with jax.enable_x64(False):
            got = np.asarray(_model(ssp, emission_type).predict_state({}).derived["sed_dust_ir"])

        assert not np.all(np.isfinite(got)), (
            f"{emission_type} is now float32-clean ({reason} no longer applies) — "
            "move it into TEMPLATE_MODELS and update "
            "docs/dev/float32-tier-b-boundary.md"
        )


def _registered_emission_names():
    """Grammar-selectable emission components, i.e. what ``dust.emission.type`` accepts.

    Deliberately reads the forward registry and **not**
    ``DUST_EMISSION_MODELS``. The latter is a lazily-populated function cache:
    measured at 17 entries in a fresh process and 18 after a few models had been
    built, because loading a model registers it on first use. A completeness
    guard keyed on it passes or fails depending on what ran before it — the one
    property a guard must not have. The registry is populated by
    ``__init_subclass__`` at import and measured identical before and after
    those same builds.
    """
    from tengri.components.dust.emission._component_base import EmissionComponent
    from tengri.forward.component_factory import _REGISTRY

    return {
        k for k, v in _REGISTRY.items() if isinstance(v, type) and issubclass(v, EmissionComponent)
    }


def test_emission_inventory_is_complete():
    """Every selectable emission component must be accounted for by name.

    The load-bearing guard of this file. Without it the inventory can only fail
    in the direction someone happens to look: the lists are hand-written, so a
    newly registered model is covered by nothing while the file's own claim that
    "every registered emission model is float32-clean" quietly stays worded as
    if it were checked.

    Two-way, like the flux-scale guard: a name that leaves the registry must
    also leave the lists, so they cannot accumulate ghosts.
    """
    registered = _registered_emission_names()
    accounted = set(TEMPLATE_MODELS) | set(NOT_YET_FLOAT32)

    unaccounted = sorted(registered - accounted)
    assert not unaccounted, (
        f"selectable emission components covered by no list: {unaccounted}. Measure each "
        "in pure float32, then add it to TEMPLATE_MODELS (clean) or NOT_YET_FLOAT32 "
        "(broken, with the measured reason). Leaving it out overstates what float32 "
        "delivers, in the direction nobody checks."
    )

    ghosts = sorted(accounted - registered - LEGACY_ALIASES)
    assert not ghosts, (
        f"listed but no longer a registered component: {ghosts} — drop them from the "
        "inventory, or add them to LEGACY_ALIASES if they are old spellings that still "
        "resolve through the loader cache"
    )


def test_photometry_is_finite_in_pure_float32(synthetic_ssp_wide):
    """The end-to-end goal: a float32 forward pass must yield finite fluxes."""
    ssp = _physical_ssp(synthetic_ssp_wide)
    model = _model(ssp, "dale2014")
    sed64 = np.asarray(model.predict_state({}).sed_intrinsic, dtype=np.float64)
    assert np.all(np.isfinite(sed64)), "setup: float64 SED is not finite"
    assert np.any(sed64 != 0.0), (
        "`sed64` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )

    with jax.enable_x64(False):
        sed32 = np.asarray(_model(ssp, "dale2014").predict_state({}).sed_intrinsic)

    assert np.all(np.isfinite(sed32)), (
        f"total SED non-finite in pure float32 "
        f"({float(np.isfinite(sed32).mean()):.2%} finite) — dust IR is the last "
        "out-of-range consumer in the chain"
    )
    assert np.any(sed32 != 0.0), (
        "`sed32` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


@pytest.mark.parametrize("emission_type", ENERGY_BALANCED_MODELS)
def test_dust_ir_is_actually_added_to_the_sed(synthetic_ssp_wide, emission_type):
    """The published ``sed_dust_ir`` must be what the total SED actually gained.

    ``apply`` rescales the emitted SED and the published dict on separate
    lines, so the two can drift: a rescale applied to one and not the other
    leaves ``sed_dust_ir`` correct while the SED every observable is built from
    is ~43 decades too faint. Verified load-bearing — dropping only the
    ``sed_out`` rescale passes every other test in this file and fails here.

    The forward model's additive contract makes this exact rather than
    approximate: ``sed_intrinsic`` is the sum of the published component SEDs
    on the shared grid (measured residual 0.0), so any scale applied to one and
    not the other shows up immediately.
    """
    ssp = _physical_ssp(synthetic_ssp_wide)
    state = _model(ssp, emission_type).predict_state({})
    total = np.asarray(state.sed_intrinsic, dtype=np.float64)

    parts = [
        np.asarray(state.derived[name], dtype=np.float64)
        for name in ("sed_dust_attenuated", "sed_dust_ir", "sed_nebular")
        if name in state.derived
    ]
    parts = [p for p in parts if p.shape == total.shape]
    assert len(parts) >= 2, "setup: expected the attenuated and IR components on the SED grid"

    dust_ir = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    assert float(np.abs(dust_ir).max()) > 0.0, "setup: nothing was emitted"

    residual = float(np.abs(total - sum(parts)).max() / np.abs(total).max())
    assert residual < 1.0e-12, (
        f"{emission_type}: sed_intrinsic is not the sum of its published parts "
        f"(residual {residual:.3e} of peak) — the dust IR was scaled into the "
        "published dict but not into the SED, or vice versa"
    )
