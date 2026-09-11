# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the orchestrator → legacy-Quantities bridges.

The :func:`tengri.forward.state_to_*_quantities` helpers and their
``SEDModel.predict_*_components`` wrappers convert a
:class:`tengri.protocols.ForwardState` into the legacy
:class:`SFHQuantities` / :class:`SEDQuantities` NamedTuples (and the
new :class:`RadioQuantities` / :class:`XRayQuantities` /
:class:`IonizingQuantities` mirrors).

This module pins their behavior with a fixed parameter set so future
changes to the bridges or the underlying components surface as test
failures rather than silent drift.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

from tengri.components.dust.two_component import DustSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.stellar import StellarSEDComponent
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.xray.component import XRaySEDComponent
from tengri.forward import (
    IonizingQuantities,
    RadioQuantities,
    XRayQuantities,
    run_components,
    state_to_ionizing_quantities,
    state_to_radio_quantities,
    state_to_sed_quantities,
    state_to_sfh_quantities,
    state_to_xray_quantities,
)
from tengri.forward.orchestrator import default_params_dict
from tengri.protocols.component import ForwardState

# Bare-stellar SSP — required by Cue (wNE SSPs now raise CueWNESSPError).
_SSP_PATH = pathlib.Path("data/fsps_prsc_miles_chabrier.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_PATH.exists():
        pytest.skip(f"SSP file not present at {_SSP_PATH}")
    return load_ssp_data(str(_SSP_PATH))


@pytest.fixture(scope="module")
def state(ssp):
    chain = [
        StellarSEDComponent(ssp_data=ssp),
        DustSEDComponent(),
        RadioSEDComponent(),
        XRaySEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))
    params = {
        # tsnorm SFH
        "sfh_tsnorm_log_total_mass": jnp.asarray(10.0),  # 1e10 Msun galaxy (#673)
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        "met_logzsol": jnp.asarray(-0.5),
        # Dust
        "dust_tau_bc": jnp.asarray(0.5),
        "dust_tau_diff": jnp.asarray(0.2),
        "dust_slope": jnp.asarray(-0.7),
        "dust_T": jnp.asarray(35.0),
        "dust_beta_ir": jnp.asarray(1.6),
        # Radio + X-ray at their declared defaults, rather than a literal that
        # cannot follow them (#1832).
        **default_params_dict([RadioSEDComponent(), XRaySEDComponent()]),
        # The one deliberate departure from the declared defaults.
        "xray_delta_alpha_ox": jnp.asarray(-1.4),
        "redshift": jnp.asarray(0.0),
    }
    return run_components(chain, state0, params)


# ── SFHQuantities ────────────────────────────────────────────────────


def test_sfh_quantities_all_fields_finite(state):
    sfh_q = state_to_sfh_quantities(state)
    for f in sfh_q._fields:
        v = float(getattr(sfh_q, f))
        assert jnp.isfinite(v), f"SFH.{f} = {v} (expected finite)"


def test_sfh_quantities_physical_ranges(state):
    sfh_q = state_to_sfh_quantities(state)
    # 10**10 Msun-class galaxy with peak SFH 10 Msun/yr at 2 Gyr
    assert 1e9 < float(sfh_q.stellar_mass) < 1e11
    assert 0.0 < float(sfh_q.sfr_10myr) < 100.0
    assert 0.0 < float(sfh_q.sfr_100myr) < 100.0
    # ssfr in 1/yr; reasonable galaxy range
    assert 1e-12 < float(sfh_q.ssfr) < 1e-7
    # Mass-weighted age positive, < age of universe
    assert 0.0 < float(sfh_q.mass_weighted_age_gyr) < 14.0


# ── SEDQuantities ────────────────────────────────────────────────────


def test_sed_quantities_all_15_fields_populated(state):
    sed_q = state_to_sed_quantities(state)
    nans = [f for f in sed_q._fields if not bool(jnp.isfinite(getattr(sed_q, f)))]
    assert nans == [], f"SED fields with NaN: {nans}"


def test_sed_quantities_physical_ranges(state):
    sed_q = state_to_sed_quantities(state)
    # L_bol ~ 10^10 Lsun for our setup
    assert 1e9 < float(sed_q.l_bol) < 1e12
    # Dust attenuation moves L_TIR below L_bol
    assert 0.0 < float(sed_q.l_tir) < float(sed_q.l_bol)
    # UV slope reddened by dust → typically negative (e.g. -1 to -2 for SF + dust)
    assert -3.0 < float(sed_q.uv_slope_beta) < 0.0
    # Dn4000 in plausible range
    assert 0.5 < float(sed_q.dn4000) < 3.0
    # Pre-dust UV brighter than post-dust (tau_bc=0.5)
    assert float(sed_q.fuv_flux_intrinsic) > float(sed_q.fuv_flux)


# ── Radio / XRay / Ionizing ──────────────────────────────────────────


def test_radio_quantities_finite_and_physical(state):
    rq = state_to_radio_quantities(state)
    assert isinstance(rq, RadioQuantities)
    # 1.4 GHz luminosity in plausible SF-galaxy range
    assert 1e25 < float(rq.l_1p4ghz) < 1e30
    # FIR-radio correlation parameter in plausible range
    assert 1.0 < float(rq.q_ir) < 5.0


def test_xray_quantities_finite_and_physical(state):
    xq = state_to_xray_quantities(state)
    assert isinstance(xq, XRayQuantities)
    # XRB luminosity in plausible range
    assert 1e37 < float(xq.l_x_xrb) < 1e42
    # No AGN component → l_x_agn is exactly 0 (not NaN)
    assert float(xq.l_x_agn) == 0.0
    assert float(xq.l_x_total) == float(xq.l_x_xrb)


def test_ionizing_quantities_finite(state):
    iq = state_to_ionizing_quantities(state)
    assert isinstance(iq, IonizingQuantities)
    # nion magnitude is set by stellar; the BakedIn SSP suppresses it
    # but it should still be positive and finite.
    assert float(iq.q_h) > 0.0
    assert float(iq.xi_ion) > 0.0


# ── Bridges work under jit ───────────────────────────────────────────


def test_state_to_sfh_quantities_jit_compatible(state):
    out = jax.jit(state_to_sfh_quantities)(state)
    out_eager = state_to_sfh_quantities(state)
    for f in out._fields:
        assert jnp.allclose(getattr(out, f), getattr(out_eager, f), rtol=1e-12), (
            f"JIT vs eager differ at {f}"
        )


def test_state_to_sed_quantities_jit_compatible(state):
    out = jax.jit(state_to_sed_quantities)(state)
    out_eager = state_to_sed_quantities(state)
    for f in out._fields:
        assert jnp.allclose(getattr(out, f), getattr(out_eager, f), rtol=1e-12), (
            f"JIT vs eager differ at {f}"
        )


# ── Emission-lines bridge ────────────────────────────────────────────


def _build_cue_state(ssp, *, cue_full_catalog: bool):
    """Run Stellar + Cue-nebular chain, with the requested catalog scope.

    ``cue_full_catalog=False`` (the ``NebularSEDComponentConfig`` default)
    selects Cue's legacy 128-line CLOUDY/FSPS-matched subset
    (``weights.line_old_idx``); ``True`` selects the full ~271-species
    Cue-trained catalog. C IV (``civ_1549``) is Cue-only and is not one of
    the 128 legacy indices, so it is absent under ``False`` and present
    under ``True`` -- the split :func:`state_with_cue` /
    :func:`state_with_cue_full_catalog` fixtures below exist to pin both
    sides of that (#2192, #2236).
    """
    from tengri.components.nebular.component import (
        NebularSEDComponent,
        NebularSEDComponentConfig,
    )
    from tengri.components.nebular.cue import CueBackend
    from tengri.forward import state_to_emission_lines  # noqa: F401 — used elsewhere

    cue_path = pathlib.Path("data/cue_weights.npz").resolve()
    if not cue_path.exists():
        pytest.skip(f"Cue weights not present at {cue_path}")
    cue = CueBackend(weights_path=str(cue_path), ssp_data=ssp)
    chain = [
        StellarSEDComponent(ssp_data=ssp),
        NebularSEDComponent(
            config=NebularSEDComponentConfig(backend="cue", cue_full_catalog=cue_full_catalog),
            backend=cue,
        ),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave)
    params = {
        "sfh_tsnorm_log_total_mass": jnp.asarray(10.0),  # 1e10 Msun galaxy (#673)
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        "met_logzsol": jnp.asarray(-0.5),
        "neb_logU": jnp.asarray(-2.5),
        "neb_logZ_gas": jnp.asarray(-0.3),
        "neb_fesc": jnp.asarray(0.0),
        "neb_fesc_lya": jnp.asarray(0.0),
        "ionspec_index1": jnp.asarray(15.0),
        "ionspec_index2": jnp.asarray(5.0),
        "ionspec_index3": jnp.asarray(0.0),
        "ionspec_index4": jnp.asarray(0.0),
        "ionspec_logLratio1": jnp.asarray(2.0),
        "ionspec_logLratio2": jnp.asarray(0.5),
        "ionspec_logLratio3": jnp.asarray(0.5),
        "gas_logn": jnp.asarray(2.0),
        "gas_logno": jnp.asarray(0.0),
        "gas_logco": jnp.asarray(0.0),
        "redshift": jnp.asarray(0.0),
    }
    return run_components(chain, state0, params)


@pytest.fixture(scope="module")
def state_with_cue(ssp):
    """Cue nebular backend, default 128-line legacy CLOUDY/FSPS subset.

    C IV (``civ_1549``) is Cue-only and not one of the 128 legacy indices,
    so it is absent from this catalog by construction (#2192, #2236).
    """
    return _build_cue_state(ssp, cue_full_catalog=False)


@pytest.fixture(scope="module")
def state_with_cue_full_catalog(ssp):
    """Cue nebular backend, ``cue_full_catalog=True`` (the ~271-species menu).

    C IV (``civ_1549``) is present here, unlike :func:`state_with_cue`'s
    128-line legacy subset (#2192, #2236).
    """
    return _build_cue_state(ssp, cue_full_catalog=True)


def test_emission_lines_published_by_cue(state_with_cue):
    """Cue backend should populate state.derived line catalog."""
    assert "line_waves" in state_with_cue.derived
    assert "line_lums" in state_with_cue.derived
    # Cue's output is a many-line catalog (~100+).
    assert state_with_cue.derived["line_waves"].shape[0] > 50


def test_state_to_emission_lines_headlines_finite_except_civ_on_legacy_subset(state_with_cue):
    """10 of 11 headline lines are finite on Cue's default 128-line subset.

    ``civ_1549`` is the one exception: C IV is a Cue-only line, absent from
    the legacy 128-index CLOUDY/FSPS subset ``state_with_cue`` builds
    (:func:`_build_cue_state`), so no catalog entry lies within the 5 A
    match tolerance (``_LINE_MATCH_TOL_AA`` in
    ``tengri.utils.sed_quantities``) and ``extract_line_luminosity``
    returns NaN -- the intended #2192 answer for a genuinely absent line,
    not a lookup bug. This test used to assert all 11 finite: that was
    stale from before #2192 (81d27d39f) made the lookup honest -- an
    unconditional nearest-neighbor previously matched both C IV components
    to HeII 1640 (~90-92 A away) and summed it twice, finite but wrong.
    ``tests/regression/bug/test_bug_1889_lineproperties_parity.py`` already
    pins the corrected NaN answer for this same default-config case (#2236).
    ``all_waves``/``all_lums`` are arrays (skipped here; see
    ``test_state_to_emission_lines_publishes_full_catalog``).
    """
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue)
    scalar_fields = [f for f in lines._fields if f not in ("all_waves", "all_lums")]
    nans = [f for f in scalar_fields if not bool(jnp.isfinite(getattr(lines, f)))]
    assert nans == ["civ_1549"], (
        f"Expected only civ_1549 NaN on Cue's default 128-line subset, got: {nans}"
    )


def test_state_to_emission_lines_all_headlines_finite_on_full_catalog(
    state_with_cue_full_catalog,
):
    """All 11 headline lines are finite when Cue exposes its full catalog.

    ``cue_full_catalog=True`` selects the ~271-species Cue-trained menu,
    which carries C IV (``sorted_line_wav[9]=1548.19``,
    ``[10]=1550.77`` A vacuum) within the 5 A match tolerance, so
    ``civ_1549`` -- the one line NaN under the default 128-line subset
    (see the sibling test above) -- is finite here too (#2236).
    """
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue_full_catalog)
    scalar_fields = [f for f in lines._fields if f not in ("all_waves", "all_lums")]
    nans = [f for f in scalar_fields if not bool(jnp.isfinite(getattr(lines, f)))]
    assert nans == [], f"Lines with NaN: {nans}"


def test_state_to_emission_lines_publishes_full_catalog(state_with_cue):
    """Cue exposes the full ~138-line catalog via all_waves/all_lums (#303)."""
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue)
    assert lines.all_waves.size > 50, (
        f"Expected >50 species exposed via all_waves, got {lines.all_waves.size}"
    )
    assert lines.all_waves.shape == lines.all_lums.shape
    # HeII 1640 was a canonical example in the issue: must be queryable.
    heii = float(lines.get(1640.4, tol_aa=5.0))
    assert jnp.isfinite(heii), "HeII 1640 should be in the catalog"


def test_state_to_emission_lines_balmer_decrement(state_with_cue):
    """Halpha / Hbeta ≈ 2.85 (case-B recombination at T_e=10⁴ K)."""
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue)
    ratio = float(lines.halpha / lines.hbeta)
    # Cue's intrinsic Balmer decrement should be in [2.5, 3.5] —
    # case-B is 2.86; small departures are physical.
    assert 2.0 < ratio < 4.0, f"Halpha/Hbeta = {ratio} (expected ~2.85)"


def test_state_to_emission_lines_no_catalog_returns_nan(state):
    """Chain without nebular catalog (no Cue/Cloudy) → NaN headlines + empty all_*."""
    from tengri.forward import state_to_emission_lines

    # The ``state`` fixture has no nebular component → no line_waves.
    assert "line_waves" not in state.derived
    lines = state_to_emission_lines(state)
    scalar_fields = [f for f in lines._fields if f not in ("all_waves", "all_lums")]
    for f in scalar_fields:
        assert not bool(jnp.isfinite(getattr(lines, f))), (
            f"Lines.{f} should be NaN when no catalog published"
        )
    assert lines.all_waves.size == 0
    assert lines.all_lums.size == 0
