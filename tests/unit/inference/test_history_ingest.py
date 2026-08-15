# SPDX-License-Identifier: BSD-3-Clause
"""#1677: validation of simulation SFH / Z(t) histories at the ingest boundary.

Pure numpy — no SSP grid file, no model build, no JAX. Everything here is what
:func:`tengri.inference.history_ingest.ingest_histories` decides *before*
anything compiles, which is the only place it can still decide anything: the
metallicity lookup clips onto the SSP grid inside JIT, where no Python exception
can be raised.

The end-to-end counterparts — that these refusals actually fire through
``Catalog.from_histories``, and that an accepted history predicts the SED it
should — are in ``tests/regression/bug/test_issue_1677_metallicity_histories.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.config.exceptions import (
    MetallicityUnitWarning,
    OutOfSSPGridWarning,
    measurements_of,
)
from tengri.inference.history_ingest import (
    MET_UNITS,
    ingest_histories,
    met_to_logzsol,
)
from tengri.parameters.translate import LOG10_ZSUN

# A three-galaxy catalog on a shared 40-node cosmic-time grid, anchored at t=0
# with SFR=0 so nothing extrapolates past the Big Bang.
_T = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])
_N_T = _T.shape[0]
_N = 3

#: A synthetic SSP metallicity grid, absolute log10(Z). With LOG10_ZSUN=-1.848
#: this spans logzsol [-2.152, +0.548] — the same span as the ``synthetic_ssp_wide``
#: fixture, so the numbers here and in the regression file agree.
_SSP_LGMET = np.array([-4.0, -2.65, -1.3])
_GRID_LO = float(_SSP_LGMET.min()) - LOG10_ZSUN
_GRID_HI = float(_SSP_LGMET.max()) - LOG10_ZSUN


def _histories(levels=(1.0, 5.0, 20.0)):
    """(t, sfr) for N galaxies whose SFHs are one shape times a constant."""
    shape = np.ones(_N_T)
    shape[0] = 0.0
    t = np.broadcast_to(_T, (len(levels), _N_T)).copy()
    sfr = np.stack([shape * float(s) for s in levels])
    return t, sfr


def _met(lo=-1.5, hi=-0.1, n=_N):
    """A rising in-grid metallicity history, shared across galaxies."""
    return np.broadcast_to(np.linspace(lo, hi, _N_T), (n, _N_T)).copy()


# ── unit conversion ──────────────────────────────────────────────────


def test_logzsol_is_the_identity() -> None:
    met = _met()
    assert np.array_equal(met_to_logzsol(met, "logzsol"), met)


def test_log_z_abs_removes_the_solar_offset() -> None:
    """The offset must be the one the forward model adds back, or the round trip lies.

    ``_tabulated_lgmet_on_ssp_ages`` lifts ``met_history`` to absolute log10(Z)
    with ``+ LOG10_ZSUN``. Ingest converting with anything else would put stars
    at a metallicity the caller never asked for and nothing would flag it.
    """
    logzsol = _met()
    assert np.allclose(met_to_logzsol(logzsol + LOG10_ZSUN, "log_z_abs"), logzsol)


def test_z_mass_fraction_round_trips() -> None:
    """Zsun = 10**LOG10_ZSUN = 0.0142 (Asplund 2009), the value MIST/DSPS use."""
    logzsol = _met()
    z_mass = 10.0 ** (logzsol + LOG10_ZSUN)
    assert np.allclose(met_to_logzsol(z_mass, "z_mass_fraction"), logzsol)


def test_z_mass_fraction_refuses_primordial_zero() -> None:
    """Z=0 is a modeling decision, not a data error — do not pick a floor for the user.

    Primordial gas is genuinely Z=0 in most simulations. ``log10(0)`` is ``-inf``,
    which would arrive at the grid check as a nonsense number; silently flooring
    it would place stars at a metallicity nobody chose.
    """
    z_mass = np.full((_N, _N_T), 1e-3)
    z_mass[1, 0] = 0.0
    with pytest.raises(ValueError, match="metallicity floor"):
        met_to_logzsol(z_mass, "z_mass_fraction")


def test_unknown_unit_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="is not a metallicity unit") as exc:
        met_to_logzsol(_met(), "dex")
    for unit in MET_UNITS:
        assert unit in str(exc.value), f"{unit!r} missing from the recovery advice"


# ── shape and finiteness ─────────────────────────────────────────────


def test_a_shared_1d_metallicity_broadcasts_like_t_gyr() -> None:
    """One chemical-evolution track across many mass scalings is the common sim case."""
    t, sfr = _histories()
    shared = ingest_histories(t_gyr=_T, sfr=sfr, met=_met(n=1)[0], ssp_lgmet=_SSP_LGMET)
    explicit = ingest_histories(t_gyr=t, sfr=sfr, met=_met(), ssp_lgmet=_SSP_LGMET)
    assert np.array_equal(shared.met, explicit.met)
    assert np.array_equal(shared.t_gyr, explicit.t_gyr)


def test_a_nan_metallicity_node_is_refused_by_position() -> None:
    """Measured before the fix: a NaN node left the predicted flux bit-identical.

    It was dropped entirely — no NaN out, no warning, no change. The message has
    to name the cell, because a caller with 10^5 galaxies cannot find it otherwise.
    """
    t, sfr = _histories()
    met = _met()
    met[1, 3] = np.nan
    with pytest.raises(ValueError, match=r"galaxy 1, node 3"):
        ingest_histories(t_gyr=t, sfr=sfr, met=met, ssp_lgmet=_SSP_LGMET)


def test_a_nan_sfr_is_refused_although_it_passes_the_negativity_check() -> None:
    """``np.any(sfr < 0.0)`` is ``False`` for NaN — the pre-#1677 guard let it through."""
    t, sfr = _histories()
    sfr[2, 5] = np.nan
    assert not np.any(sfr < 0.0), "the premise: the old guard does not see this"
    with pytest.raises(ValueError, match=r"galaxy 2, node 5"):
        ingest_histories(t_gyr=t, sfr=sfr, ssp_lgmet=_SSP_LGMET)


def test_negative_sfr_is_still_refused() -> None:
    t, sfr = _histories()
    sfr[0, 7] = -1.0
    with pytest.raises(ValueError, match="negative entries"):
        ingest_histories(t_gyr=t, sfr=sfr, ssp_lgmet=_SSP_LGMET)


def test_non_monotonic_time_is_still_refused() -> None:
    t, sfr = _histories()
    t[0, 5], t[0, 6] = t[0, 6], t[0, 5]
    with pytest.raises(ValueError, match="strictly increasing"):
        ingest_histories(t_gyr=t, sfr=sfr, ssp_lgmet=_SSP_LGMET)


def test_mismatched_metallicity_shape_names_both():
    t, sfr = _histories()
    with pytest.raises(ValueError, match=r"\(3, 40\)"):
        ingest_histories(t_gyr=t, sfr=sfr, met=_met()[:, :-3], ssp_lgmet=_SSP_LGMET)


# ── the SSP grid ─────────────────────────────────────────────────────


def test_out_of_grid_metallicity_raises_by_default() -> None:
    """The clamp is invisible: logzsol=-6 returns the grid-edge template verbatim."""
    t, sfr = _histories()
    with pytest.raises(ValueError, match="fall outside the SSP metallicity grid"):
        ingest_histories(t_gyr=t, sfr=sfr, met=np.full((_N, _N_T), -6.0), ssp_lgmet=_SSP_LGMET)


def test_the_grid_edges_themselves_are_accepted() -> None:
    """An inclusive bound, or every history clipped to the grid would then be refused."""
    t, sfr = _histories()
    for edge in (_GRID_LO, _GRID_HI):
        ingest_histories(t_gyr=t, sfr=sfr, met=np.full((_N, _N_T), edge), ssp_lgmet=_SSP_LGMET)


def test_the_out_of_grid_report_is_mass_weighted() -> None:
    """Node counts do not say whether the clamp matters; mass formed does.

    Here only the first node is off-grid, and it is the ``SFR = 0`` anchor, so it
    carries none of the stellar mass. The count and the mass share must therefore
    disagree — that disagreement is the whole point of reporting both.
    """
    t, sfr = _histories()
    met = _met()
    met[:, 0] = -6.0
    with pytest.warns(OutOfSSPGridWarning) as rec:
        ingest_histories(t_gyr=t, sfr=sfr, met=met, ssp_lgmet=_SSP_LGMET, on_out_of_grid="warn")
    m = measurements_of(rec[0].message)
    assert m["n_outside"] == _N, "one node per galaxy is off-grid"
    assert m["mass_fraction_outside"] == pytest.approx(0.0, abs=1e-12), (
        "the off-grid node is the SFR=0 anchor, so it carries no stellar mass"
    )
    assert m["worst_logzsol"] == pytest.approx(-6.0)


def test_every_node_is_checked_including_ones_that_formed_no_stars() -> None:
    """A zero-SFR node is not exempt, and the reason is interpolation.

    ``_tabulated_lgmet_on_ssp_ages`` interpolates Z(t) onto the SSP *age* grid, so
    a wild value at a node that formed nothing still drags the metallicity at
    neighboring ages that did. Filtering the check by mass would therefore let a
    real error through; the mass share is a diagnostic, never a filter.
    """
    t, sfr = _histories()
    met = _met()
    met[:, 0] = -6.0  # the SFR=0 anchor
    assert np.all(sfr[:, 0] == 0.0), "the premise: this node formed no stars"
    with pytest.raises(ValueError, match="fall outside"):
        ingest_histories(t_gyr=t, sfr=sfr, met=met, ssp_lgmet=_SSP_LGMET)


def test_ignore_restores_the_pre_1677_silence() -> None:
    t, sfr = _histories()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest_histories(
            t_gyr=t,
            sfr=sfr,
            met=np.full((_N, _N_T), -6.0),
            ssp_lgmet=_SSP_LGMET,
            on_out_of_grid="ignore",
        )


def test_no_ssp_grid_means_no_grid_check() -> None:
    """A component built without SSP data has no grid; skip rather than guess one."""
    t, sfr = _histories()
    ingest_histories(t_gyr=t, sfr=sfr, met=np.full((_N, _N_T), -6.0), ssp_lgmet=None)


def test_an_unknown_policy_is_refused_before_any_work() -> None:
    t, sfr = _histories()
    with pytest.raises(ValueError, match="is not a policy"):
        ingest_histories(t_gyr=t, sfr=sfr, met=_met(), on_out_of_grid="clip")


# ── the units heuristic ──────────────────────────────────────────────


def test_a_mass_fraction_read_as_logzsol_warns() -> None:
    """The case ``on_out_of_grid`` is structurally unable to see.

    A metal mass fraction is a small *positive* number, and small positive
    log10(Z/Zsun) values are legal near-solar metallicities — so these land
    inside the grid and the range check cannot fire. Dynamic range is what
    separates them: real enrichment moves Z(t) by orders of magnitude.
    """
    t, sfr = _histories()
    z_mass = 10.0 ** (_met() + LOG10_ZSUN)  # 4.5e-4 .. 1.1e-2
    assert np.all((z_mass > _GRID_LO) & (z_mass < _GRID_HI)), (
        "the premise: as logzsol these are in-grid, so the grid check is blind to them"
    )
    with pytest.warns(MetallicityUnitWarning, match="signature of a metal mass fraction"):
        ingest_histories(t_gyr=t, sfr=sfr, met=z_mass, ssp_lgmet=_SSP_LGMET)


def test_a_correctly_declared_mass_fraction_does_not_warn() -> None:
    """Running on the converted values is what keeps this free of false alarms."""
    t, sfr = _histories()
    z_mass = 10.0 ** (_met() + LOG10_ZSUN)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", MetallicityUnitWarning)
        ingest_histories(
            t_gyr=t, sfr=sfr, met=z_mass, met_unit="z_mass_fraction", ssp_lgmet=_SSP_LGMET
        )


def test_an_ordinary_logzsol_history_does_not_warn() -> None:
    t, sfr = _histories()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", MetallicityUnitWarning)
        ingest_histories(t_gyr=t, sfr=sfr, met=_met(), ssp_lgmet=_SSP_LGMET)


# ── bookkeeping ──────────────────────────────────────────────────────


def test_mass_formed_is_the_trapezoidal_integral() -> None:
    """Carried so a caller can check ingest against the simulation's own catalog mass."""
    t, sfr = _histories()
    h = ingest_histories(t_gyr=t, sfr=sfr, met=_met(), ssp_lgmet=_SSP_LGMET)
    assert np.allclose(h.mass_formed, np.trapezoid(sfr, t, axis=1) * 1e9, rtol=1e-12)


def test_a_single_node_history_has_no_span_and_no_mass() -> None:
    """Degenerate but reachable — a one-snapshot catalog. It must not divide by zero."""
    h = ingest_histories(t_gyr=np.array([5.0]), sfr=np.array([[3.0]]), ssp_lgmet=_SSP_LGMET)
    assert h.mass_formed == pytest.approx(0.0)


def test_met_is_none_when_no_history_is_supplied() -> None:
    t, sfr = _histories()
    assert ingest_histories(t_gyr=t, sfr=sfr, ssp_lgmet=_SSP_LGMET).met is None


# ── gas-phase metallicity, the second and separate knob ──────────────
#
# ``met`` is the metallicity the *stars* formed from and selects the SSP
# templates. ``met_gas`` is the metallicity of the *ionized gas* and drives
# nebular emission. They are different physical quantities — pristine inflow
# decouples them in real galaxies — so ingest keeps them apart and converts
# both through the one declared unit.


def _gas_kwargs(**kw):
    t, sfr = _histories()
    return {"t_gyr": t, "sfr": sfr, "ssp_lgmet": _SSP_LGMET, **kw}


def test_gas_metallicity_is_carried_separately_from_the_stellar_history() -> None:
    h = ingest_histories(**_gas_kwargs(met=_met(), met_gas=np.full(_N, -0.4)))
    assert h.met.shape == (_N, _N_T), "stellar Z stays a full history"
    assert h.met_gas.shape == (_N,), "gas Z collapses to one value per galaxy"
    assert np.allclose(h.met_gas, -0.4)


def test_gas_metallicity_may_be_supplied_without_a_stellar_history() -> None:
    """The two are independent in both directions, not merely in one."""
    h = ingest_histories(**_gas_kwargs(met_gas=np.full(_N, -0.4)))
    assert h.met is None
    assert np.allclose(h.met_gas, -0.4)


def test_a_gas_track_is_read_at_the_observed_epoch() -> None:
    """Nebular emission comes from stars younger than ~10 Myr.

    Only the present-day gas metallicity is therefore observable, so a track is
    reduced to its **last** node — last because ``t_gyr`` is cosmic time and
    ascending, the same orientation the SFH requires.
    """
    track = np.broadcast_to(np.linspace(-2.0, 0.15, _N_T), (_N, _N_T)).copy()
    h = ingest_histories(**_gas_kwargs(met=_met(), met_gas=track))
    assert np.allclose(h.met_gas, 0.15)


def test_a_shared_1d_gas_track_is_read_at_the_observed_epoch_too() -> None:
    track = np.linspace(-2.0, 0.15, _N_T)
    h = ingest_histories(**_gas_kwargs(met=_met(), met_gas=track))
    assert np.allclose(h.met_gas, 0.15)


def test_an_ambiguous_1d_gas_input_is_refused_rather_than_guessed() -> None:
    """When N == n_t a 1-D input could be per-galaxy or a shared track.

    The two readings give different physics and neither is recoverable from the
    other afterwards, so this is the one shape ingest refuses instead of picking.
    """
    n = _N_T
    shape = np.ones(_N_T)
    shape[0] = 0.0
    t = np.broadcast_to(_T, (n, _N_T)).copy()
    sfr = np.stack([shape] * n)
    with pytest.raises(ValueError, match="could equally be"):
        ingest_histories(t_gyr=t, sfr=sfr, met_gas=np.full(n, -0.3), ssp_lgmet=_SSP_LGMET)


def test_gas_metallicity_uses_the_same_declared_unit() -> None:
    """A snapshot stores both metallicities the same way, so one flag covers both."""
    z_mass = 10.0 ** (-0.4 + LOG10_ZSUN)
    h = ingest_histories(
        **_gas_kwargs(
            met=10.0 ** (_met() + LOG10_ZSUN),
            met_gas=np.full(_N, z_mass),
            met_unit="z_mass_fraction",
        )
    )
    assert np.allclose(h.met_gas, -0.4)
    assert np.allclose(h.met, _met())


def test_a_non_finite_gas_metallicity_is_refused() -> None:
    gas = np.full(_N, -0.4)
    gas[1] = np.nan
    with pytest.raises(ValueError, match="met_gas"):
        ingest_histories(**_gas_kwargs(met=_met(), met_gas=gas))


def test_gas_metallicity_is_not_checked_against_the_stellar_grid() -> None:
    """It is read on the nebular backend's grid, a different axis entirely.

    ``on_out_of_grid`` guards the SSP metallicity axis. Applying that span to the
    gas-phase value would refuse a legitimate CLOUDY/Cue input for failing a
    constraint that does not apply to it.
    """
    ingest_histories(**_gas_kwargs(met=_met(), met_gas=np.full(_N, -6.0)))


def test_gas_metallicity_defaults_to_none_so_the_model_keeps_its_own() -> None:
    h = ingest_histories(**_gas_kwargs(met=_met()))
    assert h.met_gas is None
