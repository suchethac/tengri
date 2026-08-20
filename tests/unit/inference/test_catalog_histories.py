# SPDX-License-Identifier: BSD-3-Clause
"""#1396 W6: a catalog driven by tabulated SFH/Z histories, not by free params.

``Catalog.predict`` used to extract exactly ``fwd.spec.free_params`` and stack
them with ``np.stack(param_arrays, axis=1)``. Both assumptions break for a
simulation-driven catalog:

* the ``table`` SFH declares **zero** free parameters ("the table IS the SFH"),
  so ``sfh_t_gyr`` / ``sfh_sfr`` / ``met_history`` were never read out of the
  table at all — and the forward then refused with the #996 runtime check,
  loudly, which is how we know nobody silently got default-SFH photometry;
* ``np.stack(..., axis=1)`` is scalar-only by construction, so a ``(N,)``
  scalar could not coexist with an ``(N, n_t)`` history even once the names
  were carried through.

Both are replaced by one uniform column channel: a mapping of name to array
whose leading axis is the galaxy, with **any** trailing shape. Scalars are
simply the ``(N,)`` case, which is what the docstring already promised.

Flux equality here is always checked as a **ratio**, never with bare
``np.allclose``: these fluxes are of order 1e-13, far below ``allclose``'s
default ``atol=1e-8``, which would call every pair of them equal.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

# Cosmic time [Gyr] for the tabulated histories, and the redshift they are
# evaluated at. z=0.05 -> cosmic age ~13.11 Gyr, so the grid ends at 13.0.
#
# The grid starts at t=0 with SFR=0 deliberately. A table that starts at, say,
# t=0.5 Gyr with a nonzero SFR is held at that value when the component
# extrapolates back to the SSP's oldest age (13.8 Gyr), which places ~16% of the
# stellar mass *before the Big Bang* — the component says so, loudly. That
# truncation is linear in the SFR level, so the ratio assertions below would
# still pass while silently measuring a truncated history. Anchoring at zero
# removes the extrapolation instead of tolerating it.
_Z_OBS = 0.05
_T_GYR = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])


@pytest.fixture
def fwd_table_sfh(synthetic_ssp_wide, synthetic_tophat_obs):
    """3-band ForwardModel whose SFH is tabulated at runtime (#996)."""
    from tengri import FIXED, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "table"},
            dust={
                "law": "power_law",
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.5,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(_Z_OBS),
        )
        return ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)


def _flat_histories(sfrs):
    """(N, n_t) histories that are one common shape scaled by each SFR level.

    Zero at t=0 (see ``_T_GYR``), then flat. Because every galaxy is the *same*
    shape times a constant, the stellar mass — and so the flux — is exactly
    linear in that constant, which is what the ratio assertions below exploit.
    """
    n, n_t = len(sfrs), _T_GYR.shape[0]
    shape = np.ones(n_t)
    shape[0] = 0.0  # anchor at the Big Bang so nothing extrapolates past it
    t = np.broadcast_to(_T_GYR, (n, n_t)).copy()
    sfr = np.stack([shape * float(s) for s in sfrs])
    return t, sfr


def test_table_sfh_declares_no_free_params(fwd_table_sfh):
    """The premise: the free_params-only extraction cannot see this model's SFH."""
    assert not [p for p in fwd_table_sfh.spec.free_params if p.startswith("sfh_")]


def test_the_test_histories_are_not_truncated(fwd_table_sfh):
    """Assert the setup: no stellar mass falls outside cosmic time.

    Without this, every ratio assertion below would still pass on a history that
    was silently truncated — truncation of a scaled common shape is itself
    linear in the scale, so it cancels out of a ratio.
    """
    import jax.numpy as jnp

    from tengri.components.stellar.component import SFHBeforeBigBangWarning

    t, sfr = _flat_histories([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", SFHBeforeBigBangWarning)
        fwd_table_sfh.predict_photometry(
            {
                "dust_tau_diff": jnp.asarray(0.2),
                "sfh_t_gyr": jnp.asarray(t[0]),
                "sfh_sfr": jnp.asarray(sfr[0]),
            }
        )


def test_predict_carries_array_valued_history_columns(fwd_table_sfh):
    """A (N,) scalar and an (N, n_t) history must coexist in one column channel."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    cat = Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu")

    flux = np.asarray(
        cat.predict(
            {
                "dust_tau_diff": np.array([0.2, 0.2, 0.2]),
                "sfh_t_gyr": t,
                "sfh_sfr": sfr,
            }
        )
    )

    assert flux.shape == (3, fwd_table_sfh.observation.photometry.n_filters)
    assert np.all(np.isfinite(flux))


def test_predict_actually_uses_the_histories(fwd_table_sfh):
    """The load-bearing test: flux must scale with the tabulated SFR.

    A flat SFH makes the stellar mass — and so the flux — exactly linear in the
    SFR level, so a 1 / 5 / 20 ladder must come back as a 1 / 5 / 20 ratio. If
    the history columns were dropped and the model fell back to defaults, all
    three rows would be identical.
    """
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    cat = Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu")

    flux = np.asarray(
        cat.predict(
            {
                "dust_tau_diff": np.array([0.2, 0.2, 0.2]),
                "sfh_t_gyr": t,
                "sfh_sfr": sfr,
            }
        )
    )

    # Ratio, never np.allclose — these fluxes are ~1e-13 and allclose's default
    # atol=1e-8 would declare all three rows equal regardless of the histories.
    assert np.allclose(flux[1] / flux[0], 5.0, rtol=1e-6), f"row1/row0 = {flux[1] / flux[0]}"
    assert np.allclose(flux[2] / flux[0], 20.0, rtol=1e-6), f"row2/row0 = {flux[2] / flux[0]}"


def test_predict_matches_the_single_galaxy_forward(fwd_table_sfh):
    """Per-row parity: the catalog path must equal predict_photometry row by row."""
    import jax.numpy as jnp

    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    taus = np.array([0.1, 0.2, 0.3])
    cat = Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu")

    batched = np.asarray(cat.predict({"dust_tau_diff": taus, "sfh_t_gyr": t, "sfh_sfr": sfr}))

    for i in range(3):
        one = np.asarray(
            fwd_table_sfh.predict_photometry(
                {
                    "dust_tau_diff": jnp.asarray(taus[i]),
                    "sfh_t_gyr": jnp.asarray(t[i]),
                    "sfh_sfr": jnp.asarray(sfr[i]),
                }
            )
        )
        assert np.allclose(batched[i] / one, 1.0, rtol=1e-9), f"row {i} diverges from the forward"


def test_predict_chunking_does_not_change_the_answer(fwd_table_sfh):
    """Chunk boundaries must not perturb the result (5 galaxies, chunk_size=2)."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 2.0, 3.0, 4.0, 5.0])
    cols = {"dust_tau_diff": np.full(5, 0.2), "sfh_t_gyr": t, "sfh_sfr": sfr}
    cat = Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu")

    whole = np.asarray(cat.predict(cols, chunk_size=1024))
    chunked = np.asarray(cat.predict(cols, chunk_size=2))
    assert np.allclose(chunked / whole, 1.0, rtol=1e-12)


def test_predict_names_a_missing_column(fwd_table_sfh):
    """A missing free parameter must be named, not surface as a bare KeyError."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0])
    cat = Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu")

    with pytest.raises(ValueError, match="dust_tau_diff"):
        cat.predict({"sfh_t_gyr": t, "sfh_sfr": sfr})


def test_predict_rejects_ragged_leading_axis(fwd_table_sfh):
    """Columns disagreeing on N must fail loudly, naming both lengths."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 2.0, 3.0])
    cat = Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu")

    with pytest.raises(ValueError, match="same leading"):
        cat.predict({"dust_tau_diff": np.full(2, 0.2), "sfh_t_gyr": t, "sfh_sfr": sfr})


def test_scalar_only_catalog_still_predicts(synthetic_ssp_wide, synthetic_tophat_obs):
    """Regression guard: the ordinary all-scalar catalog path is unaffected."""
    from tengri import FIXED, FREE, Catalog, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED, "tau_bc": 0.5},
            neb={"type": "none"},
            redshift=Fixed(_Z_OBS),
        )
        fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)

    cat = Catalog(fwd, None, flux_unit="cgs_fnu")
    key = jax.random.PRNGKey(0)
    rows = [dict(fwd.spec.sample(jax.random.fold_in(key, i))) for i in range(3)]
    cols = {name: np.array([float(r[name]) for r in rows]) for name in fwd.spec.free_params}

    flux = np.asarray(cat.predict(cols))
    assert flux.shape == (3, fwd.observation.photometry.n_filters)
    assert np.all(np.isfinite(flux))


# ── Catalog.from_histories — the simulation-catalog constructor (#1396 §8.1) ──


def test_from_histories_predicts(fwd_table_sfh):
    """The headline path: histories in, photometry out."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    cat = Catalog.from_histories(
        fwd_table_sfh, t_gyr=t, sfr=sfr, params={"dust_tau_diff": np.full(3, 0.2)}
    )

    flux = np.asarray(cat.predict())
    assert flux.shape == (3, fwd_table_sfh.observation.photometry.n_filters)
    assert np.allclose(flux[2] / flux[0], 20.0, rtol=1e-6)


def test_from_histories_broadcasts_a_shared_time_grid(fwd_table_sfh):
    """A single (n_t,) grid shared by every galaxy must broadcast to (N, n_t)."""
    from tengri import Catalog

    _t, sfr = _flat_histories([1.0, 5.0, 20.0])
    shared = Catalog.from_histories(
        fwd_table_sfh, t_gyr=_T_GYR, sfr=sfr, params={"dust_tau_diff": np.full(3, 0.2)}
    )
    per_galaxy = Catalog.from_histories(
        fwd_table_sfh, t_gyr=_t, sfr=sfr, params={"dust_tau_diff": np.full(3, 0.2)}
    )

    assert np.allclose(np.asarray(shared.predict()) / np.asarray(per_galaxy.predict()), 1.0)


def test_from_histories_equals_explicit_columns(fwd_table_sfh):
    """from_histories is sugar, not a second code path — it must agree exactly."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    tau = np.array([0.1, 0.2, 0.3])

    sugar = np.asarray(
        Catalog.from_histories(
            fwd_table_sfh, t_gyr=t, sfr=sfr, params={"dust_tau_diff": tau}
        ).predict()
    )
    explicit = np.asarray(
        Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu").predict(
            {"dust_tau_diff": tau, "sfh_t_gyr": t, "sfh_sfr": sfr}
        )
    )
    assert np.allclose(sugar / explicit, 1.0, rtol=1e-12)


def test_from_histories_rejects_a_non_table_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """A parametric-SFH model cannot consume histories — say so, and name the fix."""
    from tengri import FIXED, FREE, Catalog, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED, "tau_bc": 0.5},
            neb={"type": "none"},
            redshift=Fixed(_Z_OBS),
        )
        fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)

    t, sfr = _flat_histories([1.0])
    with pytest.raises(ValueError, match=r"sfh=\{'type': 'table'\}"):
        Catalog.from_histories(fwd, t_gyr=t, sfr=sfr)


def test_from_histories_rejects_non_monotonic_time(fwd_table_sfh):
    """Cosmic time must increase — a scrambled grid is a silent-garbage input."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0])
    t[0, 5], t[0, 6] = t[0, 6], t[0, 5]
    with pytest.raises(ValueError, match="increas"):
        Catalog.from_histories(fwd_table_sfh, t_gyr=t, sfr=sfr)


def test_from_histories_rejects_negative_sfr(fwd_table_sfh):
    """A negative SFR is unphysical and would quietly subtract stellar mass."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0])
    sfr[0, 3] = -1.0
    with pytest.raises(ValueError, match="negative"):
        Catalog.from_histories(fwd_table_sfh, t_gyr=t, sfr=sfr)


def test_from_histories_rejects_mismatched_n_t(fwd_table_sfh):
    """sfr and t_gyr must agree on n_t, naming both."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 2.0])
    with pytest.raises(ValueError, match="n_t"):
        Catalog.from_histories(fwd_table_sfh, t_gyr=t, sfr=sfr[:, :-3])


def test_from_histories_rejects_mismatched_n_galaxies(fwd_table_sfh):
    """Per-galaxy scalars must agree with the history galaxy count."""
    from tengri import Catalog

    t, sfr = _flat_histories([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same leading"):
        Catalog.from_histories(
            fwd_table_sfh, t_gyr=t, sfr=sfr, params={"dust_tau_diff": np.full(2, 0.2)}
        )


def test_from_histories_rejects_met_without_a_table_metallicity(fwd_table_sfh):
    """met= needs metallicity_model='table'; this fixture is delta, so refuse.

    This assertion has been round the loop twice, which is worth recording
    because the second trip looked exactly like the first. #1677 raised advice
    naming ``met={'type': 'table'}``; #1678 repointed it at
    ``stellar={'met_mode': 'table'}`` because no ``met`` group existed and the
    advice therefore raised; #1720 added the ``met`` group, making the original
    form the working one again.

    So a string pinned here is only as true as the grammar underneath it, and
    the test cannot tell the difference. That is why the property with teeth
    lives next door in
    :func:`test_the_met_advice_is_a_form_the_grammar_accepts`, which feeds the
    advice to ``parse_groups`` instead of comparing it — the one check that
    fails when the grammar moves rather than following it.
    """
    from tengri import Catalog

    t, sfr = _flat_histories([1.0])
    with pytest.raises(ValueError, match=r"met=\{'type': 'table'\}"):
        Catalog.from_histories(fwd_table_sfh, t_gyr=t, sfr=sfr, met=np.zeros_like(sfr))


def test_the_met_advice_is_a_form_the_grammar_accepts(fwd_table_sfh):
    """Pull the advice out of the raised message and run it (#1677).

    Matching the text only proves the wording did not change. This proves the
    reader can act on it — the property the old assertion could not have had,
    because the string it pinned raised.
    """
    from tengri import Catalog
    from tengri.parameters.groups import parse_groups

    t, sfr = _flat_histories([1.0])
    with pytest.raises(ValueError) as excinfo:
        Catalog.from_histories(fwd_table_sfh, t_gyr=t, sfr=sfr, met=np.zeros_like(sfr))

    spec = parse_groups(met={"type": "table"})
    assert spec.met_mode == "table"
    assert "met={'type': 'table'}" in str(excinfo.value)


def test_predict_without_columns_needs_from_histories(fwd_table_sfh):
    """predict() with no argument is only meaningful on a from_histories catalog."""
    from tengri import Catalog

    with pytest.raises(ValueError, match="from_histories"):
        Catalog(fwd_table_sfh, None, flux_unit="cgs_fnu").predict()


# ── Catalog.simulate — photometry + lines + properties (#1396 §8.1) ──


def _histories_catalog(fwd, sfrs=(1.0, 5.0, 20.0)):
    from tengri import Catalog

    t, sfr = _flat_histories(list(sfrs))
    return Catalog.from_histories(
        fwd, t_gyr=t, sfr=sfr, params={"dust_tau_diff": np.full(len(sfrs), 0.2)}
    )


def test_simulate_photometry_equals_predict(fwd_table_sfh):
    """simulate() must not be a second forward path — its photometry IS predict()."""
    cat = _histories_catalog(fwd_table_sfh)
    mock = cat.simulate()

    assert np.allclose(mock.photometry / np.asarray(cat.predict()), 1.0, rtol=1e-12)
    assert mock.photometry.shape == (3, fwd_table_sfh.observation.photometry.n_filters)


def test_simulate_properties_scale_with_the_history(fwd_table_sfh):
    """stellar_mass is exactly linear in a scaled SFH — the physics check.

    Emission lines cannot carry this check on the synthetic SSP (it is a smooth
    power law with no lines baked in, so a continuum-subtracted line flux is
    ~0), but formed stellar mass can, and it exercises the same column channel.
    """
    cat = _histories_catalog(fwd_table_sfh)
    mock = cat.simulate(properties=("stellar_mass",))

    mass = np.asarray(mock.properties["stellar_mass"])
    assert mass.shape == (3,)
    assert np.all(mass > 0.0), f"a positive SFR must form mass, got {mass}"
    assert np.allclose(mass[1] / mass[0], 5.0, rtol=1e-6), f"ratio {mass[1] / mass[0]}"
    assert np.allclose(mass[2] / mass[0], 20.0, rtol=1e-6), f"ratio {mass[2] / mass[0]}"


def test_simulate_lines_match_the_single_galaxy_measurement(fwd_table_sfh):
    """Per-row parity for the line channel: the batching must not perturb it.

    The synthetic SSP has no emission lines, so the *values* here are continuum
    residuals rather than physics — which is exactly why this asserts agreement
    with the single-galaxy call rather than a physical scaling.
    """
    import jax.numpy as jnp

    cat = _histories_catalog(fwd_table_sfh)
    mock = cat.simulate(lines=("Halpha", "OIII_5007"))

    assert set(mock.lines) == {"Halpha", "OIII_5007"}
    for name in ("Halpha", "OIII_5007"):
        assert mock.lines[name].shape == (3,)
        assert np.all(np.isfinite(mock.lines[name]))

    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    from tengri.observation.line_measurement import DESI_LINES

    defs = tuple(d for d in DESI_LINES if d.name in ("Halpha", "OIII_5007"))
    # The single-galaxy call needs the COMPLETE params dict: the window-LUT line
    # path reaches compute_joint_weights, which reads params["met_logzsol"]
    # directly and does not merge Fixed values for itself. Catalog.simulate
    # merges them once for the whole table; here we mirror that with the fixed
    # values from the spec, so the two sides are genuinely comparable.
    fixed = {
        k: jnp.asarray(v)
        for k, v in fwd_table_sfh.spec.get_fixed_values().items()
        if np.asarray(v).ndim == 0
    }
    for i in range(3):
        one = np.asarray(
            fwd_table_sfh.measure_line_fluxes(
                {
                    **fixed,
                    "dust_tau_diff": jnp.asarray(0.2),
                    "sfh_t_gyr": jnp.asarray(t[i]),
                    "sfh_sfr": jnp.asarray(sfr[i]),
                },
                defs,
                approx=True,
            )
        )
        got = np.array([mock.lines[d.name][i] for d in defs])
        assert np.allclose(got, one, rtol=1e-9, atol=0.0) or np.allclose(got, one, atol=1e-30), (
            f"row {i}: batched {got} vs single-galaxy {one}"
        )


def test_simulate_to_table_is_flat_and_named(fwd_table_sfh):
    """to_table() must give flat (N,) columns, ingest-compatible like #1313's."""
    cat = _histories_catalog(fwd_table_sfh)
    table = cat.simulate(lines=("Halpha",), properties=("stellar_mass",)).to_table()

    for name in fwd_table_sfh.observation.photometry.names:
        assert name in table, f"missing flux column {name!r}"
        assert np.asarray(table[name]).shape == (3,)
    assert np.asarray(table["Halpha"]).shape == (3,)
    assert np.asarray(table["stellar_mass"]).shape == (3,)
    assert all(np.asarray(v).ndim == 1 for v in table.values()), "columns must be flat"


def test_simulate_rejects_an_unknown_line_name(fwd_table_sfh):
    """A typo'd line name must name the available ones, not silently drop it."""
    cat = _histories_catalog(fwd_table_sfh)
    with pytest.raises(ValueError, match="Halpha"):
        cat.simulate(lines=("Halpah",))


def test_simulate_noise_points_at_the_open_issue(fwd_table_sfh):
    """noise= is #1312 and not implemented — refuse rather than ignore it."""
    cat = _histories_catalog(fwd_table_sfh)
    with pytest.raises(NotImplementedError, match="1312"):
        cat.simulate(noise=object())


# ── #1396 acceptance criterion 6: the parametric round-trip ───────────────
#
# "Round-trip validation: a tabulated history that reproduces a parametric SFH
# gives photometry matching the parametric model's, to tolerance — the test
# that proves the histories are actually being used."
#
# This is the one criterion of #1396 that was never written, and its absence is
# how #1522 shipped green: every other test in this file asserts a **ratio**
# between galaxies, and the truncation there multiplies every galaxy by the
# same factor, which a ratio divides straight out. This test compares against
# an external reference instead — the same analytic SFH, evaluated by the other
# arm of the code — so a common factor has nowhere to hide.

_RT_LOG_MASS = 10.0  # log10(M_formed / Msun)
_RT_TAU_GYR = 2.0  # delayed-tau timescale [Gyr]
_RT_AGE_GYR = 10.0  # lookback time of formation [Gyr]

# Measured on e107f0600: the matched arms differ by 1.9e-4 in color and 1.0e-4
# in normalization. That residual is a fixed difference between the two arms'
# quadratures, NOT table sampling — it is invariant over n_t = 128…4096
# (1.73e-4 → 1.86e-4). The tolerance sits 5x above it; see the negative control
# below for what it can still resolve.
_RT_RTOL = 1e-3


@pytest.fixture(scope="module")
def age_reddening_ssp():
    """A synthetic SSP whose spectral SLOPE varies with age.

    ``synthetic_ssp_wide`` cannot carry this test. Its flux is *separable* —

        base(wave) * (1 + 0.15*(age - mean)) * (1 + 0.10*(lgmet - mean))

    — so the wavelength shape is identical at every age and age only rescales
    the amplitude. On that fixture the SED **shape** cannot respond to the star
    formation history at all: measured, doubling tau moves the colors by
    3.6e-14 (machine noise), so any "the colors match" assertion passes no
    matter what the code does. Here each age gets its own power-law index, so
    young ages are blue and old ages red, and the colors become a real
    observable that the SFH drives.
    """
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 7.0, 1600)  # 100 Å – 1 mm, as synthetic_ssp_wide
    lg_age = jnp.linspace(-3.0, 1.14, 25)  # log10(age/Gyr): ~1 Myr – 13.8 Gyr
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    slope = -2.0 + 0.6 * (lg_age - lg_age.mean())  # redder with age
    shape = (wave[None, :] / 5000.0) ** slope[:, None]
    flux = shape[None, :, :] * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=lg_age,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def roundtrip_arms(age_reddening_ssp, synthetic_tophat_obs):
    """The two arms of the round-trip: one parametric model, one tabulated.

    Same SSP, same filters, same redshift, same dust (off) — the SFH
    *representation* is the only difference between them.
    """
    from tengri import FIXED, ForwardModel, SEDModel
    from tengri.cosmology import age_at_z
    from tengri.parameters.priors import Fixed

    def _build(sfh):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed = SEDModel.build(
                ssp_data=age_reddening_ssp,
                observation=synthetic_tophat_obs,
                sfh=sfh,
                dust={
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": FIXED,
                    "tau_bc": 0.0,
                    "tau_diff": 0.0,
                },
                neb={"type": "none"},
                redshift=Fixed(_Z_OBS),
            )
            return ForwardModel.build(sed=sed, observation=synthetic_tophat_obs), sed

    par_fwd, par_sed = _build(
        {
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": _RT_LOG_MASS,
            "tau_gyr": _RT_TAU_GYR,
            "age_gyr": _RT_AGE_GYR,
        }
    )
    tab_fwd, _ = _build({"type": "table"})
    return par_fwd, par_sed, tab_fwd, float(age_at_z(_Z_OBS))


def _delayed_history(t_univ_gyr, tau_gyr, n_t=512):
    """Sample the delayed-tau SFH as a cosmic-time table [Gyr], [Msun/yr].

    ``sfhdelayed`` renormalizes to ``10**log_total_mass`` by a trapezoid over
    the lookback grid it is handed, so that grid must **ascend** — hand it a
    descending one and the normalization comes back negative (measured: a mass
    of -3.8e58 Msun). Build in ascending lookback, then flip to ascending
    cosmic time, which is the orientation ``from_histories`` wants.
    """
    from tengri.components.stellar.sfh.mean_sfh import sfhdelayed

    t_lb_gyr = np.linspace(0.0, t_univ_gyr, n_t)
    sfr_lb = np.asarray(sfhdelayed(t_lb_gyr * 1e9, _RT_LOG_MASS, tau_gyr * 1e9, _RT_AGE_GYR * 1e9))
    return (t_univ_gyr - t_lb_gyr)[::-1], np.maximum(sfr_lb[::-1], 0.0)


def _parametric_prediction(par_fwd, par_sed):
    import jax.numpy as jnp

    params = {k: jnp.asarray(v) for k, v in par_fwd.spec.get_fixed_values().items()}
    flux = np.asarray(par_fwd.predict_photometry(params))
    mass = float(
        np.asarray(par_sed.predict_properties(params, names=("stellar_mass",))["stellar_mass"])
    )
    return flux, mass


def _tabulated_prediction(tab_fwd, t_gyr, sfr):
    from tengri import Catalog

    mock = Catalog.from_histories(tab_fwd, t_gyr=t_gyr[None, :], sfr=sfr[None, :]).simulate(
        properties=("stellar_mass",)
    )
    return np.asarray(mock.photometry)[0], float(np.asarray(mock.properties["stellar_mass"])[0])


def test_the_roundtrip_ssp_actually_has_color_evolution(roundtrip_arms):
    """Guard the guard: the fixture must be non-separable, or criterion 6 is void.

    If someone swaps this SSP back to a separable one, the round-trip below
    keeps passing while testing nothing. Pin the property the test depends on:
    the parametric model's five bands must span a real color range.
    """
    par_fwd, par_sed, _tab, _t = roundtrip_arms
    flux, _mass = _parametric_prediction(par_fwd, par_sed)

    colors = flux / flux[0]
    assert colors.min() < 0.5, f"SSP has no color evolution; bands span {colors}"


def test_roundtrip_tabulated_history_matches_the_parametric_model(roundtrip_arms):
    """#1396 criterion 6: the same SFH, tabulated, must give the same photometry.

    Both arms are handed the identical analytic delayed-tau SFH. The parametric
    arm evaluates it internally on the SSP age grid; the tabulated arm receives
    it as (t, SFR) columns through ``Catalog.from_histories``. Photometry,
    colors and formed mass must all agree.
    """
    par_fwd, par_sed, tab_fwd, t_univ = roundtrip_arms

    f_par, m_par = _parametric_prediction(par_fwd, par_sed)
    t_gyr, sfr = _delayed_history(t_univ, _RT_TAU_GYR)
    f_tab, m_tab = _tabulated_prediction(tab_fwd, t_gyr, sfr)

    # Ratio, never bare allclose — these fluxes are ~1e-13, far below the
    # default atol=1e-8, which would call any two of them equal.
    assert np.allclose(f_tab / f_par, 1.0, rtol=_RT_RTOL), f"flux ratio {f_tab / f_par}"
    assert np.allclose(m_tab / m_par, 1.0, rtol=_RT_RTOL), f"mass ratio {m_tab / m_par}"

    colors = (f_tab / f_tab[0]) / (f_par / f_par[0])
    assert np.allclose(colors, 1.0, rtol=_RT_RTOL), f"color ratio {colors}"


def test_roundtrip_rejects_a_history_that_does_not_match(roundtrip_arms):
    """Negative control: the round-trip must FAIL on the wrong history.

    Without this the tolerance above is unfalsifiable — and on the shipped
    ``synthetic_ssp_wide`` it genuinely would have been. Feeding a tau twice
    the parametric model's moves the colors by 9.2e-2, some 92x the tolerance;
    a 1 % tau error already lands at 1.3e-3, just outside it. That is the
    test's measured resolving power.
    """
    par_fwd, par_sed, tab_fwd, t_univ = roundtrip_arms

    f_par, _m_par = _parametric_prediction(par_fwd, par_sed)
    t_gyr, sfr = _delayed_history(t_univ, 2.0 * _RT_TAU_GYR)
    f_tab, _m_tab = _tabulated_prediction(tab_fwd, t_gyr, sfr)

    colors = (f_tab / f_tab[0]) / (f_par / f_par[0])
    assert not np.allclose(colors, 1.0, rtol=_RT_RTOL), (
        f"a 2x tau error must not pass the round-trip; color ratio {colors}"
    )
