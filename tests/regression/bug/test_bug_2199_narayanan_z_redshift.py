# SPDX-License-Identifier: BSD-3-Clause
"""``narayanan_z`` never saw the model redshift, and scaled the wrong way (#2199).

https://github.com/suchethac/tengri/issues/2199

``narayanan_z`` declares ``redshift`` and is the one attenuation law whose shape
depends on it. Neither dust path forwarded it: ``_law_shape_params`` admits only
the four tabled shape parameters and names already spelled ``dust_*``, and
``resolve_bc_diff_law_params`` had no passthrough at all, so the bare
``redshift`` that both dust ``apply()`` methods already receive was dropped
before the law was called. Measured on the tree before this fix, on the fixture
below: the model-evaluated ``A(1500)/A(5500)`` was 3.3215 at z = 0 and 3.3215 at
z = 6, and ``jax.grad`` of that ratio with respect to a free ``redshift`` was
exactly 0.0 on both screens.

The scaling was wrong as well as unreachable. The shipped coefficients were
``delta(z) = -0.2 - 0.1 z`` and ``E_b(z) = max(0, 1 - 0.15 z)``; no such closed
form appears in the paper, and the slope term steepened the curve with redshift
while Narayanan et al. (2018) Section 5.1 reports the median curves becoming
*grayer* toward z ~ 6. The law now interpolates a table fitted to the paper's
own published median curves by ``scripts/fit_narayanan2018_medians.py``, and
this file pins both halves.

References
----------
.. [1] D. Narayanan, C. Conroy, R. Davé, B. D. Johnson and G. Popping,
   "A Theory for the Variation of Dust Attenuation Laws in Galaxies,"
   ApJ, 869, 70 (2018). arXiv:1805.06905.
   https://doi.org/10.3847/1538-4357/aaed25
.. [2] M. Kriek and C. Conroy, "The Dust Attenuation Law in Distant Galaxies:
   Evidence for Variation with Spectral Type," ApJL, 775, L16 (2013).
   https://doi.org/10.1088/2041-8205/775/1/L16
"""

from __future__ import annotations

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SSPData,
    Uniform,
    WavePrecomp,
)
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Optical depth used to read k(lambda) back out of a pair of predicted SEDs.
_TAU = 0.6


def _filter_fixture_warnings() -> None:
    """Silence the two warnings this fixture provably raises, and only those.

    Measured by building every (screen, law) combination in this file under
    ``warnings.simplefilter("always")`` and collecting the categories: a
    ``BakedInNebularWarning`` (the synthetic SSP declares baked-in nebular
    emission) and a ``SFHBeforeBigBangWarning`` (the default DPL forms mass
    before the Big Bang at the fixture's redshifts). Neither is about dust.
    A blanket ``simplefilter("ignore")`` would also hide a warning this change
    introduced, which is the one thing a dust test must not hide.
    """
    from tengri.components.nebular.baked_in import BakedInNebularWarning
    from tengri.components.stellar.component import SFHBeforeBigBangWarning

    warnings.simplefilter("error")
    warnings.simplefilter("ignore", BakedInNebularWarning)
    warnings.simplefilter("ignore", SFHBeforeBigBangWarning)


#: Bit-identity references for the laws that do NOT declare ``redshift``.
#:
#: Captured on 0ec4d492c, the merge base of this branch, before any source edit:
#: this file was written with empty lists here and run, and the assertion
#: message below printed the measured vectors. The seam change adds ``redshift``
#: to a law's live shape parameters only when that law's own signature declares
#: it, so every other law must be untouched to the last bit; these literals are
#: what says so.
_UNTOUCHED_LAW_PHOTOMETRY: dict[tuple[str, str], tuple[float, ...]] = {
    ("single_component", "calzetti"): (
        8.878652556210589e-14,
        8.371162333895889e-14,
        6.182259702195155e-14,
        3.4508040030179384e-14,
        2.228794472287952e-14,
    ),
    ("single_component", "kriek_conroy"): (
        8.892444564771738e-14,
        8.348554091854069e-14,
        6.013305551395232e-14,
        3.423894018017045e-14,
        2.2269787817749572e-14,
    ),
    ("two_component", "calzetti"): (
        8.878652556210589e-14,
        8.371162333895889e-14,
        6.182259702195155e-14,
        3.4508040030179384e-14,
        2.228794472287952e-14,
    ),
    ("two_component", "kriek_conroy"): (
        8.892444564771738e-14,
        8.348554091854068e-14,
        6.013305551395232e-14,
        3.423894018017045e-14,
        2.2269787817749572e-14,
    ),
}


@pytest.fixture(scope="module")
def uv_ssp() -> SSPData:
    """A smooth power-law SSP cube, so the dust curve is the only structure."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1200)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-30, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def uv_obs() -> Observation:
    """Bands bracketing the 2175 A bump and the UV slope."""

    def _tophat(center: float, frac: float = 0.10, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 2175.0, 2800.0, 4400.0, 6200.0)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _dust_group(screen: str, law: str, **shape) -> dict:
    """The ``dust_attenuation`` group for one screen, with its optical depths freed."""
    group: dict = {"type": screen, "law": law, **shape}
    if screen == "single_component":
        group["dust_tau_v"] = Uniform(0.0, 2.0)
    else:
        group["dust_tau_bc"] = Uniform(0.0, 2.0)
        group["dust_tau_diff"] = Uniform(0.0, 2.0)
    return group


def _build(uv_ssp, uv_obs, screen: str, law: str, redshift, approx=None, **shape):
    """One model, one screen, one law, at a pinned or freed redshift."""
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        return SEDModel.build(
            ssp_data=uv_ssp,
            observation=uv_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation=_dust_group(screen, law, **shape),
            redshift=redshift,
            approx=approx,
        )


def _params(model, screen: str, tau: float = _TAU) -> dict:
    """Sampled parameters with the optical depths pinned, so only k(lambda) varies."""
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    if screen == "single_component":
        params["dust_tau_v"] = jnp.asarray(tau)
    else:
        # Birth cloud off: the age mixing collapses and the diffuse screen is the
        # whole transmission, exp(-tau_diff k(lambda)).
        params["dust_tau_bc"] = jnp.asarray(0.0)
        params["dust_tau_diff"] = jnp.asarray(tau)
    return params


def _photometry(
    uv_ssp, uv_obs, screen: str, law: str, z: float, approx=None, **shape
) -> np.ndarray:
    model = _build(uv_ssp, uv_obs, screen, law, Fixed(z), approx=approx, **shape)
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        return np.asarray(model.predict_photometry(_params(model, screen)))


def _curve(model, screen: str, wavelengths, extra: dict | None = None):
    """k(lambda) as the model evaluates it, read back from two predicted SEDs.

    The screen multiplies the intrinsic SED by ``exp(-tau k(lambda))``, so two
    predictions differing only in optical depth give k on the model's own path,
    without reaching into any private attribute. Rest-frame wavelengths, so the
    cosmological factors cancel exactly and nothing but the curve is left.
    """
    bare = {**_params(model, screen, tau=0.0), **(extra or {})}
    dusty = {**_params(model, screen, tau=_TAU), **(extra or {})}
    sed_bare = model.predict(bare).rest_sed(jnp.asarray(wavelengths))
    sed_dusty = model.predict(dusty).rest_sed(jnp.asarray(wavelengths))
    return -jnp.log(sed_dusty / sed_bare) / _TAU


def _fit_table() -> dict:
    """The fit script's own output, shipped beside the repackaged curves."""
    from tengri import data_path

    try:
        path = data_path("attenuation/narayanan2018_kc13_fits.json")
    except FileNotFoundError:  # pragma: no cover - source checkouts always have it
        pytest.skip("narayanan2018_kc13_fits.json not installed")
    return json.loads(path.read_text())


def _median_curves() -> tuple[np.ndarray, np.ndarray]:
    """The repackaged medians on an ascending wavelength axis."""
    from tengri import data_path

    try:
        path = data_path("attenuation/narayanan2018_median_curves.dat")
    except FileNotFoundError:  # pragma: no cover - source checkouts always have it
        pytest.skip("narayanan2018_median_curves.dat not installed")
    table = np.loadtxt(path)
    wavelength = 1e4 / table[:, 0]
    order = np.argsort(wavelength)
    return wavelength[order], table[:, 1:].T[:, order]


# ── (i) z = 0 is the fitted z = 0 row, exactly ───────────────────────


@pytest.mark.parametrize("screen", ["single_component", "two_component"])
def test_at_z0_the_law_is_kriek_conroy_at_the_fitted_z0_row(screen, uv_ssp, uv_obs):
    """At z = 0 the law must be the KC13 curve carrying the fitted z = 0 values.

    Not "close to": the law *is* ``kriek_conroy`` evaluated at the table's first
    row, so a model built with those two numbers pinned by hand has to agree to
    round-off. Anything looser would let a wrong table row through.
    """
    row = _fit_table()["fits"][0]
    assert row["z"] == 0
    theirs = _photometry(
        uv_ssp,
        uv_obs,
        screen,
        "kriek_conroy",
        0.0,
        dust_delta=Fixed(row["dust_delta"]),
        dust_bump_strength=Fixed(row["dust_bump_strength"]),
    )
    ours = _photometry(uv_ssp, uv_obs, screen, "narayanan_z", 0.0)
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)


def test_the_module_table_is_the_fit_script_output_element_for_element():
    """Every row of the hand-copied table must equal the JSON the script wrote.

    The z = 0 identity above pins one row through the model. This pins all
    seven directly, so a digit mistyped in row 4 fails here by name rather than
    surviving until somebody fits a z = 4 galaxy.
    """
    from tengri.components.dust.attenuation import (
        _NARAYANAN_BUMP_STRENGTH,
        _NARAYANAN_DELTA,
        _NARAYANAN_Z_NODES,
    )

    fits = _fit_table()["fits"]
    assert [row["z"] for row in fits] == [float(z) for z in _NARAYANAN_Z_NODES]
    np.testing.assert_array_equal(
        np.asarray(_NARAYANAN_DELTA), np.asarray([row["dust_delta"] for row in fits])
    )
    np.testing.assert_array_equal(
        np.asarray(_NARAYANAN_BUMP_STRENGTH),
        np.asarray([row["dust_bump_strength"] for row in fits]),
    )


# ── (ii) z > 0 reaches the law, and the curve gets grayer ────────────


@pytest.mark.parametrize("screen", ["single_component", "two_component"])
def test_the_model_redshift_reaches_the_law(screen, uv_ssp, uv_obs):
    """The model-evaluated curve at z = 2 must differ from z = 0, and from KC13.

    Cosmological dimming cannot be what moves: k(lambda) here is a ratio of two
    predictions at the same redshift, on the same rest-frame wavelengths. Before
    the fix the two curves agreed to the last bit at every wavelength, because
    the law was evaluated at z = 0 whatever the model said.

    The second comparison is against ``kriek_conroy`` at *its own* published
    defaults (delta = 0, bump = 1), the curve ``narayanan_z`` would collapse
    onto if the table were ignored. It held before the fix too and is pinned
    here so that a future change cannot satisfy the first assertion by making
    the two laws the same curve.
    """
    probe = jnp.asarray([1200.0, 1500.0, 2175.0, 3000.0, 5500.0, 8000.0])
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        model_z0 = _build(uv_ssp, uv_obs, screen, "narayanan_z", Fixed(0.0))
        model_z2 = _build(uv_ssp, uv_obs, screen, "narayanan_z", Fixed(2.0))
        model_kc = _build(uv_ssp, uv_obs, screen, "kriek_conroy", Fixed(2.0))
        at_z0 = np.asarray(_curve(model_z0, screen, probe))
        at_z2 = np.asarray(_curve(model_z2, screen, probe))
        kc_defaults = np.asarray(_curve(model_kc, screen, probe))
    assert np.max(np.abs(at_z2 - at_z0)) > 1e-3, (
        f"{screen}: the attenuation curve is unchanged between z=0 and z=2 "
        f"(max |dk| = {np.max(np.abs(at_z2 - at_z0)):.3e}). The model redshift is not "
        "reaching the law (#2199)."
    )
    assert np.max(np.abs(at_z2 - kc_defaults)) > 1e-3, (
        f"{screen}: narayanan_z at z=2 collapsed onto kriek_conroy at its own "
        f"defaults (max |dk| = {np.max(np.abs(at_z2 - kc_defaults)):.3e})."
    )


@pytest.mark.parametrize("screen", ["single_component", "two_component"])
def test_the_curve_gets_grayer_with_redshift(screen, uv_ssp, uv_obs):
    """A(1500)/A(5500) must fall from z = 0 to z = 6.

    Narayanan et al. (2018) Section 5.1: the median attenuation curves become
    grayer with redshift, and significantly so toward z ~ 6. Read straight off
    the repackaged medians, the ratio is 5.46 at z = 0 and 2.35 at z = 6. The
    shipped ``delta(z) = -0.2 - 0.1 z`` did the opposite, steepening the curve.
    """
    probe = jnp.asarray([1500.0, 5500.0])
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        model_z0 = _build(uv_ssp, uv_obs, screen, "narayanan_z", Fixed(0.0))
        model_z6 = _build(uv_ssp, uv_obs, screen, "narayanan_z", Fixed(6.0))
        k_z0 = np.asarray(_curve(model_z0, screen, probe))
        k_z6 = np.asarray(_curve(model_z6, screen, probe))
    ratio_z0 = float(k_z0[0] / k_z0[1])
    ratio_z6 = float(k_z6[0] / k_z6[1])
    assert ratio_z6 < ratio_z0, (
        f"{screen}: A(1500)/A(5500) is {ratio_z6:.4f} at z=6 against {ratio_z0:.4f} at "
        "z=0. The curve steepens with redshift; the paper's medians get grayer."
    )


@pytest.mark.parametrize("screen", ["single_component", "two_component"])
def test_the_precomputed_photometry_path_carries_the_same_row(screen, uv_ssp, uv_obs):
    """The ``WavePrecomp`` LUT must carry the z = 2 row, not only the exact path.

    Every fit surface resolves ``approx="auto"`` to ``WavePrecomp`` for
    photometry, and that path reaches the screen through
    ``compute_transmission`` rather than ``apply``. The two resolve their law
    parameters through the same function, so fixing one without the other would
    leave every actual fit on the z = 0 curve while the exploratory path looked
    correct. Pinned against the z = 0 row as well, so a LUT frozen at build time
    fails rather than passing on a coincidence.
    """
    from tengri import WavePrecomp

    fits = _fit_table()["fits"]
    ours = _photometry(uv_ssp, uv_obs, screen, "narayanan_z", 2.0, approx=WavePrecomp())
    for row in (fits[2], fits[0]):
        theirs = _photometry(
            uv_ssp,
            uv_obs,
            screen,
            "kriek_conroy",
            2.0,
            approx=WavePrecomp(),
            dust_delta=Fixed(row["dust_delta"]),
            dust_bump_strength=Fixed(row["dust_bump_strength"]),
        )
        offset = float(np.max(np.abs(ours / theirs - 1.0)))
        if row["z"] == 2:
            assert offset < 1e-10, (
                f"{screen}: the precomputed photometry differs from the fitted z=2 row "
                f"by {offset:.3e}; the LUT is not on the model's redshift."
            )
        else:
            assert offset > 1e-3, (
                f"{screen}: the precomputed photometry at z=2 matches the z=0 row to "
                f"{offset:.3e}. The LUT was frozen before the redshift was known."
            )


def test_the_spectral_index_window_lut_carries_the_same_row(uv_ssp, uv_obs):
    """The FeaturePrecomp index LUT reaches the screen by its own route.

    ``predict_spectral_indices(approx=True)`` contracts precomputed SSP window
    integrals with ``DustSEDComponent.compute_transmission``, the second of the
    two places that resolve law parameters for the two-component screen. The
    windows are rest-frame, so redshift cannot move this index except through
    the attenuation curve.
    """
    from tengri import SpectralIndexDef

    index = SpectralIndexDef(
        name="uv_break",
        index_type="break",
        continuum=((1500.0, 1600.0), (3000.0, 3100.0)),
    )
    fits = _fit_table()["fits"]

    def measure(law: str, z: float, **shape) -> float:
        model = _build(uv_ssp, uv_obs, "two_component", law, Fixed(z), **shape)
        with warnings.catch_warnings():
            _filter_fixture_warnings()
            values = model.predict_spectral_indices(
                _params(model, "two_component"), (index,), approx=True
            )
        return float(np.asarray(values)[0])

    ours = measure("narayanan_z", 2.0)
    for row in (fits[2], fits[0]):
        theirs = measure(
            "kriek_conroy",
            2.0,
            dust_delta=Fixed(row["dust_delta"]),
            dust_bump_strength=Fixed(row["dust_bump_strength"]),
        )
        offset = abs(ours / theirs - 1.0)
        if row["z"] == 2:
            assert offset < 1e-10, (
                f"the index LUT differs from the fitted z=2 row by {offset:.3e}; "
                "compute_transmission is not on the model's redshift."
            )
        else:
            assert offset > 1e-3, (
                f"the index LUT at z=2 matches the z=0 row to {offset:.3e}; the screen "
                "behind it never saw the redshift."
            )


def test_the_flat_parameters_escape_hatch_also_reaches_the_law(uv_ssp, uv_obs):
    """A spec built the expert way carries no group provenance, and must still work.

    ``SEDModel.build`` records who asked for each ``dust_*`` value, and that
    record is what decides whether a shape parameter is offered to the curve.
    The flat ``Parameters(...)`` form records nothing, so every name there reads
    as ``registry_default``. ``redshift`` is exempt from that decision -- it is
    not a ``dust_attenuation`` key and has no per-law default to protect -- and
    this is the path where the exemption is the only thing carrying it: measured
    without it, the flat spec's ``live_shape_params`` is empty and the screen
    serves a curve frozen at z = 0.
    """
    from tengri import Parameters

    probe = jnp.asarray([1500.0, 5500.0])

    def ratio(z: float) -> float:
        with warnings.catch_warnings():
            _filter_fixture_warnings()
            spec = Parameters(
                mean_sfh_type="dpl",
                dust_model="single_component",
                dust_law_bc="narayanan_z",
                dust_tau_v=Uniform(0.0, 2.0),
                redshift=Fixed(z),
            )
            model = SEDModel(spec, uv_ssp, observation=uv_obs)
            params = dict(spec.sample(jax.random.PRNGKey(0)))
            bare = model.predict({**params, "dust_tau_v": jnp.asarray(0.0)})
            dusty = model.predict({**params, "dust_tau_v": jnp.asarray(_TAU)})
            k = -jnp.log(dusty.rest_sed(probe) / bare.rest_sed(probe)) / _TAU
        return float(k[0] / k[1])

    at_z0, at_z6 = ratio(0.0), ratio(6.0)
    assert at_z6 < at_z0, (
        f"flat Parameters: A(1500)/A(5500) is {at_z6:.4f} at z=6 against {at_z0:.4f} at "
        "z=0. The expert path never handed the law its redshift."
    )


# ── (iii) the table is the paper's medians, not a remembered number ──


def test_the_law_reproduces_the_repackaged_median_curves():
    """Each z = 0..6 curve must match the published median to the fitted rms.

    The transcription guard. The module-level table in ``attenuation.py`` is a
    hand-copy of ``narayanan2018_kc13_fits.json``; a mistyped digit shows up
    here as a residual larger than the fit's own, and nowhere else. The 1.5x
    headroom is round-off room on an exact copy, not fit slack.
    """
    from tengri.components.dust.attenuation import narayanan_z

    fits = _fit_table()
    lo, hi = fits["fit_range_angstrom"]
    wavelength, medians = _median_curves()
    inside = (wavelength >= lo) & (wavelength <= hi)
    wave = jnp.asarray(wavelength[inside])

    for row in fits["fits"]:
        curve = row["norm"] * np.asarray(narayanan_z(wave, redshift=float(row["z"])))
        residual = curve - medians[row["z"]][inside]
        rms = float(np.sqrt(np.mean(residual**2)))
        assert rms <= 1.5 * row["rms"], (
            f"z={row['z']}: the law reproduces the published median curve to rms "
            f"{rms:.6f}, against the fit's own {row['rms']:.6f}. The interpolation "
            "table no longer matches narayanan2018_kc13_fits.json."
        )


def test_the_table_interpolates_between_its_nodes_and_clips_outside():
    """Between nodes the law must interpolate; outside 0 <= z <= 6 it must hold.

    Extrapolating a fit past the data it was fitted to is how a z = 12 galaxy
    gets a curve nobody measured; the table is clipped instead.
    """
    from tengri.components.dust.attenuation import narayanan_z

    wave = jnp.asarray([1500.0, 2175.0, 5500.0])
    half = np.asarray(narayanan_z(wave, redshift=0.5))
    at_0 = np.asarray(narayanan_z(wave, redshift=0.0))
    at_1 = np.asarray(narayanan_z(wave, redshift=1.0))
    assert np.all(half >= np.minimum(at_0, at_1) - 1e-12)
    assert np.all(half <= np.maximum(at_0, at_1) + 1e-12)

    np.testing.assert_allclose(
        np.asarray(narayanan_z(wave, redshift=9.0)),
        np.asarray(narayanan_z(wave, redshift=6.0)),
    )
    np.testing.assert_allclose(np.asarray(narayanan_z(wave, redshift=-1.0)), at_0)


# ── the energy-balance LUT is the third caller of the same resolver ──


@pytest.fixture(scope="module")
def ir_obs() -> Observation:
    """Optical bands that feed the energy balance, plus the 100 um band it heats."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build_ir(uv_ssp, ir_obs, law: str, redshift, approx, **shape):
    """Two-component screen with dale2014 IR re-emission, the LUT's own case."""
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        return SEDModel.build(
            ssp_data=uv_ssp,
            observation=ir_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": law,
                "dust_tau_bc": Uniform(0.0, 1.0),
                "dust_tau_diff": Uniform(0.0, 1.5),
                **shape,
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=redshift,
            approx=approx,
        )


def _ir_photometry(model) -> np.ndarray:
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        params = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(0))}
        params["dust_tau_bc"] = jnp.asarray(0.0)
        params["dust_tau_diff"] = jnp.asarray(_TAU)
        return np.asarray(model.predict_photometry(params))


@pytest.mark.parametrize("z", [0.0, 2.0, 6.0])
def test_the_energy_balance_lut_is_built_at_the_model_redshift(z, uv_ssp, ir_obs):
    """``L_ir`` under ``WavePrecomp`` must come from the curve at the model's z.

    The LUT bakes one attenuation curve at build time from the *fixed* values,
    and it is the third caller of ``resolve_bc_diff_law_params``. Left without
    ``redshift``, it integrated the absorbed luminosity under the z = 0 curve
    while ``apply()`` attenuated the starlight with the z-scaled one, so the
    model violated its own energy balance. Measured on this fixture before the
    fix, 100 um band against the exact path: 2.25e-4 at z = 0, **1.09e-2** at
    z = 2 and **2.11e-1** at z = 6.

    The bar is ``kriek_conroy`` at the same redshift: that law reads no
    redshift, so its LUT was always right, and whatever residual it shows is
    the LUT's own interpolation error rather than a wiring defect. Measured
    after the fix, 100 um: narayanan_z 2.25e-4 / 2.87e-4 / 5.76e-4 at
    z = 0 / 2 / 6, against kriek_conroy 5.60e-4 at all three.
    """
    ours = _ir_photometry(_build_ir(uv_ssp, ir_obs, "narayanan_z", Fixed(z), WavePrecomp()))
    exact = _ir_photometry(_build_ir(uv_ssp, ir_obs, "narayanan_z", Fixed(z), None))
    reference = _ir_photometry(_build_ir(uv_ssp, ir_obs, "kriek_conroy", Fixed(z), WavePrecomp()))
    reference_exact = _ir_photometry(_build_ir(uv_ssp, ir_obs, "kriek_conroy", Fixed(z), None))

    # The 100 um band is the one carrying L_ir; the optical bands also carry
    # WavePrecomp's own blue-band approximation, which is a different budget.
    ours_ir = abs(float(ours[-1] / exact[-1] - 1.0))
    reference_ir = abs(float(reference[-1] / reference_exact[-1] - 1.0))
    budget = max(reference_ir, 1e-6) * 1.5
    assert ours_ir <= budget, (
        f"z={z}: narayanan_z's IR band drifts {ours_ir:.3e} from the exact path, "
        f"against {reference_ir:.3e} for kriek_conroy at the same redshift. The "
        "energy-balance LUT was not built at the model redshift (#2199)."
    )


def test_a_free_redshift_disables_the_lut_only_for_a_law_that_reads_it(uv_ssp, ir_obs):
    """A free z is a free curve-shape parameter exactly when the law reads z.

    A build-time LUT cannot hold a curve that moves with a sampled parameter,
    which is why a free ``dust_delta`` disables it. ``redshift`` is not spelled
    ``dust_*``, so the existing filter could not see it. ``kriek_conroy`` reads
    no redshift, so its LUT must survive a free z: a blanket "free redshift
    disables the LUT" would cost every photometric-redshift fit the
    optimization for nothing.
    """
    free_z = Uniform(0.1, 5.0)
    ours = _build_ir(uv_ssp, ir_obs, "narayanan_z", free_z, WavePrecomp())
    reads_no_z = _build_ir(uv_ssp, ir_obs, "kriek_conroy", free_z, WavePrecomp())
    free_shape = _build_ir(
        uv_ssp, ir_obs, "kriek_conroy", Fixed(2.0), WavePrecomp(), dust_delta=Uniform(-1.0, 0.4)
    )
    fixed_z = _build_ir(uv_ssp, ir_obs, "narayanan_z", Fixed(2.0), WavePrecomp())

    def lut(model):
        return getattr(model, "_energy_balance_lut_cache", None) is not None

    assert not lut(free_shape), "a free dust_delta must already disable the LUT"
    assert not lut(ours), (
        "a free redshift left the LUT engaged on narayanan_z, so every sample "
        "would share one baked curve (#2199)."
    )
    assert lut(reads_no_z), "a free redshift disabled the LUT for a law that ignores z"
    assert lut(fixed_z), "a fixed redshift must keep the LUT on narayanan_z"

    # The public effect, not only the cache: with the LUT off, the free-redshift
    # model must still track the exact path.
    exact = _build_ir(uv_ssp, ir_obs, "narayanan_z", free_z, None)
    with warnings.catch_warnings():
        _filter_fixture_warnings()
        params = {**ours.spec.get_fixed_values(), **ours.spec.sample(jax.random.PRNGKey(0))}
        params["dust_tau_bc"] = jnp.asarray(0.0)
        params["dust_tau_diff"] = jnp.asarray(_TAU)
        params["redshift"] = jnp.asarray(2.0)
        a = np.asarray(ours.predict_photometry(params))
        b = np.asarray(exact.predict_photometry(params))
    assert abs(float(a[-1] / b[-1] - 1.0)) < 1e-2


# ── (iv) a free redshift stays differentiable through the law ────────


@pytest.mark.parametrize("screen", ["single_component", "two_component"])
def test_gradient_wrt_a_free_redshift_reaches_the_law(screen, uv_ssp, uv_obs):
    """``jax.grad`` must be finite on the photometry and nonzero through the curve.

    Two objectives, because they fail differently. The photometry sum is finite
    both before and after: it guards the new ``jnp.interp`` against returning
    NaN on the fit table. The curve ratio is the one that was broken - measured
    at exactly 0.0 on both screens before the fix, the signature of a parameter a
    fit cannot move, so a photometric redshift run on this law explored a
    direction the likelihood could not see.
    """
    model = _build(uv_ssp, uv_obs, screen, "narayanan_z", Uniform(0.1, 5.0))
    assert "redshift" in model.spec.free_params
    params = _params(model, screen)

    def photometry_sum(z):
        return jnp.sum(model.predict_photometry({**params, "redshift": z}))

    probe = jnp.asarray([1500.0, 5500.0])

    def curve_ratio(z):
        k = _curve(model, screen, probe, extra={"redshift": z})
        return k[0] / k[1]

    with warnings.catch_warnings():
        _filter_fixture_warnings()
        grad_photometry = float(jax.grad(photometry_sum)(jnp.asarray(2.0, dtype=jnp.float64)))
        grad_curve = float(jax.grad(curve_ratio)(jnp.asarray(2.0, dtype=jnp.float64)))

    assert np.isfinite(grad_photometry), f"{screen}: non-finite photometry gradient in z"
    assert np.isfinite(grad_curve), f"{screen}: non-finite curve gradient in z"
    assert grad_curve != 0.0, (
        f"{screen}: d[A(1500)/A(5500)]/dz is exactly zero through narayanan_z. The "
        "redshift never reaches the law, so a photo-z fit returns the prior (#2199)."
    )


# ── (v) every other law is untouched, to the bit ─────────────────────


@pytest.mark.parametrize(("screen", "law"), sorted(_UNTOUCHED_LAW_PHOTOMETRY))
def test_a_law_that_does_not_read_redshift_is_bit_identical(screen, law, uv_ssp, uv_obs):
    """Threading z must not perturb a law whose signature never names it.

    ``select_law_kwargs`` narrows the seeded ``redshift`` away for every law but
    ``narayanan_z``. The references were captured on the merge base before any
    source edit, so this is a before/after comparison and not a self-consistency
    check.
    """
    reference = np.asarray(_UNTOUCHED_LAW_PHOTOMETRY[(screen, law)])
    measured = _photometry(uv_ssp, uv_obs, screen, law, 0.5)
    assert np.array_equal(measured, reference), (
        f"{screen}/{law}: photometry moved. Expected {reference.tolist()}, got "
        f"{measured.tolist()}."
    )
