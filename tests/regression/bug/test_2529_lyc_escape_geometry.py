# SPDX-License-Identifier: BSD-3-Clause
"""Age-selective LyC escape geometry on dust_attenuation (#2529).

tengri's pre-#2529 ``neb_fesc`` (the nebular escape fraction) only ever
gated the nebular-reprocessing budget (:func:`tengri.components.lyc.
lyc_shares`); the dust screen itself was applied uniformly to every photon
regardless of ``neb_fesc``, so even at ``neb_fesc=1`` the "escaping" LyC
still carried the full birth-cloud optical depth. #2529 observed that a
literal hole in the birth cloud is geometric, not wavelength-selective: a
covering fraction ``neb_fesc`` of the young population's light should
bypass the birth-cloud screen at every wavelength, not just below the
Lyman limit.

``dust_attenuation={'lyc_escape_geometry': ...}`` adds that geometry as a
third structural axis alongside the existing ``lyc_reprocessed_by`` /
``lyc_in_energy_balance`` pair: ``'screened'`` (default, bit-identical to
every test and model built before this feature existed) /
``'birth_cloud_holes'`` (FSPS ``frac_obrun``-like: the hole still crosses
the diffuse ISM) / ``'clear'`` (Synthesizer ``fesc``-like: a fully clear
sightline). The one shared formula is
:func:`tengri.components.lyc.escape_geometry_transmission`; see its
docstring for the derivation this module's identities pin.

**'screened' bit-identical, without a bootstrap-head git-archive
checkout**: ``DustSEDComponent.apply`` (and the WavePrecomp LUT/photometry
publishers) special-case ``lyc_escape_geometry == 'screened'`` as a literal
bypass -- the EXACT same code path a caller who never heard of this key
runs, not a mathematically-equivalent reimplementation. A model with the
key explicit at its default is therefore GUARANTEED, by construction, to
be the identical computation as a model with the key entirely absent
(which is exactly the pre-#2529 / bootstrap-head grammar). This is a
stronger equivalence than a snapshot diff, and mirrors
``test_lyc_key_family.py``'s own "internal config mapping" choice for the
same reason (a second checkout buys nothing a construction argument
doesn't already give for free). :class:`TestScreenedIsExplicitDefault`
pins the equivalence as an executable check regardless.

Markers
-------
- ``@pytest.mark.regression_bug`` -- Regression for #2529.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed
from tengri.components.lyc import escape_geometry_transmission, lyc_shares
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.regression_bug

# ---------------------------------------------------------------------------
# Shared model builders
# ---------------------------------------------------------------------------


def _dust_attenuation(
    *,
    lyc_escape_geometry: str = "screened",
    lyc_reprocessed_by: str = "young",
    lyc_in_energy_balance: bool = False,
    dust_type: str = "two_component",
    tau_bc: float = 0.8,
    tau_diff: float = 0.3,
) -> dict:
    d: dict = {
        "type": dust_type,
        "all_params": Fixed(DEFAULT),
    }
    if dust_type == "two_component":
        d.update(
            law_bc="calzetti",
            law_diff="calzetti",
            tau_bc=Fixed(tau_bc),
            tau_diff=Fixed(tau_diff),
            lyc_escape_geometry=lyc_escape_geometry,
            lyc_reprocessed_by=lyc_reprocessed_by,
        )
    elif dust_type == "single_component":
        d.update(law="calzetti", tau_v=Fixed(tau_diff))
        if lyc_escape_geometry != "screened":
            d["lyc_escape_geometry"] = lyc_escape_geometry
    if lyc_in_energy_balance:
        d["lyc_in_energy_balance"] = True
    return d


def _build(
    ssp,
    obs,
    *,
    fesc: float = 0.3,
    geometry: str = "screened",
    reprocessed_by: str = "young",
    eb: bool = False,
    dust_type: str = "two_component",
    tau_bc: float = 0.8,
    tau_diff: float = 0.3,
    age_gyr: float = 0.003,
    approx=None,
    with_dust_emission: bool = False,
):
    """Young-dominated star-forming model with a live photoionized nebular
    backend (so ``neb_fesc`` / ``lyc_fesc`` actually propagate).

    ``with_dust_emission=True`` is required for ``approx=WavePrecomp()`` to
    actually build and consult the energy-balance LUT for ``log_L_absorbed``
    / ``L_ir`` at all: ``SEDModel._energy_balance_lut`` gates the whole LUT
    construction on ``needs_l_ir = has_dust_emission or
    _chain_consumes(chain, "L_ir")``, so a model with no dust_emission block
    silently computes ``log_L_absorbed`` through the EXACT path regardless
    of ``approx`` -- and a LUT-vs-exact comparison without this is
    comparing the exact path to itself, unable to catch the LUT ever
    disagreeing (the gap a RED run against this test file's own mutation
    (b) surfaced).
    """
    kwargs = {}
    if with_dust_emission:
        kwargs["dust_emission"] = {"type": "modified_blackbody", "all_params": Fixed(DEFAULT)}
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(age_gyr),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_fesc": Fixed(fesc),
            "neb_fdust_frac": Fixed(0.2),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_dust_attenuation(
            lyc_escape_geometry=geometry,
            lyc_reprocessed_by=reprocessed_by,
            lyc_in_energy_balance=eb,
            dust_type=dust_type,
            tau_bc=tau_bc,
            tau_diff=tau_diff,
        ),
        **kwargs,
        redshift=Fixed(0.0),
        approx=approx,
    )


def _log_l_absorbed(model) -> float:
    return float(np.asarray(model.predict_state({}).derived["log_L_absorbed"]))


def _sed_attenuated(model) -> np.ndarray:
    return np.asarray(model.predict_state({}).derived["sed_dust_attenuated"])


# ---------------------------------------------------------------------------
# The module function: identities pinned directly (JIT/grad-friendly, no
# SSP grid discretization in the way -- see the module docstring in lyc.py
# for why the full-model equivalents below only hold at y_age -> 0 or 1
# to a finite grid's own resolution, not bit-for-bit).
# ---------------------------------------------------------------------------


class TestEscapeGeometryTransmissionIdentities:
    """Direct, exact-arithmetic identities on
    :func:`tengri.components.lyc.escape_geometry_transmission`.
    """

    y_age = jnp.array([1.0, 0.7, 0.3, 0.0])
    T_bc = jnp.array([0.5, 0.5, 0.5, 0.5])
    T_diff = jnp.array([0.8, 0.8, 0.8, 0.8])

    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    def test_fesc_zero_matches_screened_power_blend(self, geometry):
        T = escape_geometry_transmission(self.y_age, self.T_bc, self.T_diff, 0.0, geometry)
        expected = jnp.power(self.T_bc, self.y_age) * self.T_diff
        np.testing.assert_allclose(np.asarray(T), np.asarray(expected), rtol=1e-10)

    def test_fesc_one_birth_cloud_holes_equals_diffuse_for_any_age(self):
        T = escape_geometry_transmission(
            self.y_age, self.T_bc, self.T_diff, 1.0, "birth_cloud_holes"
        )
        np.testing.assert_allclose(np.asarray(T), np.asarray(self.T_diff), rtol=1e-10)

    def test_fesc_one_clear_is_young_weighted_unattenuated_blend(self):
        T = escape_geometry_transmission(self.y_age, self.T_bc, self.T_diff, 1.0, "clear")
        expected = self.y_age + (1.0 - self.y_age) * self.T_diff
        np.testing.assert_allclose(np.asarray(T), np.asarray(expected), rtol=1e-10)
        # fully young -> exactly unattenuated
        assert float(T[0]) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    def test_old_stars_unaffected_by_any_fesc(self, geometry):
        for fesc in (0.0, 0.3, 1.0):
            T = escape_geometry_transmission(
                jnp.array(0.0), self.T_bc[0], self.T_diff[0], fesc, geometry
            )
            assert float(T) == pytest.approx(float(self.T_diff[0]), abs=1e-10)

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="birth_cloud_holes"):
            escape_geometry_transmission(self.y_age, self.T_bc, self.T_diff, 0.3, "screened")
        with pytest.raises(ValueError, match="birth_cloud_holes"):
            escape_geometry_transmission(self.y_age, self.T_bc, self.T_diff, 0.3, "bogus")

    def test_transmission_bounded_in_unit_interval(self):
        key = jax.random.PRNGKey(0)
        for geometry in ("birth_cloud_holes", "clear"):
            for _ in range(5):
                key, sub = jax.random.split(key)
                y = jax.random.uniform(sub, (7,))
                key, sub = jax.random.split(key)
                tbc = jax.random.uniform(sub, (7,))
                key, sub = jax.random.split(key)
                tdiff = jax.random.uniform(sub, (7,))
                key, sub = jax.random.split(key)
                fesc = float(jax.random.uniform(sub, ()))
                T = escape_geometry_transmission(y, tbc, tdiff, fesc, geometry)
                assert bool(jnp.all(T >= -1e-10))
                assert bool(jnp.all(T <= 1.0 + 1e-10))

    def test_gradient_wrt_fesc_finite_and_nonzero(self):
        def f(fesc):
            T = escape_geometry_transmission(
                self.y_age, self.T_bc, self.T_diff, fesc, "birth_cloud_holes"
            )
            return jnp.sum(T)

        g = float(jax.grad(f)(0.3))
        assert np.isfinite(g), "a corrupt (NaN/inf) fesc gradient breaks any gradient-based fit"
        assert g != 0.0, (
            "T_hole != T_bc**y*T_diff for a nonzero y_age slice (birth_cloud_holes with "
            "T_bc != T_diff), so d(sum T)/d(fesc) is genuinely nonzero here -- a zero "
            "would mean the hole term silently dropped out, the #2100 failure shape"
        )


# ---------------------------------------------------------------------------
# Full-model 'screened' equivalence
# ---------------------------------------------------------------------------


class TestScreenedIsExplicitDefault:
    """Explicit ``lyc_escape_geometry='screened'`` == the key omitted
    entirely, for exact and WavePrecomp, SED/photometry/log_L_absorbed,
    young/all x EB on/off -- see module docstring for why this stands in
    for a bootstrap-head snapshot.
    """

    @pytest.mark.parametrize("reprocessed_by", ["young", "all"])
    @pytest.mark.parametrize("eb", [False, True])
    @pytest.mark.parametrize("approx", [None, WavePrecomp()])
    def test_explicit_screened_matches_omitted_key(
        self, synthetic_ssp_wide, synthetic_tophat_obs, reprocessed_by, eb, approx
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m_default = _build(
            ssp,
            obs,
            fesc=0.4,
            geometry="screened",
            reprocessed_by=reprocessed_by,
            eb=eb,
            approx=approx,
        )
        atten = _dust_attenuation(lyc_reprocessed_by=reprocessed_by, lyc_in_energy_balance=eb)
        del atten["lyc_escape_geometry"]  # the omitted-key spelling
        m_omitted = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(0.003),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            neb={
                "type": "cue",
                "neb_fesc": Fixed(0.4),
                "neb_fdust_frac": Fixed(0.2),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation=atten,
            redshift=Fixed(0.0),
            approx=approx,
        )
        assert m_default.spec.dust_lyc_escape_geometry == "screened"
        assert m_omitted.spec.dust_lyc_escape_geometry == "screened"

        np.testing.assert_array_equal(
            np.asarray(m_default.predict_photometry({})),
            np.asarray(m_omitted.predict_photometry({})),
        )
        s_default = m_default.predict_state({})
        s_omitted = m_omitted.predict_state({})
        np.testing.assert_array_equal(
            np.asarray(s_default.derived["sed_dust_attenuated"]),
            np.asarray(s_omitted.derived["sed_dust_attenuated"]),
        )
        assert float(np.asarray(s_default.derived["log_L_absorbed"])) == float(
            np.asarray(s_omitted.derived["log_L_absorbed"])
        )


# ---------------------------------------------------------------------------
# Full-model identities at fesc in {0, 0.3, 1}
# ---------------------------------------------------------------------------


class TestFullModelIdentities:
    def test_fesc_zero_all_geometries_identical(self, synthetic_ssp_wide, synthetic_tophat_obs):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m_screened = _build(ssp, obs, fesc=0.0, geometry="screened")
        phot_screened = np.asarray(m_screened.predict_photometry({}))
        for geometry in ("birth_cloud_holes", "clear"):
            m = _build(ssp, obs, fesc=0.0, geometry=geometry)
            phot = np.asarray(m.predict_photometry({}))
            np.testing.assert_allclose(phot, phot_screened, rtol=1e-8)

    def test_energy_ordering_clear_le_holes_le_screened(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        L = {
            geometry: _log_l_absorbed(_build(ssp, obs, fesc=0.6, geometry=geometry))
            for geometry in ("clear", "birth_cloud_holes", "screened")
        }
        assert L["clear"] <= L["birth_cloud_holes"] + 1e-6
        assert L["birth_cloud_holes"] <= L["screened"] + 1e-6

    @pytest.mark.parametrize("eb", [False, True])
    def test_fesc_one_birth_cloud_holes_matches_tau_bc_zero_reference(
        self, synthetic_ssp_wide, synthetic_tophat_obs, eb
    ):
        # At fesc=1, 'birth_cloud_holes' young light == young SED x T_diff
        # for EVERY age (escape_geometry_transmission's own construction,
        # not merely for the mass near y_age ~ 1) -- the SAME answer as
        # pinning tau_bc=0 outright (no birth-cloud screen at all: T_bc^y
        # with tau_bc=0 is 1 for any y, too), so this holds to machine
        # precision regardless of how much of the SSP age grid actually
        # saturates y_age -> 1.
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m_holes = _build(ssp, obs, fesc=1.0, geometry="birth_cloud_holes", eb=eb, age_gyr=0.003)
        m_ref = _build(ssp, obs, fesc=1.0, geometry="screened", eb=eb, age_gyr=0.003, tau_bc=0.0)
        sed_holes = _sed_attenuated(m_holes)
        sed_ref = _sed_attenuated(m_ref)
        np.testing.assert_allclose(sed_holes, sed_ref, rtol=1e-6)

    @pytest.mark.parametrize("eb", [False, True])
    def test_fesc_one_clear_matches_fully_unattenuated_reference(
        self, synthetic_ssp_wide, synthetic_tophat_obs, eb
    ):
        # tau_diff=0.0 on BOTH models isolates the identity from how much of
        # the SSP age grid actually saturates y_age -> 1 (the old
        # population, and the covered young sub-beam, both reduce to
        # T_diff = 1 here identically, so 'clear' at fesc=1 collapses to
        # EXACTLY 1 at every age via escape_geometry_transmission's own
        # construction -- not merely approximately for the mass near
        # y_age ~ 1). tau_bc stays nonzero on ``m_clear`` so the test still
        # exercises the birth-cloud bypass, not a tau_bc=0 no-op.
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m_clear = _build(ssp, obs, fesc=1.0, geometry="clear", eb=eb, age_gyr=0.003, tau_diff=0.0)
        m_ref = _build(
            ssp,
            obs,
            fesc=1.0,
            geometry="screened",
            eb=eb,
            age_gyr=0.003,
            tau_bc=0.0,
            tau_diff=0.0,
        )
        sed_clear = _sed_attenuated(m_clear)
        sed_ref = _sed_attenuated(m_ref)
        np.testing.assert_allclose(sed_clear, sed_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# LyC closure: escaped + screen-absorbed + gas + HII-dust == total
# ---------------------------------------------------------------------------


class TestLycClosure:
    @pytest.mark.parametrize("geometry", ["screened", "birth_cloud_holes", "clear"])
    @pytest.mark.parametrize("eb", [False, True])
    def test_budget_shares_sum_to_one(self, geometry, eb):
        # lyc_shares is the one place the ionizing-photon budget is split;
        # unaffected by lyc_escape_geometry (the geometry governs where
        # the dust screen puts the COVERED fraction's photons, not how the
        # total budget is partitioned into escape/dust/gas in the first
        # place -- see lyc.py's escape_geometry_transmission docstring).
        for fesc in (0.0, 0.3, 1.0):
            f_esc, f_dust, f_gas = lyc_shares(fesc, 0.2)
            total = float(f_esc) + float(f_dust) + float(f_gas)
            assert total == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# LUT-vs-exact (energy balance) and WavePrecomp-vs-exact (photometry)
# ---------------------------------------------------------------------------


class TestWavePrecompParity:
    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    def test_lut_stellar_matches_hand_rolled_exact_unit(self, synthetic_ssp_wide, geometry):
        """Unit-level LUT-vs-exact at the stellar-absorbed-luminosity level,
        bypassing the full forward model entirely: isolates the LUT's own
        fesc-affine construction from the nebular-absorbed term (which the
        model-level comparisons below also exercise, but which can dominate
        ``log_L_absorbed`` enough to mask a broken stellar LUT at ordinary
        dust/mass fiducials -- the gap a RED run against this test's own
        mutation (b) surfaced). Single-node tau grids at exactly the query
        point eliminate the LUT's OTHER (bilinear tau-interpolation) error
        source, isolating the fesc-affine one this test targets.
        """
        from tengri.components.dust.attenuation import resolve_dust_law
        from tengri.components.dust.energy_balance_precompute import (
            build_energy_balance_lut,
            lut_l_absorbed_stellar_log10,
        )
        from tengri.components.dust.two_component import _young_indicator
        from tengri.components.lyc import LYMAN_LIMIT_AA, edge_trapezoid

        ssp = synthetic_ssp_wide
        ssp_wave = jnp.asarray(ssp.ssp_wave)
        ssp_flux = jnp.asarray(ssp.ssp_flux)  # (n_met, n_age, n_wave)
        ssp_ages_yr = (10.0 ** jnp.asarray(ssp.ssp_lg_age_gyr)) * 1e9

        tau_bc_q, tau_diff_q, fesc_q = 0.8, 0.3, 0.6
        tau_bc_grid = jnp.asarray([tau_bc_q])
        tau_diff_grid = jnp.asarray([tau_diff_q])

        met_idx, age_idx = 0, 0  # the youngest available SSP age node
        joint_weights = jnp.zeros(ssp_flux.shape[:2]).at[met_idx, age_idx].set(1.0)

        lut = build_energy_balance_lut(
            ssp_flux,
            ssp_wave,
            ssp_ages_yr,
            law_bc="calzetti",
            law_diff="calzetti",
            t_birth_yr=1e7,
            transition_width_dex=0.3,
            lyc_in_energy_balance=False,
            tau_bc_grid=tau_bc_grid,
            tau_diff_grid=tau_diff_grid,
            lyc_escape_geometry=geometry,
        )
        log_mag_lut, sign_lut = lut_l_absorbed_stellar_log10(
            lut, joint_weights, jnp.asarray(0.0), tau_bc_q, tau_diff_q, fesc=fesc_q
        )
        lut_val = float(10 ** float(log_mag_lut)) * float(sign_lut)

        y_age = _young_indicator(ssp_ages_yr, 1e7, 0.3)
        k_bc = resolve_dust_law("calzetti")(ssp_wave)
        k_diff = resolve_dust_law("calzetti")(ssp_wave)
        t_bc_raw = jnp.exp(-tau_bc_q * k_bc)
        t_diff_raw = jnp.exp(-tau_diff_q * k_diff)
        t_eb_raw = escape_geometry_transmission(
            y_age[age_idx], t_bc_raw, t_diff_raw, fesc_q, geometry
        )
        intrinsic = ssp_flux[met_idx, age_idx, :]
        mask_nonlyc = ssp_wave >= LYMAN_LIMIT_AA
        B = edge_trapezoid(intrinsic * mask_nonlyc, ssp_wave, side="all", edge_aa=LYMAN_LIMIT_AA)
        G = edge_trapezoid(
            intrinsic * t_eb_raw * mask_nonlyc, ssp_wave, side="all", edge_aa=LYMAN_LIMIT_AA
        )
        exact_val = float(B - G)

        assert lut_val == pytest.approx(exact_val, rel=1e-9)

    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    @pytest.mark.parametrize("eb", [False, True])
    @pytest.mark.parametrize("fesc", [0.0, 0.3, 1.0])
    def test_log_l_absorbed_lut_matches_exact(
        self, synthetic_ssp_wide, synthetic_tophat_obs, geometry, eb, fesc
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m_exact = _build(
            ssp, obs, fesc=fesc, geometry=geometry, eb=eb, approx=None, with_dust_emission=True
        )
        m_lut = _build(
            ssp,
            obs,
            fesc=fesc,
            geometry=geometry,
            eb=eb,
            approx=WavePrecomp(),
            with_dust_emission=True,
        )
        L_exact = _log_l_absorbed(m_exact)
        L_lut = _log_l_absorbed(m_lut)
        assert L_exact == pytest.approx(L_lut, abs=1e-6)

    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    @pytest.mark.parametrize("fesc", [0.0, 0.3, 1.0])
    def test_photometry_wave_precomp_matches_exact(
        self, synthetic_ssp_wide, synthetic_tophat_obs, geometry, fesc
    ):
        # Same tolerance order as the existing #1122 sub-band quadrature
        # convergence floor (K=5 worst case ~0.6%, see WavePrecomp's own
        # docstring) -- this feature does not introduce a NEW approximation,
        # it reuses that one.
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m_exact = _build(ssp, obs, fesc=fesc, geometry=geometry, approx=None)
        m_lut = _build(ssp, obs, fesc=fesc, geometry=geometry, approx=WavePrecomp())
        phot_exact = np.asarray(m_exact.predict_photometry({}))
        phot_lut = np.asarray(m_lut.predict_photometry({}))
        rel = np.max(np.abs(phot_exact - phot_lut) / np.maximum(np.abs(phot_exact), 1e-30))
        assert rel < 1e-3


# ---------------------------------------------------------------------------
# Cross-code: FSPS frac_obrun toy
# ---------------------------------------------------------------------------


class TestFSPSFracObrunParity:
    """Hand-computed FSPS ``frac_obrun`` transmission for a two-age toy,
    compared against :func:`escape_geometry_transmission` with
    ``'birth_cloud_holes'``.

    FSPS's ``frac_obrun`` (Conroy, Gunn & White 2009; Conroy & Gunn 2010)
    is the fraction of young (OB) stars that have escaped their birth
    cloud; FSPS's dust/nebular machinery treats that fraction as seeing
    ONLY the diffuse screen at every wavelength (``add_dust_emission``
    applies ``dust2`` alone to the escaped population) and contributing no
    nebular emission. For a population fully inside its birth cloud
    (``y_age=1``), this is EXACTLY tengri's ``'birth_cloud_holes'`` formula
    at ``f_esc = frac_obrun``: ``T = frac_obrun * T_diff + (1 - frac_obrun)
    * T_bc * T_diff``.
    """

    def test_two_age_toy_matches_frac_obrun_formula(self):
        T_bc = jnp.array([0.4, 0.6])  # two toy wavelengths
        T_diff = jnp.array([0.7, 0.85])
        frac_obrun = 0.35
        y_age = jnp.array([1.0, 1.0])  # both toy ages fully inside the birth cloud

        T_tengri = escape_geometry_transmission(
            y_age, T_bc, T_diff, frac_obrun, "birth_cloud_holes"
        )
        # Hand-computed FSPS frac_obrun transmission: a frac_obrun fraction
        # sees only T_diff; the remainder sees the full T_bc * T_diff screen.
        T_fsps = frac_obrun * T_diff + (1.0 - frac_obrun) * (T_bc * T_diff)
        np.testing.assert_allclose(np.asarray(T_tengri), np.asarray(T_fsps), rtol=1e-12)


# ---------------------------------------------------------------------------
# Key validation + refusal + round-trip (extends test_lyc_key_family.py's
# dedicated contract for the lyc_ key family itself)
# ---------------------------------------------------------------------------


class TestKeyValidationAndRefusal:
    def test_unknown_value_lists_allowed(self, synthetic_ssp_wide, synthetic_tophat_obs):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        with pytest.raises(
            ValueError, match=r"screened.*birth_cloud_holes.*clear|birth_cloud_holes"
        ):
            _build(ssp, obs, geometry="bogus")

    def test_non_screened_on_single_component_raises(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        with pytest.raises(ValueError, match="birth-cloud screen"):
            _build(ssp, obs, geometry="clear", dust_type="single_component")

    def test_non_screened_with_reprocessed_by_all_raises(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        with pytest.raises(ValueError, match="double-count"):
            _build(ssp, obs, geometry="birth_cloud_holes", reprocessed_by="all")

    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    def test_round_trip(self, synthetic_ssp_wide, synthetic_tophat_obs, geometry):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m = _build(ssp, obs, geometry=geometry)
        groups = m.spec.to_groups()
        atten = groups["dust_attenuation"]
        assert atten["lyc_escape_geometry"] == geometry

        m2 = tengri.SEDModel.build(ssp, observation=obs, **groups)
        assert m2.spec.dust_lyc_escape_geometry == geometry
        np.testing.assert_array_equal(
            np.asarray(m.predict_photometry({})), np.asarray(m2.predict_photometry({}))
        )

    def test_screened_default_not_emitted_in_round_trip(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m = _build(ssp, obs, geometry="screened")
        groups = m.spec.to_groups()
        assert "lyc_escape_geometry" not in groups["dust_attenuation"]


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


class TestGradient:
    @pytest.mark.parametrize("geometry", ["birth_cloud_holes", "clear"])
    def test_photometry_gradient_wrt_neb_fesc_finite_and_nonzero(
        self, synthetic_ssp_wide, synthetic_tophat_obs, geometry
    ):
        ssp, obs = synthetic_ssp_wide, synthetic_tophat_obs
        m = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(0.003),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            neb={
                "type": "cue",
                "neb_fesc": FREE,
                "neb_fdust_frac": Fixed(0.2),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation=_dust_attenuation(lyc_escape_geometry=geometry),
            redshift=Fixed(0.0),
        )

        def f(fesc):
            return jnp.sum(m.predict_photometry({"neb_fesc": fesc}))

        g = float(jax.grad(f)(0.3))
        assert np.isfinite(g), "a corrupt (NaN/inf) fesc gradient breaks any gradient-based fit"
        assert g != 0.0, (
            "the hole/screened mix is genuinely fesc-dependent at tau_bc=0.8 != 0 here "
            "-- a zero gradient would mean the geometry correction silently dropped out "
            "of the traced computation, the #2100 failure shape"
        )
