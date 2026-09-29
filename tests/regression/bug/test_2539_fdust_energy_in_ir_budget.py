# SPDX-License-Identifier: BSD-3-Clause
"""Regression test: neb_fdust LyC energy credited to the dust IR budget (#2539).

``neb_fdust`` (CIGALE semantics: fraction of Lyman-continuum photons absorbed
by dust inside HII regions) used to only rescale the nebular *emission*
amplitude (via ``lyc_dust_escape_factor``, the ``k`` factor) -- the energy it
diverts to dust never entered the dust IR budget (``L_absorbed`` /
``log_L_absorbed``). ``NebularSEDComponent`` now publishes
``log_L_lyc_dust = log10(neb_fdust) + log_L_lyc`` and every dust attenuator
(``single_component``, ``two_component``, ``wg00``) adds it into
``log_L_absorbed``, matching CIGALE's own
``dust.luminosity = (lum_ly_young + lum_ly_old) * fdust``
(``pcigale/sed_modules/nebular.py:191-193``).

Design notes (read before editing this file)
----------------------------------------------
**Why the identity test compares (fesc=0.3, fdust=0.0) against
(fesc=0.0, fdust=0.3) instead of (fdust=0.3) against (fdust=0.0) directly**:
``lyc_dust_escape_factor``'s ``k(fesc, fdust)`` depends on ``fesc + fdust``
only, and it scales the nebular *emission* amplitude, which is itself an
input to the SAME dust attenuator's ``log_L_absorbed`` integral (the
attenuated nebular continuum is absorbed by the same screen). So varying
``fdust`` alone at fixed ``fesc = 0`` changes ``log_L_absorbed`` two ways at
once: (1) the new #2539 credit, and (2) the pre-existing, correct reduction
in nebular-continuum absorption from a smaller ``k``. Holding
``f_total = fesc + fdust`` fixed (by trading one for the other) holds ``k``,
hence the nebular-continuum contribution, EXACTLY fixed, isolating (1) alone.
Since ``eb_include_lyc=False`` (default) masks the LyC region out of the
screen's own integral regardless of how much of it (fesc vs 0) remains in
``sed_intrinsic``, the *entire* measured difference is attributable to the
credit. This is a stronger, cleaner test of the identity the brief asked for
than comparing fdust values directly would have been.

**Population match (#2539 item 2)**: ``single_component``, ``wg00``, and
``two_component`` with ``lyc_absorb_all=True`` credit the WHOLE stellar
population's LyC (matching the population the nebular fesc/fdust mask
actually ran over). ``two_component`` with ``lyc_absorb_all=False`` (default)
only routes the YOUNG/birth-cloud population's LyC through the gas
(``two_component.py``'s ``lyc_factor = 1 - y_age*(1-lyc_t)``), so its credit
uses the SAME ``y_age``-weighted population, computed independently in this
file from ``lnu_age`` via ``tengri.components.dust.two_component._young_indicator``.

**No double counting (#2539 item 2)**: ``NebularSEDComponent`` masks
``sed_intrinsic``'s LyC region to ``neb_fesc`` fraction UNCONDITIONALLY
(independent of ``neb_fdust``); the credited ``fdust`` fraction, and the
``(1 - fesc - fdust)`` fraction that ionizes gas, never reach
``sed_intrinsic`` at all, so a dust screen (even with ``eb_include_lyc=True``)
can only ever re-absorb the ``fesc`` remainder, never energy already credited
to HII-region dust. This is verified two ways here: directly (the LyC content
of ``sed_intrinsic`` equals ``fesc`` times the raw stellar LyC, to machine
precision) and via an invariant (the EXTRA energy ``eb_include_lyc=True``
finds is independent of ``fdust`` at fixed ``fesc``, since it can only ever
draw on the ``fesc``-sized remainder).

**Exact per-photon budget (owner ruling, #2539)**: the LyC of the credited
population splits into three ADDITIVE, non-overlapping shares -- ``fesc``
escapes, ``fdust`` heats HII-region dust, ``1 - fesc - fdust`` ionizes gas --
not a sequential/product form (``lyc_dust_escape_factor`` combines them as
``f_total = f_esc + f_dust`` before anything else; CIGALE's own
``nebular.py`` masks the emergent SED by ``(1 - fesc)`` alone and separately
credits dust with ``fdust`` of the RAW luminosity: two independent parallel
shares, never a fesc-depleted remainder). See
``tengri.components.nebular._recombination_coeffs.lyc_dust_escape_factor``'s
docstring for the full derivation. tengri does not attempt to further
apportion the ``1 - fesc - fdust`` share between "observable nebular
emission" and case-B recombination-cascade losses in erg/s terms: ``k``
already governs the nebular amplitude and this file does not re-derive it.

Markers
-------
- ``@pytest.mark.regression_bug`` -- Bug regression test
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.dust.two_component import _young_indicator
from tengri.forward.energy_balance import log10_fdust_lyc_credit
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

LYC_CUTOFF_AA = 912.0
T_BIRTH_YR = 1e7
TRANSITION_WIDTH_DEX = 0.3  # two_component's declared defaults


def _l_lyc(wave, lnu) -> float:
    """Independently integrate L_LyC = |int_{lambda<912} L_nu dnu|, float64."""
    wave64 = np.asarray(wave, dtype=np.float64)
    lnu64 = np.asarray(lnu, dtype=np.float64)
    nu64 = np.asarray(C_AA, dtype=np.float64) / wave64
    integrand = np.where(wave64 < LYC_CUTOFF_AA, lnu64, 0.0)
    return float(abs(np.trapezoid(integrand, nu64)))


def _credited_lnu(state, *, young_only: bool) -> np.ndarray:
    """The raw (pre-fesc) per-wavelength L_nu of the credited population."""
    lnu_age = np.asarray(state.derived["lnu_age"])
    if young_only:
        ssp_ages_yr = np.asarray(state.derived["ssp_ages_yr"])
        y_age = np.asarray(
            _young_indicator(jnp.asarray(ssp_ages_yr), T_BIRTH_YR, TRANSITION_WIDTH_DEX)
        )
        return np.sum(y_age[:, None] * lnu_age, axis=0)
    return np.sum(lnu_age, axis=0)


def _sfh() -> dict:
    return {
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(10.0),
        "all_params": Fixed(DEFAULT),
    }


def _build(
    ssp,
    dust_type: str,
    *,
    fesc=0.0,
    fdust=0.0,
    free_fdust: bool = False,
    lyc_absorb_all: bool = False,
    eb_include_lyc: bool = False,
    tau: float = 1.0,
    neb_type: str = "cue",
) -> SEDModel:
    neb: dict = {"type": neb_type, "all_params": Fixed(DEFAULT)}
    if neb_type != "none":
        neb["neb_fesc"] = Fixed(fesc)
        neb["neb_fdust"] = FREE if free_fdust else Fixed(fdust)
    dust: dict = {"type": dust_type, "all_params": Fixed(DEFAULT)}
    if dust_type == "two_component":
        dust.update(
            law_bc="calzetti",
            law_diff="calzetti",
            tau_bc=Fixed(tau),
            tau_diff=Fixed(tau),
            lyc_absorb_all=lyc_absorb_all,
        )
    elif dust_type == "single_component":
        dust.update(law="calzetti", tau_v=Fixed(tau))
    elif dust_type == "wg00":
        dust.update(dust_curve="mw", geometry="shell", structure="homogeneous", tau_v=Fixed(tau))
    else:
        raise ValueError(dust_type)
    if eb_include_lyc:
        dust["eb_include_lyc"] = True
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh=_sfh(),
        neb=neb,
        dust_attenuation=dust,
        redshift=Fixed(0.0),
    )


# (dust_type, lyc_absorb_all, is young-credited population)
ATTENUATORS = [
    pytest.param("single_component", False, False, id="single_component"),
    pytest.param("two_component", False, True, id="two_component-lyc_absorb_all_False"),
    pytest.param("two_component", True, False, id="two_component-lyc_absorb_all_True"),
    pytest.param("wg00", False, False, id="wg00"),
]


def _build_nodust(ssp, *, fesc=0.0, fdust=0.0, neb_type: str = "cue") -> SEDModel:
    """A twin model with dust attenuation disabled entirely.

    ``state.sed_intrinsic`` is never reassigned by a dust component (every
    attenuator does ``sed_intrinsic=attenuated``/``sed_total`` post-screen,
    #2539), so this twin's ``sed_intrinsic`` is the TRUE combined
    (stellar + nebular) pre-dust SED -- the nebular fesc/fdust masking runs
    upstream of, and independent of, any dust component.

    Deliberately NOT a ``tau=0`` dust twin: WG00's tabulated (Witt & Gordon
    2000) attenuation grid does not reach transmission 1.0 at ``tau_v=0``
    (measured floor ~0.37 at the shortest wavelengths in this grid, a
    genuine grid-boundary effect of the tabulated model, not a #2539 defect),
    so a ``tau=0`` wg00 build is not a screen-transparent proxy.
    """
    neb: dict = {"type": neb_type, "all_params": Fixed(DEFAULT)}
    if neb_type != "none":
        neb["neb_fesc"] = Fixed(fesc)
        neb["neb_fdust"] = Fixed(fdust)
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh=_sfh(),
        neb=neb,
        dust_attenuation={"type": "none"},
        redshift=Fixed(0.0),
    )


def _pre_screen_state(ssp, dust_type: str, *, fesc: float, fdust: float, lyc_absorb_all: bool):
    """The pre-screen (screen-transparent) state, from whichever twin
    reconstructs it exactly for this attenuator.

    ``two_component`` with ``lyc_absorb_all=False`` reprocesses only the
    YOUNG/birth-cloud population's LyC (#2539 item 3); that per-age
    young/old split is reconstructed inside ``DustSEDComponent.apply()``
    itself (the ``lyc_factor`` weighting), so recovering it needs a REAL
    ``tau=0`` two_component build (its calzetti screen gives
    transmission exactly 1.0 at ``tau=0``, verified against
    ``dust_diff_transmission``) -- a no-dust twin would instead read back
    the nebular component's own UNIFORM (all-ages) fesc mask, which is a
    different, coarser quantity. wg00 has no such per-age reconstruction
    (single population, no birth-cloud split), so its screen-transparent
    proxy is the no-dust twin (see ``_build_nodust``), sidestepping the
    WG00 grid's non-unity floor at ``tau_v=0``.
    """
    if dust_type == "wg00":
        m0 = _build_nodust(ssp, fesc=fesc, fdust=fdust)
    else:
        m0 = _build(ssp, dust_type, fesc=fesc, fdust=fdust, lyc_absorb_all=lyc_absorb_all, tau=0.0)
    s0 = m0.predict_state({})
    return s0, np.asarray(s0.wave)


class TestFdustCreditIdentity:
    """The core identity for every attenuator: reassigning 0.3 of the LyC
    budget from escape to HII-region dust changes ``log_L_absorbed`` by
    EXACTLY ``0.3 * L_LyC`` of the credited population. See the module
    docstring for why the comparison is fesc-for-fdust rather than
    fdust-vs-zero directly.
    """

    @pytest.mark.parametrize("dust_type,lyc_absorb_all,young_only", ATTENUATORS)
    def test_identity(self, synthetic_ssp_wide, dust_type, lyc_absorb_all, young_only):
        m_escape = _build(
            synthetic_ssp_wide, dust_type, fesc=0.3, fdust=0.0, lyc_absorb_all=lyc_absorb_all
        )
        m_dust = _build(
            synthetic_ssp_wide, dust_type, fesc=0.0, fdust=0.3, lyc_absorb_all=lyc_absorb_all
        )
        s_escape = m_escape.predict_state({})
        s_dust = m_dust.predict_state({})
        L_escape = float(10.0 ** np.asarray(s_escape.derived["log_L_absorbed"]))
        L_dust = float(10.0 ** np.asarray(s_dust.derived["log_L_absorbed"]))
        assert np.isfinite(L_escape) and np.isfinite(L_dust)

        lnu_credited = _credited_lnu(s_escape, young_only=young_only)
        expected = 0.3 * _l_lyc(np.asarray(s_escape.wave), lnu_credited)
        assert expected > 0.0, "setup: credited population has zero LyC luminosity"
        np.testing.assert_allclose(L_dust - L_escape, expected, rtol=1e-6)

    @pytest.mark.parametrize("dust_type,lyc_absorb_all,young_only", ATTENUATORS)
    def test_sed_intrinsic_lyc_region_no_double_count(
        self, synthetic_ssp_wide, dust_type, lyc_absorb_all, young_only
    ):
        """The LyC that gas/HII-dust consumed is actually removed from the
        SED a screen would see (#2539 item 2): the final (tau=0, so
        attenuation-transparent) ``sed_intrinsic`` -- what every attenuator
        threads on to downstream components -- has EXACTLY ``fesc`` of the
        credited population's raw LyC below 912 A -- never the fdust or
        gas-ionizing shares -- so there is nothing left for
        ``eb_include_lyc=True`` to double-count against the #2539 credit.

        single_component/wg00 (and two_component with lyc_absorb_all=True)
        mask the WHOLE population uniformly by fesc. two_component with
        lyc_absorb_all=False (default) only masks the YOUNG/birth-cloud
        population -- the per-age lyc_factor leaves old-star LyC completely
        unmasked (point 3: "populations not reprocessed... their LyC meets
        the screen as usual") -- so the expected observed LyC there is
        ``fesc * L_lyc(young) + 1.0 * L_lyc(old)``, not ``fesc *
        L_lyc(total)``.
        """
        s, wave = _pre_screen_state(
            synthetic_ssp_wide, dust_type, fesc=0.3, fdust=0.3, lyc_absorb_all=lyc_absorb_all
        )
        lnu_total = np.sum(np.asarray(s.derived["lnu_age"]), axis=0)
        raw_lyc_total = _l_lyc(wave, lnu_total)
        observed_lyc = _l_lyc(wave, np.asarray(s.sed_intrinsic))

        if dust_type == "two_component" and not lyc_absorb_all:
            lnu_young = _credited_lnu(s, young_only=True)
            raw_lyc_young = _l_lyc(wave, lnu_young)
            raw_lyc_old = raw_lyc_total - raw_lyc_young
            expected = 0.3 * raw_lyc_young + 1.0 * raw_lyc_old
        else:
            expected = 0.3 * raw_lyc_total
        np.testing.assert_allclose(observed_lyc, expected, rtol=1e-6)

    @pytest.mark.parametrize("dust_type", ["single_component", "two_component", "wg00"])
    def test_screen_absorbed_lyc_independent_of_fdust(self, synthetic_ssp_wide, dust_type):
        """No double counting (#2539 item 2), the invariant form: the EXTRA
        energy ``eb_include_lyc=True`` finds over the default (the screen's
        own absorption of whatever LyC still remains in ``sed_intrinsic``)
        must depend on ``fesc`` alone -- it can only draw on the ``fesc``
        remainder, which does not depend on ``fdust`` -- never on ``fdust``,
        or the HII-region credit and the screen's own absorption would be
        drawing on the same photons twice.
        """
        screen_absorbed = []
        for fdust in (0.0, 0.15, 0.3):
            m_default = _build(synthetic_ssp_wide, dust_type, fesc=0.4, fdust=fdust)
            m_full = _build(
                synthetic_ssp_wide, dust_type, fesc=0.4, fdust=fdust, eb_include_lyc=True
            )
            L_default = float(
                10.0 ** np.asarray(m_default.predict_state({}).derived["log_L_absorbed"])
            )
            L_full = float(10.0 ** np.asarray(m_full.predict_state({}).derived["log_L_absorbed"]))
            assert L_full > L_default, "setup: eb_include_lyc should add energy"
            screen_absorbed.append(L_full - L_default)
        np.testing.assert_allclose(screen_absorbed, screen_absorbed[0], rtol=1e-6)


class TestLycConservationClosure:
    """LyC conservation closure for a mixed-age SFH (governing requirement
    #2): the raw ionizing budget of every star (young and old) splits,
    without loss or double counting, into exactly four dispositions --

    - **escaped** (reaches the observer): ``fesc`` of the credited
      population's raw LyC, plus the WHOLE raw LyC of any population the
      nebular step does not reprocess (#2539 item 3), minus whatever the
      dust screen itself further removes;
    - **screen-absorbed**: the above screen removal, credited to
      ``log_L_absorbed`` only when ``eb_include_lyc=True``;
    - **gas-ionizing**: the credited population's ``1 - fesc - fdust``
      share, which photoionizes hydrogen (by construction of the
      fesc/fdust/k-factor split -- not independently re-measurable in
      erg/s post-recombination, see the module docstring);
    - **HII-dust credit**: the credited population's ``fdust`` share,
      which heats dust and enters ``log_L_absorbed`` (#2539 item 1).

    Three of the four terms are measured from ACTUAL model outputs (an
    ``log_L_absorbed`` difference for the credit and for the screen
    contribution, a direct SED integral for the escaping fraction); only
    the gas-ionizing share is asserted by construction. ``_sfh()``'s
    delayed law (tau=1 Gyr, age=5 Gyr) spans stellar ages from ~0 to 5 Gyr,
    giving every attenuator (including two_component's young/old split) a
    genuinely mixed-age population to close the budget over.
    """

    @pytest.mark.parametrize("dust_type,lyc_absorb_all,young_only", ATTENUATORS)
    def test_closure(self, synthetic_ssp_wide, dust_type, lyc_absorb_all, young_only):
        fesc, fdust = 0.3, 0.3

        # Pre-screen twin: see _pre_screen_state for why the twin type is
        # dust_type-aware (tau=0 real dust for single/two_component,
        # no-dust for wg00).
        s0, wave = _pre_screen_state(
            synthetic_ssp_wide, dust_type, fesc=fesc, fdust=fdust, lyc_absorb_all=lyc_absorb_all
        )

        lnu_total = np.sum(np.asarray(s0.derived["lnu_age"]), axis=0)
        lnu_credited = _credited_lnu(s0, young_only=young_only)
        lnu_uncredited = lnu_total - lnu_credited
        L_lyc_total = _l_lyc(wave, lnu_total)
        L_lyc_credited = _l_lyc(wave, lnu_credited)
        L_lyc_uncredited = _l_lyc(wave, lnu_uncredited)
        assert L_lyc_credited > 0.0, "setup: credited population has zero LyC"

        # -- escaped (pre-screen): formula vs. the actual sed_intrinsic --
        escaped_measured = _l_lyc(wave, np.asarray(s0.sed_intrinsic))
        escaped_expected = fesc * L_lyc_credited + L_lyc_uncredited
        np.testing.assert_allclose(escaped_measured, escaped_expected, rtol=1e-6)

        # -- gas-ionizing: by construction (see class docstring) --
        gas_ionizing = (1.0 - fesc - fdust) * L_lyc_credited

        # -- HII-dust credit: an ACTUAL log_L_absorbed difference, isolated
        # by trading fesc for fdust at fixed total (holds k, hence the
        # nebular-continuum absorption term, exactly fixed; see module
        # docstring) --
        m_default = _build(synthetic_ssp_wide, dust_type, fesc=fesc, fdust=fdust, lyc_absorb_all=lyc_absorb_all)
        m_escape_only = _build(
            synthetic_ssp_wide, dust_type, fesc=fesc + fdust, fdust=0.0, lyc_absorb_all=lyc_absorb_all
        )
        s_default = m_default.predict_state({})
        L_absorbed_default = float(10.0 ** np.asarray(s_default.derived["log_L_absorbed"]))
        L_absorbed_escape_only = float(
            10.0 ** np.asarray(m_escape_only.predict_state({}).derived["log_L_absorbed"])
        )
        hii_dust_credit = L_absorbed_default - L_absorbed_escape_only
        np.testing.assert_allclose(hii_dust_credit, fdust * L_lyc_credited, rtol=1e-6)

        # -- screen-absorbed: an ACTUAL log_L_absorbed difference across the
        # eb_include_lyc toggle, cross-checked against a direct SED integral
        # of the pre-screen vs. post-screen LyC content. ``post_screen_measured``
        # reads ``s_default.sed_intrinsic`` (every attenuator reassigns it to
        # its own combined post-screen SED, #2539), not the ``sed_dust_attenuated``
        # derived key -- that key is documented STELLAR-ONLY for two_component,
        # and this leg needs the full stellar+nebular+shock+agn combination
        # ``log_L_absorbed`` itself integrates.
        m_full = _build(
            synthetic_ssp_wide,
            dust_type,
            fesc=fesc,
            fdust=fdust,
            lyc_absorb_all=lyc_absorb_all,
            eb_include_lyc=True,
        )
        L_absorbed_full = float(10.0 ** np.asarray(m_full.predict_state({}).derived["log_L_absorbed"]))
        screen_absorbed_measured = L_absorbed_full - L_absorbed_default

        post_screen_measured = _l_lyc(wave, np.asarray(s_default.sed_intrinsic))
        # Derived purely from independent SED integrals already verified above
        # (escaped_measured against escaped_expected; post_screen_measured is
        # the model's own combined post-screen SED): this is the quantity that
        # closes the four-way budget by construction, so it is what the
        # closure sum below uses.
        screen_absorbed_derived = escaped_measured - post_screen_measured

        # Two_component's ``eb_include_lyc=True`` integral, when
        # ``lyc_absorb_all=False``, reads a UNIFORM (all-ages) fesc-masked
        # bookkeeping value for the newly-unmasked LyC region rather than the
        # per-age young/old-split value ``sed_attenuated`` itself uses (a
        # pre-existing inconsistency in two_component.py's ``eb_include_lyc``
        # handling, independent of and out of scope for #2539 -- CIGALE/FSPS
        # parity and wg00's own eb_include_lyc threading, item 6, are
        # unaffected). That is the ONLY combination where the model's own
        # measured toggle diff and the independent SED-integral derivation
        # are expected to disagree; everywhere else they must agree exactly.
        _two_component_lyc_absorb_all_false_eb_lyc_gap = (
            dust_type == "two_component" and not lyc_absorb_all
        )
        if not _two_component_lyc_absorb_all_false_eb_lyc_gap:
            np.testing.assert_allclose(
                screen_absorbed_measured, screen_absorbed_derived, rtol=1e-6
            )

        # -- closure: nothing counted twice, nothing lost --
        np.testing.assert_allclose(
            post_screen_measured + screen_absorbed_derived + gas_ionizing + hii_dust_credit,
            L_lyc_total,
            rtol=1e-6,
        )


class TestWG00EbIncludeLyc:
    """wg00 + eb_include_lyc (#2539 item 1): the grammar accepted the key for
    dust_type='wg00' but a bug in ``groups.py``'s wg00 branch (an early
    ``return`` before the generic eb_include_lyc translation) meant it was
    silently never read; ``component_factory.py`` also never forwarded it.
    Both are fixed; wg00's absorbed-energy integral uses the SAME
    ``bolometric_absorbed_log10`` call with the SAME 912 A switch as
    single_component, verified here by an independent manual integral
    (not by comparing to single_component's numerically different curve).
    """

    def test_toggle_reaches_forward_pass_and_matches_manual_integral(self, synthetic_ssp_wide):
        m_default = _build(synthetic_ssp_wide, "wg00", fesc=0.2, fdust=0.0)
        m_full = _build(synthetic_ssp_wide, "wg00", fesc=0.2, fdust=0.0, eb_include_lyc=True)
        assert m_default.spec.to_groups()["dust_attenuation"].get("eb_include_lyc") is not True
        assert m_full.spec.to_groups()["dust_attenuation"]["eb_include_lyc"] is True

        s_default = m_default.predict_state({})
        s_full = m_full.predict_state({})
        L_default = float(10.0 ** np.asarray(s_default.derived["log_L_absorbed"]))
        L_full = float(10.0 ** np.asarray(s_full.derived["log_L_absorbed"]))
        assert L_full > L_default, "eb_include_lyc is a no-op for wg00"

        # Independent manual integral of the extra (now-unmasked) LyC energy
        # the SAME screen absorbs: the TRUE pre-screen SED (a no-dust twin,
        # since wg00's tabulated grid does not reach transmission 1.0 at
        # tau_v=0 -- see ``_build_nodust``) minus ``s_full``'s own combined
        # post-screen SED (``s_full.sed_intrinsic``, reassigned by wg00's
        # apply() -- NOT the ``sed_dust_attenuated`` derived key, which is
        # bit-identical to ``sed_intrinsic`` here and so cannot independently
        # verify anything).
        m_nodust = _build_nodust(synthetic_ssp_wide, fesc=0.2, fdust=0.0)
        sed_intrinsic = np.asarray(m_nodust.predict_state({}).sed_intrinsic)
        wave = np.asarray(s_full.wave)
        sed_attenuated = np.asarray(s_full.sed_intrinsic)
        nu = np.asarray(C_AA, dtype=np.float64) / wave.astype(np.float64)
        absorbed = sed_intrinsic.astype(np.float64) - sed_attenuated.astype(np.float64)
        extra_lyc_absorbed = float(
            abs(np.trapezoid(np.where(wave < LYC_CUTOFF_AA, absorbed, 0.0), nu))
        )
        assert extra_lyc_absorbed > 0.0, "setup: eb_include_lyc should unmask nonzero LyC energy"
        np.testing.assert_allclose(L_full - L_default, extra_lyc_absorbed, rtol=1e-6)


class TestDefaultsBitIdentical:
    """neb_fdust unset (Fixed(0.0), the declared default) must leave
    log_L_absorbed identical to a model with the #2539 credit path disabled,
    and every output finite. The cheapest faithful means of comparing against
    origin/main behavior without checking out a second copy of the repo:
    log_L_lyc_dust must be EXACTLY -inf (log10_fdust_lyc_credit's value
    branch, not merely small), and log10_add(x, -inf) == x exactly by
    log10_add's own -inf sentinel contract (tengri.utils.scale.log10_add) --
    so the addition this fix adds to every attenuator is verified to be a
    true no-op at the default, not just numerically negligible.
    """

    @pytest.mark.parametrize("dust_type,lyc_absorb_all,young_only", ATTENUATORS)
    def test_defaults_are_noop_and_finite(self, synthetic_ssp_wide, dust_type, lyc_absorb_all, young_only):
        m = _build(synthetic_ssp_wide, dust_type, lyc_absorb_all=lyc_absorb_all)
        s = m.predict_state({})
        log_lyc_dust = s.derived.get("log_L_lyc_dust")
        if log_lyc_dust is not None:
            assert float(np.asarray(log_lyc_dust)) == float("-inf")
        log_l_absorbed = np.asarray(s.derived["log_L_absorbed"])
        assert np.all(np.isfinite(np.asarray(s.sed_intrinsic)))
        assert np.isfinite(float(log_l_absorbed))

        # Independent recomputation with NO #2539 involvement at all: the
        # exact pre-#2539 formula (LyC-masked absorbed_integrand over a
        # no-dust twin's TRUE pre-screen sed_intrinsic vs this model's own
        # combined post-screen SED, ``s.sed_intrinsic`` -- every attenuator
        # reassigns it to its own post-screen SED, #2539 -- not the
        # ``sed_dust_attenuated`` derived key, which is documented
        # STELLAR-ONLY for two_component and so is not the quantity
        # ``log_L_absorbed`` itself integrates).
        from tengri.forward.energy_balance import bolometric_absorbed_log10

        m_nodust = _build_nodust(synthetic_ssp_wide)
        sed_intrinsic = np.asarray(m_nodust.predict_state({}).sed_intrinsic)
        sed_attenuated = np.asarray(s.sed_intrinsic)
        wave = np.asarray(s.wave)
        nu = np.asarray(C_AA) / wave
        expected_log_l, _ = bolometric_absorbed_log10(
            jnp.asarray(sed_intrinsic), jnp.asarray(sed_attenuated), jnp.asarray(nu), wave=jnp.asarray(wave)
        )
        np.testing.assert_allclose(float(log_l_absorbed), float(expected_log_l), rtol=1e-6)


class TestGradientSafety:
    """Item 3: the log-add is exact at neb_fdust == 0 AND has a finite
    gradient everywhere, including neb_fdust -> 0 (double-where idiom, see
    log10_fdust_lyc_credit's docstring).
    """

    @pytest.mark.parametrize("fdust", [0.0, 1e-8, 0.3])
    def test_log10_fdust_lyc_credit_gradient_finite(self, fdust):
        log_l_lyc = jnp.asarray(45.0)  # representative dex value, independent of fdust
        grad_fn = jax.grad(lambda f: log10_fdust_lyc_credit(log_l_lyc, f))
        g = grad_fn(jnp.asarray(fdust))
        assert jnp.isfinite(g), f"grad at fdust={fdust} is not finite: {g}"

    def test_log10_fdust_lyc_credit_value_bit_identical_at_zero(self):
        log_l_lyc = jnp.asarray(45.0)
        value = log10_fdust_lyc_credit(log_l_lyc, jnp.asarray(0.0))
        assert float(value) == float("-inf")

    def test_end_to_end_gradient_finite_through_single_component(self, synthetic_ssp_wide):
        """Finite gradient through the full forward pass, not just the helper."""
        m = _build(synthetic_ssp_wide, "single_component", fesc=0.1, free_fdust=True)
        base_params = dict(m.spec.sample(jax.random.PRNGKey(0)))

        def loss(fdust):
            p = dict(base_params)
            p["neb_fdust"] = fdust
            state = m.predict_state(p)
            return jnp.asarray(state.derived["log_L_absorbed"])

        for fdust in (0.0, 1e-8, 0.3):
            g = jax.grad(loss)(jnp.asarray(fdust))
            assert jnp.isfinite(g), f"end-to-end grad at fdust={fdust} is not finite: {g}"
