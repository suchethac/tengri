# SPDX-License-Identifier: BSD-3-Clause
r"""``state.derived["line_lums"]`` is [erg/s] for **every** backend (#1559).

The published key declares its unit::

    DerivedKey("line_lums", "erg/s", "Line luminosities")

and :meth:`SEDModel.predict_line_fluxes` consumes it as such — it applies only
``1 / (4 pi d_L^2)``, with a comment recording that inserting an ``L_sun`` factor
there was "a 33.6-dex unit error that made every joint photometry+line-flux fit
unusable against real data".

Three of the four backends did not honor it. Cue returned [erg/s]; CB19,
CloudyGrid and MappingsPhoto returned [Lsun], and :class:`NebularSEDComponent`
applied no conversion to either — so their line fluxes came out a factor
``L_sun = 3.839e33`` too faint, silently, selected only by which backend the user
named.

**Why no existing test saw it.** Every nebular test checks one backend against
itself — ratios, monotonicity in ``fesc``, parity against that backend's own
upstream tabulation. All of those are invariant under a global scale, so a
33.6-dex error is exactly the defect they are blind to by construction. The
per-backend suites were all green.

**The invariant used here** is Case B recombination, which ties a line to the
ionizing budget that powers it and is therefore backend-independent::

    L(Halpha) = 1.37e-12 erg * Q_H          (Osterbrock & Ferland 2006, Table 4.4)

so ``L(Halpha) / Q_H`` is a *number with units of erg*, near 1e-12 for any
photoionized gas. Escape fractions, dust in the HII region, temperature and
density move it by well under a decade. A [Lsun]-for-[erg/s] mixup moves it by
33.6. The two cannot be confused.

Measured on this fixture before the fix: Cue -12.70 dex (0.84 below Case B, the
physics), CB19 **-48.03** (35.3 below). The band below is set from those
measurements, not from theory.
"""

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.contract

#: H-alpha, vacuum (CLAUDE.md: vacuum wavelengths throughout).
_HALPHA_AA = 6564.61

#: Case B, Osterbrock & Ferland (2006) Table 4.4, T = 1e4 K, n_e = 100 cm^-3.
_CASE_B_ERG = 1.37e-12

#: Accepted band for ``L(Halpha) / Q_H`` [erg]. Case B is 1.37e-12; the widest
#: *physical* departure measured here is Cue at 2.0e-13 (0.84 dex low, on a
#: synthetic SSP far outside its training range), so ~2.5 further decades of
#: slack costs nothing. A unit error lands at 3.6e-46 — thirty decades outside.
_RATIO_MIN, _RATIO_MAX = 1.0e-15, 1.0e-10

_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}
_BACKENDS = ("cue", "cb19")


def _model(ssp, backend):
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    scaled = SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * 1.0e-17,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )
    return SEDModel.build(
        ssp_data=scaled,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": backend, "all_params": Fixed(DEFAULT)},
    )


def _halpha_over_qh(ssp, backend):
    """``(L(Halpha) / Q_H`` [erg], matched wavelength [A])``, or None if no catalog."""
    with jax.enable_x64(True):
        derived = _model(ssp, backend).predict_state(_PARAMS).derived
        if "line_lums" not in derived:
            return None
        waves = np.asarray(derived["line_waves"], dtype=np.float64)
        lums = np.asarray(derived["line_lums"], dtype=np.float64)
        q_h = float(np.asarray(derived["nion"], dtype=np.float64))

    i = int(np.argmin(np.abs(waves - _HALPHA_AA)))
    assert abs(waves[i] - _HALPHA_AA) < 2.0, (
        f"{backend}: nearest line to H-alpha is {waves[i]:.2f} A — the catalog does not "
        "contain it, so this test is measuring some other species"
    )
    assert q_h > 0, f"{backend}: Q_H is {q_h}, so the ratio is meaningless"
    return lums[i] / q_h, waves[i]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_published_line_lums_are_erg_per_second(synthetic_ssp_wide, backend):
    """Every backend's published catalog must sit on the erg/s side of the seam."""
    result = _halpha_over_qh(synthetic_ssp_wide, backend)
    if result is None:
        pytest.skip(f"{backend} publishes no discrete line catalog")
    ratio, _ = result

    assert _RATIO_MIN < ratio < _RATIO_MAX, (
        f"{backend}: L(H-alpha)/Q_H = {ratio:.3e} erg ({np.log10(ratio):.2f} dex), outside "
        f"[{_RATIO_MIN:.0e}, {_RATIO_MAX:.0e}]. Case B is {_CASE_B_ERG:.2e} "
        f"({np.log10(_CASE_B_ERG):.2f} dex). Being low by ~33.6 dex means this backend is "
        "publishing [Lsun] into a key whose DerivedKey declares [erg/s], so every line "
        "flux it produces is a factor L_sun = 3.839e33 too faint (#1559)"
    )


def test_every_backend_declares_its_own_lsun_convention():
    r"""The conversion constant is the backend's, not a global — and 0.287% proves it.

    "One seam, one constant" is the obvious design and it is wrong here. Cue's
    network was trained against ``L_sun = 3.839e33``; the CLOUDY/CB19/MAPPINGS
    grids are tabulated against IAU 2015's ``3.828e33``. Applying either value
    to the other backend puts a **systematic 0.287%** on every one of its lines.

    That is 2.5 decades inside the units band above, so
    :func:`test_published_line_lums_are_erg_per_second` cannot see it — it was
    caught by measuring Cue's H-alpha before and after the refactor and noticing
    it had moved when it should have been bit-identical.

    ``NebularSEDComponent`` reads ``backend.lsun_erg`` with a ``getattr``
    default, so a third-party backend still works. This test makes that default
    unreachable for the in-tree four, which is the only place it could do harm
    silently.
    """
    from tengri.components.nebular.cloudy_cb19 import CB19Backend
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.components.nebular.cue import CueBackend
    from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
    from tengri.utils.physics_constants import L_SUN, L_SUN_CUE

    expected = {
        CueBackend: L_SUN_CUE,
        CB19Backend: L_SUN,
        CloudyGridBackend: L_SUN,
        MappingsPhotoStellarBackend: L_SUN,
    }
    for cls, want in expected.items():
        declared = cls.__dict__.get("lsun_erg")
        assert declared is not None, (
            f"{cls.__name__} does not declare lsun_erg on the class itself, so the "
            "component falls back to the IAU default. For a backend tabulated in a "
            "different convention that is a silent systematic error (#1559)"
        )
        assert declared == want, (
            f"{cls.__name__}.lsun_erg is {declared:.4e}, expected {want:.4e} — a "
            f"{abs(declared / want - 1) * 100:.3f}% systematic on every line it publishes"
        )

    assert L_SUN_CUE != L_SUN, (
        "L_SUN_CUE now equals the IAU value, so this test proves nothing. If Cue was "
        "retrained on IAU 2015, delete the override rather than the test"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_component_actually_reads_the_backends_constant(synthetic_ssp_wide, backend):
    """Declaring ``lsun_erg`` is worthless if the component ignores it.

    :func:`test_every_backend_declares_its_own_lsun_convention` checks the
    classes; this checks the *wiring*, by overriding the attribute on a live
    instance and requiring the published luminosities to move by exactly that
    factor. A component that hardcoded a module-level constant would produce an
    unchanged catalog and fail here — which is the state this file was written
    against, where the ratio would come back 1.0 instead of 3.839e33.
    """
    with jax.enable_x64(True):
        model = _model(synthetic_ssp_wide, backend)
        instance = model._nebular_backend
        declared = float(instance.lsun_erg)

        published = np.asarray(model.predict_state(_PARAMS).derived["line_lums"], dtype=np.float64)
        # Instance attribute shadows the class attribute; the model is discarded
        # after this test, so nothing else sees it.
        instance.lsun_erg = 1.0
        unscaled = np.asarray(model.predict_state(_PARAMS).derived["line_lums"], dtype=np.float64)

    positive = (unscaled > 0) & np.isfinite(unscaled) & np.isfinite(published)
    assert positive.any(), f"{backend}: no usable lines to compare"

    recovered = published[positive] / unscaled[positive]
    assert np.allclose(recovered, declared, rtol=1e-12), (
        f"{backend}: overriding backend.lsun_erg changed the published line_lums by "
        f"{np.median(recovered):.4e}, not the declared {declared:.4e}. The component is "
        "not reading the backend's constant — so a backend tabulated in a non-IAU "
        "convention (Cue) is silently converted with the wrong one (#1559)"
    )


def test_agn_nlr_cue_converts_between_the_two_lsun_conventions():
    r"""The AGN NLR path crosses the same 0.287% seam, in the other direction.

    :func:`~tengri.components.nebular.agn_nebular.agn_nlr_cue` promises [Lsun]
    **IAU 2015** — the unit its Feltre and Synthesizer siblings return and the
    one its consumers multiply back out by. Cue's catalog is [Lsun] in *Cue's*
    convention. So the function owes a ``L_SUN_CUE / L_SUN`` conversion::

        nlr_out == raw_cue_catalog * covering_fraction * (L_SUN_CUE / L_SUN)

    Before #1559 that arrived as ``* L_SUN_CUE`` inside the backend and
    ``/ L_SUN`` here. Moving the first half to the component made the second
    half *look* like a leftover, and deleting it biased every NLR line low by
    0.287% — which #1073's bound test (lines cannot outshine the accretion
    luminosity that powers them) is orders of magnitude too loose to notice.
    """
    from pathlib import Path

    weights = Path(__file__).resolve().parents[2] / "data" / "cue_weights.npz"
    if not weights.exists():
        pytest.skip("Cue weights not found at data/cue_weights.npz")

    from tengri.components.nebular.agn_nebular import agn_nlr_cue
    from tengri.components.nebular.cue import CueBackend
    from tengri.utils.physics_constants import L_SUN, L_SUN_CUE

    backend = CueBackend(str(weights))
    covering, l_acc, alpha = 0.1, 1.0e45, -1.7
    gas = dict(gas_logn=3.0, gas_logz=0.0, gas_logno=0.0, gas_logco=0.0)

    with jax.enable_x64(True):
        _, nlr = agn_nlr_cue(
            backend,
            l_acc_erg=l_acc,
            covering_fraction=covering,
            alpha_pl=alpha,
            neb_logU=-3.0,
            **gas,
        )
        nlr = np.asarray(nlr, dtype=np.float64)

    assert np.isfinite(nlr).any() and (nlr > 0).any(), "no usable NLR lines to check"

    # Independent reconstruction: what the function should have returned, built
    # from the constants rather than from the function's own arithmetic.
    ratio_expected = L_SUN_CUE / L_SUN
    assert abs(ratio_expected - 1.0) > 1e-4, (
        "L_SUN_CUE and L_SUN now agree, so this test cannot fail. If Cue was retrained "
        "on IAU 2015, delete the conversion rather than the test"
    )

    # Re-derive from Cue directly at the *same* ionizing input — same Q_H and
    # the same alpha_pl-derived spectrum shape — so the only thing left between
    # the two numbers is the constant chain. Omitting ionspec here would let the
    # backend fall back to its defaults and the ratio would measure physics, not
    # units.
    from tengri.components.nebular.agn_nebular import (
        _log_qh_from_lacc,
        agn_ionspec_from_alpha_pl,
    )

    with jax.enable_x64(True):
        _, raw = backend.predict_nebular_line_luminosities(
            gas_logu=-3.0,
            gas_logqion=_log_qh_from_lacc(l_acc, alpha),
            **gas,
            **agn_ionspec_from_alpha_pl(alpha),
        )
        raw = np.asarray(raw, dtype=np.float64)

    usable = (raw > 0) & np.isfinite(raw) & np.isfinite(nlr)
    assert usable.any(), "no overlapping finite lines"
    recovered = nlr[usable] / (raw[usable] * covering)

    assert np.allclose(recovered, ratio_expected, rtol=1e-9), (
        f"agn_nlr_cue scales Cue's catalog by {np.median(recovered):.9f}, expected "
        f"{ratio_expected:.9f} = L_SUN_CUE/L_SUN. A bare {np.median(recovered):.3f} means "
        "the conversion between Cue's L_sun and IAU's was dropped, biasing every NLR "
        "line by 0.287% (#1559 / #1073)"
    )


def test_the_backends_agree_with_each_other_on_the_unit(synthetic_ssp_wide):
    """The cross-backend check the per-backend suites cannot make.

    Absolute luminosities legitimately differ between backends — different
    photoionization codes, and a synthetic SSP outside everyone's training range.
    The *ratio to Q_H* is the part that must not differ by decades.
    """
    ratios = {}
    for backend in _BACKENDS:
        result = _halpha_over_qh(synthetic_ssp_wide, backend)
        if result is not None:
            ratios[backend] = result[0]

    assert len(ratios) >= 2, (
        f"only {len(ratios)} backend(s) published a line catalog ({sorted(ratios)}), so "
        "there is nothing to cross-check — this test is vacuous. Fix the fixture rather "
        "than deleting it"
    )

    spread = np.log10(max(ratios.values())) - np.log10(min(ratios.values()))
    assert spread < 3.0, (
        f"backends disagree on L(H-alpha)/Q_H by {spread:.1f} dex: "
        f"{ {k: f'{np.log10(v):.2f} dex' for k, v in ratios.items()} }. Photoionization "
        "physics does not span that; a units mismatch at the publish seam does (#1559)"
    )
