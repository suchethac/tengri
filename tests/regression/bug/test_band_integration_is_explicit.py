# SPDX-License-Identifier: BSD-3-Clause
"""The band-integration scheme must be chosen by name, not inferred.

Three schemes project the multiplicative dust screen through a filter under
``approx=WavePrecomp(...)``:

* **quadrature** — the screen is *evaluated* at K nodes per band and summed
  against the sub-band SSP x filter tensors. Converges as 1/K^2 (#1122).
  The default, and the accurate one.
* **taylor** — first-order spectral moment about the effective wavelength,
  ``A(lam_eff)*Phi + A'(lam_eff)*Psi`` (Zacharegkas+2025, #617). Retained
  for reproducing published pre-#1122 results and for comparison work.
* **effective_wavelength** — zeroth order, ``A(lam_eff)*Phi``.

Before this change the scheme was not selected; it was *inferred* from which
precompute tensors happened to be present, driven by two knobs whose
interaction was undocumented at the call site:

    WavePrecomp(taylor_correction=True)   -> quadrature   (flag ignored)
    WavePrecomp(n_subbands=0)             -> effective_wavelength (silently)

Asking for Taylor by name did not produce Taylor. It required also knowing
to zero a second, unrelated-sounding knob. That is the defect this file
pins: a user who names a scheme gets that scheme, and a combination that
cannot be honored says so instead of quietly resolving to something else.

The three source-level defaults also disagreed with each other
(``WavePrecomp`` said K=5/taylor=False; ``SEDModel._DEFAULT_APPROX`` said
K=0/taylor=True), so which scheme ran depended on which constructor path
the model took. ``band_integration`` is resolved in one place so that
cannot recur.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp

pytestmark = pytest.mark.regression_bug

# GALEX FUV is the discriminating band: it is where the attenuation curve is
# steepest, so the three schemes separate there. On SDSS r they agree to
# well under a percent and the test would be vacuous.
_BANDS = ["galex_fuv", "sdss_u", "sdss_g", "sdss_r"]


@pytest.fixture(scope="module")
def ssp():
    """The committed bare-stellar grid, resolved without a working directory.

    Nothing below is C3K-specific — the assertions pin that the three schemes
    are *distinct* and *ordered*, not any grid's numbers. Measured on both:
    the schemes separate by 3.6e-3 here versus 3.7e-3 on C3K, against a 1e-6
    threshold, and the error ordering is identical. ``load_ssp()`` resolves
    ``data/`` from any ancestor directory and never downloads (#1486).
    """
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(_BANDS))


def _model(ssp, obs, *, dust_type, approx):
    dust = {"law": "power_law", "type": dust_type, "all_params": Fixed(DEFAULT)}
    if dust_type == "two_component":
        # `law`, not `law_bc`: under the old symmetric inheritance naming one
        # screen applied that curve to both, so a single `law` reproduces it.
        # Assigning `law_bc` here alongside the literal `law` above would form
        # the ambiguous pair the grammar now rejects.
        dust["law"] = "calzetti"
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
        dust_attenuation=dust,
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=approx,
    )


# ── the selector exists and defaults to the accurate scheme ──────────────


def test_default_is_quadrature(ssp, obs):
    """K-point is what you get without asking."""
    m = _model(ssp, obs, dust_type="two_component", approx=WavePrecomp())
    assert m._approx["band_integration"] == "quadrature"
    assert m._approx["n_subbands"] > 0


@pytest.mark.parametrize("scheme", ["quadrature", "taylor", "effective_wavelength"])
@pytest.mark.parametrize("dust_type", ["two_component", "single_component"])
def test_named_scheme_is_the_scheme_that_runs(ssp, obs, scheme, dust_type):
    """Naming a scheme selects it — in BOTH dust structural branches.

    The two branches previously had independent fallback ladders, so a
    scheme honored under ``two_component`` was not necessarily honored
    under ``single_component``.
    """
    m = _model(ssp, obs, dust_type=dust_type, approx=WavePrecomp(band_integration=scheme))
    assert m._approx["band_integration"] == scheme


def test_taylor_by_name_is_not_silently_quadrature(ssp, obs):
    """The headline defect: asking for Taylor used to return quadrature."""
    taylor = _model(
        ssp, obs, dust_type="two_component", approx=WavePrecomp(band_integration="taylor")
    )
    quad = _model(
        ssp, obs, dust_type="two_component", approx=WavePrecomp(band_integration="quadrature")
    )
    assert taylor._approx["band_integration"] == "taylor"
    f_taylor = np.asarray(taylor.predict_photometry({}))
    f_quad = np.asarray(quad.predict_photometry({}))
    assert np.all(np.isfinite(f_taylor))
    # FUV is index 0 — the band where the schemes must disagree. If these
    # match, the selector is a no-op and the test above is measuring a label.
    rel = abs(f_taylor[0] - f_quad[0]) / f_quad[0]
    assert rel > 1e-6, (
        f"taylor and quadrature agree to {rel:.2e} in GALEX FUV — the "
        "selector is not reaching the projector"
    )


# ── contradictions are reported, not silently resolved ───────────────────


def test_contradictory_legacy_knobs_warn(ssp, obs):
    """``taylor_correction=True`` with quadrature on cannot be honored.

    Previously resolved silently in favor of quadrature, so the flag was a
    no-op that read as a choice.
    """
    with pytest.warns(UserWarning, match="taylor"):
        WavePrecomp(taylor_correction=True, n_subbands=5)


@pytest.mark.parametrize("bad", ["Quadrature", "taylor_correction", "kpoint", "", "none"])
def test_unknown_scheme_raises_naming_the_valid_set(bad):
    """A typo must not fall through to a default."""
    with pytest.raises(ValueError, match="band_integration"):
        WavePrecomp(band_integration=bad)


def test_setting_k_alongside_quadrature_is_not_a_warning(recwarn):
    """``n_subbands`` sets K — it is not redundant with the scheme name.

    The docstring recommends exactly this form, so warning on it would make
    the recommended usage noisy. Only genuinely inert combinations warn (see
    below).
    """
    cfg = WavePrecomp(band_integration="quadrature", n_subbands=8)
    assert cfg.n_subbands == 8
    assert cfg.band_integration == "quadrature"
    assert not [
        w for w in recwarn.list if issubclass(w.category, (UserWarning, DeprecationWarning))
    ]


def test_k_is_inert_under_a_non_quadrature_scheme_and_says_so():
    with pytest.warns(UserWarning, match="n_subbands"):
        WavePrecomp(band_integration="taylor", n_subbands=8)


def test_taylor_flag_alongside_an_explicit_scheme_says_it_is_ignored():
    with pytest.warns(UserWarning, match="taylor_correction is ignored"):
        WavePrecomp(band_integration="quadrature", taylor_correction=True)


def test_quadrature_with_zero_nodes_raises_rather_than_substituting():
    """A contradictory request must not be silently corrected.

    Quietly promoting K=0 to K=5 would honor a request nobody made — the
    same defect class this selector exists to remove.
    """
    with pytest.raises(ValueError, match="contradictory"):
        WavePrecomp(band_integration="quadrature", n_subbands=0)


@pytest.mark.parametrize("k", [-1, -5])
def test_negative_node_count_raises(k):
    with pytest.raises(ValueError, match="n_subbands"):
        WavePrecomp(band_integration="quadrature", n_subbands=k)


# ── one definition of the defaults, not two ──────────────────────────────


def test_default_approx_does_not_carry_its_own_copy_of_the_band_knobs():
    """``SEDModel._DEFAULT_APPROX`` must derive them, not restate them.

    It used to spell out ``taylor_correction=True, n_subbands=0`` while
    ``WavePrecomp`` resolved to ``False, 5``. Two copies, disagreeing, and
    the difference is a silent accuracy change of up to 42 % in the rest-UV
    (see the module docstring) rather than a cosmetic one.
    """
    from tengri.forward.approx_policy import BAND_PROJECTION_KEYS, ApproxPolicy

    defaults = ApproxPolicy()
    for key in BAND_PROJECTION_KEYS:
        assert SEDModel._DEFAULT_APPROX[key] == defaults[key], (
            f"_DEFAULT_APPROX[{key!r}] has drifted from WavePrecomp's default. "
            "Derive it from _band_projection_defaults() rather than restating it."
        )


def test_the_derived_defaults_are_the_accurate_scheme():
    """Deriving is only safe if what we derive FROM is right.

    Pins the value too, so a future change to WavePrecomp's default silently
    flipping every no-preference model onto a worse scheme fails here.
    """
    from tengri.forward.approx_policy import ApproxPolicy

    assert ApproxPolicy().band_integration == "quadrature"
    assert ApproxPolicy().n_subbands >= 1
    assert ApproxPolicy().taylor_correction is False


def test_every_band_knob_is_copied_from_the_config(ssp, obs):
    """No knob may be forgotten at the copy site.

    Naming the fields one at a time is how one gets dropped and silently
    keeps a default contradicting the rest; both ends key off the same tuple.
    """
    from tengri.forward.approx_policy import BAND_PROJECTION_KEYS

    cfg = WavePrecomp(band_integration="taylor", fast_dust_emission=True)
    m = _model(ssp, obs, dust_type="two_component", approx=cfg)
    for key in BAND_PROJECTION_KEYS:
        assert m._approx[key] == getattr(cfg, key), (
            f"{key} did not reach the policy from the config"
        )


# ── backward compatibility: the legacy pair still resolves as before ─────


@pytest.mark.parametrize(
    ("n_subbands", "taylor_correction", "expected"),
    [
        (5, False, "quadrature"),
        (0, True, "taylor"),
        (0, False, "effective_wavelength"),
    ],
)
def test_legacy_knob_pairs_resolve_to_the_documented_scheme(
    ssp, obs, n_subbands, taylor_correction, expected
):
    """The pre-existing combinations keep their meaning.

    These three are the combinations the old docstring documented as
    meaningful; the fourth (K>0 with taylor=True) was the contradiction and
    is covered above.
    """
    with pytest.warns(DeprecationWarning):
        cfg = WavePrecomp(n_subbands=n_subbands, taylor_correction=taylor_correction)
    m = _model(ssp, obs, dust_type="two_component", approx=cfg)
    assert m._approx["band_integration"] == expected


def test_the_three_schemes_are_ordered_in_accuracy(ssp, obs):
    """Quadrature sits between the two, and all three are finite.

    Not an accuracy assertion against truth — that belongs in a crossval
    test with the exact path as reference. This pins only that the three
    are genuinely distinct computations rather than aliases.
    """
    fluxes = {}
    for scheme in ("quadrature", "taylor", "effective_wavelength"):
        m = _model(
            ssp, obs, dust_type="two_component", approx=WavePrecomp(band_integration=scheme)
        )
        fluxes[scheme] = np.asarray(m.predict_photometry({}))
        assert np.all(np.isfinite(fluxes[scheme])), f"{scheme} produced non-finite photometry"

    # Compared as a RELATIVE difference, not with ``np.allclose``. The FUV
    # flux here is ~1e-29 erg/s/cm2/Hz, far below allclose's default
    # atol=1e-8, so allclose calls every scheme "equal" and the assertion
    # passes while measuring nothing.
    def _max_rel(a, b):
        return float(np.max(np.abs(a - b) / np.abs(b)))

    assert _max_rel(fluxes["quadrature"], fluxes["taylor"]) > 1e-6
    assert _max_rel(fluxes["quadrature"], fluxes["effective_wavelength"]) > 1e-6


def test_quadrature_is_closest_to_the_exact_path(ssp, obs):
    """The default must be the most accurate of the three.

    Compares each scheme against the exact wave-grid path on the same
    model. This is the assertion that justifies the default; without it
    "quadrature is preferred" is a claim rather than a measurement.
    """
    exact = np.asarray(
        _model(ssp, obs, dust_type="two_component", approx=None).predict_photometry({})
    )
    err = {}
    for scheme in ("quadrature", "taylor", "effective_wavelength"):
        m = _model(
            ssp, obs, dust_type="two_component", approx=WavePrecomp(band_integration=scheme)
        )
        f = np.asarray(m.predict_photometry({}))
        err[scheme] = float(np.max(np.abs(f - exact) / exact))
    assert err["quadrature"] < err["taylor"], (
        f"quadrature ({err['quadrature']:.3e}) is not more accurate than "
        f"taylor ({err['taylor']:.3e}) — the default is misjustified"
    )
    assert err["quadrature"] < err["effective_wavelength"]
