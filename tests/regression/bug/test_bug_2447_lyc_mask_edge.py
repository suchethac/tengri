# SPDX-License-Identifier: BSD-3-Clause
"""The exact-path Lyman-continuum mask must step at LYMAN_LIMIT_AA (one edge).

Regression for #2447 (refs #2439, #2427). One Lyman edge (L3): the whole
model shares a single physical edge, ``LYMAN_LIMIT_AA`` = 911.76 A
(:mod:`tengri.components.lyc`), and a single step model for the grid cell
straddling it (last-ionizing-node value to the edge, first-non-ionizing-node
value from it -- never a linear ramp). ``NebularSEDComponent.apply`` masks
the stellar Lyman continuum with ``where(ionizing_mask(wave_rest), neb_fesc,
1)`` on the dense SSP wavelength grid (``components/nebular/component.py``);
masking per node is exact PROVIDED every consumer that separates or spans
the edge reads it through the step model rather than a plain trapezoid/
linear rule. The photometric band integral
(``observation.photometry._filter_integral_union``) integrates on the union
of the redshifted SSP grid and the filter's own wavelength table: once a
filter's rest-frame node spacing near the edge is finer than the SSP grid's
(true for real broadband curves like GALEX NUV, 0.33 A rest spacing at z=2,
against the MIST/C3K, h=1.83 A, or MILES, h=10 A, grids), a NAIVE linear
resample of the already-masked node values blends the step across that span
instead of stepping at the edge -- the #2447 defect, measured up to -2.0651%
/ +11.3813% on the issue's own model before any fix. The exact path now
resamples with :func:`tengri.components.lyc.edge_interp` (a step in the
bracket cell, ordinary linear interpolation elsewhere) and inserts the edge
into the quadrature grid as a zero-width node pair
(:func:`tengri.components.lyc.edge_bracket_values`) so an ordinary
trapezoid integrates it exactly -- no separate correction term, because the
masking (and any dust/IGM multiplicative factor) is already baked into
``state.sed_intrinsic`` per node; see ``lnu_filter_integral``'s
``has_lyc_edge`` kwarg and ``observation.photometry.project_photometry``.

Reference construction: an INDEPENDENT (does not call
:mod:`tengri.components.lyc` or :func:`tengri.observation.photometry._filter_integral_union`)
brute-force band flux, ``_reference_band_flux``, built directly from the
already-computed ``state.sed_intrinsic``/``state.wave`` of the SAME model
under test -- no second model build on a perturbed SSP grid. An earlier
version of this reference rebuilt the whole model (SFH, Q_H, mass
normalization, nebular, dust) on an SSP grid with the edge inserted as an
extra node pair; that construction is sensitive to how OTHER, unrelated
internal quadratures (Q_H, SFH age-weighting, mass normalization) react to
the perturbed node count, which measured as a several-1e-6 floor having
nothing to do with the filter integral itself. Building the reference from
the FINAL ``sed_intrinsic`` array instead isolates exactly the question
this file is about (does the filter integral treat the edge exactly) and
matches ``lnu_filter_integral(..., has_lyc_edge=True)`` to round-off in
every case measured (fesc in {0, 0.3, 1}, dust in {none, single, two
component}, z in {2, 3, 6}): fesc=1 is NOT a no-op for this check, because
the raw (pre-fesc) stellar continuum carries its own genuine break at the
edge (the #537 feature every LyC-adjacent quantity in
:mod:`tengri.components.lyc` shares), independent of nebular masking.

The synthetic fixture below uses a coarser top-hat filter (~11 A rest
spacing, matching
``tests/regression/bug/test_bug_2439_precomp_lyc_mask.py``'s own
``_straddle_filter``); the real-grid class exercises the finer, real GALEX
NUV / SDSS u tables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.components.lyc import LYMAN_LIMIT_AA
from tengri.observation.photometry import FilterCurve, lnu_filter_integral
from tengri.observation.spectroscopy import Spectroscopy
from tests._data_skip import requires_cue_weights
from tests.regression.bug.test_bug_2439_precomp_lyc_mask import _make_lyc_ssp

pytestmark = pytest.mark.regression_bug

LYC = float(LYMAN_LIMIT_AA)

# Measured (2026-09, L3): the independent reference matches
# ``lnu_filter_integral(..., has_lyc_edge=True)`` to ~1e-16 (float64
# round-off) on the synthetic fixture across every (fesc, dust, z)
# combination exercised below. rtol is set to 1e-9, ~1e7x looser than the
# measured floor (covers any platform/BLAS-ordering difference) while
# staying ~2500x tighter than the pre-L3 2.5e-6 (itself already far below
# the pre-#2447 defect, percent-scale).
_SYNTHETIC_RTOL = 1e-9


def _reference_band_flux(sed_rest, wave_rest, filter_wave_obs, filter_trans, z):
    r"""Independent brute-force step-model band flux (#2447/one Lyman edge, L3).

    Does not call :mod:`tengri.components.lyc` or
    :mod:`tengri.observation.photometry`: reimplements the same "insert the
    edge as a zero-width node pair, integrate with an ordinary trapezoid"
    construction from scratch, as a genuine second opinion on
    ``lnu_filter_integral(..., has_lyc_edge=True)``.

    Parameters
    ----------
    sed_rest : array_like, shape (n_wave,)
        Rest-frame ``state.sed_intrinsic`` [erg/s/Hz] -- already carries
        whatever masking (fesc<1 or fesc=1) and dust/IGM multiplicative
        factor the model under test applied, per node.
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom] (``state.wave``).
    filter_wave_obs : array_like, shape (n_filt,)
        Filter wavelength grid, observed frame [Angstrom].
    filter_trans : array_like, shape (n_filt,)
        Filter transmission (dimensionless).
    z : float
        Redshift.

    Returns
    -------
    float
        Filter-weighted rest-frame L_nu [erg/s/Hz] (BESSELL convention,
        ``w = 1/lambda``, matching ``lnu_filter_integral``'s default).
    """
    sed = np.asarray(sed_rest, dtype=np.float64)
    wave = np.asarray(wave_rest, dtype=np.float64)
    fw = np.asarray(filter_wave_obs, dtype=np.float64)
    ft = np.asarray(filter_trans, dtype=np.float64)
    wave_obs = wave * (1.0 + z)
    edge_obs = LYC * (1.0 + z)

    n_wave = wave.shape[0]
    n_ion = int(np.sum(wave < LYC))
    idx_a = int(np.clip(n_ion - 1, 0, n_wave - 1))
    idx_b = int(np.clip(n_ion, 0, n_wave - 1))
    y_a, y_b = sed[idx_a], sed[idx_b]
    wave_a_obs, wave_b_obs = wave_obs[idx_a], wave_obs[idx_b]

    def _edge_interp(x):
        linear = np.interp(x, wave_obs, sed, left=0.0, right=0.0)
        in_bracket = (x >= wave_a_obs) & (x <= wave_b_obs)
        step = np.where(x < edge_obs, y_a, y_b)
        return np.where(in_bracket, step, linear)

    grid0 = np.sort(np.concatenate([wave_obs, fw]))
    l_base = _edge_interp(grid0)
    all_x = np.concatenate([grid0, [edge_obs, edge_obs]])
    all_y = np.concatenate([l_base, [y_a, y_b]])
    order = np.argsort(all_x, kind="stable")
    grid = all_x[order]
    l_on_grid = all_y[order]

    trans_on_grid = np.interp(grid, fw, ft, left=0.0, right=0.0)
    weight = trans_on_grid / grid  # BESSELL: w = 1/lambda
    num = np.trapezoid(l_on_grid * weight, grid)
    den = np.trapezoid(weight, grid)
    return num / den


def _straddle_filter(z: float, frac: float = 0.35, n: int = 60, name: str = "straddle"):
    """Top-hat filter whose REST-frame footprint straddles LYMAN_LIMIT_AA at ANY z.

    Identical construction to ``test_bug_2439_precomp_lyc_mask.py``'s own
    ``_straddle_filter``: rest-frame node spacing ~LYC*2*frac/(n-1) = ~11 A,
    coarser than the synthetic SSP's own ~6.6 A spacing near the edge, so the
    filter's table does not subdivide the straddling SSP interval (the
    precondition the trapezoid-exact node fix needs -- see the module
    docstring).
    """

    center_obs = LYC * (1.0 + z)
    wave = jnp.linspace(center_obs * (1.0 - frac), center_obs * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


def _dust_group(dust: str | None) -> dict:
    if dust is None:
        return {"type": "none"}
    if dust == "single":
        return {
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": 0.4,
        }
    if dust == "two_component":
        return {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.5,
            "tau_diff": 0.3,
        }
    raise ValueError(dust)  # pragma: no cover - parametrize controls this


def _build(ssp, z, fesc, dust, igm, filters=None, wave_obs=None, resample="auto"):
    """Build the issue's model shape: delayed SFH, Cue nebular, optional dust/IGM."""
    igm_group = {"type": "none"} if igm is None else {"type": igm, "all_params": Fixed(DEFAULT)}
    if filters is not None:
        observation = Observation(photometry=Photometry(filters=tuple(filters)))
    else:
        observation = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs, resample=resample))
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 3.0,
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": 0.0, "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_fesc": Fixed(float(fesc))},
        dust_attenuation=_dust_group(dust),
        igm=igm_group,
        redshift=Fixed(float(z)),
        approx=None,
    )


@pytest.fixture(scope="module")
def lyc_ssp():
    return _make_lyc_ssp()


def _has_lyc_edge(state) -> bool:
    """Whether ``state`` carries a live per-node Lyman-continuum mask.

    Mirrors ``observation.photometry.project_photometry``'s own guard
    (#2447/one Lyman edge, L3): ``lyc_transmission`` is published by
    ``NebularSEDComponent`` whenever a photoionized backend ran, and that
    presence alone is enough for ``lnu_filter_integral``'s ``has_lyc_edge``
    -- there is nothing to reconstruct, since the masking (and any dust/IGM
    factor) is already baked into ``state.sed_intrinsic`` per node.
    """
    return state.derived is not None and state.derived.get("lyc_transmission") is not None


# ── Band flux: exact path vs. the independent brute-force reference ───────


@pytest.mark.parametrize("igm", [None, "inoue"])
@pytest.mark.parametrize("dust", [None, "single", "two_component"])
@pytest.mark.parametrize("z", [2.0, 3.0, 6.0])
@pytest.mark.parametrize("fesc", [0.0, 0.3, 1.0])
def test_band_flux_matches_split_at_edge(lyc_ssp, fesc, z, dust, igm):
    filt = _straddle_filter(z)
    m_exact = _build(lyc_ssp, z, fesc, dust, igm, filters=[filt])
    s_exact = m_exact.predict_state({})

    exact = float(
        lnu_filter_integral(
            s_exact.sed_intrinsic,
            s_exact.wave,
            filt.wave,
            filt.trans,
            z,
            has_lyc_edge=_has_lyc_edge(s_exact),
        )
    )
    ref = _reference_band_flux(s_exact.sed_intrinsic, s_exact.wave, filt.wave, filt.trans, z)

    assert ref != 0.0, "reference band flux is exactly zero -- fixture cannot discriminate"
    rel = abs(exact - ref) / abs(ref)
    assert rel < _SYNTHETIC_RTOL, (
        f"z={z} fesc={fesc} dust={dust} igm={igm}: exact={exact:.10e} vs "
        f"edge-exact reference={ref:.10e}, rel={rel:.3e} exceeds "
        f"{_SYNTHETIC_RTOL:.1e}"
    )


# ── Spectrum channel: pixels straddling the edge ────────────────────────────


def _step_model_nodes(wave, sed):
    """``(wave, sed)`` with the Lyman edge inserted as a zero-width node pair.

    The ionizing-side node value is held up to the edge and the non-ionizing
    one from it, so ordinary piecewise-linear interpolation of the returned
    arrays IS the step model (#2447). Independent of :mod:`tengri.components.lyc`.
    """
    k = int(np.searchsorted(wave, LYC))
    y_a, y_b = sed[k - 1], sed[k]
    return (
        np.concatenate([wave[:k], [LYC, LYC], wave[k:]]),
        np.concatenate([sed[:k], [y_a, y_b], sed[k:]]),
    )


def _reference_point(wave, sed, x):
    """Step-model point samples at ``x`` (the caller keeps every ``x`` off the edge itself)."""
    wn, sn = _step_model_nodes(wave, sed)
    k = int(np.searchsorted(wn, LYC))
    below = np.interp(x, wn[: k + 1], sn[: k + 1], left=0.0)
    above = np.interp(x, wn[k + 1 :], sn[k + 1 :], right=0.0)
    return np.where(x < LYC, below, above)


def _reference_pixel_mean(wave, sed, x):
    """Step-model mean over each pixel, edges at midpoints between centers."""
    wn, sn = _step_model_nodes(wave, sed)
    mid = 0.5 * (x[1:] + x[:-1])
    edges = np.concatenate([[x[0] - 0.5 * (x[1] - x[0])], mid, [x[-1] + 0.5 * (x[-1] - x[-2])]])
    out = np.empty(x.size)
    for j in range(x.size):
        lo, hi = edges[j], edges[j + 1]
        inner = (wn > lo) & (wn < hi)
        xs = np.concatenate([[lo], wn[inner], [hi]])
        ys = np.concatenate([[np.interp(lo, wn, sn)], sn[inner], [np.interp(hi, wn, sn)]])
        out[j] = np.trapezoid(ys, xs) / (hi - lo)
    return out


@pytest.mark.parametrize("resample", ["point", "conserving"])
@pytest.mark.parametrize("z", [2.0, 6.0])
def test_spectrum_pixels_straddling_edge(lyc_ssp, z, resample):
    """Every spectrum pixel reads the Lyman edge as the step model, in both resample modes.

    ``state.sed_intrinsic`` carries the Lyman-continuum mask per node, so the
    SED is a step at 911.76 A, not the linear ramp across the SSP cell that
    straddles it. Photometry integrates that step exactly (above); the
    spectrum projection must read the same step, whether it point-samples the
    model (``resample="point"``) or averages it over each pixel
    (``"conserving"``, the bin integral). The reference inserts the edge as a
    zero-width node pair and samples or integrates the result with ordinary
    numpy -- an implementation independent of :mod:`tengri.components.lyc`.
    Pixels are ~14 A wide in the rest frame, wider than the 6.5 A straddling
    cell, so a ramp misplaces up to ~20% of a pixel's flux next to the edge.
    """
    from tengri.cosmology import luminosity_distance
    from tengri.units import lnu_to_fnu

    # An even pixel count keeps every center off the edge, where a point sample
    # of a step is undefined (it reads either side at round-off); the pixel
    # holding the edge still straddles it in the bin integral.
    wave_obs = jnp.linspace(LYC * (1.0 + z) * 0.7, LYC * (1.0 + z) * 1.3, 40)
    m_exact = _build(lyc_ssp, z, 0.3, None, None, wave_obs=wave_obs, resample=resample)
    s_exact = m_exact.predict_state({})
    assert _has_lyc_edge(s_exact), "fixture bug: no Lyman-continuum mask to exercise"

    rest = np.asarray(wave_obs) / (1.0 + z)
    assert np.any(rest < LYC) and np.any(rest >= LYC), "fixture bug: pixels miss the edge"

    spec = np.asarray(m_exact.predict_spectrum({}))
    wave, sed = np.asarray(s_exact.wave), np.asarray(s_exact.sed_intrinsic)
    reference = _reference_point if resample == "point" else _reference_pixel_mean
    spec_ref = np.asarray(
        lnu_to_fnu(jnp.asarray(reference(wave, sed, rest)), float(luminosity_distance(z)), z)
    )

    nonzero = spec_ref != 0.0
    rel = np.abs((spec[nonzero] - spec_ref[nonzero]) / spec_ref[nonzero])
    assert np.max(rel) < _SYNTHETIC_RTOL, (
        f"z={z}, resample={resample}: worst pixel relative error {np.max(rel):.3e} "
        f"against the step-model reference"
    )


# ── Gradient finiteness and analytic match wrt neb_fesc ────────────────────


def test_gradient_wrt_fesc_finite_and_analytic(lyc_ssp):
    """d(band flux)/d(neb_fesc) is finite and matches central finite differences.

    Guards the affine (linear-in-fesc) construction of ``T_a``/``T_b``: no
    clamp or ``jnp.where`` branch should poison the gradient even though the
    straddling-interval geometry (``h_l``, ``h``, ``h_r``, ``s``) does not
    itself depend on ``fesc``. ``neb_fesc`` must be a FREE parameter here
    (not ``Fixed``, which is baked into the model at build time and carries
    no gradient path through ``predict_state``'s ``params`` argument) so the
    differentiation happens the way an inference backend actually uses it.
    """
    from tengri import Uniform

    z = 2.0
    filt = _straddle_filter(z)
    model = SEDModel.build(
        ssp_data=lyc_ssp,
        observation=Observation(photometry=Photometry(filters=(filt,))),
        sfh={
            "type": "delayed",
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 3.0,
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": 0.0, "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_fesc": Uniform(0.0, 1.0)},
        dust_attenuation={"type": "none"},
        igm={"type": "none"},
        redshift=Fixed(z),
        approx=None,
    )
    fesc_name = next(p for p in model.spec.free_params if p.endswith("neb_fesc"))

    def straddle_flux(fesc_value):
        state = model.predict_state({fesc_name: fesc_value})
        return lnu_filter_integral(
            state.sed_intrinsic,
            state.wave,
            filt.wave,
            filt.trans,
            z,
            has_lyc_edge=_has_lyc_edge(state),
        )

    for fesc0 in (0.1, 0.5, 0.9):
        grad = float(jax.grad(straddle_flux)(fesc0))
        assert np.isfinite(grad), f"fesc={fesc0}: gradient is not finite ({grad})"
        assert grad != 0.0, (
            f"fesc={fesc0}: gradient is identically zero -- the fesc correction "
            "is not wired into the differentiated path"
        )
        h = 1e-4
        fd = (straddle_flux(fesc0 + h) - straddle_flux(fesc0 - h)) / (2.0 * h)
        rel = abs(grad - float(fd)) / max(abs(float(fd)), 1e-300)
        assert rel < 1e-4, (
            f"fesc={fesc0}: analytic grad {grad:.6e} vs central-difference "
            f"{float(fd):.6e}, rel={rel:.3e}"
        )


# ── Real-grid class: the issue's own bands, on real data ───────────────────


@requires_cue_weights
class TestRealGridIssueRows:
    """The issue's own bands/redshifts on real SSP grids + real Cue weights.

    Unlike the synthetic fixture above, GALEX NUV's real filter table is far
    finer (rest-frame node spacing ~0.33 A at z=2) than either real grid's
    spacing near the edge (MIST/C3K h=1.83 A, MILES h=10 A), so the filter's
    own nodes subdivide the straddling SSP interval and a naive linear
    resample of the already-masked node values would blend the step across
    it instead of stepping at the edge.

    MAKE-IT-EXACT seam (photometry-side, #2447 follow-up; one Lyman edge,
    L3): rather than inserting the edge into the SSP grid itself (the
    whole-model change the module docstring's mechanism discussion once
    ruled out here), ``observation.photometry._filter_integral_union``
    resamples ``state.sed_intrinsic`` (already masked, whatever masking
    rule ran -- stellar AND nebular together, one array) with
    :func:`tengri.components.lyc.edge_interp` directly on ITS OWN (finer)
    union grid and integrates the step exactly via a zero-width edge node
    pair (:func:`tengri.components.lyc.edge_bracket_values`) -- no separate
    correction term, no unmasked SED needed, no re-derivation of the
    nebular continuum's own shape near the edge (it is part of the SAME
    ``sed_intrinsic`` array the step model applies to, not a distinct term).
    ``lnu_filter_integral``'s ``has_lyc_edge`` kwarg (built by this module's
    ``_has_lyc_edge`` helper) opts a caller into it; ``project_photometry``
    (the production forward path) does so automatically whenever a nebular
    component published ``lyc_transmission``.

    Measured (2026-09, L3) residual against ``_reference_band_flux`` (the
    independent, ``state.sed_intrinsic``-level reference): round-off
    (1e-16 to exactly 0.0) on real MILES data, WITH the issue's own Cue
    nebular backend (fesc=0, maximal reprocessing) and at fesc=1, down from
    the original (pre-#2447) -2.0651% / +11.3813% and a pre-L3 intermediate
    2-6e-4 (measured against the OLD, whole-model-rebuild reference, which
    could not isolate the nebular continuum's own near-edge structure from
    grid-perturbation side effects in Q_H/SFH-weighting elsewhere in that
    reference's construction -- see ``_reference_band_flux``'s docstring).
    MIST/C3K rows are present but typically skip (data not tracked in this
    repo checkout); MILES (``fsps_prsc_miles_chabrier.h5``) is git-tracked.
    """

    MILES_PATH = "fsps_prsc_miles_chabrier"
    MIST_C3K_PATH = "fsps_mist_c3k_a_chabrier"

    @staticmethod
    def _load(name):
        import tengri as _t

        try:
            return _t.load_ssp(name)
        except FileNotFoundError:
            pytest.skip(f"{name}.h5 not found in TENGRI_DATA_DIR")

    def _measure(self, ssp, z, filt_name, fesc, ceiling):
        filt = Photometry.from_names([filt_name]).filters[0]
        m_exact = _build(ssp, z, fesc, None, None, filters=[filt])
        s_exact = m_exact.predict_state({})
        exact = float(
            lnu_filter_integral(
                s_exact.sed_intrinsic,
                s_exact.wave,
                filt.wave,
                filt.trans,
                z,
                has_lyc_edge=_has_lyc_edge(s_exact),
            )
        )
        ref = _reference_band_flux(s_exact.sed_intrinsic, s_exact.wave, filt.wave, filt.trans, z)
        assert ref != 0.0, f"z={z} {filt_name}: reference flux is exactly zero"
        rel = abs(exact - ref) / abs(ref)
        assert rel < ceiling, (
            f"z={z} {filt_name} fesc={fesc}: exact-vs-edge-exact-reference "
            f"residual {rel * 100:.4f}% exceeds the {ceiling * 100:.2f}% "
            f"ratchet (filter-oversampling floor; see class docstring)."
        )
        return rel

    def test_miles_z2_galex_nuv(self):
        # Measured (2026-09, L3): 2.13e-16, round-off.
        ssp = self._load(self.MILES_PATH)
        self._measure(ssp, 2.0, "galex_nuv", 0.0, ceiling=1e-9)

    def test_miles_z3_sdss_u(self):
        # Measured: 0.0, round-off. See test_miles_z2_galex_nuv.
        ssp = self._load(self.MILES_PATH)
        self._measure(ssp, 3.0, "sdss_u", 0.0, ceiling=1e-9)

    def test_miles_fesc1_floor(self):
        """fesc=1 must ALSO collapse to round-off, not just the fesc<1 case.

        Unlike the pre-L3 reference, ``_reference_band_flux`` treats
        ``state.sed_intrinsic`` (stellar + nebular together) as the stepped
        quantity, matching what ``has_lyc_edge=True`` itself does -- so
        there is no separate "nebular continuum's own shape" residual to
        bound at fesc=1 either: measured 2.49e-16, round-off.
        """
        ssp = self._load(self.MILES_PATH)
        rel = self._measure(ssp, 2.0, "galex_nuv", 1.0, ceiling=1e-9)
        assert rel < 1e-9, f"fesc=1 floor {rel:.3e} not near round-off"

    def test_mist_c3k_z2_galex_nuv(self):
        # Round-off expected (see test_miles_z2_galex_nuv); skips if the
        # MIST/C3K grid is not present in this checkout.
        ssp = self._load(self.MIST_C3K_PATH)
        self._measure(ssp, 2.0, "galex_nuv", 0.0, ceiling=1e-9)

    def test_mist_c3k_z3_sdss_u(self):
        # See test_mist_c3k_z2_galex_nuv.
        ssp = self._load(self.MIST_C3K_PATH)
        self._measure(ssp, 3.0, "sdss_u", 0.0, ceiling=1e-9)
