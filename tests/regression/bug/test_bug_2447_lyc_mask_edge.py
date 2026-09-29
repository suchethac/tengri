# SPDX-License-Identifier: BSD-3-Clause
"""The exact-path Lyman-continuum mask must step at 912 A, not at a grid node.

Regression for #2447 (refs #2439, #2427). ``NebularSEDComponent.apply``
masks the stellar Lyman continuum with ``where(wave_rest < 912, neb_fesc, 1)``
on the dense SSP wavelength grid (``components/nebular/component.py``). The
SSP grid has no node exactly at 912 A -- FSPS MIST/C3K brackets it at
911.5716/913.3967 A (h = 1.8256 A) -- so the step lands on whichever node
sits just below 912 A, and a trapezoid band integral over the masked
spectrum ramps transmission linearly across that interval instead of
stepping at the physical edge. Measured on the issue's own model (delayed
SFH, log M=10, tau=1 Gyr, age=3 Gyr, solar, Cue nebular default
``neb_fesc=0``, no dust, z=2, GALEX NUV, ``fsps_mist_c3k_a_chabrier``):
exact path -2.0651% against a reference with 912 A inserted as a grid node,
while the #2439 LUT's algebraic split is exact (+0.0000%).

``_lyman_edge_transmission`` (``components/nebular/component.py``) replaces
the two nodes bracketing 912 A with the trapezoid-exact split weights
derived in its own docstring, leaving every other node's plain step
untouched. That derivation is exact for a SINGLE trapezoid panel over the
straddling interval; it is not exact when a consumer's own quadrature grid
places EXTRA nodes strictly inside that (tiny, ~2-10 A) interval, because
the corrected node values are then linearly interpolated across it rather
than stepping at 912 A. The dense-path band integral
(``observation.photometry._filter_integral_union``) integrates on the union
of the redshifted SSP grid and the filter's own wavelength table, so this
happens whenever a filter's rest-frame node spacing near 912 A is finer than
the SSP grid's -- true for real broadband curves like GALEX NUV (0.33 A rest
spacing at z=2) against the MIST/C3K (h=1.83 A) or MILES (h=10 A) grids.
The synthetic fixture below uses a coarser top-hat filter (~11 A rest
spacing, matching ``tests/regression/bug/test_bug_2439_precomp_lyc_mask.py``'s
own ``_straddle_filter``) so the single-panel exactness claim can be
verified to a tight tolerance; the real-grid class documents the larger,
measured residual this filter-oversampling effect leaves on GALEX NUV /
SDSS u specifically, and ratchets it rather than asserting the brief's
aspirational 1e-4 (see that class's docstring for the measurement).

Reference construction: rather than algebraically reconstructing the split
(error-prone once dust and IGM are layered on), ``_augment_ssp_at_edge``
inserts 912 A into the SSP's OWN wavelength grid as two coincident-limit
nodes (912- carrying the unmasked flux, 912+ the same unmasked flux -- the
LyC step is then applied automatically and exactly by whichever masking
rule is live, since there is no more straddling interval to misplace), with
every per-(age, metallicity) flux value linearly interpolated at 912 A from
the original grid. Running the FULL model (same SFH, dust, IGM, nebular
backend, fesc) on this augmented grid gives a black-box reference that
handles dust attenuation and IGM absorption exactly like the model under
test, with no separate re-derivation of either.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve, lnu_filter_integral
from tengri.observation.spectroscopy import Spectroscopy
from tests._data_skip import requires_cue_weights
from tests.regression.bug.test_bug_2439_precomp_lyc_mask import _make_lyc_ssp

pytestmark = pytest.mark.regression_bug

LYC = 912.0

# Measured (2026-09) worst-case relative residual between the exact path and
# the augmented-grid reference on the synthetic fixture, at fesc=0 (the
# largest departure from a no-op), across z in {2, 3, 6}: 1.31e-6. This is
# NOT purely the node-wise fix's own error -- comparing the SAME model built
# on the original vs. the augmented grid with the mask disabled entirely
# (fesc=1, or no nebular component at all) already shows a ~2-6e-7 floor
# from the extra grid nodes alone shifting quadrature rounding elsewhere in
# the pipeline (SFH-weighted age integration, mass normalization). rtol is
# set to 2.5e-6, comfortably covering the measured 1.31e-6 worst case with
# margin while staying far below the pre-fix defect (percent-scale).
_SYNTHETIC_RTOL = 2.5e-6


def _augment_ssp_at_edge(ssp: SSPData, eps: float = 1e-6) -> SSPData:
    """Insert 912 A into ``ssp``'s wavelength grid as two coincident-limit nodes.

    Parameters
    ----------
    ssp : SSPData
        Source SSP grid.
    eps : float, optional
        Separation [Angstrom] between the two inserted nodes. Must be
        nonzero (``jnp.interp`` requires strictly-ascending nodes) and far
        smaller than the grid's own spacing near 912 A; the reference is
        insensitive to its exact value once it is this small (verified by
        an eps sweep from 1e-2 to 1e-8 A during development: the residual
        against the exact path converges and stays flat below ~1e-5 A).

    Returns
    -------
    SSPData
        A new grid with ``ssp_wave`` two elements longer: ``..., lambda_a,
        912-eps, 912+eps, lambda_b, ...`` where ``lambda_a < 912 <=
        lambda_b`` in the original grid. ``ssp_flux`` gains a matching pair
        of columns, both equal to the ORIGINAL grid's own linear
        interpolation of the flux at 912 A (so the two inserted nodes carry
        the same unmasked value -- any LyC step is then applied by whichever
        masking rule runs on this grid, not baked in here).
    """
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)  # (n_met, n_age, n_wave)
    insert_at = int(np.searchsorted(wave, LYC))
    new_wave = np.concatenate([wave[:insert_at], [LYC - eps, LYC + eps], wave[insert_at:]])
    f_edge = np.apply_along_axis(lambda f: np.interp(LYC, wave, f), -1, flux)  # (n_met, n_age)
    new_flux = np.concatenate(
        [flux[..., :insert_at], f_edge[..., None], f_edge[..., None], flux[..., insert_at:]],
        axis=-1,
    )

    return SSPData(
        ssp_wave=jnp.asarray(new_wave),
        ssp_flux=jnp.asarray(new_flux),
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _straddle_filter(z: float, frac: float = 0.35, n: int = 60, name: str = "straddle"):
    """Top-hat filter whose REST-frame footprint straddles 912 A at ANY z.

    Identical construction to ``test_bug_2439_precomp_lyc_mask.py``'s own
    ``_straddle_filter``: rest-frame node spacing ~912*2*frac/(n-1) = ~11 A,
    coarser than the synthetic SSP's own ~6.6 A spacing near 912 A, so the
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


def _build(ssp, z, fesc, dust, igm, filters=None, wave_obs=None):
    """Build the issue's model shape: delayed SFH, Cue nebular, optional dust/IGM."""
    igm_group = {"type": "none"} if igm is None else {"type": igm, "all_params": Fixed(DEFAULT)}
    if filters is not None:
        observation = Observation(photometry=Photometry(filters=tuple(filters)))
    else:
        observation = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))
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


@pytest.fixture(scope="module")
def lyc_ssp_edge(lyc_ssp):
    return _augment_ssp_at_edge(lyc_ssp)


# ── Band flux: exact path vs. the augmented-grid (edge-exact) reference ────


@pytest.mark.parametrize("igm", [None, "inoue"])
@pytest.mark.parametrize("dust", [None, "single", "two_component"])
@pytest.mark.parametrize("z", [2.0, 3.0, 6.0])
@pytest.mark.parametrize("fesc", [0.0, 0.3, 1.0])
def test_band_flux_matches_split_at_edge(lyc_ssp, lyc_ssp_edge, fesc, z, dust, igm):
    filt = _straddle_filter(z)
    m_exact = _build(lyc_ssp, z, fesc, dust, igm, filters=[filt])
    m_ref = _build(lyc_ssp_edge, z, fesc, dust, igm, filters=[filt])

    s_exact = m_exact.predict_state({})
    s_ref = m_ref.predict_state({})

    exact = float(
        lnu_filter_integral(s_exact.sed_intrinsic, s_exact.wave, filt.wave, filt.trans, z)
    )
    ref = float(lnu_filter_integral(s_ref.sed_intrinsic, s_ref.wave, filt.wave, filt.trans, z))

    assert ref != 0.0, "reference band flux is exactly zero -- fixture cannot discriminate"
    rel = abs(exact - ref) / abs(ref)
    assert rel < _SYNTHETIC_RTOL, (
        f"z={z} fesc={fesc} dust={dust} igm={igm}: exact={exact:.10e} vs "
        f"edge-exact reference={ref:.10e}, rel={rel:.3e} exceeds "
        f"{_SYNTHETIC_RTOL:.1e}"
    )


# ── Spectrum channel: pixels straddling the edge ────────────────────────────


@pytest.mark.parametrize("z", [2.0, 6.0])
def test_spectrum_pixels_straddling_edge(lyc_ssp, lyc_ssp_edge, z):
    """Exact-path spectrum pixels match the edge-exact reference off the node pair.

    ``predict_spectrum`` (``approx=None``) resamples ``state.sed_intrinsic``
    directly onto the pixel grid (``observation.spectrum.project_spectrum``),
    unlike the SpectrumPrecomp-only ``stellar_spec_lnu_precomp`` mechanism in
    ``nebular/component.py`` (which masks AFTER interpolating the UNMASKED
    spectrum -- exact for any pixel by construction, since a point sample has
    no interval to split). A pixel landing OUTSIDE the SSP grid's own
    straddling interval ``[lambda_a, lambda_b)`` is unaffected by either the
    pre- or post-fix node values (interpolation there uses two OTHER,
    unmodified nodes) and must match the edge-exact reference tightly. A
    pixel landing INSIDE that interval cannot be made exact by adjusting only
    the two bracketing node values -- no single straight line reproduces a
    step at an interior point -- so it is checked separately, only for
    finiteness and that the fix does not make it WORSE than the naive,
    unfixed mask (measured: 33.6% relative error for the one pixel that
    lands exactly on 912 A rest at z=2 before this fix, vs. single-digit
    percent after -- see the mutation-test section of the PR report, not
    reproduced here to avoid depending on unfixed source).
    """
    wave = np.asarray(lyc_ssp.ssp_wave, dtype=np.float64)
    insert_at = int(np.searchsorted(wave, LYC))
    lam_a, lam_b = wave[insert_at - 1], wave[insert_at]

    wave_obs = jnp.linspace(LYC * (1.0 + z) * 0.7, LYC * (1.0 + z) * 1.3, 41)
    fesc = 0.3
    m_exact = _build(lyc_ssp, z, fesc, None, None, wave_obs=wave_obs)
    m_ref = _build(lyc_ssp_edge, z, fesc, None, None, wave_obs=wave_obs)

    rest = np.asarray(wave_obs) / (1.0 + z)
    below = rest < LYC
    assert np.any(below) and np.any(~below), "fixture bug: pixel grid no longer straddles 912 A"

    spec_exact = np.asarray(m_exact.predict_spectrum({}))
    spec_ref = np.asarray(m_ref.predict_spectrum({}))
    assert np.all(np.isfinite(spec_exact)), "exact-path spectrum has non-finite pixels"

    inside = (rest >= lam_a) & (rest < lam_b)
    outside_nonzero = (~inside) & (spec_ref != 0.0)
    rel = np.abs(
        (spec_exact[outside_nonzero] - spec_ref[outside_nonzero]) / spec_ref[outside_nonzero]
    )
    assert np.max(rel) < 1e-9, (
        f"z={z}: off-interval pixels worst relative error {np.max(rel):.3e} too large "
        f"(pixels inside the SSP straddling interval [{lam_a:.4f}, {lam_b:.4f}) A are "
        "excluded by construction -- see docstring)"
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
        return lnu_filter_integral(state.sed_intrinsic, state.wave, filt.wave, filt.trans, z)

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
    spacing near 912 A (MIST/C3K h=1.83 A, MILES h=10 A), so the filter's own
    nodes subdivide the straddling SSP interval and the single-trapezoid-
    panel exactness the node-wise fix relies on (module docstring) no longer
    holds exactly -- measured (2026-09, post #2447 fix) residual against the
    augmented-grid reference: MIST/C3K z=2 GALEX NUV 6.7e-4, MILES z=2 GALEX
    NUV 1.8e-2, both z=3 SDSS u rows below 1e-2. These ratchets bound that
    residual (a real, understood filter-oversampling effect, not the #2447
    node-quantization defect itself, which this same measurement shows fell
    from -2.0651% to +0.0671% on MIST/C3K) from getting WORSE or from the fix
    being silently disabled again (which reopens the multi-percent defect),
    not from shrinking to the brief's aspirational 1e-4 -- that would need
    inserting 912 A into the SSP grid itself (a much larger change than this
    fix's nebular-component scope; see the module docstring's mechanism
    discussion).
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
        ssp_edge = _augment_ssp_at_edge(ssp)
        m_exact = _build(ssp, z, fesc, None, None, filters=[filt])
        m_ref = _build(ssp_edge, z, fesc, None, None, filters=[filt])
        s_exact = m_exact.predict_state({})
        s_ref = m_ref.predict_state({})
        exact = float(
            lnu_filter_integral(s_exact.sed_intrinsic, s_exact.wave, filt.wave, filt.trans, z)
        )
        ref = float(lnu_filter_integral(s_ref.sed_intrinsic, s_ref.wave, filt.wave, filt.trans, z))
        assert ref != 0.0, f"z={z} {filt_name}: reference flux is exactly zero"
        rel = abs(exact - ref) / abs(ref)
        assert rel < ceiling, (
            f"z={z} {filt_name} fesc={fesc}: exact-vs-edge-exact-reference "
            f"residual {rel * 100:.4f}% exceeds the {ceiling * 100:.2f}% "
            f"ratchet (filter-oversampling floor; see class docstring)."
        )
        return rel

    def test_miles_z2_galex_nuv(self):
        ssp = self._load(self.MILES_PATH)
        self._measure(ssp, 2.0, "galex_nuv", 0.0, ceiling=0.05)

    def test_miles_z3_sdss_u(self):
        ssp = self._load(self.MILES_PATH)
        self._measure(ssp, 3.0, "sdss_u", 0.0, ceiling=0.02)

    def test_miles_fesc1_floor(self):
        """fesc=1 must collapse to near the ordinary (mask-free) floor."""
        ssp = self._load(self.MILES_PATH)
        rel = self._measure(ssp, 2.0, "galex_nuv", 1.0, ceiling=0.01)
        assert rel < 0.001, f"fesc=1 floor {rel * 100:.4f}% not near-zero"

    def test_mist_c3k_z2_galex_nuv(self):
        ssp = self._load(self.MIST_C3K_PATH)
        self._measure(ssp, 2.0, "galex_nuv", 0.0, ceiling=0.003)

    def test_mist_c3k_z3_sdss_u(self):
        ssp = self._load(self.MIST_C3K_PATH)
        self._measure(ssp, 3.0, "sdss_u", 0.0, ceiling=0.003)
