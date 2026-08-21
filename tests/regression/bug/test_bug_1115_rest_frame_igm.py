# SPDX-License-Identifier: BSD-3-Clause
"""Rest-frame photometry carried a line-of-sight absorber (#1115).

``phot_rest_fnu`` — and therefore ``Observables.mag_absolute`` — is the SED
reprojected at z=0, d_L=10 pc: *the galaxy as it is*. The IGM sits between us and
the source, so it is not part of the galaxy's rest-frame SED. The exact path fed it
the IGM-attenuated SED anyway, which made an object's **absolute** magnitude depend
on how far away it happens to be. (The galaxy's own LyC absorption — ``neb_fesc``,
dust — is already in the rest SED and correctly stays.)

The projection happens at z=0, so a filter's OWN wavelengths are read as REST
wavelengths, while the IGM transmission is stored on the rest grid as
T(λ_rest·(1+z)). The corruption was therefore confined to filters with support at
rest λ < 1216 Å — blueward of Lyα, where Madau/Inoue absorption begins. The (1+z)
cancels, so that boundary is redshift-invariant; its depth is not.

**Every shipped filter is redward of Lyα** (the bluest, GALEX FUV, starts at 1341 Å),
so a test built from them passes VACUOUSLY — both paths already agree there. These
tests use a synthetic Lyman-continuum band, which is exactly the band an
escape-fraction study would add, and guard the vacuity explicitly.

Scope: this is about the IGM only. The two paths *also* disagree about what
``phot_rest_fnu`` means at z > 0 (the exact path reads the filter as a rest-frame
band; the LUT reads it as the observed band rescaled to 10 pc, i.e. no
K-correction). That is a separate and larger divergence, tracked on its own — do not
read these tests as asserting the two paths agree numerically.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.igm.igm import igm_absorption
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

KEY = jax.random.PRNGKey(0)

#: Redshifts where the Lyman forest bites. Pre-fix, the rest-900 Å band moved by
#: -5.2 % / -30.0 % / -95.9 % (the last is ~-3.5 mag in M_AB).
REDSHIFTS = [1.0, 3.0, 6.0]

LYC_REST = 900.0  # Å — blueward of Lyα (1216 Å): the IGM reaches it
CTRL_REST = 5500.0  # Å — redward: T is exactly 1 at every z, so it must be inert


def _band(name: str, center: float, width: float, n: int = 80) -> FilterCurve:
    wave = jnp.linspace(center - width / 2.0, center + width / 2.0, n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.5
    return FilterCurve(wave=wave, trans=trans, name=name)


def _obs() -> Observation:
    """A rest-frame LyC band plus a redward control.

    The filter's own wavelengths are what ``phot_rest_fnu`` sees, so these are rest
    wavelengths by construction.
    """
    return Observation(
        photometry=Photometry(
            filters=(
                _band("lyc_900", LYC_REST, 120.0),
                _band("ctrl_5500", CTRL_REST, 800.0),
            )
        )
    )


def _build(ssp, z: float, approx, *, igm: bool = True, obs=None) -> SEDModel:
    extra = {"igm": {"type": "inoue"}} if igm else {"igm": {"type": "none"}}
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs if obs is not None else _obs(),
        sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "*": FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.5,
        },
        redshift=Fixed(z),
        approx=approx,
        **extra,
    )


def _rest_igm_on_off(ssp, z: float, approx, obs=None):
    """phot_rest_fnu with the IGM on and off, from ONE shared param dict.

    ``spec.sample()`` returns FIXED params too, so a dict sampled from a
    differently-configured model would override this one's fixed dust.
    """
    m_on = _build(ssp, z, approx, igm=True, obs=obs)
    p = dict(m_on.spec.sample(KEY))
    on = np.asarray(m_on.predict_observables(p).phot_rest_fnu)
    off = np.asarray(
        _build(ssp, z, approx, igm=False, obs=obs).predict_observables(p).phot_rest_fnu
    )
    return on, off


# ── the vacuity guard comes FIRST: without it the rest prove nothing ──────────


@pytest.mark.parametrize("z", REDSHIFTS)
def test_the_igm_actually_reaches_this_band(synthetic_ssp_wide, z):
    """The band must sit where the IGM bites, and the IGM must be switched on.

    Both assertions below are of the form "turning the IGM on does not move X". They
    would pass trivially if the IGM were absent, if it were transparent at this
    wavelength, or if the band had no flux. Pin all three: T at the band must be well
    below 1, and the OBSERVED flux must move a lot.
    """
    t = float(igm_absorption(np.array([LYC_REST * (1.0 + z)]), z)[0])
    assert t < 0.95, f"T = {t:.4f} at rest {LYC_REST} Å, z={z} — the IGM barely bites here"

    m_on = _build(synthetic_ssp_wide, z, None)
    p = dict(m_on.spec.sample(KEY))
    obs_on = np.asarray(m_on.predict_observables(p).phot_fnu)
    obs_off = np.asarray(
        _build(synthetic_ssp_wide, z, None, igm=False).predict_observables(p).phot_fnu
    )
    moved = abs(obs_on[0] / obs_off[0] - 1.0)
    assert moved > 1e-3, (
        f"the OBSERVED flux barely moved ({moved:.2e}) when the IGM was switched on — "
        f"the IGM is not reaching this model, so the rest-frame assertions are vacuous"
    )


# ── the bug ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("z", REDSHIFTS)
def test_exact_path_rest_frame_photometry_carries_no_igm(synthetic_ssp_wide, z):
    """The exact path must not redden the rest frame with a line-of-sight absorber.

    Pre-fix (``sed_atten`` fed to the rest projection): -5.2 % at z=1, -30.0 % at
    z=3, -95.9 % at z=6 — an absolute magnitude that moved ~3.5 mag with distance.

    Neuter-check: restore ``sed_atten`` in the ``phot_rest`` block of
    ``Observation.predict`` and this goes red at every z.
    """
    on, off = _rest_igm_on_off(synthetic_ssp_wide, z, None)
    err = abs(on[0] / off[0] - 1.0)
    assert err < 1e-12, (
        f"rest-frame LyC flux moved {err * 100:+.2f}% when the IGM was toggled at "
        f"z={z} — phot_rest_fnu is projected at z=0 and must carry no IGM"
    )


@pytest.mark.parametrize("z", REDSHIFTS)
def test_lut_path_rest_frame_photometry_carries_no_igm(synthetic_ssp_wide, z):
    """The LUT path already had this right; keep it that way.

    ``predict_via_precomp`` applies the IGM band factor to the observed flux only.
    #1135 folded the IGM into the sub-band quadrature weights and had to keep the
    IGM-free tensor alongside the folded one precisely to preserve this.
    """
    on, off = _rest_igm_on_off(synthetic_ssp_wide, z, WavePrecomp())
    err = abs(on[0] / off[0] - 1.0)
    assert err < 1e-12, f"LUT rest-frame LyC flux moved {err * 100:+.2f}% at z={z}"


@pytest.mark.parametrize("z", REDSHIFTS)
def test_the_redward_control_band_is_inert(synthetic_ssp_wide, z):
    """Redward of Lyα the transmission is exactly 1, so nothing may move there —
    on either path, before or after. Bounds the blast radius to rest λ < 1216 Å."""
    t = float(igm_absorption(np.array([CTRL_REST * (1.0 + z)]), z)[0])
    assert t == pytest.approx(1.0, abs=1e-12), f"T = {t} redward of Lyα at z={z}"
    for approx in (None, WavePrecomp()):
        on, off = _rest_igm_on_off(synthetic_ssp_wide, z, approx)
        assert abs(on[1] / off[1] - 1.0) < 1e-12


# ── the zero-diff claim: no shipped filter reaches blueward of Lyα ────────────


def test_no_shipped_filter_has_throughput_blueward_of_lyman_alpha():
    """The fix is a zero-diff change for every current user — which is why it lands
    now, before someone adds a Lyman-continuum band for escape-fraction work and
    quietly gets an answer that scales with source redshift.

    Asserts the premise directly, over every filter shipped in ``data/filters/``:
    none has any throughput below rest 1216 Å. If a future filter breaks this, the
    zero-diff claim in the changelog stops being true and this says so.
    """
    from tengri.observation.filters import list_available_filters, load_filter

    lya = 1216.0
    offenders = []
    for name in list_available_filters():
        try:
            fc = load_filter(name)
        except Exception:
            continue
        w = np.asarray(fc.wave)
        t = np.asarray(fc.trans)
        below = t[w < lya]
        if below.size and float(np.max(below)) > 0.0:
            offenders.append((name, float(np.min(w[t > 0.0]))))
    assert not offenders, (
        f"filters with throughput blueward of Lyα: {offenders}. The #1115 fix is no "
        f"longer zero-diff for them (it is still CORRECT — their rest-frame "
        f"photometry simply stops carrying an observed-frame absorber)."
    )


@pytest.mark.parametrize("z", [1.0, 3.0, 6.0])
def test_shipped_filters_are_unchanged_by_the_fix(synthetic_ssp_wide, z):
    """The behavioral half of the zero-diff claim: with real filters, toggling the
    IGM leaves rest-frame photometry bit-identical on both paths."""
    obs = Observation(
        photometry=Photometry.from_names(["galex_fuv", "galex_nuv", "des_g", "wise_w1"])
    )
    for approx in (None, WavePrecomp()):
        on, off = _rest_igm_on_off(synthetic_ssp_wide, z, approx, obs=obs)
        np.testing.assert_allclose(on, off, rtol=0.0, atol=0.0)
