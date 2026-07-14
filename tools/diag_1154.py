"""#1154 round 2: does the ionspec cache COLLISION cause the 21%, on linux?

Round 1 killed the platform hypothesis: the scipy-fit ionspec table is BIT-IDENTICAL
on linux/x86 and macOS/arm64 (different numpy/scipy/jax), and in a clean process the
Cue fast path reproduces the exact forward to 0.42% on BOTH. So the 21% needs something
a clean process does not have.

The candidate is the collision fixed in #1156: `_ssp_fingerprint` hashed ssp_flux as
(shape, dtype) only, so the bare-stellar grid and its wNE twin shared an ionizing-
spectrum cache key. In the full suite, another test loads the wNE grid first and the
bare model is then served a wNE ionizing spectrum.

I tested that on macOS and got 0.074% — both paths read the same table, so a poisoned
table seemed to move them together. But I never tested it on LINUX, which is the only
platform the failure was ever seen on. That is the gap this closes.

Reproduces test_fast_nebular_wiring's config EXACTLY (8 bands, n_grid=16, logU-only
free axis, z=0.15, the same 8 PRNG draws) and compares PHOTOMETRY, which is what
actually reports 2.09e-01.

  A. clean cache                        -> expect ~0.1%
  B. wNE loaded first, OLD colliding key -> 20.9% ?
  C. wNE loaded first, NEW key (#1156)   -> must be immune

Disposable. Delete with the branch.
"""

import os
import platform
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TENGRI_JAX_CACHE_DIR"] = tempfile.mkdtemp(prefix="diag1154_")

import warnings

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, SEDModel, Uniform, WavePrecomp, load_ssp_data
from tengri.components.nebular import ionizing_spectrum as ion
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.photometry_config import Photometry

BARE = "data/fsps_prsc_miles_chabrier.h5"
WNE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

# test_fast_nebular_wiring's config, verbatim
_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
Z = 0.15
_CUE = {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}

print("=" * 78)
print(f"platform : {platform.platform()}  machine={platform.machine()}")
print("=" * 78)

NEW_FP = ion._ssp_fingerprint  # the #1156 (fixed) key


def OLD_FP(w, f, m):
    """The pre-#1156 key: ssp_flux enters as (shape, dtype) only."""
    return (
        tuple(f.shape),
        str(f.dtype),
        bytes(np.asarray(w).tobytes()),
        bytes(np.asarray(m).tobytes()),
    )


def _build(ssp):
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=_LINE_DATA)
    dust = {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,
        "tau_diff": Fixed(0.25),
        "tau_bc": Fixed(0.4),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust=dust,
            neb=_CUE,
            redshift=Fixed(Z),
            approx=WavePrecomp(),
        )


def worst_photometry():
    """Exactly what test_fast_photometry_and_lines_match_exact measures."""
    bare = load_ssp_data(BARE)
    m_exact = _build(bare)
    m_fast = _build(bare)
    m_fast.enable_fast_nebular(_LW, n_grid=16)
    wp, worst = 0.0, None
    for i in range(8):
        p = dict(m_exact.spec.sample(jax.random.PRNGKey(600 + i)))
        pe = np.asarray(m_exact.predict_photometry(p))
        pf = np.asarray(m_fast.predict_photometry(p))
        rel = np.abs(pf - pe) / (np.abs(pe) + 1e-40)
        if rel.max() > wp:
            wp, worst = rel.max(), (i, int(rel.argmax()), pe, pf, rel)
    i, b, pe, pf, rel = worst
    r = abs(pe[_BANDS.index("des_r")]) + 1e-300
    print(f"    worst band {_BANDS[b]!r} on draw {i}: exact={pe[b]:.5e} fast={pf[b]:.5e}")
    print(f"    {'band':>10} {'exact':>13} {'fast':>13} {'rel':>9} {'/des_r':>9}")
    for k, band in enumerate(_BANDS):
        print(
            f"    {band:>10} {pe[k]:13.5e} {pf[k]:13.5e} {rel[k]:9.3%} {abs(pe[k]) / r:9.2e}"
        )
    return wp


def scenario(tag, *, poison, fingerprint):
    ion._IONSPEC_TABLE_CACHE.clear()
    ion._ssp_fingerprint = fingerprint
    if poison:
        wne = load_ssp_data(WNE)
        ion.precompute_ionizing_params_table(wne.ssp_wave, wne.ssp_flux, wne.ssp_lgmet)
        n = len(ion._IONSPEC_TABLE_CACHE)
        print(f"\n{tag}\n    (cache primed from the wNE grid: {n} entry)")
    else:
        print(f"\n{tag}")
    w = worst_photometry()
    print(f"    -> worst fast-vs-exact photometry = {w:.3%}   (test asserts < 3%)")
    ion._IONSPEC_TABLE_CACHE.clear()
    return w


a = scenario("A  clean cache, fixed key", poison=False, fingerprint=NEW_FP)
b = scenario("B  wNE first, OLD colliding key (pre-#1156)", poison=True, fingerprint=OLD_FP)
c = scenario("C  wNE first, NEW key (#1156 — must be immune)", poison=True, fingerprint=NEW_FP)
ion._ssp_fingerprint = NEW_FP

print("\n" + "=" * 78)
print("VERDICT   (CI reported 2.09e-01 = 20.9%; tolerance 3%)")
print(f"  A clean, fixed key          : {a:7.3%}")
print(f"  B wNE first, OLD key        : {b:7.3%}")
print(f"  C wNE first, NEW key (#1156): {c:7.3%}")
if b > 0.03 and c < 0.03:
    print("\n  -> #1154 IS the ionspec cache collision. #1156 already fixes it.")
elif b < 0.03:
    print("\n  -> the collision does NOT reproduce it here either. #1154 is something else.")
print("=" * 78)
