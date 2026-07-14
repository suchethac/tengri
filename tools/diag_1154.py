"""#1154 attribution: was it the CACHE COLLISION (#1156) or the BROKEN SSP (#1153)?

Two variables moved between the failing and passing linux runs, and I must not credit
the wrong one:

  cs/ci-shard-contract : OLD SSP (missing ssp_mass_remaining) + NO #1156  -> 20.9% FAIL
  main (today)         : NEW SSP (#1153 data fix)            + #1156      -> passes

A mutation test already refutes the collision: strip #1156's flux digest, poison the
cache with the wNE grid (disk cache isolated), and the bare model's fast-vs-exact
agreement is UNCHANGED. So the collision corrupts both paths' physics together and does
not open a fast-vs-exact gap.

That leaves the SSP. #1153 found data/fsps_prsc_miles_chabrier.h5 was shipped WITHOUT
the `ssp_mass_remaining` dataset — every physics array byte-identical, one key absent,
so tengri silently fell back to a generic mass-remaining table.

This run holds #1156 FIXED and swaps only the SSP:

  A. current SSP  (#1153's corrected file)   -> expect ~0.07%
  B. OLD SSP      (as shipped in #1146/1151) -> 20.9% ?

If B fails, #1154 was the incomplete SSP, #1153 already fixed it, and #1156 — while a
real bug — is innocent of this one.
"""

import os
import platform
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TENGRI_JAX_CACHE_DIR"] = tempfile.mkdtemp(prefix="diag1154_")
os.environ["TENGRI_ALLOW_WNE_CUE"] = "1"

import warnings

import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, SEDModel, Uniform, WavePrecomp, load_ssp_data
from tengri.components.nebular import ionizing_spectrum as ion
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.photometry_config import Photometry

NEW_SSP = "data/fsps_prsc_miles_chabrier.h5"
OLD_SSP = "data/_old_fsps_prsc_miles_chabrier.h5"  # written by the workflow from git

_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
_CUE = {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}

print("=" * 78)
print(f"platform : {platform.platform()}  machine={platform.machine()}")
print("=" * 78)

for tag, path in (("NEW (main)", NEW_SSP), ("OLD (#1146)", OLD_SSP)):
    if not os.path.exists(path):
        print(f"{tag:12s} MISSING: {path}")
        continue
    with h5py.File(path) as f:
        keys = sorted(f.keys())
    print(f"{tag:12s} {path}")
    print(f"             keys: {keys}")
    print(f"             ssp_mass_remaining present: {'ssp_mass_remaining' in keys}")


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
            redshift=Fixed(0.15),
            approx=WavePrecomp(),
        )


def worst(tag, path):
    ion._IONSPEC_TABLE_CACHE.clear()
    ssp = load_ssp_data(path)
    m_exact, m_fast = _build(ssp), _build(ssp)
    m_fast.enable_fast_nebular(_LW, n_grid=16)
    wp, w = 0.0, None
    for i in range(8):
        p = dict(m_exact.spec.sample(jax.random.PRNGKey(600 + i)))
        pe = np.asarray(m_exact.predict_photometry(p))
        pf = np.asarray(m_fast.predict_photometry(p))
        rel = np.abs(pf - pe) / (np.abs(pe) + 1e-40)
        if rel.max() > wp:
            wp, w = rel.max(), (i, int(rel.argmax()), pe, pf, rel)
    i, b, pe, pf, rel = w
    r = abs(pe[_BANDS.index("des_r")]) + 1e-300
    print(f"\n{tag}")
    print(f"    worst band {_BANDS[b]!r} draw {i}")
    print(f"    {'band':>10} {'exact':>13} {'fast':>13} {'rel':>9} {'/des_r':>9}")
    for k, band in enumerate(_BANDS):
        print(f"    {band:>10} {pe[k]:13.5e} {pf[k]:13.5e} {rel[k]:9.3%} {abs(pe[k]) / r:9.2e}")
    print(f"    -> worst fast-vs-exact = {wp:.3%}   (test asserts < 3%)")
    ion._IONSPEC_TABLE_CACHE.clear()
    return wp


a = worst("A  NEW SSP (#1153's corrected file), #1156 fix present", NEW_SSP)
b = worst("B  OLD SSP (missing ssp_mass_remaining), #1156 fix present", OLD_SSP)

print("\n" + "=" * 78)
print("VERDICT   (#1154 reported 20.9%; tolerance 3%)")
print(f"  A new SSP : {a:7.3%}")
print(f"  B old SSP : {b:7.3%}")
if b > 0.03 >= a:
    print("\n  -> #1154 WAS THE INCOMPLETE SSP. #1153 fixed it. #1156 is innocent of this.")
elif b <= 0.03:
    print("\n  -> the old SSP is NOT enough to reproduce it either. Cause still unknown;")
    print("     the remaining variable is the full-suite context.")
print("=" * 78)
