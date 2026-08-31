# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Custom filter curves, and 7DT
#
# > ⚠️ **Experimental.** A research demonstration using experimental APIs that may change between releases.
#
# Bringing your own bandpasses. Three registration routes, the one unit mistake
# that does not raise, and an ADU-plus-zeropoint table taken through to a fit.
#
# 7DT is the worked example throughout: 3 broad (`g`, `r`, `i`) plus 20 medium
# bands on a 25 nm grid, `m400`-`m875`. Those 23 ship with tengri as `7dt_*` and
# need no registration at all. Everything before the last two sections is about
# what to do when your filters are *not* already built in.

# %%
from _setup import effective_wavelengths_um, quiet

quiet()

import tempfile
import warnings
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import (
    builders,
    DEFAULT,
    Fixed,
    ForwardModel,
    FREE,
    generate_mock,
    Observation,
    Photometry,
    plot,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.observation.filters import load_filter
from tengri.observation.filters.custom import (
    register_filter,
    register_filter_from_file,
    unregister_filter,
)

plot.setup_style()

C_DATA, C_MODEL, C_BAD = "#c3372a", "#3a76d9", "0.55"
WORK = Path(tempfile.mkdtemp(prefix="tengri_filters_"))

# %% [markdown]
# ## What a delivered curve looks like
#
# Rarely two whitespace columns in Angstrom. The 7DT delivery is one CSV per
# band, `lam,trans`, wavelength in **nanometers** on a 0.1 nm grid, zero-padded
# to 300-1000 nm. We fabricate one here in exactly that shape.

# %%
lam_nm = np.arange(300.0, 1000.1, 0.1)
trans = 0.34 * np.exp(-0.5 * ((lam_nm - 401.0) / 10.3) ** 2)

delivered = WORK / "m400.csv"
delivered.write_text(
    "lam,trans\n" + "\n".join(f"{w:.5f},{t:.5f}" for w, t in zip(lam_nm, trans, strict=True))
)
print(delivered.read_text()[:64].replace("\n", " | "))

# %% [markdown]
# `load_filter` accepts comma- or whitespace-separated columns, with or without
# a header row, and strips `#` comments. Detection is by parse attempt, not by
# file extension, because a `.csv` full of whitespace and a `.dat` full of
# commas both turn up.

# %% [markdown]
# ## Route 1: in-memory, from arrays
#
# `register_filter` puts a curve in the process-local registry under a name.
# From that point it resolves anywhere a built-in does.

# %%
register_filter("demo_from_arrays", lam_nm, trans, wave_unit="nm", overwrite=True)
fc = load_filter("demo_from_arrays")
print(f"{fc.name}: {fc.wave.min():.0f}-{fc.wave.max():.0f} AA, peak T = {fc.trans.max():.3f}")

# %% [markdown]
# ## Route 2: in-memory, from a file
#
# Same registry, one less step. `wave_unit` is the whole reason this is safe;
# see the next section.

# %%
register_filter_from_file("demo_from_file", delivered, wave_unit="nm", overwrite=True)
print(np.allclose(load_filter("demo_from_file").wave, load_filter("demo_from_arrays").wave))

# %% [markdown]
# ## Route 3: `$TENGRI_FILTER_DIR`
#
# A `:`-separated directory list, searched by file stem, no code call at all.
# This is the route that survives being handed to someone else: a config naming
# `m400` resolves on any machine that has the directory. The other two live and
# die with the process.
#
# Note that it has no place to put `wave_unit` -- a bare directory of files
# carries no metadata -- so **files dropped here must already be in Angstrom.**

# %%
import os

filter_dir = WORK / "curves"
filter_dir.mkdir(exist_ok=True)
np.savetxt(filter_dir / "m400_dir.dat", np.column_stack([lam_nm * 10.0, trans]))
os.environ["TENGRI_FILTER_DIR"] = str(filter_dir)

print(f"resolved by stem: {load_filter('m400_dir').wave.max():.0f} AA")

# %% [markdown]
# ## The mistake that does not raise
#
# Every other unit error in tengri fails loudly. This one does not, because
# nanometers-read-as-Angstrom is a *valid* array of numbers -- it just describes
# the extreme UV, where the ISM is opaque and no bandpass exists.
#
# Registering without saying the unit trips a heuristic: a curve lying wholly
# between 100 Å and 1340 Å (the blue edge of GALEX FUV, the bluest filter
# tengri ships) is almost certainly nanometers.

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    register_filter("demo_unstated", lam_nm, trans, overwrite=True)

print(str(caught[0].message) if caught else "no warning")

# %% [markdown]
# What it costs if you miss it. The curve sits at 300-1000 Å, the SED has
# essentially no flux there, and the predicted photometry is not wrong so much
# as meaningless -- while every fit still converges and every error bar still
# looks reasonable.

# %%
wrong = load_filter("demo_unstated")
right = load_filter("demo_from_arrays")
print(f"unstated : {wrong.wave.min():7.0f} - {wrong.wave.max():7.0f} AA")
print(f"nm stated: {right.wave.min():7.0f} - {right.wave.max():7.0f} AA")
print(f"ratio    : {float(right.wave.max() / wrong.wave.max()):.0f}x")

for name in ("demo_from_arrays", "demo_from_file", "demo_unstated"):
    unregister_filter(name)

# %% [markdown]
# The heuristic is a backstop, not a guarantee. It cannot catch microns at all
# (an optical curve in microns lands at 0.5-0.7 Å, which is a real NuSTAR band),
# and it cannot catch a nanometer set running past 1340 nm without also firing
# on GALEX FUV. **State the unit.**

# %% [markdown]
# ## The 23 bundled 7DT bands
#
# These need none of the above. They ship in the package as total system
# response -- detector QE and optics folded in, which is what the photometry was
# measured through -- and resolve by name like any SVO band. See
# `tengri/data/filters_7dt/PROVENANCE.md`.

# %%
BANDS_BROAD = ["7dt_g", "7dt_r", "7dt_i"]
BANDS_MEDIUM = [f"7dt_m{lam}" for lam in range(400, 900, 25)]
FILTERS = BANDS_MEDIUM + BANDS_BROAD

print(tengri.list_filters(survey="7dt")[:4])
print(f"\n{len(FILTERS)} bands")

# %% [markdown]
# The peak transmission envelope is not flat across the medium bands: 0.34 at
# `m400`, 0.66 near 500 nm, 0.07 at `m875`. That is the QE, and it is harmless
# for AB photometry -- a constant scale on $T$ cancels in the ratio of the two
# bandpass integrals, so only the shape matters.

# %%
fig, ax = plt.subplots(figsize=(9.0, 3.4))
for name in BANDS_MEDIUM:
    fc = load_filter(name)
    ax.plot(np.asarray(fc.wave) / 1e4, np.asarray(fc.trans), lw=1.0, color=C_MODEL)
for name, ls in zip(BANDS_BROAD, ["-", "--", ":"], strict=True):
    fc = load_filter(name)
    ax.plot(np.asarray(fc.wave) / 1e4, np.asarray(fc.trans), lw=1.4, ls=ls, color=C_DATA)
ax.set(xlabel=r"$\lambda_{\rm obs}$ [$\mu$m]", ylabel="transmission", xlim=(0.32, 0.98))
ax.set_title("7DT: 20 medium (blue) over g, r, i (red)", fontsize=10)
fig.tight_layout()

# %% [markdown]
# ## ADU and a zeropoint, taken to a fit
#
# Instrument tables rarely arrive in $F_\nu$. The 7DT photometry is mean surface
# flux per pixel in ADU with a per-band AB zeropoint, which is one step from a
# unit tengri ingests:
#
# $$m_{\rm AB} = \mathrm{ZP}_b - 2.5\log_{10} f_{\rm ADU}, \qquad
#   \sigma_m = 1.0857\,\sigma_f / f_{\rm ADU}.$$
#
# Then `flux_unit="ab_mag"`. There is no ADU option and there should not be: a
# zeropoint is per-band calibration, not a unit.
#
# We synthesize the ADU here from a known model so the answer is checkable.

# %%
ZP = dict.fromkeys(FILTERS, 23.89)  # the delivered zeropoints sit within 0.03 mag

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
obs = Observation(photometry=Photometry.from_names(FILTERS))

sed_model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    approx=WavePrecomp(),
    sfh=builders.sfh.tsnorm(all_params=FREE),
    dust_attenuation=builders.dust.two_component(
        all_params=Fixed(DEFAULT), law="calzetti", tau_diff=Uniform(0.0, 1.0)
    ),
    dust_emission=builders.dust.emission.modified_blackbody(all_params=Fixed(DEFAULT)),
    neb=builders.neb.none(),
    met={"logzsol": Uniform(-1.5, 0.3)},
    redshift=Fixed(0.0062),  # NGC 1380, Fornax
)
forward = ForwardModel.build(sed=sed_model)
print(f"free parameters ({sed_model.spec.n_free}): {', '.join(sed_model.spec.free_params)}")

# %%
key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(11), 3)
truth = sed_model.spec.sample(key_truth)
mock = generate_mock(sed_model, truth, key=key_mock, snr=30.0)

fnu = np.asarray(mock["flux_obs"])
sigma = np.asarray(mock["noise"])

# Round-trip to the instrument's own units, then back, exactly as the delivered
# table would have to be handled.
mag = -2.5 * np.log10(fnu) - 48.60
adu = 10.0 ** (-0.4 * (mag - np.array([ZP[b] for b in FILTERS])))
adu_err = adu * np.log(10.0) / 2.5 * (1.0857 * sigma / fnu)

print(f"ADU range: {adu.min():.1f} - {adu.max():.1f}")

# %% [markdown]
# `ingest_catalog` takes explicit `flux_cols` and `err_cols`, so a table with
# `flux_<band>` / `err_<band>` columns needs no renaming. (The `read_catalog`
# convenience reader hardcodes a `<band>_err` suffix *and* silently defaults a
# missing redshift to 0.0 -- do not use it for this.)

# %%
from tengri.inference.catalog_ingest import ingest_catalog

mag_obs = np.array([ZP[b] for b in FILTERS]) - 2.5 * np.log10(adu)
mag_err = 1.0857 * adu_err / adu

table = {"redshift": np.array([0.0062])}
for j, band in enumerate(FILTERS):
    table[f"flux_{band}"] = np.array([mag_obs[j]])
    table[f"err_{band}"] = np.array([mag_err[j]])

arrays = ingest_catalog(
    table,
    photometry=obs.photometry,
    flux_unit="ab_mag",
    flux_cols=[f"flux_{b}" for b in FILTERS],
    err_cols=[f"err_{b}" for b in FILTERS],
    redshift_col="redshift",
)
print(f"round-trip max |dF/F|: {np.abs(arrays.flux[0] / fnu - 1.0).max():.2e}")

# %% [markdown]
# The round trip closes to machine precision, so the ADU-to-AB-to-$F_\nu$ chain
# introduces nothing of its own.
#
# On errors: if the delivered `err` column already contains the zeropoint
# uncertainty (7DT's does), pass it through and leave `NoiseModel`'s
# `calibration_floor` at 0. Setting both counts the same term twice. Note that
# no per-fit floor represents the *coherence* of a zeropoint error across bins
# of one galaxy -- it is one shared offset, not independent noise. That matters
# for joint or radial analyses, not for fitting bins one at a time.

# %%
map_result = forward.fit(arrays.flux[0], arrays.noise[0], method="map", key=key_fit, n_steps=300)

for pname in sed_model.spec.free_params:
    print(
        f"  {pname:24} fit {float(map_result.params[pname]):+8.3f}   truth {float(truth[pname]):+8.3f}"
    )

# %% [markdown]
# Mass and SFH width come back tightly; metallicity and dust do not. That is
# the age-metallicity-dust degeneracy, not a filter problem -- 23 optical bands
# with no UV and no NIR do not break it, and a MAP point estimate reports none
# of the covariance that would show you so. Run `method="mcmc_nuts"` if you care
# about the answer rather than the plumbing.

# %% [markdown]
# ## Scale
#
# One MAP fit here. A Voronoi-binned program is ~300 bins per galaxy, and ~100
# galaxies is ~30,000 SEDs. `Fitter.fit_batch` shares the XLA compile cache
# across galaxies, so the first fit pays compilation and the rest do not --
# which makes the per-fit cost, not the compile cost, the thing to measure
# before committing to a full run.

# %%
del os.environ["TENGRI_FILTER_DIR"]
unregister_filter("m400_dir")
print("done")
