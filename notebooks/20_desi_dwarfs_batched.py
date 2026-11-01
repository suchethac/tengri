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
# # Batched spectroscopy of DESI-like dwarf galaxies
#
# ## What you will do
# Fit four dwarf galaxies observed with a DESI-like spectrograph as one batched program. Each
# galaxy has a banded resolution, a banded pixel covariance, three calibration segments (one per
# camera), emission lines marginalized jointly with the calibration, a Milky Way foreground, and a
# photometry set that is shared across galaxies.
#
# ## What you need
# The `fsps_prsc_miles_chabrier` SSP grid in the repository's `data/` directory. The notebook is
# sized for a laptop CPU. The mocks are synthetic: the truth of every galaxy is known.
#
# ## What you will have
# A MAP estimate and a posterior for every galaxy, recovered against the truth, with the compile and
# fit timings of the batched stack. The stack compiles the forward model once per bucket of galaxies.
# Each galaxy's data enters the compiled function as traced arrays, and its redshift comes from the
# record rather than from the sampler.
#
# This notebook is experimental: the batched spectroscopic stack and its options are still open to
# change, and the interface shown here is the one this notebook was written against.

# %%
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from _setup import FIG_DIR, quiet

quiet()

import tengri
from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
    make_agebins_from_zred,
    plot,
)
from tengri.forward.batched_spectrum import (
    batched_eline_design,
    batched_line_set,
    check_template_model,
    make_batched_predict,
)
from tengri.inference.batched_spectro import fit_spectra_catalog_mcmc, fit_spectra_map_vmap
from tengri.observation.banded import BandedMatrix, gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.line_list import LineList

warnings.filterwarnings("ignore", message=".*SFHBeyondOldest.*")
plot.setup_style()
print(f"JAX backend: {jax.devices()[0].platform}   x64: {jax.config.jax_enable_x64}")

# %% [markdown]
# ## The scenario
#
# Four galaxies with true log10 M between 7.5 and 9.5 and redshifts between 0.02 and 0.10. Each
# spectrum is one window of the 3600–9800 Å grid at 0.8 Å, with a per-galaxy pixel mask that
# removes the sky lines (5577, 6300 and 6364 Å), the camera join (5780 and 7570 Å) and the Ca K line.
# That leaves 4,100 to 4,500 fitted pixels per galaxy. Each window, with its masked pixels, is at most
# 4,608 pixels, so all four fit in one bucket. The resolution rises from R = 2000 to R = 5000 across
# the grid. Pixel noise is correlated over three pixels
# through a banded covariance, which has the lag structure that constant-velocity rebinning
# produces. The signal to noise per pixel is 3 to 10.
#
# Each galaxy carries the seven-filter union: DECam *grz* and WISE W1. The SDSS *grz* curves stand
# in for the BASS and MzLS bands, since the repository ships no BASS or MzLS curves. Each galaxy
# observes one of four band sets, so the photometry presence differs between galaxies while the
# template is shared. Each galaxy also has its own Milky Way E(B−V).

# %%
N_GAL = 4
SEED = 20261101
N_WAVE = 7751
WAVE = 3600.0 + 0.8 * np.arange(N_WAVE)
Z_RANGE = (0.02, 0.10)
LOGM_RANGE = (7.5, 9.5)
Z_TEMPLATE_MAX = 0.12
FILTERS = ["des_g", "des_r", "des_z", "sdss_g", "sdss_r", "sdss_z", "wise_w1"]
SKY_LINES = (5577.3, 6300.3, 6363.8)
CAMERA_JOIN = (5780.0, 7570.0)
CA_K = 3934.78
PRIOR_SIGMA = 1e-27
KERNEL = np.array([0.3, 1.0, 0.3])
KERNEL_NORM = float(np.sum(KERNEL**2))
RHO = (
    float(KERNEL[0] * KERNEL[1] + KERNEL[1] * KERNEL[2]) / KERNEL_NORM,
    float(KERNEL[0] * KERNEL[2]) / KERNEL_NORM,
)
PHOT_SETS = {0: [0, 1, 2], 1: [0, 1, 2, 6], 2: [3, 4, 5], 3: [3, 4, 5, 6]}
FIT_TARGET_PIXELS = (4100, 4500)


def resolution_curve(wave):
    """Resolving power R at each observed wavelength [Angstrom]."""
    return 2000.0 + 3000.0 * (np.asarray(wave) - 3600.0) / (9800.0 - 3600.0)


def camera_ids(wave):
    """Calibration segment (0 blue, 1 red, 2 near-infrared) of each pixel."""
    return np.digitize(np.asarray(wave), CAMERA_JOIN)


def pixel_mask(lo, hi, z):
    """Good-pixel mask of the window [lo, hi) at redshift z: sky, camera join, Ca K and edges."""
    w = WAVE[lo:hi]
    bad = np.zeros(w.size, bool)
    bad[:10] = True
    bad[-10:] = True
    for line in SKY_LINES:
        bad |= np.abs(w - line) < 4.0
    for join in CAMERA_JOIN:
        bad |= np.abs(w - join) < 3.2
    bad |= np.abs(w - CA_K * (1 + z)) < 3.0 * (1 + z)
    return ~bad


def draw_windows(rng, z):
    """Window bounds with 4,100 to 4,500 fitted pixels that span all three cameras."""
    targets = rng.permutation(np.linspace(*FIT_TARGET_PIXELS, N_GAL)).astype(int)
    bounds, masks = [], []
    for k in range(N_GAL):
        n_fit = int(targets[k])
        for _ in range(500):
            length = n_fit + 60
            lo = int(rng.integers(0, N_WAVE - length - 1))
            for _ in range(30):
                mask = pixel_mask(lo, lo + length, z[k])
                diff = int(mask.sum()) - n_fit
                if diff == 0:
                    break
                length -= diff
            ids = camera_ids(WAVE[lo : lo + length])
            if diff == 0 and set(ids.tolist()) == {0, 1, 2}:
                break
        bounds.append((lo, lo + length))
        masks.append(mask)
    return bounds, masks


def banded_covariance(n_pix, sigma2):
    """Symmetric banded covariance with diagonal offsets 0, 1, 2 for correlated pixel noise."""
    data = np.zeros((3, n_pix))
    data[0] = sigma2
    data[1, :-1] = RHO[0] * sigma2
    data[2, :-2] = RHO[1] * sigma2
    return BandedMatrix(offsets=jnp.asarray([0, 1, 2]), data=jnp.asarray(data))


def correlated_noise(rng, n_pix, sigma):
    """Noise with the covariance of banded_covariance, from a three-tap kernel on white noise."""
    w = rng.standard_normal(n_pix + 2)
    return (
        sigma
        * (KERNEL[0] * w[:-2] + KERNEL[1] * w[1:-1] + KERNEL[2] * w[2:])
        / np.sqrt(KERNEL_NORM)
    )


def make_record(lo, hi, z, edges, mask, flux, ivar, phot, perr, ebmv, covariance=None):
    """One galaxy's observation record on its window [lo, hi)."""
    w = WAVE[lo:hi]
    return GalaxySpectrum(
        wave=w,
        flux=flux,
        ivar=ivar,
        resolution=gaussian_resolution_bands(jnp.asarray(w), resolution_curve(w), 11),
        z=float(z),
        mask=mask,
        phot_flux=phot,
        phot_err=perr,
        sfh_bin_edges_gyr=edges,
        ebmv_mw=float(ebmv),
        segment_ids=camera_ids(w),
        covariance=covariance,
    )


# %% [markdown]
# ## The template
#
# One `SEDModel` fixes everything the galaxies share: the continuity star formation history with
# six free ratios and the mass, two Calzetti dust components, a free metallicity, the Cue nebular
# lines and the photometry union. Its spectroscopy carries the camera segments, so each galaxy's
# calibration has one free scale per camera, with the scale-free prior and a fiber-fraction scale
# prior of 0.1 dex. The emission lines are marginalized jointly with the calibration. The Milky Way
# screen is switched on through `foreground`; each galaxy supplies its own E(B−V).
#
# The redshift is free in the template. A batch carries each galaxy's redshift, so `Fixed(z)` is
# refused here.


# %%
def build_template():
    """The shared SEDModel of the galaxies, with free redshift and batched-stack options."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(WAVE),
        resolution=resolution_curve(WAVE),
        resolution_matrix=gaussian_resolution_bands(jnp.asarray(WAVE), resolution_curve(WAVE), 11),
        resample="conserving",
        eline_mode="marginalized",
        eline_catalog=LineList.default_optical(),
        eline_prior_sigma=PRIOR_SIGMA,
        eline_prior_type="flat",
        eline_broad=False,
        calibration_segments=camera_ids(WAVE),
        cal_prior="scale_free",
        cal_scale_prior=(0.0, 0.1),
    )
    model = SEDModel.build(
        ssp_data=tengri.load_ssp("fsps_prsc_miles_chabrier", download=True),
        observation=Observation(
            spectroscopy=spectroscopy, photometry=Photometry.from_names(FILTERS)
        ),
        sfh={
            "type": "continuity",
            "all_params": FREE,
            "bin_edges_gyr": [float(e) for e in make_agebins_from_zred(zred=Z_TEMPLATE_MAX)],
            "log_total_mass": Uniform(6.0, 10.5),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 1.5),
            "other_params": Fixed(DEFAULT),
        },
        met={"logzsol": Uniform(-1.5, 0.2), "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        foreground={"ebmv_mw": 0.02},
        redshift=Uniform(0.0, Z_TEMPLATE_MAX),
    )
    spec = model.spec.with_fixed_value("eline_sigma_kms", 100.0)
    return SEDModel(spec, model.ssp_data, observation=model.observation)


t0 = time.perf_counter()
template = build_template()
print(f"template build: {time.perf_counter() - t0:.1f} s")
print(f"free parameters ({template.spec.n_free}): {', '.join(template.spec.free_params)}")

# %%
t0 = time.perf_counter()
check_template_model(template)
print(f"check_template_model passed in {time.perf_counter() - t0:.2f} s")

# %% [markdown]
# ## The mocks
#
# Each galaxy's truth is drawn from the scenario ranges. The continuum comes from the batched forward
# model at the truth, and the marginal line amplitudes are added on top of it, with amplitudes
# between 0.5 and 10 times the continuum. The noise follows the banded covariance. The photometry
# carries 5 percent scatter, and each galaxy keeps only the filters of its band set.
#
# The mocks are built in two passes: the first pass fixes the batch layout (the masks and line
# columns), and the second adds the data. The bucket plan of the fit is printed after the mocks.


# %%
def draw_truth(rng):
    """Truth of the galaxies: log10 M and the free parameters of the template."""
    logm = rng.permutation(np.linspace(*LOGM_RANGE, N_GAL))
    truth = {
        "sfh_cont_log_total_mass": logm,
        "dust_tau_bc": rng.uniform(0.1, 1.0, N_GAL),
        "dust_tau_diff": rng.uniform(0.05, 0.5, N_GAL),
        "met_logzsol": rng.uniform(-1.0, -0.2, N_GAL),
    }
    for i in range(6):
        truth[f"sfh_cont_ratio_{i}"] = np.clip(rng.normal(0, 0.4, N_GAL), -1, 1)
    return truth


def predict_all(template, buckets, params):
    """Continuum, photometry and line design of every galaxy, in catalog order."""
    cont = [None] * N_GAL
    phot = [None] * N_GAL
    design = [None] * N_GAL
    line_set = batched_line_set(template)
    for spec, batch, index in buckets:
        sel = jnp.asarray(index)
        bucket_params = {k: jnp.asarray(v)[sel] for k, v in params.items()}
        out = make_batched_predict(template, spec)(bucket_params, batch)
        g = np.asarray(
            jax.vmap(lambda o: batched_eline_design(line_set, o, {"eline_sigma_kms": 100.0}))(
                batch
            )
        )
        for local, i in enumerate(index):
            cont[i] = np.asarray(out["spec_fnu"])[local]
            phot[i] = np.asarray(out["phot_fnu"])[local]
            design[i] = g[local]
    return cont, phot, design


def make_mocks(template):
    """Mock spectra with their truth; returns the records, the truth, the redshifts and the windows."""
    rng = np.random.default_rng(SEED)
    z = rng.permutation(np.linspace(*Z_RANGE, N_GAL))
    bounds, masks = draw_windows(rng, z)
    edges = [make_agebins_from_zred(zred=float(zi)) for zi in z]
    truth = draw_truth(rng)
    ebmv = rng.uniform(0.01, 0.05, N_GAL)
    band_set = rng.permutation(np.arange(N_GAL) % 4)
    placeholder = [
        make_record(
            lo,
            hi,
            z[k],
            edges[k],
            masks[k],
            np.ones(hi - lo),
            np.ones(hi - lo),
            np.ones(len(FILTERS)),
            np.ones(len(FILTERS)),
            ebmv[k],
        )
        for k, (lo, hi) in enumerate(bounds)
    ]
    buckets = build_spectro_batches(placeholder, quantum=512, model=template)
    cont, phot, design = predict_all(template, buckets, truth)

    records = []
    for k, (lo, hi) in enumerate(bounds):
        n = hi - lo
        good = masks[k]
        cont_med = np.median(cont[k][:n][good])
        snr = rng.uniform(3.0, 10.0)
        sigma = cont_med / snr
        amps = cont_med * 10 ** rng.uniform(-0.3, 1.0, design[k].shape[1])
        lines = design[k][:n] @ amps
        flux = cont[k][:n] + lines + correlated_noise(rng, n, sigma)
        ivar = np.full(n, 1.0 / sigma**2)
        pflux = phot[k] * (1 + 0.05 * rng.standard_normal(phot[k].shape))
        perr = 0.05 * phot[k]
        absent = np.setdiff1d(np.arange(len(FILTERS)), PHOT_SETS[band_set[k]])
        pflux[absent] = np.nan
        perr[absent] = np.nan
        records.append(
            make_record(
                lo,
                hi,
                z[k],
                edges[k],
                masks[k],
                flux,
                ivar,
                pflux,
                perr,
                ebmv[k],
                covariance=banded_covariance(n, sigma**2),
            )
        )
    return records, truth, z, bounds, masks


t0 = time.perf_counter()
records, truth, z_gal, bounds, masks = make_mocks(template)
print(f"mocks: {len(records)} galaxies in {time.perf_counter() - t0:.1f} s")
print("fitted pixels per galaxy:", [int(m.sum()) for m in masks])
print("window redshifts:", np.round(z_gal, 3).tolist())

# %% [markdown]
# ## The bucket plan
#
# `build_spectro_batches` pads each galaxy's fitted pixel count up to a multiple of `quantum`. Each
# distinct padded count is one bucket, and each bucket compiles once. A larger `quantum` gives fewer
# buckets and more padding. With `quantum=512` the four galaxies share one padded count, so the fit
# compiles once. The padding waste below is measured against the fitted pixels.

# %%
QUANTUM = 512
buckets = build_spectro_batches(records, quantum=QUANTUM, model=template)
fitted_total = sum(int(m.sum()) for m in masks)
padded_total = sum(spec.n_max * len(index) for spec, _, index in buckets)
print(f"buckets at quantum={QUANTUM}: {len(buckets)}")
for spec, _, index in buckets:
    print(f"  n_max={spec.n_max}  galaxies={len(index)}  segments={spec.n_segments}")
print(f"padding over fitted pixels: {100 * (padded_total / fitted_total - 1):.1f} %")

# %% [markdown]
# ## Maximum a posteriori for all four
#
# `fit_spectra_map_vmap` runs an L-BFGS optimization for every galaxy in a bucket under one vmapped
# call, with at most 120 iterations; the optimizer stops earlier when the gradient tolerance is met.
# The first call includes the compile of each bucket. The second call repeats the fit with the
# compiled function. Calibration and emission lines are marginalized, so the MAP is the optimum of
# the marginal posterior.

# %%
MAP_STEPS = 120

t0 = time.perf_counter()
map_fit = fit_spectra_map_vmap(
    template, buckets, n_steps=MAP_STEPS, calibration_marginalize=True, cal_n_poly=3
)
t_first = time.perf_counter() - t0
print(f"first call (compile + fit): {t_first:.1f} s")

# %%
t0 = time.perf_counter()
map_fit = fit_spectra_map_vmap(
    template, buckets, n_steps=MAP_STEPS, calibration_marginalize=True, cal_n_poly=3
)
t_second = time.perf_counter() - t0
print(f"second call (fit only):     {t_second:.1f} s")

map_logm = np.asarray(map_fit["params"]["sfh_cont_log_total_mass"])
true_logm = truth["sfh_cont_log_total_mass"]
last_loss = np.asarray(map_fit["loss_history"][0])[-1]
print(f"{'gal':>3} {'z':>6} {'truth':>7} {'MAP':>7} {'MAP-truth':>10} {'final loss':>12}")
for i in range(N_GAL):
    print(
        f"{i:>3} {z_gal[i]:6.3f} {true_logm[i]:7.3f} {map_logm[i]:7.3f} "
        f"{map_logm[i] - true_logm[i]:+10.3f} {float(np.ravel(last_loss)[i]):12.3f}"
    )

# %% [markdown]
# ## Sampling all four
#
# `fit_spectra_catalog_mcmc` with the ChEES sampler runs every galaxy's chains, warm-started from
# the MAP. Each galaxy runs two chains, 60 warmup steps and 60 draws per chain. The draws come back
# with the chains flattened, one after the other, so the convergence check below is a split R-hat
# computed on the two halves of each galaxy's flattened draws. It is a split statistic on halves,
# not a between-chain R-hat. The interval is the 16 to 84 percent range of the draws.


# %%
def split_rhat(x):
    """Split R-hat of one parameter from the two halves of its draws, shape (n,)."""
    a, b = np.split(np.asarray(x, dtype=float), 2)
    n = a.size
    means = np.array([a.mean(), b.mean()])
    variances = np.array([a.var(ddof=1), b.var(ddof=1)])
    within = variances.mean()
    between = n * means.var(ddof=1)
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


t0 = time.perf_counter()
post = fit_spectra_catalog_mcmc(
    template,
    buckets,
    sampler="chees",
    init="map",
    precondition=1.0,
    n_warmup=60,
    n_samples=60,
    n_chains=2,
    map_steps=MAP_STEPS,
    calibration_marginalize=True,
    cal_n_poly=3,
)
t_mcmc = time.perf_counter() - t0
print(f"sampling wall time for {N_GAL} galaxies: {t_mcmc:.1f} s")

draws = np.asarray(post["draws"]["sfh_cont_log_total_mass"])
print(f"{'gal':>3} {'truth':>7} {'median':>8} {'16-84 width':>12} {'split R-hat':>12} {'div':>4}")
for i in range(N_GAL):
    lo16, med, hi84 = np.percentile(draws[i], [16, 50, 84])
    print(
        f"{i:>3} {true_logm[i]:7.3f} {med:8.3f} {hi84 - lo16:12.3f} "
        f"{split_rhat(draws[i]):12.3f} {int(post['divergences'][i]):>4}"
    )

# %% [markdown]
# ## Recovery against the truth
#
# Each point is a galaxy's posterior median with its 16–84 percent interval. The dashed line marks
# the truth.

# %%
fig, ax = plt.subplots(figsize=(4.2, 4.0))
lo16, med, hi84 = np.percentile(draws, [16, 50, 84], axis=1)
ax.errorbar(true_logm, med, yerr=[med - lo16, hi84 - med], fmt="o", color="#3a76d9", capsize=3)
edge = np.array(LOGM_RANGE)
ax.plot(edge, edge, ls="--", color="0.3", lw=1)
ax.set_xlabel(r"true $\log_{10}\,M_\ast/M_\odot$")
ax.set_ylabel(r"posterior $\log_{10}\,M_\ast/M_\odot$")
fig.tight_layout()
fig.savefig(FIG_DIR / "20_desi_dwarfs_recovery.png", dpi=120)
plt.show()

# %% [markdown]
# ## One galaxy's window
#
# The data of galaxy 0 with the MAP continuum. The marginal line amplitudes are not returned by
# `fit_spectra_map_vmap` or by the catalog sampler when `eline_mode="marginalized"`, so the model
# below carries the stellar, nebular and dust continuum, and the emission lines appear as residual
# structure. Set `eline_mode="fitted"` to return the `eline_amp_*` amplitudes.

# %%
g = 0
spec_b, batch_b, idx_b = next(
    (s, b, idx) for s, b, idx in buckets if g in set(np.asarray(idx).tolist())
)
local = int(np.where(np.asarray(idx_b) == g)[0][0])
params_b = {
    k: jnp.asarray(np.asarray(v))[jnp.asarray(idx_b)] for k, v in map_fit["params"].items()
}
model_flux = np.asarray(make_batched_predict(template, spec_b)(params_b, batch_b)["spec_fnu"])[
    local
]
n_g = bounds[g][1] - bounds[g][0]
w_g = WAVE[bounds[g][0] : bounds[g][1]]
good = masks[g]

fig, (ax_top, ax_res) = plt.subplots(
    2, 1, sharex=True, figsize=(7.0, 4.2), gridspec_kw={"height_ratios": [3, 1]}
)
ax_top.plot(w_g[good], records[g].flux[good], lw=0.5, color="0.5", label="data")
ax_top.plot(w_g[good], model_flux[:n_g][good], lw=0.9, color="#3a76d9", label="MAP continuum")
ax_top.set_ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_top.legend(frameon=False, fontsize=8)
ax_res.plot(w_g[good], (records[g].flux - model_flux[:n_g])[good], lw=0.5, color="0.3")
ax_res.set_xlabel(r"$\lambda_{\rm obs}$ [$\AA$]")
ax_res.set_ylabel("residual")
fig.tight_layout()
fig.savefig(FIG_DIR / "20_desi_dwarfs_window.png", dpi=120)
plt.show()

# %% [markdown]
# ## Real DESI coadds
#
# To fit DESI spectra, read the three camera spectra of each target with
# `tengri.io.desi.read_desi_cameras` and build the banded resolution with
# `tengri.io.desi.desi_resolution_matrix`. The cameras overlap in wavelength, so the reader keeps
# them separate and concatenates them in camera order, never sorted. The calibration segment ids
# of each pixel then follow the camera order, as they do here.
#
# Two requirements carry over from the mocks. The data and the sigma must be in the model's flux
# units, F_nu in erg/s/cm²/Hz, and sigma must be representable in float32 (above about 1e-38).
#
# This notebook fits four mock galaxies with short chains. A production run uses more galaxies and
# longer chains. `fit_spectra_catalog_mcmc` also accepts `sampler="ghmc"`, which returns a nested
# R-hat per galaxy. The batched spectroscopy page measures that on narrow DESI-like posteriors it
# does not reach NUTS, and it recommends `sampler="chees"` for catalog-wide posteriors.
