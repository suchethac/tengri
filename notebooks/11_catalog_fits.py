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
# # Catalog fits: a survey in parallel
#
# Notebooks [`05`](05_fitting_photometry.py)–[`10`](10_fastspecfit_joint_fit.py)
# fit *one* galaxy at a time. A survey is not one galaxy. Rubin LSST will deliver
# photometry for **billions** of galaxies; the DESI/Euclid value-added catalogs
# are already in the tens of millions. The unit of work is no longer a fit — it
# is a *catalog* of fits.
#
# The good news is that a catalog of independent galaxies is
# **embarrassingly parallel**: each galaxy is its own low-dimensional posterior,
# and nothing couples them. The naive way to exploit that is a Python `for` loop
# over per-galaxy fits — correct, but it pays the JIT compile and walks the
# galaxies one at a time. `Catalog` does the same fits as **one vectorized program**:
# `forward_chunk_size=K` galaxies advance their chains *together* on every
# sampler step, and the compiled graph is `O(1)` in the catalog size `N`. On a
# GPU those `K` chains fill the card's lanes at once — that is the throughput win.
#
# This notebook builds a **Rubin-LSST-style** scenario — LSST *ugrizy* plus the
# Euclid near-IR that a real LSST-era catalog carries — with **free redshift and
# dust** (a photometric-redshift fit) over an SSP that carries nebular emission,
# fits a mock catalog in parallel, and **prints the timing in detail**, because
# the whole point is how fast a catalog goes through.

# %%
from _setup import FIG_DIR, quiet

quiet()

# Notebook-specific: we pair the wNE SSP with baked-in nebular, as intended.
import warnings

warnings.filterwarnings("ignore", message=".*wNE.*")

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    plot,
)
from tengri import Catalog, ForwardModel

plot.setup_style()

C_POST, C_TRUTH = "#3a76d9", "0.15"

print(
    f"JAX backend: {jax.devices()[0].platform}   devices: {jax.device_count()}   x64: {jax.config.jax_enable_x64}"
)

# %% [markdown]
# ## The observation: LSST *ugrizy* + Euclid near-IR
#
# Rubin LSST measures six broad optical bands, *ugrizy* (0.32–1.0 μm). On its own
# that constrains a photometric redshift well below z ≈ 1, where the 4000 Å break
# still sits inside the optical. The Rubin–Euclid overlap adds the Euclid NISP
# *Y/J/H* near-IR (1.0–2.0 μm), which follows the break to higher redshift and
# pins the stellar mass — the "other bands that may be there" in a real LSST-era
# catalog. Nine bands total; the same machinery takes any set your survey has.

# %%
# A "wNE" SSP: nebular emission (lines + continuum) is baked into the templates
# at a fixed ionization parameter and escape fraction, so a broadband fit gets the
# emission-line boost in its colors with *no* per-step nebular emulator. `load_ssp`
# walks up to the repo `data/` directory to find the grid by its short alias.
ssp = tengri.load_ssp("prsc_miles_chabrier_wNE")
print(
    f"SSP: {ssp.source}\n  nebular provenance: {ssp.nebular!r}  (emission baked into the templates)"
)

LSST = ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]  # Rubin LSST optical
EUCLID = ["euclid_y", "euclid_j", "euclid_h"]  # Euclid NISP near-IR
FILTERS = LSST + EUCLID
phot_obs = Photometry.from_names(FILTERS)
print(f"{phot_obs.n_filters} bands: {', '.join(phot_obs.names)}")

# %% [markdown]
# ## A free-redshift photometry model with dust and nebular emission
#
# The model has **three free parameters** — the redshift, the stellar mass, and
# the diffuse dust optical depth — over a **stellar + nebular + dust** continuum.
# The nebular emission is not a fitted backend: it is *baked into the SSP* at a
# fixed ionization parameter and escape fraction (the wNE grid loaded above), so
# the emission lines shift through the LSST/Euclid bands with redshift and boost
# the broadband colors at **zero per-step cost** — no nebular emulator runs inside
# the sampler. Only the SFH *shape*, the metallicity, and the nebular
# logU/escape-fraction are held fixed.
#
# This is a more honest photo-*z* posterior than a dust-fixed fit: dust reddening
# and redshift both move the optical/near-IR colors, so leaving dust free lets the
# fit **marginalize the dust–redshift degeneracy** instead of pretending the dust
# is known. It is still low-dimensional enough that per-galaxy sampling (not VI)
# is cheap and correct. A flexible SFH and a fitted nebular backend are in
# notebooks [`05`](05_fitting_photometry.py) and
# [`10`](10_fastspecfit_joint_fit.py); here we keep the forward model light,
# because a catalog of thousands rewards a cheap gradient.
#
# We build on the **`WavePrecomp` fast path**. Its lookup table is tabulated over
# a redshift grid, so a *free*-redshift fit just interpolates the table — nebular
# lines and all — instead of re-integrating the SSP × filter product on every
# sampler step. One `WavePrecomp()` build is shared by every galaxy in the catalog.

# %%
Z_PRIOR = (0.05, 1.5)


def build_model():
    obs = Observation(photometry=phot_obs)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Uniform(*Z_PRIOR),  # <-- free redshift: a photo-z fit
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(8.0, 12.0)},
        met={"logzsol": Fixed(0.0)},
        # Free diffuse dust optical depth: marginalize the dust-redshift degeneracy.
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "ssp"},  # nebular emission is baked into the wNE SSP (fixed logU/fesc)
        approx=WavePrecomp(n_z=120, z_min=0.0, z_max=1.6),
    )


t0 = time.perf_counter()
model = build_model()
build_wall = time.perf_counter() - t0
print(f"Free parameters ({model.spec.n_free}): {', '.join(model.spec.free_params)}")
print(f"WavePrecomp LUT built in {build_wall:.1f} s (one-time; content-hash-cached on re-runs)")

# %% [markdown]
# ## A mock catalog
#
# `N` galaxies spread across redshift and stellar mass, each a noisy realization
# of its own truth at a fixed depth (S/N 20). The catalog is a list of
# `{"flux_obs", "noise"}` dicts, stacked into the table `Catalog` ingests. Truths are
# drawn once so we can score the recovery at the end.

# %%
N_GAL = 12
_rng = np.random.default_rng(1)
PHOT_SNR = 20.0

z_true = _rng.uniform(0.2, 1.0, N_GAL)
logm_true = _rng.uniform(9.5, 11.0, N_GAL)
tau_true = _rng.uniform(0.05, 1.0, N_GAL)  # per-galaxy diffuse dust optical depth

_fixed = model.spec.get_fixed_values()
galaxies = []
for z, lm, tau in zip(z_true, logm_true, tau_true):
    truth = {
        **_fixed,
        "redshift": jnp.asarray(z),
        "sfh_dpl_log_total_mass": jnp.asarray(lm),
        "dust_tau_diff": jnp.asarray(tau),
    }
    p = np.asarray(model.predict_photometry(truth))
    n = np.abs(p) / PHOT_SNR
    f = p + _rng.normal(size=p.shape) * n
    galaxies.append({"flux_obs": jnp.asarray(f), "noise": jnp.asarray(n)})

print(
    f"Mock catalog: {N_GAL} galaxies, {phot_obs.n_filters} bands, SNR {PHOT_SNR:.0f}\n"
    f"  z      in [{z_true.min():.2f}, {z_true.max():.2f}]\n"
    f"  logM   in [{logm_true.min():.2f}, {logm_true.max():.2f}]\n"
    f"  tau_d  in [{tau_true.min():.2f}, {tau_true.max():.2f}]"
)

# %% [markdown]
# ## Fit the catalog in parallel — timed in detail
#
# One call fits the whole catalog. `Catalog.fit(method="mcmc_hmc",
# forward_chunk_size=K)` builds a **single** JIT'd HMC program and streams the `N`
# galaxies through `jax.lax.map(..., batch_size=K)`: `K` galaxies advance their
# chains *together* on every sampler step, and the compiled graph is `O(1)` in the
# catalog size `N` — one program whether you fit 12 galaxies or 12 million. We use
# **HMC** (the catalog default): its fixed-length trajectories are cheap and
# predictable, which matters for a *catalog* — NUTS can spend a whole step building
# a deep tree on a banana-shaped photo-z posterior, and paid once per galaxy that
# adds up. We set `K = N` (fit them all at once), a diagonal mass matrix (each
# galaxy is low-D and the parallelism is *width over galaxies*), and one chain per
# galaxy. We time a single forward evaluation first, then the catalog.

# %%
FIT_KW = dict(
    method="mcmc_hmc",
    n_warmup=100,
    n_samples=120,
    n_burnin=20,
    n_leapfrog_steps=20,  # longer trajectories mix the dust-redshift degeneracy
    target_accept_rate=0.85,
    dense_mass_matrix=False,
    verbose=False,
)


# The Catalog noun takes a table of named columns; stack the per-galaxy dicts.
forward = ForwardModel.build(sed=model)
band_names = list(forward.observation.photometry.names)
flux_arr = np.stack([np.asarray(g["flux_obs"]) for g in galaxies])  # (N, n_bands)
noise_arr = np.stack([np.asarray(g["noise"]) for g in galaxies])
table = {}
for j, b in enumerate(band_names):
    table[b] = flux_arr[:, j]
    table[f"{b}_err"] = noise_arr[:, j]
cat = Catalog(forward, table, flux_unit="cgs_fnu")  # free redshift -> no redshift_col


def fit_catalog(K):
    """Fit the whole catalog with chunk size K; return (wall seconds, CatalogPosterior)."""
    t0 = time.perf_counter()
    cp = cat.fit(key=jax.random.PRNGKey(0), forward_chunk_size=K, **FIT_KW)
    jax.block_until_ready(cp.posteriors[0].samples["redshift"])
    return time.perf_counter() - t0, cp


# One forward photometry evaluation — the inner cost the sampler pays per leapfrog.
_probe = {
    **_fixed,
    "redshift": jnp.asarray(0.5),
    "sfh_dpl_log_total_mass": jnp.asarray(10.0),
    "dust_tau_diff": jnp.asarray(0.3),
}
jax.block_until_ready(model.predict_photometry(_probe))  # warm the trace
t0 = time.perf_counter()
for _ in range(100):
    jax.block_until_ready(model.predict_photometry(_probe))
fwd_ms = (time.perf_counter() - t0) / 100 * 1e3

fit_wall, catalog = fit_catalog(N_GAL)
n_div = catalog.diagnostics["n_divergent_total"]
print(
    f"fit {N_GAL} galaxies ({model.spec.n_free}-D each) in {fit_wall:.1f} s "
    f"= {fit_wall / N_GAL:.2f} s per posterior; {n_div} divergences"
)

# %% [markdown]
# ## Method: why HMC and not VI or NUTS?
#
# The native VI backends raise `NotImplementedError` on this catalog path
# (per-galaxy redshift + presence masks are incompatible with vmapped VI). On the
# batched path, use `method="mcmc_nuts"` or `method="mcmc_hmc"` (both vectorizable)
# or `method="map"` (sequential). HMC's fixed-length trajectories keep per-posterior
# cost predictable across the catalog.

# %% [markdown]
# ## The timing, in detail
#
# The whole catalog is one compiled program, `O(1)` in catalog size `N`: the compile
# is paid once (and the persistent JAX cache elides it on re-runs).
#
# **Batching speeds up per-galaxy cost** even on a single CPU. Advancing `K`
# galaxy-chains together makes every array `K` times larger, which feeds vector units
# and amortizes per-step overhead.

# %%
backend = jax.devices()[0].platform.upper()
per_gal = fit_wall / N_GAL
print(f"{'quantity':<42}{'value':>15}")
print("-" * 57)
print(f"{'backend':<42}{backend:>15}")
print(f"{'catalog size N':<42}{N_GAL:>15d}")
print(f"{'free parameters per galaxy':<42}{model.spec.n_free:>15d}")
print(f"{'WavePrecomp LUT build (one-time)':<42}{build_wall:>13.1f} s")
print(f"{'one forward photometry eval':<42}{fwd_ms:>12.2f} ms")
print(f"{'catalog fit wall (K = N)':<42}{fit_wall:>13.1f} s")
print(f"{'  time per posterior (CPU)':<42}{per_gal:>13.2f} s")
print(f"{'  throughput (posteriors / s)':<42}{N_GAL / fit_wall:>15.2f}")
print("-" * 57)
print(
    f"Each posterior is ~{FIT_KW['n_warmup'] + FIT_KW['n_samples']} HMC iterations x "
    f"{FIT_KW['n_leapfrog_steps']} leapfrog steps ~ "
    f"{(FIT_KW['n_warmup'] + FIT_KW['n_samples']) * FIT_KW['n_leapfrog_steps']} gradient\n"
    f"evaluations of the forward model ({fwd_ms:.2f} ms each) — that, not the 3 free\n"
    "parameters, is the cost. Batching K chains amortizes the per-step overhead: the\n"
    "K-sweep below shows the per-galaxy time falling as K grows, even on this CPU."
)

# %% [markdown]
# ## The parallel speedup: a chunk-size sweep
#
# `forward_chunk_size=K` sets how many galaxy-chains advance *together* per sampler
# step: `K=1` runs them one at a time (the serial baseline), `K=N` batches the whole
# catalog. `K` changes *only* the throughput, never the posterior (the vectorization
# is bit-exact). We re-fit the same catalog at a few `K` — **same sampler settings
# as above**, so the `K=N` row *is* the headline number — and time each. The wall
# includes the one-time compile a first run pays; at this sampler depth the ~4400
# gradient evaluations dwarf it, so a single wall is a stable measure.

# %%
K_VALUES = [1, 4, N_GAL]

# Same `fit_catalog` (and so the same FIT_KW) as the science fit above — the K=N
# point *is* that fit, reused rather than re-run.
sweep_wall = [fit_catalog(1)[0], fit_catalog(4)[0], fit_wall]
sweep_per_gal = [w / N_GAL for w in sweep_wall]
_serial = sweep_per_gal[0]

print(f"{'chunk K':>8}{'wall (s)':>11}{'s / galaxy':>13}{'speedup':>10}")
print("-" * 42)
for K, w, pg in zip(K_VALUES, sweep_wall, sweep_per_gal):
    tag = "  serial" if K == 1 else ("  batched" if K == N_GAL else "")
    print(f"{K:>8d}{w:>11.1f}{pg:>13.2f}{_serial / pg:>8.1f}x{tag}")
print(
    f"\nBatching all {N_GAL} chains is {_serial / sweep_per_gal[-1]:.1f}x faster per galaxy "
    f"than K=1 — on one {backend}, no GPU."
)

# %%
fig, ax = plt.subplots(figsize=(5.4, 4.1))
ax.plot(K_VALUES, sweep_per_gal, "o-", color=C_POST, mec="white", mew=0.9, ms=9, lw=1.8)
for K, pg in zip(K_VALUES, sweep_per_gal):
    ax.annotate(
        f"{_serial / pg:.1f}x",
        (K, pg),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9,
        color=C_TRUTH,
    )
ax.set_xticks(K_VALUES)
ax.set_xlabel("chunk size K  (galaxy-chains batched together)")
ax.set_ylabel("wall time per galaxy (s)")
ax.set_ylim(bottom=0)
ax.set_title(f"Per-galaxy cost falls as K grows ({backend}, {N_GAL} galaxies)")
fig.tight_layout()
fig.savefig(FIG_DIR / "11_catalog_ksweep.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Did the parallel fit recover the truth?
#
# Each galaxy in the `CatalogPosterior` is a full `Posterior` (`catalog.posteriors[i]`),
# carrying its own chain in `.samples`. The three free parameters are stacked over the
# galaxy axis, 16/50/84 percentiles computed per galaxy, and compared to the injected
# truth.
#
# The dust fit (rather than fixed) marginalizes the redshift–dust degeneracy,
# so the intervals are more honest than a dust-fixed model would report. They remain
# *conditional* on the fixed SFH shape, metallicity, and the SSP's baked-in nebular
# logU/escape-fraction. A fully flexible SFH and a *fitted* nebular backend are in
# notebooks [`05`](05_fitting_photometry.py) and
# [`10`](10_fastspecfit_joint_fit.py); this notebook is the parallel machinery and throughput.


# %%
def stack_samples(name):
    """Per-galaxy posterior draws of a free parameter, shape (N, n_samples)."""
    return np.stack([np.asarray(p.samples[name]) for p in catalog.posteriors])


z_samp = stack_samples("redshift")  # (N, n_samples)
m_samp = stack_samples("sfh_dpl_log_total_mass")  # (N, n_samples)
t_samp = stack_samples("dust_tau_diff")  # (N, n_samples)
z_lo, z_med, z_hi = np.percentile(z_samp, [16, 50, 84], axis=1)
m_lo, m_med, m_hi = np.percentile(m_samp, [16, 50, 84], axis=1)
t_lo, t_med, t_hi = np.percentile(t_samp, [16, 50, 84], axis=1)

dz = (z_med - z_true) / (1.0 + z_true)  # standard photo-z residual
sig_nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
dlogm = float(np.median(np.abs(m_med - logm_true)))
dtau = float(np.median(np.abs(t_med - tau_true)))
z_cov = int(np.sum((z_lo <= z_true) & (z_true <= z_hi)))
m_cov = int(np.sum((m_lo <= logm_true) & (logm_true <= m_hi)))
t_cov = int(np.sum((t_lo <= tau_true) & (tau_true <= t_hi)))
print(f"Photo-z:   sigma_NMAD(dz/(1+z)) = {sig_nmad:.3f}   (68% interval covers {z_cov}/{N_GAL})")
print(f"log M*:    median |Δ| = {dlogm:.3f} dex             (68% interval covers {m_cov}/{N_GAL})")
print(f"dust tau:  median |Δ| = {dtau:.3f}                 (68% interval covers {t_cov}/{N_GAL})")

# %%
fig, (axz, axm, axt) = plt.subplots(1, 3, figsize=(13.5, 4.6))

_ebar = dict(fmt="o", ms=5, color=C_POST, mec="white", mew=0.6, elinewidth=1.0, capsize=2)

axz.errorbar(z_true, z_med, yerr=[z_med - z_lo, z_hi - z_med], label="posterior 16/50/84", **_ebar)
_zline = np.array(Z_PRIOR)
axz.plot(_zline, _zline, color=C_TRUTH, lw=1.0, ls="--", label="1:1")
axz.set_xlim(0.15, 1.05)
axz.set_ylim(0.15, 1.05)
axz.set_xlabel("true redshift")
axz.set_ylabel("recovered redshift (photo-z)")
axz.set_title(rf"Photo-z recovery  ($\sigma_{{\rm NMAD}}$ = {sig_nmad:.3f})")
axz.legend(frameon=False, fontsize=8.5, loc="upper left")

axm.errorbar(logm_true, m_med, yerr=[m_med - m_lo, m_hi - m_med], **_ebar)
_mline = np.array([9.3, 11.2])
axm.plot(_mline, _mline, color=C_TRUTH, lw=1.0, ls="--")
axm.set_xlabel(r"true $\log M_\star$")
axm.set_ylabel(r"recovered $\log M_\star$")
axm.set_title(rf"Stellar-mass recovery  ($|\Delta|$ = {dlogm:.3f} dex)")

axt.errorbar(tau_true, t_med, yerr=[t_med - t_lo, t_hi - t_med], **_ebar)
_tline = np.array([0.0, 1.05])
axt.plot(_tline, _tline, color=C_TRUTH, lw=1.0, ls="--")
axt.set_xlim(0.0, 1.05)
axt.set_ylim(0.0, 1.6)
axt.set_xlabel(r"true dust $\tau_{\rm diff}$")
axt.set_ylabel(r"recovered dust $\tau_{\rm diff}$")
axt.set_title(rf"Dust recovery  ($|\Delta|$ = {dtau:.3f})")

fig.suptitle(f"Catalog of {N_GAL} galaxies fit in parallel (K={N_GAL})", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "11_catalog_recovery.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# - A catalog of independent galaxies is **embarrassingly parallel**. The default
#   is **per-galaxy HMC** (`mcmc_nuts` vectorizes too), where fixed-length
#   trajectories keep the per-posterior cost predictable.
# - **`Catalog.fit(method="mcmc_hmc", forward_chunk_size=K)`** fits the whole
#   catalog as one vectorized program: `K` galaxies advance per sampler step, the
#   compiled graph is `O(1)` in catalog size, and the compile is paid once and
#   amortized. Changing `K` changes only throughput, never the posterior — the
#   vectorization is bit-exact.
# - **The per-posterior cost is sampler iterations, not the small parameter space.**
#   Batching `K` chains amortizes per-step overhead, so per-galaxy time falls as
#   `K` grows — a factor of several on one CPU from batching alone.
# - **Free redshift rides `WavePrecomp`** — the LUT is tabulated over redshift,
#   so a photo-z fit interpolates it instead of re-integrating per step. Baking
#   nebular emission into the SSP (the wNE grid) keeps the line-boosted colors at
#   **zero per-step cost**.
# - A **GPU** extends batching much further, scaling to thousands of galaxies in
#   a single call.
