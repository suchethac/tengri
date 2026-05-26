"""
Stacking 1000 galaxies in seconds with JAX vmap
================================================

This gallery demonstrates population-scale analysis: draw 1,000 galaxy parameters
from the prior distribution, predict their rest-frame SEDs via ``jax.vmap``,
and visualize the median SED plus 16–84 percentile confidence band. The entire
computation completes on CPU in under 1 second after warm-up, showing how
effective vmap is for population-level diagnostics without per-galaxy compilation.
The results resemble a spectral stack (e.g. Eisenstein+2003 SDSS LRG stack),
revealing the mean galaxy SED morphology across a population prior.

This technique is essential for:

- **Prior predictive checks** — Does the default prior produce plausible galaxies?
- **Population diagnostics** — What is the mean SED shape for a given model?
- **Scaling studies** — Demonstrate near-linear throughput from 10s to 10,000s of galaxies.

Reference: Eisenstein et al. 2003, ApJ 585, 694–717 (SDSS DR1 spectral stacking);
Bradbury et al. 2018, arXiv:1811.02361 (JAX documentation).

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import time
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import recipes
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*wNE.*")

# Constants for SED display
C_AA_PER_S = 2.998e18

# Build the model from a standard recipe
BARE = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(ssp_data=BARE, **recipes.star_forming_photometry())

print(f"Model built with {len(model.spec.free_params)} free parameters.")

# Define population size
N_GALAXIES = 1000
key = jax.random.PRNGKey(42)

# Sample parameters for the population from the prior
print(f"Sampling {N_GALAXIES} galaxy parameters from the prior...")
t0_sample = time.perf_counter()
keys = jax.random.split(key, N_GALAXIES)
pop_params = jax.vmap(model.spec.sample)(keys)
sample_time = time.perf_counter() - t0_sample
print(f"  Sampling time: {sample_time * 1e3:.1f} ms")

# Compile the vmapped prediction function (warm-up call)
print("Compiling vmapped prediction kernel...")
t0_compile = time.perf_counter()
batch_predict = jax.jit(jax.vmap(model.predict_rest_sed))
predictions = batch_predict(pop_params)
jax.tree.map(lambda x: x.block_until_ready(), predictions)
compile_time = time.perf_counter() - t0_compile
print(f"  Compilation + first execution: {compile_time * 1e3:.1f} ms")

# Extract wavelengths and SEDs
wavelength = np.asarray(predictions.wavelength[0])  # Same for all
seds = np.asarray(predictions.sed)  # Shape: (N_GALAXIES, n_wave)

print(f"  SED shape: {seds.shape}")

# Compute median and percentile band
sed_median = np.median(seds, axis=0)
sed_p16 = np.percentile(seds, 16, axis=0)
sed_p84 = np.percentile(seds, 84, axis=0)

# Time a repeat execution (after warm-up)
print("Timing post-compilation throughput...")
t0_repeat = time.perf_counter()
n_repeats = 10
for _ in range(n_repeats):
    predictions = batch_predict(pop_params)
    jax.tree.map(lambda x: x.block_until_ready(), predictions)
repeat_time = (time.perf_counter() - t0_repeat) / n_repeats
print(
    f"  Average per execution: {repeat_time * 1e3:.2f} ms "
    f"({N_GALAXIES / repeat_time:.0f} galaxies/sec)"
)

# Plot: median SED + percentile band
fig, ax = plt.subplots(figsize=(8.0, 5.0))

# Convert to νL_ν for perceptual SED display
nu_l_nu_median = C_AA_PER_S / wavelength * sed_median
nu_l_nu_p16 = C_AA_PER_S / wavelength * sed_p16
nu_l_nu_p84 = C_AA_PER_S / wavelength * sed_p84

# Plot 16–84 percentile band as shaded region
ax.fill_between(
    wavelength,
    nu_l_nu_p16,
    nu_l_nu_p84,
    alpha=0.3,
    color="C0",
    label="16–84 percentile band",
)

# Plot median as thick line
ax.loglog(wavelength, nu_l_nu_median, color="C0", lw=2.0, label="Median (1000 galaxies)")

# Physical axis labels
ax.set(
    xlim=(500, 1e7),
    ylim=(1e37, 1e46),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
)

ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(True, alpha=0.2, which="both")

fig.tight_layout()
plt.savefig("plot_galaxy_stack_1000.png", dpi=150, bbox_inches="tight")

# Summary statistics
print(
    f"\nSummary:\n"
    f"  Population: {N_GALAXIES} galaxies\n"
    f"  Model: {recipes.star_forming_photometry.__doc__.split(chr(10))[0]}\n"
    f"  Median rest-frame L_ν range: "
    f"{nu_l_nu_median.min():.2e} — {nu_l_nu_median.max():.2e} erg/s\n"
    f"  16–84 span at 5500 Å: "
    f"{nu_l_nu_p84[np.argmin(np.abs(wavelength - 5500))] / nu_l_nu_p16[np.argmin(np.abs(wavelength - 5500))]:.2f}× "
    f"(dynamical range across prior)\n"
    f"  Throughput: {N_GALAXIES / repeat_time:.0f} galaxies/sec (post-compilation)"
)
