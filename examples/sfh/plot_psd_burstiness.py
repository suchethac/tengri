"""
PSD Parameter Space and Burstiness
===================================

Visualize how the two DRW PSD parameters -- sigma (amplitude) and
tau (damping timescale) -- map to different levels of SFH burstiness.
A 3x3 grid of GP-modulated SFHs shows the effect of each parameter.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    compute_sqrt_power_drw,
    generate_gp_fourier,
    make_log_age_grid,
    tsnorm,
)

# --- Grid setup ---
n_grid = 256
log_age_grid = make_log_age_grid(n_grid)
d_log_age = float(log_age_grid[1] - log_age_grid[0])
t_lookback = 10.0 ** log_age_grid
t_gyr = np.array(t_lookback) / 1e9

mean_sfr = tsnorm(t_lookback, log_peak_sfr=1.0, peak_lbt=6e9,
                  width=2e9, skew=0.5, trunc=3.0)

# --- Parameter grid ---
sigmas = [0.2, 0.6, 1.2]
taus_myr = [30, 200, 1000]

fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
key_base = jax.random.PRNGKey(7)

for i, sigma in enumerate(sigmas):
    for j, tau in enumerate(taus_myr):
        ax = axes[i, j]
        sqrt_p = compute_sqrt_power_drw(n_grid, d_log_age, sigma, tau * 1e6)

        # Plot 3 realizations
        for k in range(3):
            key = jax.random.fold_in(key_base, i * 100 + j * 10 + k)
            gp = generate_gp_fourier(key, sqrt_p, n_grid)
            variance = float(jnp.var(gp))
            sfr = mean_sfr * jnp.exp(gp - variance / 2.0)
            ax.plot(t_gyr, np.array(sfr), lw=0.7, alpha=0.7)

        ax.plot(t_gyr, np.array(mean_sfr), "k--", lw=1.0, alpha=0.5)
        ax.set_xlim(0, 14)

        if i == 0:
            ax.set_title(rf"$\tau = {tau}$ Myr", fontsize=10)
        if j == 0:
            ax.set_ylabel(rf"$\sigma = {sigma}$" + "\nSFR [M$_\\odot$/yr]",
                          fontsize=9)
        if i == 2:
            ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle("PSD Burstiness Grid: sigma (rows) vs tau (columns)",
             fontsize=13, y=1.01)
fig.tight_layout()
plt.savefig("plot_psd_burstiness.png", dpi=150, bbox_inches="tight")
plt.show()
