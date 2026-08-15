# Notebook Design Language

**For agents and contributors writing or editing tengri notebooks.**

## Architecture

Each notebook is a **jupytext percent-format `.py` file** in `notebooks/`. These are the source of truth — edit them directly.

### File format

- `# %%` marks code cell boundaries
- `# %% [markdown]` marks markdown cells (content prefixed with `# `)
- LaTeX works naturally: `$\alpha$`, `$$\int$$` (no escaping needed)
- Files are valid Python scripts (can be run with `python notebooks/00_quickstart.py`)

### Workflow

```bash
# Edit the .py file (in any editor, IDE, or via AI agent)
vim notebooks/00_quickstart.py

# Open in Jupyter (jupytext presents it as a notebook)
jupyter lab notebooks/00_quickstart.py

# Generate .ipynb if needed (e.g., for nbconvert/Sphinx)
jupytext --to ipynb notebooks/00_quickstart.py

# Sync all paired notebooks
cd notebooks && jupytext --sync *.py
```

### Rules for agents

1. **Edit `.py` files only** — never edit `.ipynb` files
2. Code cells: delimited by `# %%`
3. Markdown cells: delimited by `# %% [markdown]`, each line prefixed with `# `
4. To add a new cell, insert a `# %%` marker
5. Keep `_plot_style.py` — it provides `setup_style()`, `COLORS`, `safe_corner`

## API Patterns (tengri-specific)

### Data Loading
```python
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
model = Model(spec, ssp_data, filters=filters)
```

### SSPData Attributes
- `ssp_data.ssp_wave` — wavelength (Angstrom)
- `ssp_data.ssp_flux` — shape (n_met, n_age, n_wave)
- `ssp_data.ssp_lg_age_gyr` — log10(age/Gyr)
- `ssp_data.ssp_lgmet` — log10(Z/Zsun)

### Filter API
- `load_filter_set(names)` returns 3-tuple: `(filter_waves, filter_trans, filter_curves)`
- `FilterCurve` has `.wave`, `.trans`, `.name` — **NO `.wave_effective`**
- Hardcode SDSS effective wavelengths: `[3551, 4686, 6166, 7480, 8932]` Angstrom

### Model API
- `Model(spec, ssp_data, filters=filters)` — filters is 3-tuple or list of FilterCurve
- `model.predict_photometry(params)` — shape (n_filters,)
- `model.predict_spectrum(params, wave_obs)` — wave_obs passed directly
- `model.predict_sfh(params)` — returns dict with `t_gyr`, `sfr_mean`, `sfr_full`
- `model.plot_sfh_posterior(posterior, true_params=, ax=, color=, label=)`

### Spectroscopy
```python
model._wave_obs = wave_obs  # MUST set before creating Fitter
fitter = Fitter(model, spec_obs, noise, data_type="spectroscopy")
```

### Posterior API
- `result.summary()` returns `{param: {"median", "lo_68", "hi_68"}}` for samplers, `{"value"}` for MAP
- `result.plot_corner(params=, truths=, color=, fig=, label=)` — supports overlay via `fig=`
- `result.effective_sample_size()` — dict of ESS values
- `result.samples` — dict of arrays, `psd_xi` has shape `(n_samples, n_grid)`
- When iterating samples with `psd_xi`, keep it as array: `arr[k_idx]` not `float(arr[k_idx])`

### ParamSpec
- `spec.free_params` (not `free_names`) — list of free parameter names
- `spec.n_free` — count of free params (excludes psd_xi)
- `spec.sample(key)` — returns dict including `psd_xi` for stochastic models
- `spec.get_distribution(name)` — returns Distribution object
- Distribution has `.bounds` (tuple) — NOT `.high`/`.low`

### Low-level SFH Functions
- `compute_sqrt_power_drw(n_points, d_log_age, sigma_ps, tau_ps)` — in `gp_sfh.py`
- `gp_from_xi(xi, sqrt_power, n_points)` — 3 args
- `drw_variance(sigma_ps)` — 1 arg, returns sigma^2/2
- `double_powerlaw(t_lookback_yr, alpha, beta, tau_yr, norm)` — **lookback time in years**
- `psd_drw(omega, sigma_ps, tau_ps)` — in `psd_models.py`
- `drw_acf(delta_t, sigma_ps, tau_ps)` — in `psd_models.py`

## Recurring Patterns (every fitting notebook)

1. **Corner plot** with truth lines — wrap in `try/except ValueError` for degenerate posteriors
2. **SFH recovery** — linear lookback time, 16-84% fill, truth overlay
3. **Photometry → Spectroscopy flow** — show both, compare
4. **RT + geoVI side by side** — always show both samplers
5. **NUTS where feasible** — gold standard for D < 15
6. **Diagnostics**: ESS, acceptance rate, wall time
7. **Posterior predictive check**: model predictions overlaid on data

## Plotting conventions (BAGPIPES-inspired)

- **Style**: Use `tengri.plotting.setup_style()` or notebook `_plot_style.setup_style()` — 18pt axis labels, 14pt ticks, inward ticks on all four sides, 2pt lines, no legend frame.
- **SFH axis**: Lookback time with **present at the right** (reversed axis: `ax.set_xlim(13.5, 0)` so high lookback is left).
- **Primary posterior fill**: Use navajowhite or a light tint of the sampler color for 68% CI fill where a single method is highlighted; multi-method overlay uses COLORS["rt"], COLORS["geovi"], COLORS["nuts"].
- **Truth**: Solid black or near-black (`#1a1a1a`) line; dashed for markers in derived-quantity plots.
- **Corner plots**: Overlay multiple samplers in one figure where possible; use `safe_corner` or `plot_corner_comparison` to handle degenerate posteriors without crashing.

## Style

- Heavy pedagogical markdown — "Why" before "How"
- LaTeX equations in `$$...$$` blocks for key results
- Astronomer-friendly language (SFH, not "latent signal field")
- Cite papers inline: Behroozi (2025), Frank et al. (2021), etc.
- Units on all axes: Gyr, Angstrom, Msun/yr
- Use `plt.tight_layout()` and `plt.show()` after every figure

## Notebook Suite

| # | Notebook | Purpose | Status |
|---|----------|---------|--------|
| 00 | quickstart | Hook — 60-second demo | PASS |
| 01 | the_model | IFT theory | PASS |
| 02 | forward_model | SPS pipeline | PASS |
| 03 | inference_methods | Centerpiece — 5 samplers | PASS |
| 04 | recovery_tests | Mock validation | WIP |
| 05 | hierarchical | Population PSD | WIP |
| 06 | data_information | Progressive reveal | WIP |
| 07 | spectroscopic | Spectral recipes | WIP |
| 08 | psd_physics | PSD ↔ astrophysics | WIP |
| 09 | custom_models | Developer guide | WIP |
