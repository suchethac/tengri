# quickstart — audit

Counter: 2/2 images reviewed (all scripts have rendered PNG).

## sphx_glr_plot_first_fit_001.png

- **Script**: `/Users/suchethacooray/Projects/tengri/examples/quickstart/plot_first_fit.py`
- **Status**: ISSUE DETECTED (API migration needed)
- **Visual**: Excellent. Clear three-panel layout: main photometry fit with Truth vs MAP vs Observed SNR=20, plus inset SFR history subplot. No truncation, no glitches. Legend readable, colors well-contrasted.
- **Code quality**: Ruff passes cleanly. Imports correct (JAX, numpy, matplotlib). SSP path resolution via `_find_ssp()` is robust (handles sphinx-gallery cwd). Filter loading via `_FILTER_DIR` robust across 4 possible paths.
- **Newcomer clarity**: Docstring is concise and well-paced: "simplest possible tengri workflow", "define SFH model → generate mock → fit with MAP → plot". Annotations on y-axis clear ("Flux density [erg/s/cm$^2$/Hz]"). Two-liner docstring introduces the task immediately.
- **CRITICAL ISSUE — API migration**: Script uses **flat-kwarg `Parameters(...)`** constructor (lines 76–87), which is the legacy expert-escape-hatch path. CLAUDE.md states (as of 2026-05): "The **recommended path** is the nested-dict builder shipped in 2026-05 (`SEDModel.from_groups` + `tengri.recipes.*`)" and explicitly marks flat-kwarg as "still used internally, but **not the recommended user-facing path**." This is the entry-point script for all new users. **Action required:** Rewrite to use `SEDModel.from_groups(ssp_data=ssp, observation=obs, **recipes.star_forming_photometry())` or hand-rolled nested-dict form.
- **Units**: Correct throughout. Wavelength [A] on x-axis, Flux density [erg/s/cm$^2$/Hz] on y-axis, axis label formatting with LaTeX math mode. Inset shows SFR correctly.
- **Canonical names**: Uses `SEDModel`, `Observation`, `Photometry`, `Fitter` — all canonical (no deprecated `Model`, `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig` detected). Good.
- **Style notes**: figsize=(7, 4) for main, figsize optimal for readability. `setup_style()` applied. No gridlines (correct house style). Legend frameon=False. Error bars on observed data help readers understand SNR. Inset SFR subplot adds pedagogical value — shows what the fit is recovering.

## sphx_glr_plot_sed_components_001.png

- **Script**: `/Users/suchethacooray/Projects/tengri/examples/quickstart/plot_sed_components.py`
- **Status**: ISSUE DETECTED (API migration + physics redundancy)
- **Visual**: Clean two-component spectrum plot (intrinsic blue, attenuated orange, dust-absorbed shaded). Log-log axes, no truncation, dust absorption effect obvious via the filled region. Readable legend, appropriate wavelength range (0.09–3 μm), y-axis range spans 9 orders of magnitude cleanly.
- **Code quality**: Ruff passes cleanly. SSP loading and filter path resolution robust (identical to plot_first_fit.py). Manual dust-off computation (lines 97–99) creates a second parameter dict with `dust_tau_bc` and `dust_tau_diff` set to 0.
- **Newcomer clarity**: Docstring: "Predict a galaxy SED and visualize its components: the intrinsic stellar emission and the dust-attenuated total." Good. "Uses the lazy `model.predict()` API and direct SED computation to show the effect of dust attenuation on the spectrum." Pedagogically clear — two methods shown (predict_rest_sed, manual dust-off). Parameter dict values all Fixed (line 76–87), so model is deterministic and shows pure physics without fitting complexity. Good intro for newcomers.
- **CRITICAL ISSUE — API migration**: Same as plot_first_fit.py. Script uses flat-kwarg `Parameters(...)` (lines 75–87). **Action required:** Migrate to `SEDModel.from_groups(..., **recipes.star_forming_photometry())` or equivalent.
- **Secondary issue — physics redundancy**: Lines 97–99 manually zero dust parameters. This is pedagogically reasonable (shows that dust attenuation is additive), but the CLAUDE.md convention (from feedback_analysis_module_role.md and project_sed_model_split.md) notes that component isolation should be handled by the forward model's `precompute` cache, not hand-rolled in user code. However, for a quickstart teaching "how dust affects the SED," this is acceptable. A future refactor could add a convenience method like `model.predict_rest_sed(params, exclude_dust=True)` or show how to construct a dust-free model via groups dict. Current code is transparent enough for newcomers.
- **Units**: Excellent. X-axis: wavelength in μm (user-facing convention for panchromatic plots; correct for newcomers). Y-axis: L_ν [erg/s/Hz] (canonical tengri unit). Mask (900–30,000 A) sensible and documented with comment.
- **Canonical names**: Uses `SEDModel`, `Parameters`, `Observation`, `Photometry` — all canonical. No deprecated names.
- **Style notes**: figsize=(9, 4.5) for log-log spectrum (good aspect ratio). `setup_style()` applied. Log scales on both axes with clean limits (xlim 0.09–3 μm, ylim 1e20–1e29 erg/s/Hz). Legend frameon=False, loc="upper right" (dust absorption at high z makes upper right uncluttered). No gridlines. Fill_between `alpha=0.15` makes dust absorbed region visible without obscuring lines.

## Section observations

**Alignment with recent 2026-05 API changes:**
Both scripts predate the nested-dict builder rollout (shipped 2026-05-16 per CLAUDE.md and memory/project_nested_dict_builder.md). They use the legacy flat-kwarg `Parameters(...)` form. This was explicitly marked as "not the recommended user-facing path" in CLAUDE.md. **Since these are the first examples new users encounter**, the inconsistency signals backward direction: newcomers will learn the old API first, then later discovery that the new recommended path exists. This violates the principle that documentation should model best practices.

**Migration impact:**
- **plot_first_fit.py**: Rewrite lines 76–87. Replace with `SEDModel.from_groups(ssp_data=ssp, observation=obs, **recipes.star_forming_photometry())` — cleaner, shorter, introduces recipes.
- **plot_sed_components.py**: Rewrite lines 75–87. Use nested-dict builder with all parameters Fixed. E.g.:
  ```python
  model = SEDModel.from_groups(
      ssp_data=ssp,
      observation=obs,
      sfh={'type': 'tsnorm', '*': Fixed(...)},  # all Fixed values
      dust={'type': 'two_component', 'law_bc': 'calzetti', '*': Fixed(...)},
      redshift=Fixed(0.0),
  )
  ```
  Or: use recipe and then override to all-Fixed if that's clearer for pedagogy.

**Consistency check — visual rendering:**
Both PNGs render correctly. No blank figures, no truncation, no missing axis labels. Figures match their docstrings.

**Canonical naming:**
Both scripts clean. No deprecated API (`Model`, `ParamSpec`, `SpectroscopyConfig`, etc.) detected.

**Ruff/style compliance:**
Both pass `ruff check` with zero violations. Imports organized. No print statements (good). Line lengths under 99 chars.

**Units and physics accuracy:**
- plot_first_fit.py: Flux [erg/s/cm$^2$/Hz] correct. Wavelength in Angstrom (rest-frame). Inset SFR history adds clarity.
- plot_sed_components.py: L_ν [erg/s/Hz] correct. Wavelength in μm (user-facing panchromatic convention). Dust attenuation pedagogy sound.

---

## Tally

- **OK**: 1 (plot_sed_components — visual/physics sound, API migration needed)
- **MINOR**: 0
- **MAJOR**: 2 (both scripts use legacy flat-kwarg API instead of recommended nested-dict builder)
- **BROKEN**: 0

**Report path**: `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/quickstart.md`

**Action required:** Rewrite both plot_first_fit.py and plot_sed_components.py to use `SEDModel.from_groups(...)` + `tengri.recipes` as the primary example. This is the advertised recommended path as of 2026-05 and should be the first thing new users see.
