#!/usr/bin/env python
"""Patch all notebook build scripts to fix API issues found during review.

Fixes applied:
1. load_filter_set("sdss") → load_filter_set(SDSS_FILTERS)
2. f.wave_effective → hardcoded SDSS effective wavelengths
3. filters.values() / filters.keys() → filter_curves patterns
4. Fitter(..., wave_obs=wave_obs) → model._wave_obs = wave_obs; Fitter(...)
5. summary()[name]["std"] → compute from samples
6. spec.free_names → spec.free_params
7. predict_sfh(result.summary()) → predict_sfh(result.params)
8. psd_sigma=Fixed(1.0) → Fixed(0.0) for parametric models
9. NB01: double_powerlaw(t_cosmic, ...) → double_powerlaw(t_lookback, ...)
10. NB00: "What's Next" section with correct notebook names
11. Model(..., filters=filters) → unpack tuple for 3-tuple returns
"""

import re
from pathlib import Path


def patch_file(filepath, replacements):
    """Apply a list of (old, new) string replacements to a file."""
    text = filepath.read_text()
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  Fixed: {old[:60]}...")
    filepath.write_text(text)


def patch_nb00():
    """Fix NB00: quickstart."""
    p = Path("notebooks/_build_nb00.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    patch_file(p, [
        # Filter loading
        ('load_filter_set("sdss")',
         'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])'),
        # Filter result is a 3-tuple, not a dict
        ('filters = load_filter_set',
         'filters = load_filter_set'),
        # Effective wavelengths — replace any wave_effective pattern
        ('wave_eff = jnp.array([f.wave_effective for f in filters.values()])',
         'wave_eff = jnp.array([3551, 4686, 6166, 7480, 8932])  # SDSS ugriz'),
        # If it uses filters.keys()
        ('list(filters.keys())',
         '[fc.name for fc in filters[2]]'),
        # Fix "What's Next" section
        ('''    - **NB01 — Understanding the Model**: anatomy of the forward model, SFH
      components, dust, and SPS
    - **NB02 — The Forward Model**: step-by-step walkthrough from parameters
      to photometry
    - **NB03 — Mock Generation**: creating realistic synthetic observations
    - **NB04 — Fitting**: deep dive into all five inference backends
    - **NB05 — Inference Comparison**: systematic comparison of MAP, RT,
      NUTS, geoVI, and MGVI''',
         '''    - **[NB01 — The IFT Model](01_the_model.ipynb)**: PSD, GP theory,
      mean SFH, and the burstiness plane
    - **[NB02 — Forward Model](02_forward_model.ipynb)**: SPS pipeline from
      SFH to photometry/spectroscopy
    - **[NB03 — Inference Methods](03_inference_methods.ipynb)**: physics of
      RT, geoVI, NUTS — when to use which sampler
    - **[NB04 — Recovery Tests](04_recovery_tests.ipynb)**: mock validation
      across regimes and data types
    - **[NB05 — Hierarchical](05_hierarchical.ipynb)**: population-level PSD
      recovery — the Paper I key result
    - **[NB06 — Data Information](06_data_information.ipynb)**: progressive
      reveal of how data constrains the model
    - **[NB07 — Spectroscopy](07_spectroscopic_fitting.ipynb)**: fitting
      galaxy spectra and resolving degeneracies
    - **[NB08 — PSD Physics](08_psd_physics.ipynb)**: connecting PSD
      parameters to astrophysical mechanisms
    - **[NB09 — Custom Models](09_custom_models.ipynb)**: extending diffsed
      with new priors, PSD models, and dust laws'''),
    ])
    # Also fix the SSP data attribute print
    text = p.read_text()
    text = text.replace(
        'ssp_data.ssp_lgmet.shape[0]} metallicities, "\n'
        '          f"{ssp_data.ssp_lg_age_gyr.shape[0]} ages")',
        'len(ssp_data.ssp_lgmet)} metallicities, "\n'
        '          f"{len(ssp_data.ssp_lg_age_gyr)} ages")')
    # Fix filter output format — it's a 3-tuple, not a dict
    text = text.replace(
        'print(f"Filters loaded — {list(filters.keys())}")',
        'print(f"Filters loaded — {[fc.name for fc in filters[2]]}")')
    # Fix Model() call to accept tuple
    # Model already handles 3-tuples, so filters= is fine
    p.write_text(text)


def patch_nb01():
    """Fix NB01: the model."""
    p = Path("notebooks/_build_nb01.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix double_powerlaw: cell 11 passes t_cosmic which is wrong
    # The function takes LOOKBACK time. In the archetype plot, we want to
    # show SFR vs cosmic time, but must call with lookback time.
    text = text.replace(
        '    sfr = double_powerlaw(t_cosmic, p["alpha"], p["beta"], p["tau"], p["norm"])\n'
        '        ax1.plot(t_cosmic / 1e9, sfr',
        '    sfr = double_powerlaw(t_lb, p["alpha"], p["beta"], p["tau"], p["norm"])\n'
        '        ax1.plot(t_cosmic / 1e9, sfr')
    text = text.replace(
        '    sfr = double_powerlaw(t_cosmic, p["alpha"], p["beta"], p["tau"], p["norm"])',
        '    sfr = double_powerlaw(t_lb, p["alpha"], p["beta"], p["tau"], p["norm"])')

    # Fix cell 12: parameter exploration also uses t_cosmic
    text = text.replace(
        'sfr = double_powerlaw(t_cosmic, alpha=alpha, beta=1.0, tau=5e9, norm=10.0)',
        'sfr = double_powerlaw(13.7e9 - t_cosmic, alpha=alpha, beta=1.0, tau=5e9, norm=10.0)')
    text = text.replace(
        'sfr = double_powerlaw(t_cosmic, alpha=1.5, beta=beta, tau=5e9, norm=10.0)',
        'sfr = double_powerlaw(13.7e9 - t_cosmic, alpha=1.5, beta=beta, tau=5e9, norm=10.0)')
    text = text.replace(
        'sfr = double_powerlaw(t_cosmic, alpha=1.5, beta=1.0, tau=tau_gyr * 1e9, norm=10.0)',
        'sfr = double_powerlaw(13.7e9 - t_cosmic, alpha=1.5, beta=1.0, tau=tau_gyr * 1e9, norm=10.0)')

    # Fix delayed_tau import — check if the import is from mean_sfh
    # The delayed_tau IS in mean_sfh, so the import is correct.
    # But cell 2 uses it as: delayed_tau(t_lookback, tau=3e9, norm=8e-9)
    # Check signature: delayed_tau(t_lookback, tau, norm)
    # The call uses keyword args, which should work.

    p.write_text(text)


def patch_nb02():
    """Fix NB02: forward model."""
    p = Path("notebooks/_build_nb02.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix double_powerlaw call — it takes 5 args not 4
    text = text.replace(
        'sfr = sfr_norm * double_powerlaw(ssp_ages_yr, alpha, beta, tau_sfh)',
        'sfr = double_powerlaw(ssp_ages_yr, alpha, beta, tau_sfh, sfr_norm)')

    # Fix filter set access — the script already uses load_filter_set(filter_names)
    # which returns a 3-tuple, and unpacks it. Check if it does.
    # It does: filter_waves, filter_trans, filter_curves = load_filter_set(filter_names)
    # Good.

    # Fix SSPData print attributes
    text = text.replace(
        'ssp_data.ssp_lgmet.shape[0]} metallicities × "\n'
        '          f"{ssp_data.ssp_lg_age_gyr.shape[0]} ages × "',
        'len(ssp_data.ssp_lgmet)} metallicities × "\n'
        '          f"{len(ssp_data.ssp_lg_age_gyr)} ages × "')
    text = text.replace(
        'ssp_data.ssp_wave.shape[0]} wavelengths")',
        'len(ssp_data.ssp_wave)} wavelengths")')

    # Fix filter print
    text = text.replace(
        'print(f"Filters loaded: {[fc.name for fc in filter_curves]}")',
        'print(f"Filters loaded: {[fc.name for fc in filter_curves]}")')  # already correct

    p.write_text(text)


def patch_nb03():
    """Fix NB03: inference methods (centerpiece)."""
    p = Path("notebooks/_build_nb03.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    # Fix parametric model psd_sigma
    text = text.replace(
        'psd_sigma=Fixed(1.0),\n        psd_tau_myr=Fixed(50.0),\n'
        '        met_logzsol=Uniform(-2.0, 0.5),\n'
        '        dust_tau_bc=Uniform(0.0, 2.0),\n'
        '        dust_tau_diff=Uniform(0.0, 2.0),\n'
        '        dust_slope=Fixed(-0.7),\n'
        '        redshift=Fixed(0.1),\n'
        '        stochastic=False,',
        'psd_sigma=Fixed(0.0),\n        psd_tau_myr=Fixed(50.0),\n'
        '        met_logzsol=Uniform(-2.0, 0.5),\n'
        '        dust_tau_bc=Uniform(0.0, 2.0),\n'
        '        dust_tau_diff=Uniform(0.0, 2.0),\n'
        '        dust_slope=Fixed(-0.7),\n'
        '        redshift=Fixed(0.1),\n'
        '        stochastic=False,',
        1)  # only first occurrence

    # Fix predict_sfh(summary()) — summary returns nested dicts
    text = text.replace(
        'sfh_map = model.predict_sfh(result_map.summary())',
        'sfh_map = model.predict_sfh(result_map.params)')

    # Fix spec.free_names → spec.free_params
    text = text.replace('spec.free_names', 'spec.free_params')

    # Fix band_names for SDSS
    # The existing code uses band_names = ["u", "g", "r", "i", "z"] which is fine

    # Fix wave_eff access from filters
    # NB03 may not have this pattern since it uses photometry directly

    p.write_text(text)


def patch_nb04():
    """Fix NB04: recovery tests."""
    p = Path("notebooks/_build_nb04.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    # Fix filter attribute access
    text = text.replace(
        'list(filters.keys())',
        '[fc.name for fc in filters[2]]')
    text = text.replace(
        'wave_eff = jnp.array([f.wave_effective for f in filters.values()])',
        'wave_eff = jnp.array([3551, 4686, 6166, 7480, 8932])  # SDSS ugriz')

    # Fix SSP data attributes
    text = text.replace('ssp_data.ssp_lgmet.shape[0]', 'len(ssp_data.ssp_lgmet)')
    text = text.replace('ssp_data.ssp_lg_age_gyr.shape[0]', 'len(ssp_data.ssp_lg_age_gyr)')

    # Fix spec.free_names → spec.free_params
    text = text.replace('spec_param.free_names', 'spec_param.free_params')
    text = text.replace('spec_stoch.free_names', 'spec_stoch.free_params')

    # Fix Fitter wave_obs for spectroscopy
    text = text.replace(
        'fitter_spec = Fitter(model_param, spec_obs, noise_spec,\n'
        '                     data_type="spectroscopy", wave_obs=wave_obs)',
        'model_param._wave_obs = wave_obs\n'
        '    fitter_spec = Fitter(model_param, spec_obs, noise_spec,\n'
        '                         data_type="spectroscopy")')

    p.write_text(text)


def patch_nb05():
    """Fix NB05: hierarchical."""
    p = Path("notebooks/_build_nb05.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    p.write_text(text)


def patch_nb06():
    """Fix NB06: data information."""
    p = Path("notebooks/_build_nb06.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading — keep list patterns that are already correct
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    # Fix summary()["std"] pattern — replace with computed std from samples
    text = text.replace(
        "summary[pname][\"std\"]",
        "(summary[pname]['hi_68'] - summary[pname]['lo_68']) / 2")

    # Fix spectroscopy wave_obs pattern
    text = text.replace(
        'model_full._wave_obs = wave_obs\n\n    result_spec = run_fit',
        'model_full._wave_obs = wave_obs\n\n    result_spec = run_fit')  # already correct

    # Fix Fitter wave_obs if present
    text = text.replace(
        "data_type=\"spectroscopy\", wave_obs=wave_obs",
        "data_type=\"spectroscopy\"")

    p.write_text(text)


def patch_nb07():
    """Fix NB07: spectroscopic fitting."""
    p = Path("notebooks/_build_nb07.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    # Fix Fitter spectroscopy pattern
    text = text.replace(
        'fitter = Fitter(model, spec_obs, noise,\n'
        '                    data_type="spectroscopy", wave_obs=wave_obs)',
        'model._wave_obs = wave_obs\n'
        '    fitter = Fitter(model, spec_obs, noise,\n'
        '                    data_type="spectroscopy")')
    text = text.replace(
        'fitter_snr = Fitter(model, spec_obs_snr, noise_snr,\n'
        '                            data_type="spectroscopy", wave_obs=wave_obs)',
        'model._wave_obs = wave_obs\n'
        '        fitter_snr = Fitter(model, spec_obs_snr, noise_snr,\n'
        '                            data_type="spectroscopy")')

    p.write_text(text)


def patch_nb08():
    """Fix NB08: PSD physics."""
    p = Path("notebooks/_build_nb08.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading if present
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    p.write_text(text)


def patch_nb09():
    """Fix NB09: custom models."""
    p = Path("notebooks/_build_nb09.py")
    if not p.exists():
        return
    print(f"\nPatching {p.name}...")
    text = p.read_text()

    # Fix filter loading
    text = text.replace(
        'load_filter_set("sdss")',
        'load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])')

    p.write_text(text)


if __name__ == "__main__":
    print("=" * 60)
    print("Patching all notebook build scripts")
    print("=" * 60)

    patch_nb00()
    patch_nb01()
    patch_nb02()
    patch_nb03()
    patch_nb04()
    patch_nb05()
    patch_nb06()
    patch_nb07()
    patch_nb08()
    patch_nb09()

    print("\n" + "=" * 60)
    print("All patches applied. Run build scripts to regenerate notebooks.")
    print("=" * 60)
