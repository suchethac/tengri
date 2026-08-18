#!/usr/bin/env python3
"""Regenerate all 121 examples with explicit venv interpreter."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Enforce main venv
VENV_PYTHON = "/Users/suchethacooray/Projects/tengri/.venv/bin/python"
PYTHONPATH = "/Users/suchethacooray/Projects/tengri/.claude/worktrees/gallery-overhaul/src"
REPO = Path(".")

# Verify we're using the correct interpreter
result = subprocess.run([VENV_PYTHON, "-c", "import sys; print(sys.executable)"],
                       capture_output=True, text=True)
print(f"Using Python: {result.stdout.strip()}")

# All 121 examples (with plot_ prefix)
EXAMPLES = [
    "plot_agn_disc_compare", "plot_agn_feii_sweep", "plot_agn_free_param_sensitivity",
    "plot_agn_hierarchy", "plot_agn_lines_compare", "plot_agn_qsogen_emline_sweep",
    "plot_agn_torus_compare", "plot_alpha_fe_sweep", "plot_alpha_ox_relations",
    "plot_alpha_ox_sweep", "plot_alpha_sf_sweep", "plot_balmer_break_redshift_evolution",
    "plot_bandheads_age_metallicity", "plot_birth_cloud_vs_diffuse", "plot_bosa_grid",
    "plot_bpt_diagram_population", "plot_bump_delta_joint_grid", "plot_color_tracks_redshift",
    "plot_components_isolated", "plot_composable_block_toggles", "plot_continuity_vs_bursty_psd",
    "plot_cosmic_dimming_observed_flux", "plot_cue_parameter_atlas", "plot_custom_attenuation_component",
    "plot_custom_torus_extension", "plot_d4000_hdelta_diagram", "plot_diag_gradient_finite_difference",
    "plot_diag_mass_conservation_sfh", "plot_diag_redshift_rest_invariance", "plot_diag_waveprecomp_accuracy",
    "plot_dla_absorption", "plot_dla_redshift_evolution", "plot_dpl_alpha_beta_grid",
    "plot_dust_geometry_screen_vs_mixed", "plot_dust_qpah_umin_grid", "plot_fesc_sweep",
    "plot_filter_throughput_overlay", "plot_fisher_degeneracy", "plot_grahsp_paper_fig7_galaxy_attenuation",
    "plot_halpha_sfr_calibration_age", "plot_igm_models_comparison", "plot_igm_redshift",
    "plot_imf_choice_sweep", "plot_ir_library_compare", "plot_jax_gradient_sensitivity",
    "plot_kd18_disc_sweep", "plot_lae_spectrum_z6", "plot_logzsol_panchromatic",
    "plot_lyalpha_ew_vs_age", "plot_lyman_alpha_igm_attenuation", "plot_mass_to_light_band_comparison",
    "plot_mbb_temperature_beta_grid", "plot_metallicity_age_grid", "plot_mid_ir_pah_features",
    "plot_model_summary_walkthrough", "plot_nebular_backends", "plot_pahspec_starlight_sweep",
    "plot_photoz_color_degeneracy_grid", "plot_polar_dust_ebv_type12_sweep", "plot_psd_burstiness",
    "plot_q_ir_sweep", "plot_qh_vs_age_metallicity", "plot_quenching_pathway_compare",
    "plot_radio_crossover_frequency", "plot_radio_lir_relation", "plot_radio_loudness_sweep",
    "plot_radio_model_family_compare", "plot_radio_vs_agn_lbol", "plot_recipe_compare",
    "plot_recipe_custom_filter", "plot_recipe_introspection_tour", "plot_recipes_gallery",
    "plot_red_sequence_blue_cloud", "plot_relagn_spin", "plot_resolution_sweep",
    "plot_rv_av_uv_slope_degeneracy", "plot_sed_components", "plot_sed_with_igm",
    "plot_sfh_form_compare", "plot_sfh_nonparametric_compare", "plot_sfh2exp_main_plus_burst",
    "plot_shock_emission", "plot_shock_frac_sweep", "plot_skirtor_xcigale_sweep",
    "plot_spectral_indices_vs_age", "plot_ssp_age_sweep", "plot_ssp_grid",
    "plot_ssp_library_shootout", "plot_ssp_metallicity_sweep", "plot_stochastic_sfh",
    "plot_strong_line_metallicity_diagnostics", "plot_swap_nebular_backend", "plot_tdust_vs_lir",
    "plot_themis_alpha_sweep", "plot_two_burst_observability", "plot_type1_type2_unified_model",
    "plot_ulirg_to_qso_transition", "plot_usecase_age_dust_redshift_degeneracy",
    "plot_usecase_balmer_decrement_av", "plot_usecase_d4000_vs_ssfr", "plot_usecase_dropout_selection_z3",
    "plot_usecase_hubble_sequence", "plot_usecase_jwst_color_color",
    "plot_usecase_main_sequence_cosmic_evolution", "plot_usecase_sdss_lrg_stack_template",
    "plot_usecase_sfr_uv_ir_consistency", "plot_usecase_simulation_seds", "plot_usecase_uv_slope_beta",
    "plot_usecase_uvj_diagram", "plot_uv_ir_energy_balance", "plot_velocity_dispersion_sweep",
    "plot_velocity_offset_lines", "plot_warm_cold_dust_decomposition", "plot_wg00_tau_v_sweep",
    "plot_wise_agn_color_color", "plot_xray_component_decomposition", "plot_xray_model_family_compare",
    "plot_xray_nh_sweep", "plot_xray_pexrav_compton_hump", "plot_xray_sf", "plot_zh_evolution_compare",
]

BATCH_SIZE = 8

def batch_examples(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        yield examples[i:i+batch_size]

def regen_batch(batch_num, examples_batch):
    # Use venv python explicitly
    cmd = [VENV_PYTHON, "tools/regen_gallery.py"] + examples_batch
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["JAX_PLATFORMS"] = "cpu"
    env["MPLBACKEND"] = "Agg"

    print(f"\n[Batch {batch_num}] Regenerating {len(examples_batch)} examples...")
    for ex in examples_batch[:3]:
        print(f"  - {ex}")
    if len(examples_batch) > 3:
        print(f"  ... and {len(examples_batch)-3} more")

    start = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO))
    elapsed = time.time() - start

    return {
        "batch": batch_num,
        "examples": examples_batch,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }

def main():
    results = []
    total_start = time.time()

    for batch_num, batch in enumerate(batch_examples(EXAMPLES, BATCH_SIZE), 1):
        result = regen_batch(batch_num, batch)
        results.append(result)

        if result["returncode"] != 0:
            print(f"[Batch {batch_num}] EXIT CODE {result['returncode']}")
        else:
            print(f"[Batch {batch_num}] ✓ ({result['elapsed_sec']:.0f}s)")

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*60)
    print("REGENERATION SUMMARY")
    print("="*60)
    succeeded = [r for r in results if r["returncode"] == 0]
    failed = [r for r in results if r["returncode"] != 0]
    print(f"Total batches: {len(results)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")

    # Write results JSON
    results_file = REPO / "regen_fixed_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "total_examples": len(EXAMPLES),
            "batch_size": BATCH_SIZE,
            "total_time_seconds": total_elapsed,
            "batches": results,
            "interpreter": VENV_PYTHON,
        }, f, indent=2)
    print(f"Results saved to: {results_file}")

    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
