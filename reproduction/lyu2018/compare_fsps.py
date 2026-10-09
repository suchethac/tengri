"""Compare a live FSPS stellar continuum plus public templates with Tengri.

The default run builds a small three-metallicity MIST+MILES Chabrier SSP grid with python-fsps,
then evaluates one fixed delayed-\N{GREEK SMALL LETTER TAU} example in FSPS and
Tengri. It also constructs an independent NumPy reference from the public
Lyu2018 AGN and Haro 11 source tables. The caller supplies ``--output-dir``
for generated files.

This is a forward-model comparison, not an end-to-end Prospector fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

FSPS_LSUN_ERG_S = 3.839e33
IAU_LSUN_ERG_S = 3.828e33
IMF_CHABRIER = 1
MIST_SOLAR_METALLICITY = 0.0142
LOGZ_NODES = np.array([-0.2, -0.1, 0.0], dtype=np.float64)
METALLICITY_SCATTER_DEX = 0.001


def build_ssp_grid(path: Path) -> dict[str, object]:
    """Generate the small SSP grid from the active python-fsps build."""
    if not os.environ.get("SPS_HOME"):
        raise RuntimeError("Set SPS_HOME to the FSPS source/data directory.")

    import fsps

    sp = fsps.StellarPopulation(zcontinuous=1, imf_type=IMF_CHABRIER)
    libraries = tuple(
        value.decode() if isinstance(value, bytes) else str(value) for value in sp.libraries
    )
    library_names = " ".join(libraries).lower()
    if "mist" not in library_names or "miles" not in library_names:
        raise RuntimeError(f"Expected live FSPS MIST+MILES libraries; found {libraries!r}.")

    wave_ref = None
    age_ref = None
    spectra = []
    surviving = []
    for logzsol in LOGZ_NODES:
        sp.params["imf_type"] = IMF_CHABRIER
        sp.params["sfh"] = 0
        sp.params["logzsol"] = float(logzsol)
        sp.params["dust1"] = 0.0
        sp.params["dust2"] = 0.0
        sp.params["add_dust_emission"] = False
        sp.params["add_neb_emission"] = False
        sp.params["add_neb_continuum"] = False
        sp.params["fagn"] = 0.0
        sp.params["add_igm_absorption"] = False

        wave, flux = sp.get_spectrum(tage=0.0, peraa=False)
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        log_age_yr = np.asarray(sp.ssp_ages, dtype=np.float64)
        mass_remaining = np.asarray(sp.stellar_mass, dtype=np.float64)
        if flux.ndim != 2 or flux.shape != (log_age_yr.size, wave.size):
            raise RuntimeError(
                "FSPS full-age spectrum has an unexpected shape: "
                f"flux={flux.shape}, ages={log_age_yr.shape}, wavelength={wave.shape}."
            )
        if mass_remaining.shape != log_age_yr.shape:
            raise RuntimeError(
                "FSPS surviving-mass array does not match the full age grid: "
                f"{mass_remaining.shape} != {log_age_yr.shape}."
            )
        if wave_ref is None:
            wave_ref = wave
            age_ref = log_age_yr
        elif not (np.array_equal(wave, wave_ref) and np.array_equal(log_age_yr, age_ref)):
            raise RuntimeError("FSPS wavelength or age grid changed with metallicity.")
        spectra.append(flux)
        surviving.append(mass_remaining)

    assert wave_ref is not None and age_ref is not None
    solar_z_fsps = float(sp.solar_metallicity)
    if not np.isclose(solar_z_fsps, MIST_SOLAR_METALLICITY, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "The FSPS MIST build's solar metallicity does not match Tengri's "
            f"MIST convention: Zsun={solar_z_fsps:g}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.attrs["imf"] = "chabrier"
        output.attrs["isochrones"] = "MIST"
        output.attrs["spectrum_library"] = "MILES"
        output.attrs["nebular_included"] = False
        output.attrs["lsun_erg_per_s"] = FSPS_LSUN_ERG_S
        output.attrs["source"] = "Live python-fsps StellarPopulation.get_spectrum"
        output.attrs["description"] = (
            "Small local-metallicity comparison grid; not suitable for general fits."
        )
        output.attrs["units_flux"] = "Lsun/Hz/Msun_formed"
        output.attrs["units_wavelength"] = "Angstrom"
        output.create_dataset("ssp_wave", data=wave_ref)
        output.create_dataset("ssp_flux", data=np.stack(spectra))
        output.create_dataset("ssp_lg_age_gyr", data=age_ref - 9.0)
        # FSPS reports its Fortran REAL(SP) constant as 0.01420000009, while
        # Tengri's MIST coordinate uses the defining decimal 0.0142. The
        # spectra were computed at FSPS logzsol nodes relative to that same
        # compiled 0.0142; using the canonical double here keeps the solar
        # model target exactly on the solar SSP node.
        output.create_dataset("ssp_lgmet", data=np.log10(MIST_SOLAR_METALLICITY) + LOGZ_NODES)
        output.create_dataset("ssp_mass_remaining", data=np.stack(surviving))

    return {
        "libraries": libraries,
        "solar_metallicity_fsps_reported": solar_z_fsps,
        "solar_metallicity_grid_convention": MIST_SOLAR_METALLICITY,
        "wavelength_count": wave_ref.size,
        "age_count": age_ref.size,
        "flux_shape": (len(LOGZ_NODES), age_ref.size, wave_ref.size),
        "ssp_path": str(path),
    }


def parse_args() -> argparse.Namespace:
    """Parse the grid and output paths and the setup-only flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the generated SSP file, plots, and result record.",
    )
    parser.add_argument(
        "--ssp-grid",
        type=Path,
        help="Use an existing FSPS SSP HDF5 file instead of generating one.",
    )
    parser.add_argument(
        "--build-ssp-only",
        action="store_true",
        help="Generate the small live-FSPS SSP grid and exit.",
    )
    return parser.parse_args()


def _independent_templates(
    wave_aa: np.ndarray,
    *,
    data_dir: Path,
    log_agn_luminosity: float,
    tau_v: float,
    log_ir_luminosity: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Scale public template tables on the model grid using NumPy only."""
    c_aa_per_s = 2.99792458e18
    agn_path = data_dir / "lyu2018_agn.h5"
    with h5py.File(agn_path, "r") as source:
        group = source["families/norm"]
        agn_wave = np.asarray(group["wavelength_aa"], dtype=np.float64)
        tau_grid = np.asarray(group["tau_v"], dtype=np.float64)
        family_flux = np.asarray(group["flux_nu_relative"], dtype=np.float64)
        c0_stored = float(group["c0_reference"][()])

    if tau_v < tau_grid[0] or tau_v > tau_grid[-1]:
        raise ValueError(f"tau_V={tau_v:g} is outside [{tau_grid[0]}, {tau_grid[-1]}].")
    upper = int(np.searchsorted(tau_grid, tau_v, side="right"))
    upper = min(max(upper, 1), tau_grid.size - 1)
    lower = upper - 1
    tau_fraction = (tau_v - tau_grid[lower]) / (tau_grid[upper] - tau_grid[lower])
    agn_row = (1.0 - tau_fraction) * family_flux[lower] + tau_fraction * family_flux[upper]
    # Recompute the shared family normalization from its public tau=0 row.
    c0 = -float(np.trapezoid(family_flux[0], c_aa_per_s / agn_wave))
    agn_shape = np.interp(wave_aa, agn_wave, agn_row, left=0.0, right=0.0)
    agn_lnu = agn_shape * (10.0**log_agn_luminosity * IAU_LSUN_ERG_S / c0)

    haro_path = data_dir / "haro11.h5"
    with h5py.File(haro_path, "r") as source:
        raw_wave_um = np.asarray(source["source/wavelength_um"], dtype=np.float64)
        raw_flux = np.asarray(source["source/flux_jansky_like"], dtype=np.float64)
    valid = raw_wave_um > 5.0
    haro_wave = raw_wave_um[valid] * 1.0e4
    haro_flux = raw_flux[valid]
    haro_shape = np.interp(wave_aa, haro_wave, haro_flux, left=0.0, right=0.0)
    ir_integral = -float(np.trapezoid(haro_shape, c_aa_per_s / wave_aa))
    if not np.isfinite(ir_integral) or ir_integral <= 0.0:
        raise RuntimeError(f"Haro 11 has a nonpositive integral on the model grid: {ir_integral}.")
    haro_lnu = haro_shape * (10.0**log_ir_luminosity * IAU_LSUN_ERG_S / ir_integral)
    checks = {
        "agn_c0_recomputed": c0,
        "agn_c0_stored": c0_stored,
        "agn_c0_relative_difference": c0 / c0_stored - 1.0,
        "haro11_grid_integral_relative_template": ir_integral,
        "haro11_source_points_used": int(haro_wave.size),
    }
    return agn_lnu, haro_lnu, checks


def _plot_comparison(
    *,
    output_dir: Path,
    wave_aa: np.ndarray,
    reference_components: dict[str, np.ndarray],
    tengri_components: dict[str, np.ndarray],
    photometry_rows: list[tuple[str, float, float, float, float]],
) -> dict[str, float | int]:
    """Save broadband and optical comparison plots, with 5% mock errors."""
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave_um = wave_aa / 1.0e4
    ref_total = reference_components["total"]
    tng_total = tengri_components["sed_total"]
    valid = np.isfinite(ref_total) & (ref_total > 0.0) & np.isfinite(tng_total)
    ratio = np.full_like(ref_total, np.nan)
    ratio[valid] = tng_total[valid] / ref_total[valid]

    bands = [row for row in photometry_rows if np.isfinite(row[2]) and row[3] > 0.0]
    pivots = np.asarray([row[1] for row in bands])
    tng_band = np.asarray([row[2] for row in bands])
    ref_band = np.asarray([row[3] for row in bands])
    band_ratio = tng_band / ref_band
    band_residual = band_ratio - 1.0
    chi2_5pct = float(np.sum((tng_band - ref_band) ** 2 / (0.05 * ref_band) ** 2))

    fig, (ax, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    display = (wave_um >= 0.1) & (wave_um <= 1000.0) & valid
    ax.loglog(
        wave_um[display],
        ref_total[display],
        color="black",
        lw=2.0,
        label="FSPS + independent public-template reference",
    )
    ax.loglog(
        wave_um[display], tng_total[display], color="C3", lw=1.5, ls="--", label="Tengri total"
    )
    for key, label, color in (
        ("stellar", "FSPS stellar", "C0"),
        ("agn", "Lyu2018 AGN reference", "C1"),
        ("haro11", "Haro 11 reference", "C2"),
    ):
        values = reference_components[key]
        mask = display & (values > 0.0)
        ax.loglog(wave_um[mask], values[mask], color=color, lw=0.9, alpha=0.75, label=label)
    ax.errorbar(
        pivots,
        ref_band,
        yerr=0.05 * ref_band,
        fmt="o",
        ms=3.5,
        capsize=1.5,
        color="black",
        ecolor="0.45",
        label="Independent-reference bands ±5% (illustrative)",
        zorder=4,
    )
    ax.scatter(pivots, tng_band, marker="x", s=22, color="C3", label="Tengri bands", zorder=5)
    ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
    ax.set_title("Delayed-τ stars + Lyu2018 NORMAL (τV=5) + Haro 11")
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax_resid.axhspan(-0.05, 0.05, color="0.8", alpha=0.6, label="±5% illustrative error")
    resid_mask = display & np.isfinite(ratio)
    ax_resid.semilogx(wave_um[resid_mask], ratio[resid_mask] - 1.0, color="C3", lw=0.8)
    ax_resid.scatter(pivots, band_residual, marker="o", s=18, color="black", zorder=3)
    ax_resid.axhline(0.0, color="black", lw=0.7)
    ax_resid.set_xlabel("Rest wavelength [μm]")
    ax_resid.set_ylabel("Tengri / ref − 1")
    ax_resid.grid(True, which="both", alpha=0.15)
    ax_resid.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "broadband_sed.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    optical = (wave_aa >= 3500.0) & (wave_aa <= 9000.0) & valid
    optical_resid = ratio[optical] - 1.0
    fig, (ax, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(
        wave_aa[optical],
        ref_total[optical],
        color="black",
        lw=1.5,
        label="FSPS + NumPy AGN reference",
    )
    ax.plot(
        wave_aa[optical], tng_total[optical], color="C3", lw=1.1, ls="--", label="Tengri total"
    )
    ax.plot(
        wave_aa[optical],
        reference_components["stellar"][optical],
        color="C0",
        lw=0.8,
        alpha=0.8,
        label="FSPS stellar component",
    )
    ax.plot(
        wave_aa[optical],
        reference_components["agn"][optical],
        color="C1",
        lw=0.8,
        alpha=0.8,
        label="Lyu2018 AGN component",
    )
    ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
    ax.set_title("Rest-frame optical spectrum (3500–9000 Å)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax_resid.axhspan(-0.05, 0.05, color="0.8", alpha=0.6)
    ax_resid.plot(wave_aa[optical], optical_resid, color="C3", lw=0.8)
    ax_resid.axhline(0.0, color="black", lw=0.7)
    ax_resid.set_xlabel("Rest wavelength [Å]")
    ax_resid.set_ylabel("Tengri / ref − 1")
    ax_resid.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "optical_spectrum.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    band_abs = np.abs(band_residual)
    optical_abs = np.abs(optical_resid)
    return {
        "n_photometric_bands": len(bands),
        "photometry_median_absolute_fractional_residual": float(np.median(band_abs)),
        "photometry_p95_absolute_fractional_residual": float(np.percentile(band_abs, 95)),
        "photometry_maximum_absolute_fractional_residual": float(np.max(band_abs)),
        "photometry_bands_within_5_percent": int(np.count_nonzero(band_abs <= 0.05)),
        "photometry_chi2_for_illustrative_5_percent_errors": chi2_5pct,
        "optical_grid_points": int(optical_resid.size),
        "optical_median_absolute_fractional_residual": float(np.median(optical_abs)),
        "optical_p95_absolute_fractional_residual": float(np.percentile(optical_abs, 95)),
        "optical_maximum_absolute_fractional_residual": float(np.max(optical_abs)),
    }


def run_comparison(ssp_path: Path, output_dir: Path) -> dict[str, object]:
    """Evaluate the single matched-physics FSPS and Tengri forward models."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TENGRI_DISABLE_JAX_CACHE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import jax
    from reproduction._validation import BROAD_FILTERS, filter_rows
    from reproduction.prospector._drivers import prospector_driver as fsps_driver

    import tengri
    from tengri.components.stellar.sps.dsps_wrapper import (
        compute_lgmet_weights,
        interpolate_metallicity_smooth,
        load_ssp_data,
    )
    from tengri.utils.physics_constants import LOG10_ZSUN

    ssp = load_ssp_data(str(ssp_path))
    metal_nodes = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    solar_weights = np.asarray(
        compute_lgmet_weights(LOG10_ZSUN, ssp.ssp_lgmet, METALLICITY_SCATTER_DEX)
    )
    solar_flux = np.asarray(
        interpolate_metallicity_smooth(
            ssp.ssp_flux, ssp.ssp_lgmet, LOG10_ZSUN, METALLICITY_SCATTER_DEX
        )
    )
    flux_grid = np.asarray(ssp.ssp_flux)
    solar_node_difference = float(np.max(np.abs(solar_flux - flux_grid[-1])))
    if ssp.imf != "chabrier" or ssp.nebular != "bare":
        raise RuntimeError(f"Unexpected SSP metadata: imf={ssp.imf!r}, nebular={ssp.nebular!r}.")
    if not np.array_equal(solar_weights, np.array([0.0, 0.0, 1.0])):
        raise RuntimeError(
            f"The configured solar metallicity weights are not one-hot: {solar_weights}."
        )
    if not np.allclose(solar_flux, flux_grid[-1], rtol=1e-12, atol=0.0):
        raise RuntimeError(
            "The Tengri solar metallicity did not select the upper SSP node exactly."
        )

    groups = {
        "ssp_data": ssp,
        "sfh": {
            "type": "delayed",
            "tau_gyr": tengri.Fixed(1.0),
            "age_gyr": tengri.Fixed(5.0),
            "log_total_mass": tengri.Fixed(10.0),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "met": {
            "logzsol": tengri.Fixed(0.0),
            "logzsol_scatter": tengri.Fixed(METALLICITY_SCATTER_DEX),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "dust_attenuation": {
            "type": "none",
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "neb": {"type": "none", "all_params": tengri.Fixed(tengri.DEFAULT)},
        "agn": {
            "type": "lyu2018",
            "norm": "independent",
            "agn_log_lbol": tengri.Fixed(10.0),
            "agn_lyu2018_tau_v": tengri.Fixed(5.0),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "dust_emission": {
            "type": "haro11",
            "log_L_ir": tengri.Fixed(10.0),
            "other_params": tengri.Fixed(tengri.DEFAULT),
        },
        "redshift": tengri.Fixed(0.0),
        "precompute": False,
    }
    model = tengri.SEDModel.build(**groups)
    if float(model.spec.get_fixed_values()["met_logzsol_scatter"]) != METALLICITY_SCATTER_DEX:
        raise RuntimeError("The model did not retain the requested metallicity-scatter value.")
    prediction = model.predict({})
    components = prediction.sed.components
    wave_aa = np.asarray(components["wavelength"], dtype=np.float64)
    tengri_components = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in components.items()
        if key != "wavelength"
    }
    if not np.all(np.diff(wave_aa) > 0.0):
        raise RuntimeError("Tengri returned a non-increasing rest-frame wavelength grid.")

    fsps_wave_aa, fsps_lnu_per_formed_msun = fsps_driver.csp_lnu(
        logzsol=0.0,
        tau=1.0,
        tage=5.0,
        sfh=4,
        av=0.0,
        add_dust_emission=False,
        add_neb_emission=False,
        add_agn_dust=False,
        fagn=0.0,
    )
    fsps_lnu = np.asarray(fsps_lnu_per_formed_msun, dtype=np.float64) * 1.0e10
    fsps_stellar_on_grid = np.interp(
        wave_aa, np.asarray(fsps_wave_aa, dtype=np.float64), fsps_lnu, left=0.0, right=0.0
    )
    agn_lnu, haro_lnu, template_checks = _independent_templates(
        wave_aa,
        data_dir=repo_root / "src" / "tengri" / "data" / "lyu2018",
        log_agn_luminosity=10.0,
        tau_v=5.0,
        log_ir_luminosity=10.0,
    )
    reference_components = {
        "stellar": fsps_stellar_on_grid,
        "agn": agn_lnu,
        "haro11": haro_lnu,
    }
    reference_components["total"] = fsps_stellar_on_grid + agn_lnu + haro_lnu
    rows = filter_rows(
        wave_aa,
        tengri_components["sed_total"],
        reference_components["total"],
        filters=BROAD_FILTERS,
        weight="photon",
        integrate="filter",
    )
    stats = _plot_comparison(
        output_dir=output_dir,
        wave_aa=wave_aa,
        reference_components=reference_components,
        tengri_components=tengri_components,
        photometry_rows=rows,
    )
    result: dict[str, object] = {
        "case": {
            "sfh": "delayed tau",
            "age_gyr": 5.0,
            "tau_gyr": 1.0,
            "log_formed_mass_msun": 10.0,
            "logzsol": 0.0,
            "imf": "Chabrier",
            "dust_attenuation": "none",
            "agn_family": "NORMAL",
            "agn_tau_v": 5.0,
            "agn_log_reference_lsun": 10.0,
            "haro11_log_ir_lsun": 10.0,
            "redshift": 0.0,
            "amplitudes_are_illustrative": True,
        },
        "environment": {
            "hostname": os.uname().nodename,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "ssp_grid": {
            "path": str(ssp_path),
            "shape": list(flux_grid.shape),
            "wavelength_range_aa": [float(np.min(ssp.ssp_wave)), float(np.max(ssp.ssp_wave))],
            "absolute_logz_nodes": metal_nodes.tolist(),
            "solar_metallicity_interpolation": {
                "mode": model._met_interp,
                "scatter_dex": METALLICITY_SCATTER_DEX,
                "node_weights": solar_weights.tolist(),
                "max_absolute_difference_from_upper_node": solar_node_difference,
            },
        },
        "units": {
            "fsps_native_lsun_erg_s": FSPS_LSUN_ERG_S,
            "tengri_lsun_erg_s": IAU_LSUN_ERG_S,
            "stellar_conversion": (
                "The SSP loader rescales native FSPS Lsun values by 3.839/3.828; "
                "Tengri then multiplies by 3.828, preserving the FSPS cgs spectrum."
            ),
            "spectral_luminosity": "erg/s/Hz",
            "wavelength": "rest-frame Angstrom",
        },
        "reference_template_checks": template_checks,
        "photometric_rows": [
            {
                "filter": row[0],
                "pivot_um": row[1],
                "tengri_lnu": row[2],
                "reference_lnu": row[3],
                "ratio": row[4],
            }
            for row in rows
        ],
        "residual_statistics": stats,
        "notes": [
            "Reference photometry is synthesized from the independent FSPS plus NumPy "
            "template sum; it is not observed galaxy data.",
            "A 5% uncertainty is illustrative and is not a pass/fail threshold.",
            "Prospector was not run; python-fsps generated the stellar reference directly.",
            "The three-node SSP grid is intended only for this fixed-solar comparison.",
        ],
    }
    result_path = output_dir / "comparison_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Plots: {output_dir / 'broadband_sed.png'}; {output_dir / 'optical_spectrum.png'}")
    return result


def main() -> None:
    """Build the comparison grid if needed, then run the fixed example."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.build_ssp_only:
        grid_path = args.ssp_grid or args.output_dir / "fsps_mist_miles_chabrier_3z.h5"
        summary = build_ssp_grid(grid_path)
        print(f"Live FSPS SSP grid: {summary}")
        return
    grid_path = args.ssp_grid or args.output_dir / "fsps_mist_miles_chabrier_3z.h5"
    if not grid_path.is_file():
        summary = build_ssp_grid(grid_path)
        print(f"Live FSPS SSP grid: {summary}")
    run_comparison(grid_path, args.output_dir)


if __name__ == "__main__":
    main()
