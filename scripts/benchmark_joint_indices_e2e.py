"""End-to-end PopulationFitter validation for joint+indices observable.

The prior-predictive analysis (``joint_indices_discrimination.py``) projected
N=154 for 3σ σ_PSD discrimination with the joint + Lick indices + Hα/FUV
observable. This script validates that projection by running the *actual*
hierarchical VI on real mocks, measuring σ_PSD posterior tightening.

Observable per galaxy (18 components, mixed units handled per-component):
    - 10 broadband photometry (FUV, NUV, ugriz, J, H, Ks)  [flux]
    - 4 emission line integrated fluxes (Hα, Hβ, [OIII] 5007, [OII] 3727)
    - 4 Lick-style absorption indices (D4000, Hδ_A, Hβ_abs, Mg b)

All evaluated from one ``predict_spectrum`` call concatenating line windows
+ index bands. Noise: 5% relative for photometry/lines; D4000 σ=0.02;
EW indices σ=0.3 Å (typical SDSS values).

Comparison: ``data/vi_scaling_benchmark_joint.json`` has the joint-only
(14-component) cells at the same N values. The σ_PSD posterior std should
tighten by roughly the projected ratio (joint N=200, joint+indices N=154,
i.e. ~30% lower σ_PSD posterior width if the marginalization tax is
constant — and noticeably more if indices break degeneracies).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import tengri  # noqa: F401
from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
)
from tengri.inference.hierarchical import PopulationFitter
from tengri.sps.dsps_wrapper import load_ssp_data

jax.config.update("jax_enable_x64", True)

SSP_FILE = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
FILTERS = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks",
]
LINE_NAMES = ["Halpha", "Hbeta", "OIII_5007", "OII_3727"]
LINE_WAVES_REST_AA = jnp.array([6564.61, 4862.68, 5008.24, 3727.09])

Z_FIX = 0.1
LINE_WINDOW_AA = 30.0
LINE_NPIX = 41  # match reference benchmark_population_native (was 11; HLO no longer
# the dominant constraint after Spectroscopy precompute hookup — see
# joint_information_content.md for the cache-write tradeoff)

INDEX_DEFS = {
    "D4000":      {"feat": (4050, 4250), "blue": (3750, 3950), "red": None},
    "Hdelta_A":   {"feat": (4083, 4122), "blue": (4041, 4079), "red": (4128, 4161)},
    # Hbeta_abs (feat 4848-4877) was dropped: overlaps with the Hβ emission-line
    # window (4862±30 Å), causing double-counted likelihood at the same spectrum
    # pixels and a "narrow + biased" σ_PSD posterior. To keep Hβ_abs we'd need
    # explicit covariance between the line flux and the index — not worth the
    # complexity for a single index.
    "Mgb":        {"feat": (5160, 5193), "blue": (5142, 5161), "red": (5191, 5206)},
}
INDEX_NPIX = 21  # was 5 — undersampling Lick bands biased σ_PSD posterior in the
# same "narrow + biased" failure mode as n_grid=64 did for SFH; the underresolved
# trapezoid integration introduces systematic likelihood gradient signal


def make_spec() -> Parameters:
    return Parameters(
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(Z_FIX),
        mean_sfh_type=["tsnorm", "field"],
        n_grid=128,  # match reference; 64 cannot resolve τ=20 Myr fluctuations
    )


def _build_index_grid():
    layout = {}
    waves = []
    cursor = 0
    for name, defs in INDEX_DEFS.items():
        layout[name] = {}
        for k in ("feat", "blue", "red"):
            if defs[k] is None:
                continue
            lo, hi = defs[k]
            obs_w = jnp.linspace(lo, hi, INDEX_NPIX) * (1.0 + Z_FIX)
            layout[name][k] = (cursor, cursor + INDEX_NPIX)
            waves.append(obs_w)
            cursor += INDEX_NPIX
    return jnp.concatenate(waves), layout


def _compute_indices(spec_idx, layout, wave_idx):
    out = []
    for name in INDEX_DEFS:
        feat_lo, feat_hi = layout[name]["feat"]
        feat_F = spec_idx[feat_lo:feat_hi]
        feat_w = wave_idx[feat_lo:feat_hi]
        blue_lo, blue_hi = layout[name]["blue"]
        blue_F = spec_idx[blue_lo:blue_hi]
        blue_w = wave_idx[blue_lo:blue_hi]
        if name == "D4000":
            # Bruzual D4000: ratio of mean F_ν in red band over blue band.
            # tengri returns L_ν (erg/s/Hz), so this is the right ratio.
            d4 = jnp.mean(feat_F) / jnp.maximum(jnp.mean(blue_F), 1e-40)
            out.append(d4)
        else:
            red_lo, red_hi = layout[name]["red"]
            red_F = spec_idx[red_lo:red_hi]
            red_w = wave_idx[red_lo:red_hi]
            blue_F_mean = jnp.mean(blue_F)
            red_F_mean = jnp.mean(red_F)
            blue_mid = 0.5 * (blue_w[0] + blue_w[-1])
            red_mid = 0.5 * (red_w[0] + red_w[-1])
            slope = (red_F_mean - blue_F_mean) / (red_mid - blue_mid)
            cont = blue_F_mean + slope * (feat_w - blue_mid)
            ew = jnp.trapezoid(1.0 - feat_F / jnp.maximum(cont, 1e-40), feat_w)
            out.append(ew)
    return jnp.array(out)


def build_joint_grid(include_indices: bool = True):
    """Build the concatenated wave grid for the joint observable (multi-band sub-grid).

    Returns the line+index wavelength grid plus per-line and per-index layout
    metadata needed to slice and integrate the predicted spectrum back into
    line fluxes and Lick-style indices.
    """
    line_centers_obs = LINE_WAVES_REST_AA * (1.0 + Z_FIX)
    waves_per_line = jnp.stack([
        jnp.linspace(c - LINE_WINDOW_AA, c + LINE_WINDOW_AA, LINE_NPIX)
        for c in line_centers_obs
    ])
    waves_lines = waves_per_line.reshape(-1)
    if include_indices:
        waves_index, layout = _build_index_grid()
        waves_all = jnp.concatenate([waves_lines, waves_index])
    else:
        waves_all = waves_lines
        layout = None
        waves_index = None
    return waves_all, waves_per_line, waves_index, layout


def build_joint_observation(include_indices: bool = True) -> tuple[Observation, jnp.ndarray]:
    """Build an Observation with photometry + spectroscopy registered on the joint grid.

    Registering the joint wave grid as a ``Spectroscopy`` activates the
    precomputed-template spectrum fast path (``_predict_spectrum_compositional``,
    sed_model.py:2746): the SSP is rebinned from 11149 to ``len(waves_all)``
    pixels at construction, eliminating the in-graph 11149-pt interpolation
    that previously inflated HLO past the 2 GB protobuf limit.

    The returned ``waves_all`` is the SAME identity array stored on the
    spectroscopy — pass it (or ``None``) to ``predict_spectrum`` to trigger
    the fast path via the ``wave_obs is precomputed.wave_obs_pixels`` check.
    """
    waves_all, _, _, _ = build_joint_grid(include_indices=include_indices)
    obs = Observation(
        photometry=Photometry.from_names(FILTERS),
        spectroscopy=Spectroscopy(wave_obs=waves_all),
    )
    return obs, waves_all


def patch_predict_joint_indices(model: SEDModel, include_indices: bool = True) -> SEDModel:
    """Wrap predict_photometry to return concatenated (10 phot + 4 lines [+ 4 indices]).

    Reads the wave grid from ``model.observation.spectroscopy.wave_obs`` so the
    precomputed-template fast path is hit (object identity comparison in
    ``_predict_spectrum_compositional`` requires passing the same array
    reference).
    """
    _, waves_per_line, waves_index, layout = build_joint_grid(
        include_indices=include_indices
    )
    n_lines = waves_per_line.size

    if model.observation is None or model.observation.spectroscopy is None:
        raise ValueError(
            "model.observation must have a Spectroscopy registered with the "
            "joint wave grid; use build_joint_observation()."
        )
    waves_all = model.observation.spectroscopy.wave_obs  # identity-preserving

    orig_predict = model.predict_photometry

    def predict_joint(params, mode="auto"):
        phot = orig_predict(params, mode=mode)
        # Pass the identity-preserving array to hit the precomputed fast path.
        spec_all = model.predict_spectrum(params, waves_all, mode=mode)
        spec_lines = spec_all[:n_lines]
        spec_per_line = spec_lines.reshape(LINE_WAVES_REST_AA.shape[0], LINE_NPIX)
        cont = 0.5 * (spec_per_line[:, 0] + spec_per_line[:, -1])
        line_flux = jax.vmap(
            lambda f, w, c: jnp.trapezoid(f - c, w)
        )(spec_per_line, waves_per_line, cont)

        if include_indices:
            spec_idx = spec_all[n_lines:]
            indices = _compute_indices(spec_idx, layout, waves_index)
            return jnp.concatenate([phot, line_flux, indices])
        return jnp.concatenate([phot, line_flux])

    model.predict_photometry = predict_joint  # type: ignore
    return model


def model_factory(psd_sigma=1.0, psd_tau_myr=50.0, *, ssp_data, obs,
                  include_indices: bool = True):
    spec = make_spec()
    return patch_predict_joint_indices(
        SEDModel(spec, ssp_data, observation=obs),
        include_indices=include_indices,
    )


def make_galaxies(n_gal: int, ssp_data, obs: Observation, key, sigma_true=2.0,
                  tau_true=20.0, include_indices: bool = True):
    template = patch_predict_joint_indices(
        SEDModel(make_spec(), ssp_data, observation=obs),
        include_indices=include_indices,
    )
    n_phot = len(FILTERS)
    n_lines = len(LINE_NAMES)
    n_idx = len(INDEX_DEFS) if include_indices else 0

    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        true_params = template.spec.sample(k)
        true_params["sfh_field_psd_sigma"] = jnp.array(sigma_true)
        true_params["sfh_field_psd_tau_myr"] = jnp.array(tau_true)
        flux = template.predict_photometry(true_params)
        flux_arr = np.asarray(flux)

        # Per-component noise levels
        noise = np.zeros(flux_arr.shape[0])
        # 10% relative noise + absolute floor; match reference benchmark_population_native
        flux_pl = flux_arr[:n_phot + n_lines]
        noise[:n_phot + n_lines] = np.abs(flux_pl) * 0.10 + 1e-3
        # D4000: σ=0.02; EW indices: σ=0.3 Å (Hβ_abs dropped — overlaps Hβ line)
        if include_indices:
            noise[n_phot + n_lines + 0] = 0.02   # D4000
            noise[n_phot + n_lines + 1] = 0.3    # Hδ_A
            noise[n_phot + n_lines + 2] = 0.3    # Mg b

        flux_obs = flux_arr + noise * np.asarray(jax.random.normal(k, shape=flux_arr.shape))
        galaxies.append({"flux_obs": jnp.asarray(flux_obs),
                         "noise": jnp.asarray(noise)})
    return galaxies


def run_one(n_gal: int, K: int, n_iter: int = 15, n_samp: int = 3,
            include_indices: bool = True) -> dict:
    label = "joint+indices" if include_indices else "joint-only"
    print(f"\n=== N={n_gal}  K={K}  {label} ===", flush=True)
    ssp_data = load_ssp_data(SSP_FILE)
    obs, _waves_all = build_joint_observation(include_indices=include_indices)

    key = jax.random.PRNGKey(42)
    t_setup = time.time()
    galaxies = make_galaxies(n_gal, ssp_data, obs, key,
                             include_indices=include_indices)
    pop = PopulationFitter(
        lambda psd_sigma=1.0, psd_tau_myr=50.0:
            model_factory(psd_sigma, psd_tau_myr, ssp_data=ssp_data, obs=obs,
                          include_indices=include_indices),
        galaxies, data_type="photometry",
    )
    setup_s = time.time() - t_setup

    t0 = time.time()
    result = pop.run(
        method="native_vi_linear",
        n_iterations=n_iter,
        n_samples=n_samp,
        n_posterior_samples=200,
        forward_chunk_size=K,
        verbose=False,
    )
    wall_s = time.time() - t0

    shared = getattr(result, "shared_samples", {}) or {}
    # Existing benches use "psd_sigma" / "psd_tau_myr" keys (not the full
    # "sfh_field_*" names). Probe both for forward-compat.
    sigma_samp = np.asarray(shared.get("psd_sigma",
                                       shared.get("sfh_field_psd_sigma", [])))
    tau_samp = np.asarray(shared.get("psd_tau_myr",
                                     shared.get("sfh_field_psd_tau_myr", [])))

    def _sum(arr):
        if arr.size == 0:
            return {"keys_available": list(shared.keys())}
        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p16": float(np.percentile(arr, 16)),
            "p84": float(np.percentile(arr, 84)),
        }

    out = {
        "n_gal": n_gal,
        "forward_chunk_size": K,
        "setup_s": setup_s,
        "wall_s": wall_s,
        "n_iters_used": result.diagnostics.get("n_iterations"),
        "psd_sigma_summary": _sum(sigma_samp),
        "psd_tau_summary": _sum(tau_samp),
    }
    print(f"  wall={wall_s:.1f}s  iters={out['n_iters_used']}")
    ss, ts = out["psd_sigma_summary"], out["psd_tau_summary"]
    if "median" in ss:
        print(f"  σ posterior: median={ss['median']:.2f} ± {ss['std']:.2f}")
    else:
        print(f"  σ posterior: keys available = {ss.get('keys_available')}")
    if "median" in ts:
        print(f"  τ posterior: median={ts['median']:.0f} ± {ts['std']:.0f} Myr")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--out", default="analysis/joint_indices_e2e.json")
    p.add_argument("--no-indices", action="store_true",
                   help="Disable Lick indices; runs as plain joint-mode for "
                        "control comparison with vi_scaling_benchmark_joint.json.")
    args = p.parse_args()
    include_indices = not args.no_indices

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    import traceback
    for n in args.Ns:
        try:
            row = run_one(n, args.K, include_indices=include_indices)
        except Exception as exc:
            row = {"n_gal": n, "K": args.K, "error": repr(exc),
                   "traceback": traceback.format_exc()}
            print(f"ERROR at N={n}: {exc}")
            traceback.print_exc()
        rows.append(row)
        Path(args.out).write_text(json.dumps(rows, indent=2))

    print(f"\nWrote {args.out}")
    print("\n=== Joint-only baseline (data/vi_scaling_benchmark_joint.json) ===")
    bp = Path("data/vi_scaling_benchmark_joint.json")
    if bp.exists():
        baseline = json.loads(bp.read_text())
        for r in baseline:
            if r.get("n_gal") in args.Ns and r.get("forward_chunk_size") == args.K:
                ss = r.get("psd_sigma_summary") or {}
                ts = r.get("psd_tau_summary") or {}
                print(f"  N={r['n_gal']}  K={r['forward_chunk_size']}  "
                      f"σ med={ss.get('median')}  std={ss.get('std')}  "
                      f"τ med={ts.get('median')}")


if __name__ == "__main__":
    main()
