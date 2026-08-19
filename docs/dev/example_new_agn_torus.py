"""
Worked example: contributing a new AGN torus model to tengri.

Run:    .venv/bin/python examples/contrib/example_new_agn_torus.py

This script demonstrates the full collaborator workflow:
  1. Register a new AGN model with metadata (citation + status).
  2. Verify it's discoverable via tengri.list_agn_models() and tengri.describe().
  3. Generate synthetic photometry from a known galaxy.
  4. Fit it back with the new model using MAP optimization.

Copy this file as a template when contributing your own model.
"""

from __future__ import annotations

import jax.numpy as jnp

import tengri
from tengri.components.agn.unified import register_agn_model

# ---------------------------------------------------------------------------
# 1.  Register a toy AGN torus model.
# ---------------------------------------------------------------------------


@register_agn_model(
    "my_toy_torus",
    citation="Sprint demo (replace with your reference)",
    status="experimental",
    short_doc="Toy single-temperature blackbody torus — DEMO ONLY",
)
def my_toy_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = 0.1,
    agn_T_torus: float = 300.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """Single-temperature graybody torus — placeholder physics for the worked example.

    This is a toy model for demonstration. Replace the body with your own
    torus physics (e.g., Mullaney et al. 2011, Ciesla et al. 2015, etc.).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_lum_ratio : float, optional
        AGN luminosity fraction of total SED [dimensionless]. Default 0.1.
    agn_T_torus : float, optional
        Dust torus temperature [K]. Default 300.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless], range [0, 1]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — delegates to Planck function.
    """
    # Constants (cgs)
    h = 6.626e-27
    c = 2.998e10
    k = 1.381e-16

    # Convert wavelength (Angstrom) to frequency (Hz)
    nu = c / (wavelength * 1e-8)

    # Planck function: B_nu(T) = (2 h nu^3 / c^2) / (exp(h nu / k T) - 1)
    B_nu = (2 * h * nu**3 / c**2) / (jnp.exp(h * nu / (k * agn_T_torus)) - 1)

    # Scale by luminosity and apply covering factor
    # Bolometric luminosity (Lsun to erg/s)
    L_bol_erg_s = 10.0**agn_log_lbol * 3.839e33

    # Normalize to unity at 1 micron and scale by covering fraction
    norm_1um_hz = c / (1e-4)
    B_norm = (2 * h * norm_1um_hz**3 / c**2) / (jnp.exp(h * norm_1um_hz / (k * agn_T_torus)) - 1)
    L_nu = (B_nu / B_norm) * (L_bol_erg_s / 1e10) * agn_torus_frac

    return L_nu * agn_lum_ratio


# ---------------------------------------------------------------------------
# 2.  Confirm it's discoverable.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  Example: Contributing a new AGN model to tengri")
print("=" * 70)

print("\n[Step 1] Registered AGN models (experimental status)")
print("-" * 70)
for m in tengri.list_agn_models(status="experimental"):
    print(f"  {m['name']:30s} -- {m['short_doc']}")

print("\n[Step 2] Describe my_toy_torus")
print("-" * 70)
entry = tengri.describe("my_toy_torus")
for k, v in entry.items():
    print(f"  {k:15s}: {v}")

print("\n[Step 3] Available primary inference methods")
print("-" * 70)
for m in tengri.list_inference_methods(tier="primary"):
    print(f"  {m['name']}")

# ---------------------------------------------------------------------------
# 3.  Optional: end-to-end fit (requires SSP data).
# ---------------------------------------------------------------------------

print("\n[Step 4] End-to-end fit on mock data (SKIPPED — requires SSP setup)")
print("-" * 70)
print(
    """
    Uncomment the code block below and adapt to your environment.
    You will need:
      1. SSP data file (e.g., ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5)
      2. Filter transmission curves (auto-downloaded on first run)

    Minimal recipe:

        from pathlib import Path
        import jax
        import tengri

        repo_root = Path.cwd()
        ssp_file = (
            repo_root / "data" / "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
        )
        ssp = tengri.load_ssp_data(str(ssp_file))

        # Build model using SEDModel.build() with the new nested-dict API
        obs = tengri.Observation(
            photometry=tengri.Photometry.from_names(
                ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
            )
        )

        model = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": tengri.FREE},
            agn={"model": "my_toy_torus", "all_params": tengri.FREE},
            redshift=tengri.Fixed(0.05),
        )

        # Generate mock data
        key = jax.random.PRNGKey(42)
        truth = dict(model.spec.sample(key))
        truth.update({
            "sfh_tsnorm_log_total_mass": 0.5,
            "agn_log_lbol": 45.0,
            "agn_lum_ratio": 0.1,
        })
        mock = model.mock(truth, snr=100.0, key=key)

        # Fit with MAP
        forward = tengri.ForwardModel.build(sed=model, observation=obs)
        posterior = forward.fit(
            mock.flux_obs, mock.noise, method="map", n_steps=50, verbose=False
        )
        print(posterior.summary())
    """
)

print("\n" + "=" * 70)
print("  Done.")
print("=" * 70)
print("\nNext: Copy this file to examples/contrib/your_model_name.py and:")
print("  1. Replace my_toy_torus with your model function.")
print("  2. Update citation, status, and short_doc.")
print("  3. Uncomment and adapt the fit block for your parameters.")
print("=" * 70 + "\n")
