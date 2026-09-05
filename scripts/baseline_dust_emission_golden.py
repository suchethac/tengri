#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Capture golden baseline dust emission SEDs for regression testing.

This script captures the CURRENT dust-emission outputs before any refactoring
to SEDModelComponent subclasses. The outputs are frozen as golden arrays
to prove byte-for-byte fidelity during the migration.

Usage:
    python scripts/baseline_dust_emission_golden.py

Outputs:
    - tests/regression/data/dust_emission_golden/<template>.npy — golden arrays
    - tests/regression/data/dust_emission_golden/params.json — input parameters
    - tests/regression/data/dust_emission_golden/README.md — capture metadata
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main():
    """Capture golden dust emission SEDs for all templates."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS, preload_emission_model

    # Fixed wavelength grid (1e3 to 1e7 Angstrom)
    # np.linspace, not jnp.linspace: the tests rebuild this grid from
    # params.json with np.linspace, and the two disagree in the last ulp on
    # some elements. Under linear interpolation that was invisible; log-log
    # takes a log and an exp, which carries the ulp through to ~1e-14 in the
    # output — enough to trip an rtol=1e-14 bit-exact assertion.
    wave_aa = jnp.asarray(np.linspace(1e3, 1e7, 512, dtype=np.float64))
    L_ir = 1.0e44  # erg/s

    # Output directory
    outdir = Path("tests/regression/data/dust_emission_golden")
    outdir.mkdir(parents=True, exist_ok=True)

    # Registry of templates to capture (excluding aliases and non-templates)
    templates_to_capture = {
        # Analytic models
        "modified_blackbody": {
            "call_path": "closure",
            "params": {
                "dust_T": 30.0,
                "dust_beta_ir": 1.8,
                "dust_epsilon_mbb": 1.0,
                "redshift": 0.0,
            },
        },
        "casey2012": {
            "call_path": "closure",
            "params": {
                "dust_T": 35.0,
                "dust_beta_ir": 1.8,
                "dust_alpha_mir": 2.0,
                "optically_thin": False,
                "redshift": 0.0,
            },
        },
        "greybody": {
            "call_path": "closure",
            "params": {
                "dust_T": 35.0,
                "dust_beta_ir": 1.8,
                "dust_lambda_0_um": 150.0,
                "dust_epsilon_mbb": 1.0,
                "redshift": 0.0,
            },
        },
        "pah_drude": {
            "call_path": "closure",
            "params": {"redshift": 0.0},
        },
        "schreiber2016": {
            "call_path": "closure",
            "params": {
                "dust_T": 30.0,
                "dust_f_pah": 0.05,
                "redshift": 0.0,
            },
        },
        # Grid-based models
        "dale2014": {
            "call_path": "lazy_loader",
            "params": {
                "dust_alpha_dale": 2.0,
                "dust_frac_agn": 0.0,
            },
        },
        "dale2014_cigale": {
            "call_path": "lazy_loader",
            "params": {
                "dust_alpha_dale": 2.0,
                "dust_frac_agn": 0.0,
            },
        },
        "draine_li2007": {
            "call_path": "lazy_loader",
            "params": {
                "dust_umin": 1.0,
                "dust_gamma_dl": 0.01,
                "dust_qpah": 2.5,
            },
        },
        "draine_li2014": {
            "call_path": "lazy_loader",
            "params": {
                "dust_umin": 1.0,
                "dust_gamma_dl": 0.01,
                "dust_qpah": 2.5,
                "dust_alpha_dl14": 2.0,
            },
        },
        # NOTE (#871): astrodust is intentionally NOT captured here. Its
        # DUST_EMISSION_MODELS entry is the retired DL07-*costume*
        # (umin/gamma/qpah over an HD23→DL07-translated grid, with a no-op
        # dust_qpah), not the faithful Hensley & Draine 2023 model. The faithful
        # lgU component (_REGISTRY["astrodust"]) is captured + regression-tested from
        # the component directly in tests/regression/test_dust_goldens_852.py, whose
        # frozen golden lives at
        # tests/regression/data/dust_emission_golden/astrodust.npy.
        "bosa": {
            "call_path": "lazy_loader",
            "params": {"dust_log_ssfr": -10.0},
        },
        "themis": {
            "call_path": "lazy_loader",
            "params": {
                "dust_umin": 1.0,
                "dust_gamma_dl": 0.01,
                "dust_qhac": 0.17,
                "dust_alpha": 2.0,
            },
        },
        "schreiber2018": {
            "call_path": "lazy_loader",
            "params": {
                "dust_T": 30.0,
                "dust_f_pah": 0.05,
            },
        },
        # NOTE (#852): draine2021_pah is intentionally NOT captured here. Its
        # DUST_EMISSION_MODELS entry is a DEPRECATED ALIAS to pah_drude (a
        # different, analytic model; #693), so capturing it via this loader path
        # would freeze the WRONG physics. The real tabulated PAHspec component
        # (_REGISTRY["draine2021_pah_ir"]) is captured + regression-tested from
        # the component directly in tests/regression/test_dust_goldens_852.py, whose
        # frozen golden lives at
        # tests/regression/data/dust_emission_golden/draine2021_pah_ir.npy.
    }

    # Metadata for the report
    metadata = {
        "wave_spec": {
            "min_aa": float(wave_aa[0]),
            "max_aa": float(wave_aa[-1]),
            "n_wave": len(wave_aa),
        },
        "L_ir_erg_s": L_ir,
        "templates": {},
        "skipped": {},
    }

    captured_templates = []
    skipped_templates = []

    # Capture each template
    for template_name, info in sorted(templates_to_capture.items()):
        print(f"Capturing {template_name}...", end=" ", flush=True)

        try:
            # Preload to avoid tracer leaks
            if info["call_path"] == "lazy_loader":
                preload_emission_model(template_name)

            # Get the callable
            fn = DUST_EMISSION_MODELS[template_name]

            # Call the model
            L_nu = fn(wave_aa, L_ir, **info["params"])

            # Convert to numpy and verify shape
            L_nu_np = np.array(L_nu, dtype=np.float64)
            assert L_nu_np.shape == (512,), f"Expected shape (512,), got {L_nu_np.shape}"

            # Save as .npy
            npy_path = outdir / f"{template_name}.npy"
            np.save(npy_path, L_nu_np)

            # Record in metadata
            metadata["templates"][template_name] = {
                "params": info["params"],
                "call_path": info["call_path"],
                "output_shape": [int(s) for s in L_nu_np.shape],
            }

            captured_templates.append(template_name)
            print("OK")

        except FileNotFoundError as e:
            print("SKIPPED (data file not found)")
            skipped_templates.append((template_name, str(e)))
            metadata["skipped"][template_name] = {"reason": "data file not found", "error": str(e)}

        except Exception as e:
            print(f"SKIPPED (error: {type(e).__name__})")
            skipped_templates.append((template_name, str(e)))
            metadata["skipped"][template_name] = {
                "reason": type(e).__name__,
                "error": str(e),
            }

    # Save metadata
    params_path = outdir / "params.json"
    with open(params_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write README
    readme_path = outdir / "README.md"
    readme_content = f"""# Dust Emission Golden Baseline

Captured dust emission SEDs at commit ccb1b6eda, before migration to
SEDModelComponent subclasses.

## Captured Templates ({len(captured_templates)})

{chr(10).join(f"- `{t}`" for t in sorted(captured_templates))}

## Skipped Templates ({len(skipped_templates)})

{chr(10).join(f"- `{t}`: {reason}" for t, reason in sorted(skipped_templates))}

## Input Parameters

See `params.json` for exact wavelength grid, L_ir, and per-template parameters.

## Generation

```bash
python scripts/baseline_dust_emission_golden.py
```

All outputs use 64-bit JAX arrays (jax_enable_x64=True).
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # Summary
    print(f"\n{'=' * 60}")
    print("Golden baseline capture complete.")
    print(f"  Captured: {len(captured_templates)} templates")
    print(f"  Skipped: {len(skipped_templates)} templates")
    print(f"  Output directory: {outdir}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
