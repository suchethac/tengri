# SPDX-License-Identifier: BSD-3-Clause
"""Representative SEDModel builds for the compile_signature() policy tests.

Mirrors ``bench/scripts/bench_compile_signature.py`` builds A-F (tests must
not import ``bench/``, so the SSP-resolution helpers are copied here rather
than shared) and adds a seventh (G: spectroscopy with a resolution matrix and
a covariance) plus one extra, non-canonical helper (H) used only to
materialize a handful of attributes that are absent from A-G: the
conditionally-set radio scalars, the fast-nebular grid table, and the
full-state-chain memo. See ``test_sed_model_policy_complete`` in
``test_compile_signature_invariants.py`` for how these are assembled into
one policy-completeness check.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    WavePrecomp,
)
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.banded import gaussian_resolution_bands
from tests._data_skip import DATA_DIR

# ── SSP resolution (the same walk as bench/scripts/bench_compile_signature.py;
# tests must not import bench/) ────────────────────────────────────────────

SSP_NAME = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_FALLBACK_WNE_NAME = "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

#: A genuinely bare-stellar (no baked-in nebular emission) SSP file. The
#: bench script's own "first ssp_*.h5 without _wNE_ in the name" heuristic
#: picks ``ssp_prsc_bc03_chabrier.h5`` on this checkout, which the Cue
#: backend's own Q_H heuristic flags as wNE-like despite the filename
#: (measured max log10(Q_H) 62.4, far above the ~47-50 bare-stellar range) --
#: the exact "substring/filename heuristics lie" class this repo has hit
#: before. ``fsps_prsc_miles_chabrier.h5`` is the file CueBackend's own error
#: message names as a known-good bare-stellar grid, so build H (the one
#: representative that attaches Cue) uses it explicitly instead of trusting
#: the filename convention.
BARE_STELLAR_NAME = "fsps_prsc_miles_chabrier.h5"


def _find_ssp_file(name: str) -> Path:
    """Resolve an SSP file under ``data/``, skipping cleanly if absent."""
    candidate = DATA_DIR / name
    if not candidate.is_file():
        pytest.skip(f"SSP file not found: {candidate} (needed for a compile_signature build)")
    return candidate


def _first_bare_ssp() -> Path | None:
    """The bench script's own bare-stellar fallback: first ``ssp_*.h5`` without ``_wNE_``."""
    for path in sorted(glob.glob(str(DATA_DIR / "ssp_*.h5"))):
        if "_wNE_" not in os.path.basename(path):
            return Path(path)
    return None


def resolve_ssp_data(requirement: str):
    """Load SSP data based on recipe requirement (mirrors the bench script)."""
    if requirement == "bare-stellar":
        bare = _first_bare_ssp()
        if bare is None:
            return load_ssp_data(str(_find_ssp_file(SSP_NAME)))
        return load_ssp_data(str(bare))
    return load_ssp_data(str(_find_ssp_file(SSP_NAME)))


def _bare_stellar_for_nebular():
    """A bare-stellar SSP safe for a Q_H-linear nebular backend (Cue)."""
    return load_ssp_data(str(_find_ssp_file(BARE_STELLAR_NAME)))


# ── A-F: the same builds as bench/scripts/bench_compile_signature.py ──────


def build_photometry_star_forming():
    """Build A: Star-forming photometry."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    dust_atten = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def build_spectroscopy_simple():
    """Build B: Spectroscopy with simple grammar."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=np.linspace(3800.0, 9000.0, 1500)))
    dust_atten = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        redshift=Fixed(0.1),
    )


def build_agn_dust_emission():
    """Build C: AGN + dust emission."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def build_nebular_shock():
    """Build D: Nebular + shock (no nebular for compatibility, mirrors the bench script)."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    dust_atten = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        shock={"norm": "frac", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        redshift=Fixed(0.1),
    )


def build_per_screen_laws_themis():
    """Build E: Per-screen laws + THEMIS."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    dust_atten = {
        "type": "two_component",
        "law_bc": "conroy2010",
        "law_diff": "cardelli",
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        dust_emission={"type": "themis", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def build_from_config():
    """Build F: from_config (deprecated config path)."""
    import warnings

    ssp_path = _first_bare_ssp() or _find_ssp_file(SSP_NAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return SEDModel.from_config(
                str(ssp_path),
                filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
                redshift=0.1,
                approx=WavePrecomp(),
            )
        except TypeError:
            return SEDModel.from_config(
                str(ssp_path),
                filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
                redshift=0.1,
            )


# ── G: new representative (#2163) ──────────────────────────────────────────


def build_spectroscopy_resolution_matrix_covariance():
    """Build G: spectroscopy with a banded resolution matrix and a covariance.

    New for #2163, to exercise ``Spectroscopy.resolution_matrix`` and
    ``Spectroscopy.covariance`` through the policy-driven signature (both
    already carry their own row in ``Spectroscopy``'s cache_key() ledger,
    #2163 E.2; this build exercises them through the model-level ledger too).
    """
    ssp_data = resolve_ssp_data("bare-stellar")
    wave_obs = jnp.asarray(np.linspace(3800.0, 9000.0, 300))
    resolution_matrix = gaussian_resolution_bands(wave_obs, 2500.0, n_diag=21)
    covariance = jnp.eye(wave_obs.shape[0]) * (0.05**2)
    obs = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=wave_obs,
            resolution=2500.0,
            resolution_matrix=resolution_matrix,
            covariance=covariance,
        )
    )
    dust_atten = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        redshift=Fixed(0.1),
    )


#: The seven E.3 representative builds, in order (A-G).
ALL_BUILDS = (
    ("A", build_photometry_star_forming),
    ("B", build_spectroscopy_simple),
    ("C", build_agn_dust_emission),
    ("D", build_nebular_shock),
    ("E", build_per_screen_laws_themis),
    ("F", build_from_config),
    ("G", build_spectroscopy_resolution_matrix_covariance),
)


def predict_for_build(model):
    """Run the predict/predict_photometry-or-spectrum pass a representative needs
    before its lazily-created memo caches exist, and return the sampled params.
    """
    params = model.spec.sample(jax.random.PRNGKey(0))
    model.predict(params)
    if model.observation is not None and model.observation.can_do_spectroscopy:
        model.predict_spectrum(params)
    else:
        model.predict_photometry(params)
    return params


# ── H: extra, non-canonical helper (not one of the seven E.3 builds) ──────
#
# Materializes attributes that never appear on A-G: the conditionally-set
# radio scalars (_radio_include_freefree/_radio_sfr_mode/_radio_agn_model,
# set only `if self._uses_radio`), the fast-nebular grid table
# (_nebular_grid_table, set only by enable_fast_nebular(), which itself
# requires a Q_H-linear nebular backend none of A-G attaches), and the
# full-state-chain memo (_cached_full_state_chain, populated by a
# predict_state() call that reads more than the projected observables).
# Without this, test_sed_model_policy_complete's ledger rows for those
# attributes would be "stale" (in the policy, absent from every object it
# checks) rather than exercised.


def build_kitchen_sink_for_completeness():
    """H: Cue nebular + AGN + radio + shock, for ledger-completeness only."""
    ssp_data = _bare_stellar_for_nebular()
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"all_params": Fixed(DEFAULT)},
        shock={"norm": "frac", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def exercise_kitchen_sink(model):
    """Drive H through enable_fast_nebular + a full predict_state, for the ledger."""
    params = model.spec.sample(jax.random.PRNGKey(0))
    model.predict_photometry(params)
    model.enable_fast_nebular([6564.61, 4862.68])
    model.predict_photometry(params)
    _ = model.available_properties
    model.predict_state(params)
    return params
