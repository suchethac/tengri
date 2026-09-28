#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Declarative metadata for the six demonstration configurations.

Separated from `configs.py` because that module imports tengri, and so JAX, to
build models. Everything here is plain data, so a consumer that only needs to
know what a configuration claims to be -- a figure reading posterior NPZs, the
provenance audit -- can read it without paying for the import.

`configs.py` re-exports both names, so `from configs import CONFIGS` keeps
working.
"""

from __future__ import annotations

SSP_FOR_CONFIG = {
    "I": "fsps_mist_c3k_a_chabrier",
    "II": "fsps_prsc_c3k_a_chabrier",
    "III": "fsps_mist_miles_chabrier",
    "IV": "fsps_prsc_miles_chabrier",
    "V": "bpss_stars_c3k_a_chabrier",
    "VI": "fsps_mist_c3k_a_chabrier",
}

CONFIGS = {
    "I": {
        "key": "I",
        "dust_param": "dust_tau_diff",
        "name": "continuity, MIST/C3K",
        "sfh": "continuity, 7 bins",
        "sfh_type": "continuity",
        "library": "FSPS MIST/C3K",
        "attenuation": "Kriek+13, 2-comp",
        "dust_ir": "Draine+2014",
        "nebular": "Cue",
        "agn": False,
        "ssp_grid": SSP_FOR_CONFIG["I"],
        "n_free": None,
    },
    "II": {
        "key": "II",
        "dust_param": "dust_tau_v",
        "name": "double power law, PARSEC/C3K",
        "sfh": "double power law",
        "sfh_type": "dpl",
        "library": "FSPS PARSEC/C3K",
        "attenuation": "Calzetti, 1-comp",
        "dust_ir": "Dale+2014",
        "nebular": "Cue",
        "agn": False,
        "ssp_grid": SSP_FOR_CONFIG["II"],
        "n_free": None,
    },
    "III": {
        "key": "III",
        "dust_param": "dust_tau_diff",
        "name": "delayed-tau, MIST/MILES",
        "sfh": "delayed-tau",
        "sfh_type": "delayed",
        "library": "FSPS MIST/MILES",
        "attenuation": "Charlot+2000, 2-comp",
        "dust_ir": "THEMIS",
        "nebular": "Cue",
        "agn": False,
        "ssp_grid": SSP_FOR_CONFIG["III"],
        "n_free": None,
    },
    "IV": {
        "key": "IV",
        "dust_param": "dust_tau_diff",
        "name": "Dirichlet, PARSEC/MILES",
        "sfh": "Dirichlet, 7 bins",
        "sfh_type": "dirichlet",
        "library": "FSPS PARSEC/MILES",
        "attenuation": "Kriek+13, 2-comp",
        "dust_ir": "Casey+2012",
        "nebular": "Cloudy, free logU",
        "agn": False,
        "ssp_grid": SSP_FOR_CONFIG["IV"],
        "n_free": None,
    },
    "V": {
        "key": "V",
        "dust_param": "dust_tau_v",
        "name": "log-normal, BPASS",
        "sfh": "log-normal",
        "sfh_type": "lnorm",
        "library": "BPASS C3K",
        "attenuation": "SMC, 1-comp",
        "dust_ir": "Dale+2014",
        "nebular": "Cue",
        "agn": False,
        "ssp_grid": SSP_FOR_CONFIG["V"],
        "n_free": None,
    },
    "VI": {
        "key": "VI",
        "dust_param": "dust_tau_diff",
        "name": "continuity + AGN disc and torus",
        "sfh": "continuity, 7 bins",
        "sfh_type": "continuity",
        "library": "FSPS MIST/C3K",
        "attenuation": "Kriek+13, 2-comp",
        "dust_ir": "Draine+2014",
        "nebular": "Cue",
        "agn": True,
        "ssp_grid": SSP_FOR_CONFIG["VI"],
        "n_free": None,
    },
}
