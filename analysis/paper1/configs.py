"""Three SED model configurations for Paper I analysis."""

from __future__ import annotations

import jax

import tengri
from tengri import FIXED, FREE, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.cosmology import age_at_z

jax.config.update("jax_enable_x64", True)


def load_ssp_for(key: str) -> tengri.SSPData:
    """Load SSP grid for configuration key I/II/III."""
    ssp_names = {
        "I": "bc03_pdva_stelib_chabrier",
        "II": "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0",
        "III": "fsps_mist_c3k_a_chabrier",
    }
    return tengri.load_ssp(ssp_names[key])


def config_I(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """Config I: single-screen Calzetti, CUE nebular at fixed logU=-2.5."""
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "tau_gyr": Uniform(0.1, 20.0),
            "age_gyr": Uniform(1.0, age_at_z(z)),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": Uniform(-2.0, 0.3),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={"type": "dale2014", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED, "neb_logU": Fixed(-2.5)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_II(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """Config II: DPL SFH, two-component Calzetti, DL07."""
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "dpl",
            "all_params": FIXED,
            "alpha": Uniform(0.5, 5.0),
            "beta": Uniform(0.3, 3.0),
            "tau_gyr": Uniform(0.5, 13.0),
            "age_gyr": Uniform(1.0, age_at_z(z)),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": Uniform(-2.0, 0.3),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
        dust_emission={"type": "dl07", "all_params": FIXED},
        neb={"type": "ssp"},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_III(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """Config III: n_bins=7 (8 edges, 6 ratios), Kriek & Conroy, CUE nebular with free logU."""
    bin_edges_gyr = tengri.make_agebins_from_zred(z, n_bins=7)
    sfh_dict = {
        "type": "continuity",
        "all_params": FIXED,
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": Uniform(-2.0, 0.3),
        "bin_edges_gyr": bin_edges_gyr,
    }
    for i in range(6):
        sfh_dict[f"ratio_{i}"] = FREE
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=sfh_dict,
        dust_attenuation={
            "type": "two_component",
            "law": "kriek_conroy",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
        dust_emission={"type": "dl07", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED, "neb_logU": Uniform(-4.0, -1.0)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


CONFIGS = {
    "I": {
        "key": "I",
        "name": "energy-balance parametric",
        "ssp_grid": "bc03_pdva_stelib_chabrier",
        "n_free": None,
    },
    "II": {
        "key": "II",
        "name": "two-component dust, double power law",
        "ssp_grid": "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0",
        "n_free": None,
    },
    "III": {
        "key": "III",
        "name": "nonparametric",
        "ssp_grid": "fsps_mist_c3k_a_chabrier",
        "n_free": None,
    },
}

if __name__ == "__main__":
    test_z, test_filters = (
        1.0,
        [
            "hst_f435w",
            "hst_f606w",
            "hst_f775w",
            "hst_f814w",
            "hst_f850lp",
            "hst_f105w",
            "hst_f125w",
            "hst_f160w",
            "vista_ks",
            "irac_36",
            "irac_45",
            "irac_58",
            "irac_80",
        ],
    )
    obs, key = (
        tengri.Observation(photometry=tengri.Photometry.from_names(test_filters)),
        jax.random.PRNGKey(0),
    )
    for cfg_key in ["I", "II", "III"]:
        ssp = load_ssp_for(cfg_key)
        model = globals()[f"config_{cfg_key}"](ssp, obs, test_z)
        n_free = len(model.spec.free_params)
        CONFIGS[cfg_key]["n_free"] = n_free
        print(f"Config {cfg_key}: {n_free} free params: {model.spec.free_params}")
        pred = model.predict_photometry(model.spec.sample(key=key))
        assert (pred > 0).all() and pred.shape == (len(test_filters),)
    print("✓ All configs built and tested successfully")
