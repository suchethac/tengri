"""The six demonstration configurations for Paper I.

These implement ``tab:configs`` in the manuscript exactly. The set is a *suite*
of common SED recipes run on the same galaxies, likelihood, and sampler -- not a
single-axis ablation off one backbone -- so that the spread in inferred
properties measures model-choice uncertainty under realistic recipe changes.

Star formation history, stellar library, dust attenuation, and dust infrared
emission all vary across the suite. Configurations I--V leave the AGN off;
Configuration VI turns on a QSOgen disc and a SKIRTOR torus over a host matched
to Configuration I, so that the AGN contrast is the only difference between the
two rows.

===  ===================  ====================  ======================  =============  ==================  ===
ID   SFH                  Library               Attenuation             Dust IR        Nebular             AGN
===  ===================  ====================  ======================  =============  ==================  ===
I    Continuity, 7 bins   FSPS MIST/C3K         Kriek+13, 2-comp        Draine+2014    Cue                 off
II   Double power law     FSPS PARSEC/C3K       Calzetti, 1-comp        Dale+2014      Cue                 off
III  Delayed-tau          FSPS MIST/MILES       Charlot+2000, 2-comp    THEMIS         Cue                 off
IV   Dirichlet, 7 bins    FSPS PARSEC/MILES     Kriek+13, 2-comp        Casey+2012     Cloudy, free logU   off
V    Log-normal           BPASS C3K             SMC, 1-comp             Dale+2014      Cue                 off
VI   Continuity, 7 bins   FSPS MIST/C3K         Kriek+13, 2-comp        Draine+2014    Cue                 on
===  ===================  ====================  ======================  =============  ==================  ===

All six use the log-age cloud-in-cell age kernel (the default) and Inoue+2014 IGM
attenuation at the catalog redshift.

Two priors carry a deliberate choice and are documented where they are set:

* **Age.** Age priors of the parametric histories run from 1 Gyr to the cosmic age
  at the galaxy redshift. The upper bound keeps the likelihood from going flat
  beyond the Big Bang mask; the lower bound is a restriction, not a convenience:
  below 1 Gyr the CANDELS photometry admits a second, young and dusty solution
  that the sampler cannot traverse together with the older one (187 divergences,
  R-hat 1.06 on galaxy 13097 with a 0.1 Gyr bound, versus 1-3 edge divergences
  with 1 Gyr).

* **Metallicity.** The prior spans each library's own metallicity grid rather than
  a hardcoded ceiling. Every configuration in this suite draws on a different
  stellar library, and the libraries do not share a grid edge; a literal copied
  between them either truncates the posterior inside the grid, leaving unused
  headroom, or runs past the edge and manufactures divergences. See
  :func:`met_prior_for`.
"""

from __future__ import annotations

from pathlib import Path

import jax
from config_metadata import CONFIGS, SSP_FOR_CONFIG

import tengri
from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.cosmology import age_at_z

jax.config.update("jax_enable_x64", True)

# SSPData.ssp_lgmet is absolute log10(Z); the user-facing met_logzsol is
# log10(Z/Zsun). Asplund+2009 solar reference, matching dsps_wrapper.SSPData.
LOG10_ZSUN = -1.848

# Distance held between the metallicity prior and the library's outermost grid
# node, in dex. See met_prior_for for why the edge itself is the wrong place to
# stop. Small against grid spans of 2.8-3.6 dex across this suite.
MET_EDGE_INSET_DEX = 0.02

# Grid per configuration. The suite spans MIST and PARSEC isochrones crossed with
# C3K and MILES spectral libraries, plus BPASS for binary-star evolution.
# Configuration VI reuses Configuration I's library so the AGN is the only change.
N_SFH_BINS = 7

#: Photoionization grid for Configuration IV, named rather than discovered.
#:
#: Left unset, the Cloudy backend searches the data directory and takes what it
#: finds. On a machine holding several grids that is a silent choice: measured
#: 2026-09-20, Configuration IV resolved to ``cloudy_grid_mist.h5`` -- an MIST
#: grid under a PARSEC/MILES configuration, exactly the isochrone mismatch the
#: backend's own error text warns against. Nothing raised, and the free
#: parameter count was unchanged, so the mismatch was invisible to every check
#: that did not look at the resolved path.
#:
#: The grid is derived, not distributed: ``data/.gitignore`` excludes
#: ``cloudy_grid_*.h5`` and they are built by
#: ``scripts/convert_fsps_cloudy_grid.py --isoc prsc`` from the FSPS
#: ``ZAU_ND_prsc.lines``/``.cont`` files in ``$SPS_HOME/nebular/`` or
#: ``data/cloudy_raw/``. Naming it here turns a missing grid into a loud
#: failure at build time rather than a quiet substitution.
CLOUDY_GRID_NAME_FOR_IV = "cloudy_grid_prsc.h5"


def cloudy_grid_for_iv() -> str:
    """Absolute path to Configuration IV's PARSEC photoionization grid.

    Resolved explicitly rather than left to the backend's directory search,
    which takes whatever it finds first and gave an MIST grid to this
    PARSEC/MILES configuration without raising.

    Raises:
        FileNotFoundError: naming the command that builds the grid. It is
            derived rather than distributed, so a missing file is a setup step
            that has not been run, not a broken install -- and saying so beats
            a search that silently succeeds with the wrong isochrone.
    """
    import os

    candidates = []
    env_dir = os.environ.get("TENGRI_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / CLOUDY_GRID_NAME_FOR_IV)
    here = Path(__file__).resolve()
    # analysis/paper1/configs.py -> paper1 -> analysis -> repo root, then any
    # ancestor: a worktree's data/ is often the main checkout's.
    for parent in here.parents:
        candidates.append(parent / "data" / CLOUDY_GRID_NAME_FOR_IV)

    for path in candidates:
        if path.is_file():
            return str(path)

    raise FileNotFoundError(
        f"Configuration IV needs {CLOUDY_GRID_NAME_FOR_IV}, which is derived rather "
        f"than distributed (data/.gitignore excludes cloudy_grid_*.h5). Build it "
        f"with:\n\n"
        f"    python scripts/convert_fsps_cloudy_grid.py --sps-home --isoc prsc\n"
        f"    # or, from local FSPS ASCII files:\n"
        f"    python scripts/convert_fsps_cloudy_grid.py --input-dir data/cloudy_raw --isoc prsc\n\n"
        f"It must be the PARSEC grid: Configuration IV is FSPS PARSEC/MILES, and "
        f"an MIST-derived grid pairs the wrong ionizing spectrum with the "
        f"stellar library. Searched: {[str(c) for c in candidates[:4]]}"
    )


def load_ssp_for(key: str) -> tengri.SSPData:
    """Load the stellar library for configuration key I/II/III/IV/V/VI."""
    return tengri.load_ssp(SSP_FOR_CONFIG[key])


def met_prior_for(ssp_data: tengri.SSPData, inset: float = MET_EDGE_INSET_DEX) -> Uniform:
    """Metallicity prior spanning this library's own grid, in log10(Z/Zsun).

    Derived from the grid rather than hardcoded. A ceiling copied from another
    library is wrong in two ways: too low truncates the posterior and leaves
    headroom the library actually has, too high lets the sampler evaluate the
    library outside its own support.

    The bounds are then held ``inset`` dex inside the grid. This is a
    **forward-model** guard, not a sampler one, and an earlier version of this
    docstring confused the two. The library is interpolated, so beyond the last
    node the prediction clips to a constant and the metallicity gradient goes to
    zero; a chain reaching there reads flat likelihood as a maximum (#442). The
    inset keeps the prior support inside the grid so that region is never
    evaluated. It is small against the several-dex span of every grid here, so
    it costs essentially none of the headroom it protects.

    What it does **not** do is relieve a hard bound in the sampled coordinate,
    which is what the earlier text claimed. ``Uniform.unstandardize`` is the
    Gaussian CDF, so a bound sits at :math:`\\xi = \\pm\\infty` and NUTS never
    integrates up against a wall. Measured on galaxy 79 configuration I, the
    metallicity coordinate has standard deviation 0.458 -- better conditioned
    than the unit-normal prior -- so the divergences on that cell are not the
    truncation. Crediting this inset for them would be attributing a sampler
    symptom to a forward-model fix.

    Separately worth knowing for the science: that cell pushes hard against the
    top of its library, median :math:`\\xi = +1.544` with 19% of draws beyond
    :math:`\\xi = +2` against a prior expectation of 2.3%. The galaxy wants a
    metallicity above the highest node the library carries, which is a caveat on
    any metallicity quoted for it rather than a defect.
    """
    lgmet = ssp_data.ssp_lgmet
    return Uniform(
        float(lgmet.min()) - LOG10_ZSUN + inset,
        float(lgmet.max()) - LOG10_ZSUN - inset,
    )


def _continuity_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Continuity SFH: piecewise-constant with a Student-t prior on log SFR ratios.

    Shared by Configurations I and VI so the AGN contrast is exact.
    """
    sfh = {
        "type": "continuity",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
        "bin_edges_gyr": tengri.make_agebins_from_zred(z, n_bins=N_SFH_BINS),
    }
    # n_bins bins share n_bins-1 ratios. Leaving these at Fixed(DEFAULT) would
    # pin the star formation shape and the history would not be free at all.
    for i in range(N_SFH_BINS - 1):
        sfh[f"ratio_{i}"] = FREE
    return sfh


def _kriek_conroy_two_component() -> dict:
    """Kriek & Conroy (2013) two-component attenuation, shared by I, IV, and VI."""
    return {
        "type": "two_component",
        "law": "kriek_conroy",
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 3.0),
        "tau_diff": Uniform(0.5, 3.0),
    }


def config_I(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """I: continuity SFH, FSPS MIST/C3K, Kriek+13 two-component, Draine+2014, Cue."""
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=_continuity_sfh(ssp_data, z),
        dust_attenuation=_kriek_conroy_two_component(),
        dust_emission={"type": "dl14", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_II(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """II: double power law, FSPS PARSEC/C3K, Calzetti single screen, Dale+2014, Cue."""
    # tau_gyr is the turnover timescale: once it exceeds age_gyr, the power-law
    # shape flattens by ~200x and tau becomes unobservable. Cap near the cosmic
    # age to keep the turnover in the galaxy's history. See sfh_tau_conditioning.py.
    #
    # TWO CLAIMS, and only the first is established. (a) The cap is right on
    # conditioning and on physics: a turnover longer than the age of the universe
    # at the galaxy's redshift is not a meaningful model, and the direction is
    # measurably flat there. (b) Whether it RESOLVES the 381/1200 divergences
    # observed on galaxy 79 is a separate question, and this cap was committed
    # before the cell testing it reported. If that cell comes back still frozen,
    # (a) still holds and (b) is false -- the row would need a different fix and
    # this cap should not be credited with one.
    # Measured on the committed 13 Gyr prior (2026-09-21): |dF/F| 0.018% outside
    # the age window against 300-2100% inside, with 59% of that prior sitting
    # in the flat regime at z ~ 1.07. See sfh_tau_conditioning.py.
    tau_upper = age_at_z(z)
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "alpha": Uniform(0.5, 5.0),
            "beta": Uniform(0.3, 3.0),
            "tau_gyr": Uniform(0.5, tau_upper),
            "age_gyr": Uniform(1.0, age_at_z(z)),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": met_prior_for(ssp_data),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_III(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """III: delayed-tau, FSPS MIST/MILES, Charlot+2000 two-component, THEMIS, Cue."""
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Uniform(0.1, 20.0),
            "age_gyr": Uniform(1.0, age_at_z(z)),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": met_prior_for(ssp_data),
        },
        # Charlot & Fall (2000) is a power law with a steeper birth-cloud screen
        # than diffuse: tau ~ lambda^-1.3 in the birth cloud, lambda^-0.7 in the
        # ISM. The registry has no law named "charlot_fall2000" and never did --
        # that string is a CITATION key, and resolve.py maps it to this law
        # ("power_law": "charlot_fall2000"). An earlier revision took the
        # citation key for a registry key because grepping the source found the
        # string; being present in the source is not being registered.
        #
        # The slopes are scalars rather than priors because per-screen law
        # shapes are build-time constants folded into the compile signature.
        # That suits this model: Charlot & Fall fix both exponents, so there is
        # nothing here that wants sampling.
        dust_attenuation={
            "type": "two_component",
            "law_bc": "power_law",
            "slope_bc": -1.3,
            "law_diff": "power_law",
            "slope_diff": -0.7,
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.5, 3.0),
        },
        dust_emission={"type": "themis", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_IV(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """IV: Dirichlet SFH, FSPS PARSEC/MILES, Kriek+13 two-component, Casey+2012, Cloudy.

    This is the one row whose nebular emission carries a free ionization
    parameter. Note the nebular backend is declared explicitly: an earlier
    revision of this file used ``neb={"type": "ssp"}``, which selects the
    baked-in backend and returns a zero array on the assumption that the library
    already carries the emission. The MILES grids here do not, so that model had
    no nebular emission at all, silently, and the declaration that selected it is
    also the documented way to silence the backend's only advisory.
    """
    sfh = {
        "type": "dirichlet",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
        "bin_edges_gyr": tengri.make_agebins_from_zred(z, n_bins=N_SFH_BINS),
    }
    # Free the Dirichlet bin weights; otherwise all_params=Fixed(DEFAULT) pins the
    # star formation shape and the model is not nonparametric at all.
    for i in range(N_SFH_BINS - 1):
        sfh[f"z_{i}"] = FREE
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=sfh,
        dust_attenuation=_kriek_conroy_two_component(),
        dust_emission={"type": "casey2012", "all_params": Fixed(DEFAULT)},
        neb={
            "type": "cloudy",
            "grid": cloudy_grid_for_iv(),
            "all_params": Fixed(DEFAULT),
            "neb_logU": Uniform(-4.0, -1.0),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_V(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """V: log-normal SFH, BPASS C3K, SMC single screen, Dale+2014, Cue.

    The log-normal family registers as ``lnorm``; its shape parameters are the
    peak position and width in Gyr rather than a timescale and an age.
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "lnorm",
            "all_params": Fixed(DEFAULT),
            # Peak bounded by the galaxy's age, like age_gyr (owner, 2026-09-21).
            # Uniform(0.1, 13) let more than half the draws put the peak after
            # the epoch of observation, where only the rising limb of the
            # log-normal is inside the galaxy's life and many (peak, width)
            # pairs share one slope: a flat ridge, the row-II tau > age(z)
            # direction again. On 7837, 13097 and 14099 the rung-1 posterior
            # spanned the whole peak prior (p1 0.3, p99 12.9 Gyr) with healthy
            # E-BFMI (0.87-0.99) and width never at its floor, and every retune
            # rung then adapted to a step of 0.0002-0.001 and froze (ESS 1-3).
            # The cap has a measured cost: beyond the age window the peak keeps
            # 0.48-2.70% band sensitivity (inside/outside ratio 2.96-3.26), so
            # rising-SFH solutions the data can constrain are excluded. The
            # ruling trades that for the sampler freezing measured above.
            "peak_gyr": Uniform(0.1, age_at_z(z)),
            "width_gyr": Uniform(0.1, 5.0),
            "age_gyr": Uniform(1.0, age_at_z(z)),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": met_prior_for(ssp_data),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "smc",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def config_VI(ssp_data: tengri.SSPData, observation, z: float) -> SEDModel:
    """VI: Configuration I's host with a QSOgen disc and a SKIRTOR torus.

    The host is matched to Configuration I -- same SFH, library, attenuation,
    dust emission, and nebular backend -- so that the difference between the two
    rows isolates the AGN rather than confounding it with a recipe change. This
    row is fit on every galaxy, including the three mid-infrared AGN candidates.
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=_continuity_sfh(ssp_data, z),
        dust_attenuation=_kriek_conroy_two_component(),
        dust_emission={"type": "dl14", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)},
        agn={
            "disc": {"type": "qsogen"},
            "torus": {"type": "skirtor"},
            "all_params": Fixed(DEFAULT),
            "log_lbol": Uniform(9.42, 13.42),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


CONFIG_KEYS = ["I", "II", "III", "IV", "V", "VI"]


if __name__ == "__main__":
    test_z = 1.0
    test_filters = [
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
    ]
    obs = tengri.Observation(photometry=tengri.Photometry.from_names(test_filters))
    key = jax.random.PRNGKey(0)

    for cfg_key in CONFIG_KEYS:
        ssp = load_ssp_for(cfg_key)
        met = met_prior_for(ssp)
        model = globals()[f"config_{cfg_key}"](ssp, obs, test_z)
        n_free = len(model.spec.free_params)
        CONFIGS[cfg_key]["n_free"] = n_free
        print(
            f"Config {cfg_key:>3}  {CONFIGS[cfg_key]['ssp_grid']:<32} "
            f"met_logzsol in [{met.lo:+.3f}, {met.hi:+.3f}]  "
            f"{n_free} free"
        )
        print(f"           {model.spec.free_params}")
        pred = model.predict_photometry(model.spec.sample(key=key))
        assert pred.shape == (len(test_filters),), f"{cfg_key}: wrong shape {pred.shape}"
        assert (pred > 0).all(), f"{cfg_key}: non-positive photometry"
    print("\n[ok] all six configurations build and predict positive photometry")
