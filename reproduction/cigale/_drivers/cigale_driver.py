"""Wrappers around pcigale.sed_modules for reproduction notebook.

High-level interfaces to instantiate CIGALE modules and extract SEDs,
attenuation curves, and star formation histories. Each function handles
module discovery and parameter marshalling so notebook cells stay concise.

References
----------
.. [1] Boquien, M., et al. (2019). CIGALE: Code Investigating GALaxy
       Emission. Astronomy & Astrophysics, 622, A103.
"""

import importlib

import numpy as np
from pcigale.sed import SED

from . import units as U

# Module name → class name mapping for pcigale.sed_modules.*
# (CamelCase conversion is attempted first; this map overrides)
NAME_MAP = {
    "sfhdelayed": "SFHDelayed",
    "bc03": "BC03",
    "m2005": "M2005",
    "bpassv2": "BPASSv2",
    "nebular": "NebularEmission",
    "dustatt_calzleit": "CalzLeit",
    "dustatt_modified_CF00": "ModCF00Att",
    "dustatt_modified_starburst": "ModStarburstAtt",
    "dustatt_powerlaw": "PowerLawAtt",
    "dustatt_2powerlaws": "TwoPowerLawAtt",
    "dl2007": "DL2007",
    "dl2014": "DL2014",
    "casey2012": "Casey2012",
    "dale2014": "Dale2014",
    "themis": "THEMIS",
    "schreiber2016": "Schreiber2016",
    "fritz2006": "Fritz2006",
    "skirtor2016": "SKIRTOR2016",
    "radio": "Radio",
    "redshifting": "Redshifting",
    "yang20": "Yang20",
    "sfh2exp": "Sfh2Exp",
    "sfh_buat08": "SfhBuat08",
    "sfhdelayedbq": "SFHDelayedBQ",
    "sfhfromfile": "SfhFromFile",
    "sfhperiodic": "SfhPeriodic",
    "sfhstochastic_carvajal2025": "StochasticSFH",
}


def cigale_version() -> str:
    """Installed pcigale version, for the SSP-grid provenance line.

    The repackaged grid is gitignored (``*.h5``) and rebuilt from whichever
    pcigale is installed, and §6's dust-IR ratio moves with the CIGALE module
    version. Printing it is what lets a reader tell a library-version
    difference from a physics one.
    """
    from importlib.metadata import version

    return version("pcigale")


def _get_module_class(module_name):
    """Load a pcigale module class by name.

    Parameters
    ----------
    module_name : str
        Module identifier (e.g., "sfhdelayed", "bc03"). Matches the
        filename in pcigale/sed_modules/ (without .py extension).

    Returns
    -------
    cls : type
        The SedModule subclass.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class is not found in the module.
    """
    # Try the NAME_MAP first
    class_name = NAME_MAP.get(module_name)
    if class_name is None:
        # Fallback: CamelCase conversion (e.g., "sfhdelayed" → "Sfhdelayed")
        class_name = module_name[0].upper() + module_name[1:].replace("_", "")

    mod = importlib.import_module(f"pcigale.sed_modules.{module_name}")
    return getattr(mod, class_name)


def run_chain(modules):
    """Execute a chain of pcigale modules and return the SED.

    Parameters
    ----------
    modules : list of tuple
        List of (module_name, params_dict) pairs. Each module is
        instantiated with the params dict and process() is called
        in order on a single SED object.

    Returns
    -------
    sed : pcigale.sed.SED
        The processed SED object with wavelength_grid (nm),
        luminosity (W/nm), and derived quantities.
    """
    sed = SED()
    for module_name, params in modules:
        cls = _get_module_class(module_name)
        # `name=module_name` bypasses pcigale's inspect.getfile() lookup,
        # which fails on dynamically-imported SedModule subclasses.
        module = cls(name=module_name, **params)
        module.process(sed)
    return sed


def to_lnu(sed):
    """Convert pcigale SED to erg/s/Hz on Angstrom wavelength grid.

    Parameters
    ----------
    sed : pcigale.sed.SED
        SED object with wavelength_grid (nm) and luminosity (W/nm).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength in Angstroms.
    L_nu_erg_per_hz : ndarray, shape (n_wave,)
        Luminosity density in erg/s/Hz.
    """
    return U.wnm_to_erg_per_hz_per_aa(sed.wavelength_grid, sed.luminosity)


def attenuation_curve(law_name, **params):
    """Extract attenuation curve A_λ from a pcigale dust law.

    Applies the dust attenuation module to a flat unity stellar SED
    and computes A_λ = -2.5 log10(L_attenuated / L_intrinsic).

    Parameters
    ----------
    law_name : str
        Dust law module name (e.g., "dustatt_modified_starburst").
    **params : dict
        Parameters for the dust law (e.g., E_BV=0.3).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength in Angstroms.
    A_lambda_mag : ndarray, shape (n_wave,)
        Attenuation in magnitudes.
    """
    # Build a flat stellar SED to apply attenuation to. Times are in Myr
    # for sfhdelayed; age_burst=20 / tau_burst=50 are safe defaults that
    # avoid the empty-burst-array crash in pcigale.
    sed_intrinsic = SED()
    sfh_cls = _get_module_class("sfhdelayed")
    sfh = sfh_cls(
        name="sfhdelayed",
        tau_main=1000,
        age_main=5000,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=True,
    )
    sfh.process(sed_intrinsic)

    ssp_cls = _get_module_class("bc03")
    ssp = ssp_cls(name="bc03", imf=1, metallicity=0.02, separation_age=10)
    ssp.process(sed_intrinsic)

    # Get the intrinsic spectrum
    L_intrinsic = sed_intrinsic.luminosity.copy()

    # Apply attenuation
    sed_attenuated = SED()
    sfh.process(sed_attenuated)
    ssp.process(sed_attenuated)

    dust_cls = _get_module_class(law_name)
    dust = dust_cls(name=law_name, **params)
    dust.process(sed_attenuated)

    L_attenuated = sed_attenuated.luminosity

    # Compute A_λ in magnitudes
    with np.errstate(divide="ignore", invalid="ignore"):
        A_lambda_mag = -2.5 * np.log10(L_attenuated / L_intrinsic)
    A_lambda_mag = np.nan_to_num(A_lambda_mag, nan=0.0, posinf=0.0, neginf=0.0)

    wave_aa, _ = U.wnm_to_erg_per_hz_per_aa(sed_intrinsic.wavelength_grid, L_intrinsic)
    return wave_aa, A_lambda_mag


def sfh_curve(sfh_module_name, **params):
    """Extract star formation history from a pcigale SFH module.

    Parameters
    ----------
    sfh_module_name : str
        SFH module name (e.g., "sfhdelayed", "sfh2exp").
    **params : dict
        Parameters for the SFH module.

    Returns
    -------
    t_yr : ndarray, shape (n_time,)
        Time in years (converted from the module's internal Myr grid).
    sfr_msun_per_yr : ndarray, shape (n_time,)
        Star formation rate in Msun/yr.
    """
    sed = SED()
    sfh_cls = _get_module_class(sfh_module_name)
    sfh = sfh_cls(name=sfh_module_name, **params)
    sfh.process(sed)

    # sed.sfh is an SFR array; time grid is implicit [0, 1, 2, ..., n_age-1] Myr
    # Infer age from sed.info["sfh.age"] (or len of sed.sfh for other modules)
    age_myr = sed.info.get("sfh.age", len(sed.sfh))
    t_myr = np.arange(len(sed.sfh), dtype=float)
    t_yr = t_myr * 1e6

    return t_yr, sed.sfh
