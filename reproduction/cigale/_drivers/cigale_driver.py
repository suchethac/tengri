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


def attenuation_curve(law_name, wave_aa, **params):
    """A(λ)/A_V from a pcigale dust law's own analytic curve function.

    Calls the module-level curve function each ``dustatt_*`` module exposes —
    ``a_vs_ebv`` for the Calzetti/Leitherer family, ``alambda_av`` for the
    Charlot & Fall power laws — on ``wave_aa``, and normalizes to its value at
    5500 Å.

    This reads the *law*, not a spectrum attenuated by it. Running a stellar
    SED through the module instead measures something else: ``calzleit``
    (``E_BV_old_factor``, default 0.44) and ``modified_CF00`` (birth-cloud term
    on young stars only) attenuate the young and old populations differently,
    so ``-2.5 log10(L_att/L_int)`` on a composite SED returns an
    SSP-weighted *mixture* of two curves, which depends on the SFH and the
    separation age. Against that mixture tengri's single analytic law
    disagrees by 1.7 % (calzleit) to 50 % (modified_CF00) with no physics
    behind either number. ``modified_CF00`` is the one law whose comparison
    needs two curves on both sides, and the notebook composes them explicitly.

    Parameters
    ----------
    law_name : str
        Dust law module name (e.g. ``"dustatt_modified_starburst"``).
    wave_aa : array_like, shape (n_wave,)
        Wavelength grid in Angstroms.
    **params : dict
        Keyword arguments for the module's curve function. ``a_vs_ebv`` takes
        ``bump_wave``, ``bump_width``, ``bump_ampl`` and ``power_slope`` (in
        nm and CIGALE's own units); ``alambda_av`` takes ``delta``.

    Returns
    -------
    a_over_av : ndarray, shape (n_wave,)
        A(λ) normalized to A_V, dimensionless.

    Raises
    ------
    AttributeError
        If the module exposes neither ``a_vs_ebv`` nor ``alambda_av``. Raised
        rather than swallowed: a law that stops evaluating must vanish from
        the panel loudly, not silently.
    """
    mod = importlib.import_module(f"pcigale.sed_modules.{law_name}")
    if hasattr(mod, "a_vs_ebv"):
        curve = mod.a_vs_ebv
    else:
        curve = mod.alambda_av

    wave_aa = np.asarray(wave_aa, dtype=float)
    grid_nm = np.concatenate([wave_aa, [5500.0]]) / 10.0
    a = np.asarray(curve(grid_nm, **params), dtype=float)
    return a[:-1] / a[-1]


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
