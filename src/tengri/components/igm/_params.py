# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the IGM component.

Three tuples, each the canonical source for the corresponding IGM
parameter family:

- :data:`PARAMS`: CGM damping-wing knobs (``igm_z_mid``, ``igm_dz``,
  ``igm_log_nhi``) declared by :class:`IGMSEDComponent`. Not registered
  by the flat ``Parameters(...)`` builder: these always traveled the
  SEDComponent path.
- :data:`PATCHY_PARAMS`: patchy reionization extras (``igm_x_HI``,
  ``igm_bubble_mpc``). Registered when ``igm_patchy=True``. Backs
  ``_IGM_PATCHY_PARAMS`` in ``_param_defs``.
- :data:`DLA_PARAMS`: Damped Lyman-α absorber knobs
  (``dla_log_n_hi``, ``dla_z``, ``dla_temp``, ``dla_b_turb``).
  Registered when ``dla=True``. Backs ``_DLA_PARAMS``. The ``dla_*``
  prefix is owned here because the DLA absorber is conceptually an
  IGM/line-of-sight phenomenon.

"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration, declared_default

PARAMS: tuple[ParamDeclaration, ...] = (
    # The three CGM damping-wing knobs describe a sigmoid in redshift, so their
    # ranges come from the epoch the sigmoid has to span rather than from a
    # tabulated grid. Bounds are reasoned from established landmarks and stated
    # here rather than attributed to a paper that does not set them.
    ParamDeclaration(
        "igm_z_mid",
        Fixed(7.0),
        "CGM damping-wing sigmoid midpoint redshift [dimensionless]",
        # Brackets the reionization epoch: transmission in the Lyman-alpha
        # forest shows it is essentially complete by z ~ 5.5, and it is well
        # underway by z ~ 12. A midpoint outside that is not a midpoint.
        free_prior=Uniform(5.0, 12.0, "Damping-wing midpoint redshift", default=7.0),
    ),
    ParamDeclaration(
        "igm_dz",
        Fixed(0.5),
        "CGM damping-wing sigmoid width [dimensionless]",
        # Below ~0.1 the sigmoid is a step at any redshift sampling the model
        # uses; above ~3 it is broader than the epoch it is meant to describe
        # and stops being a transition at all.
        free_prior=Uniform(0.1, 3.0, "Damping-wing width", default=0.5),
    ),
    ParamDeclaration(
        "igm_log_nhi",
        Fixed(20.0),
        "CGM plateau log10(N_HI / cm^-2) [dimensionless]",
        units="log10(cm^-2)",
        # Spans Lyman-limit systems (log N_HI ~ 17.2, where the gas first turns
        # optically thick at the Lyman edge) through the damped Lyman-alpha
        # threshold at 20.3 and a little beyond.
        free_prior=Uniform(17.0, 21.0, "CGM plateau column", units="log10(cm^-2)", default=20.0),
    ),
)

PATCHY_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "igm_x_HI",
        Fixed(0.0),
        "Volume-averaged neutral hydrogen fraction for patchy IGM "
        "(Miralda-Escude 1998; 0 = fully ionized, 1 = fully neutral)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "Volume-averaged neutral hydrogen fraction", default=0.0),
    ),
    ParamDeclaration(
        "igm_bubble_mpc",
        Fixed(10.0),
        "Ionized bubble radius in proper Mpc for patchy IGM (Mason+2018; 0.1-100)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # The interval this description already states, alongside its source.
        free_prior=Uniform(0.1, 100.0, "Ionized bubble radius", units="Mpc", default=10.0),
    ),
)

DLA_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "dla_log_n_hi",
        # Damped Lyman-α threshold: log10(N_HI / cm⁻²) = 20.3 (Wolfe+2005 DLA
        # definition). Picking the threshold as the default means "if you
        # marked dla_log_n_hi FIXED you got the canonical DLA-floor column".
        Uniform(19.0, 22.0, default=20.3),
        "log10(N_HI / cm^-2) for foreground DLA absorber (Voigt profile)",
        lambda lo, hi: lo >= 15 and hi <= 24,
        "must be in [15, 24]",
        units="log10(cm^-2)",
    ),
    ParamDeclaration(
        "dla_z",
        Fixed(0.0),
        "Redshift of DLA absorber (defaults to source z if fixed at 0)",
        # Deliberately NO free_prior. 0 is not a value here, it is a sentinel
        # meaning "put the absorber at the source redshift". A prior would give
        # that sentinel measure zero and silently change what the default means
        # -- and no interval is right in any case, since a foreground absorber
        # can sit anywhere between the observer and the source. Free it
        # explicitly against your own source redshift.
    ),
    ParamDeclaration(
        "dla_temp",
        Fixed(1e4),
        "Gas temperature of DLA absorber (K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
        # DLA gas is cool neutral hydrogen; the range spans the cold neutral
        # medium up to the warm phase, with the canonical 1e4 K as default. It
        # enters the Voigt profile only through the thermal width, which goes as
        # sqrt(T), so the observable spans a factor ~2.4 across this range.
        free_prior=Uniform(3.0e3, 3.0e4, "DLA gas temperature", units="K", default=1e4),
    ),
    ParamDeclaration(
        "dla_b_turb",
        Fixed(0.0),
        "Turbulent broadening of DLA absorber (km/s)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        units="km/s",
        # Added in quadrature with the thermal width, so 0 (the default) means
        # a purely thermal profile. The ceiling is set by where "turbulence"
        # stops being a sensible description of a single absorber's kinematics
        # and becomes unresolved velocity structure.
        free_prior=Uniform(0.0, 50.0, "DLA turbulent broadening", units="km/s", default=0.0),
    ),
)

#: Default DLA column for standalone calls, read from the declaration above
#: (ADR-0011). ``igm_absorption`` previously hardcoded 20.0, which is *below*
#: the 20.3 threshold that defines a damped Lyman-alpha system; so the default
#: "DLA" was a sub-DLA.
DEFAULT_DLA_LOG_N_HI = declared_default(DLA_PARAMS, "dla_log_n_hi")

__all__ = ["DEFAULT_DLA_LOG_N_HI", "DLA_PARAMS", "PARAMS", "PATCHY_PARAMS"]
