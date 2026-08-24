# SPDX-License-Identifier: BSD-3-Clause
"""Component pipeline: public entry point.

This namespace gathers the :class:`SEDComponent` Protocol, the
:class:`ForwardState` it threads through, the orchestrator helpers
(:func:`run_components`, :func:`merge_declared_parameters`), and the
shipped adapters.

The legacy :class:`tengri.SEDModel` tier-dispatch path is unchanged; this is an *additive* surface
for users who want to compose SED
forward models out of explicit components.

Example
-------
>>> import jax.numpy as jnp
>>> from tengri.pipeline import (
...     ForwardState,
...     RadioSEDComponent,
...     IGMSEDComponent,
...     run_components,
... )
>>> wave = jnp.logspace(2, 8, 1024)
>>> state = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave))
>>> chain = [RadioSEDComponent(), IGMSEDComponent()]
>>> params = {
...     "redshift": 0.5,
...     "radio_q_ir": 2.64,
...     "radio_alpha_sf": 0.8,
...     "radio_loudness": 0.0,
...     "radio_alpha_agn": 0.7,
...     "radio_T_e": 1e4,
...     "radio_alpha_ff": -0.1,
...     "igm_z_mid": 7.0,
...     "igm_dz": 0.5,
...     "igm_log_nhi": 20.0,
... }
>>> final = run_components(chain, state, params)

Stable surface
--------------
Names exported from this namespace are the **canonical** import paths
for the component pipeline. Their location may move as the surface
evolves; old imports keep working with a :class:`DeprecationWarning`.

See :doc:`docs/dev/api_migration_v0.x.md` for the migration table.
"""

from __future__ import annotations

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.forward.orchestrator import (
    default_params_dict,
    merge_declared_parameters,
    run_components,
    sample_params_dict,
    slice_params_for_component,
)
from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.likelihoods.marginalized import (
    CalibrationELineMarginalizedLikelihood,
    CalibrationMarginalizedLikelihood,
    CloudyELineMarginalizedLikelihood,
    ELineFittedLikelihood,
    ELineMarginalizedLikelihood,
)
from tengri.inference.likelihoods.protocol import (
    CensoredLikelihood,
    GaussianLikelihood,
    MultivariateGaussianLikelihood,
    StudentTLikelihood,
)
from tengri.inference.photometry_likelihood import PhotometryLikelihood
from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood
from tengri.observation.photometry_model import PhotometryObservationModel
from tengri.protocols import (
    BARE_NAME_ALLOWLIST,
    ForwardState,
    Likelihood,
    ObservationModel,
    ParamDeclaration,
    SEDComponent,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "CalibrationELineMarginalizedLikelihood",
    "CalibrationMarginalizedLikelihood",
    "CensoredLikelihood",
    "CloudyELineMarginalizedLikelihood",
    "CompositeLikelihood",
    "DustAttenuationSEDComponent",
    "ELineFittedLikelihood",
    "ELineMarginalizedLikelihood",
    "ForwardState",
    "GaussianLikelihood",
    "IGMSEDComponent",
    "Likelihood",
    "MultivariateGaussianLikelihood",
    "NebularSEDComponent",
    "ObservationModel",
    "ParamDeclaration",
    "PhotometryLikelihood",
    "PhotometryObservationModel",
    "RadioSEDComponent",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
    "SpectroscopyLikelihood",
    "StudentTLikelihood",
    "XRaySEDComponent",
    "default_params_dict",
    "merge_declared_parameters",
    "run_components",
    "sample_params_dict",
    "slice_params_for_component",
]


#: Names removed in #871 when the monolithic ``DustEmissionSEDComponent`` adapter
#: was retired in favor of per-template ``SEDModelComponent`` emission components.
#: No 1:1 successor exists (the adapter dispatched three templates); accessing the
#: old name raises with the migration path rather than silently aliasing to one
#: component.
_REMOVED_DUST_EMISSION_NAMES = frozenset(
    {"DustEmissionSEDComponent", "DustEmissionSEDComponentConfig", "DustEmissionSEDComponentState"}
)


def __getattr__(name: str):
    if name in _REMOVED_DUST_EMISSION_NAMES:
        raise AttributeError(
            f"{name!r} was removed in tengri #871. Dust IR emission is now authored "
            "as SEDModelComponents selected via the model grammar, e.g. "
            "SEDModel.build(dust_attenuation={'law': 'calzetti'}, "
            "dust_emission={'type': 'astrodust'}) with type in "
            "{'modified_blackbody', 'draine2021_pah', 'astrodust', 'dale2014', "
            "'casey2012', 'schreiber2016', ...}. To import a component directly use e.g. "
            "tengri.components.dust.emission.templates.astrodust.AstrodustIRSEDComponent."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
