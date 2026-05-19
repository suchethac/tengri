# SPDX-License-Identifier: BSD-3-Clause
"""Phase II-1 component pipeline — public entry point.

This namespace gathers the :class:`SEDComponent` Protocol, the
:class:`PipelineState` it threads through, the orchestrator helpers
(:func:`run_components`, :func:`merge_declared_parameters`), and every
adapter that ships in Phase II-1.

The legacy :class:`tengri.SEDModel` tier-dispatch path is unchanged —
this is an *additive* surface for users who want to compose SED
forward models out of explicit components.

Example
-------
>>> import jax.numpy as jnp
>>> from tengri.pipeline import (
...     PipelineState,
...     RadioSEDComponent,
...     IGMSEDComponent,
...     run_components,
... )
>>> wave = jnp.logspace(2, 8, 1024)
>>> state = PipelineState(wave=wave, sed_intrinsic=jnp.zeros_like(wave))
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
Names exported from this namespace are the **canonical** Phase II-1
import paths. Their location may move once Phase II-2 lands but old
imports keep working with a :class:`DeprecationWarning`.

See :doc:`docs/dev/api_migration_v0.x.md` for the migration table.
"""

from __future__ import annotations

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.dust.emission_component import DustEmissionSEDComponent
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.forward.orchestrator import (
    merge_declared_parameters,
    run_components,
    sample_params_dict,
    slice_params_for_component,
)
from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.likelihoods.marginalised import (
    CalibrationELineMarginalisedLikelihood,
    CalibrationMarginalisedLikelihood,
    CloudyELineMarginalisedLikelihood,
    ELineFittedLikelihood,
    ELineMarginalisedLikelihood,
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
    Likelihood,
    ObservationModel,
    ParamDeclaration,
    PipelineState,
    SEDComponent,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "CalibrationELineMarginalisedLikelihood",
    "CalibrationMarginalisedLikelihood",
    "CensoredLikelihood",
    "CloudyELineMarginalisedLikelihood",
    "CompositeLikelihood",
    "DustAttenuationSEDComponent",
    "DustEmissionSEDComponent",
    "ELineFittedLikelihood",
    "ELineMarginalisedLikelihood",
    "GaussianLikelihood",
    "IGMSEDComponent",
    "Likelihood",
    "MultivariateGaussianLikelihood",
    "NebularSEDComponent",
    "ObservationModel",
    "ParamDeclaration",
    "PhotometryLikelihood",
    "PhotometryObservationModel",
    "PipelineState",
    "RadioSEDComponent",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
    "SpectroscopyLikelihood",
    "StudentTLikelihood",
    "XRaySEDComponent",
    "merge_declared_parameters",
    "run_components",
    "sample_params_dict",
    "slice_params_for_component",
]
