# SPDX-License-Identifier: BSD-3-Clause
"""Read the :class:`~tengri.inference.channels.NoiseSpec` off a built likelihood cohort.

The Fitter still assembles its likelihood as a cohort of adapters. :func:`noise_spec_of`
translates that cohort into the noise model and the two engine-side column builders the
scorer needs, so the data term scores through :func:`~tengri.inference.channels.score`
without changing how the cohort is built. A likelihood that is not a shipped adapter (a
user's own) returns ``None`` and is called directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tengri.inference.channels import CalibrationSpec, LineSpec, NoiseSpec
from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.likelihoods.marginalized import (
    CalibrationELineMarginalizedLikelihood,
    CalibrationMarginalizedLikelihood,
    CloudyELineMarginalizedLikelihood,
    ELineFittedLikelihood,
    ELineMarginalizedLikelihood,
    ModelELineMarginalizedLikelihood,
)
from tengri.inference.likelihoods.protocol import (
    CensoredLikelihood,
    GaussianLikelihood,
    MultivariateGaussianLikelihood,
    StudentTLikelihood,
)

__all__ = ["ColumnBuilders", "noise_spec_of"]

_CONSTRAINT_CHANNELS = {"line_fluxes", "line_ratios", "indices"}
_MARGINAL_TYPES = (
    CalibrationMarginalizedLikelihood,
    ELineMarginalizedLikelihood,
    CloudyELineMarginalizedLikelihood,
    ModelELineMarginalizedLikelihood,
    ELineFittedLikelihood,
    CalibrationELineMarginalizedLikelihood,
)


@dataclass(frozen=True, eq=False)
class ColumnBuilders:
    """Engine-side producers of the line design columns and the model line-prior inputs.

    Parameters
    ----------
    design : callable or array_like or None
        ``design(params)`` giving the design columns, or the static columns themselves.
    prior_inputs : callable or None
        ``prior_inputs(params)`` giving ``(redshift, dl_cm, aperture_frac, transmission)``.
    """

    design: Any = None
    prior_inputs: Any = None

    def design_for(self, params):
        """Design columns at ``params``, or ``None`` without a line block."""
        if self.design is None:
            return None
        return self.design(params) if callable(self.design) else self.design

    def prior_inputs_for(self, params):
        """Model line-prior inputs at ``params``, or ``None``."""
        return None if self.prior_inputs is None else self.prior_inputs(params)


def _members(likelihood):
    if isinstance(likelihood, CompositeLikelihood):
        return tuple(m for lk in likelihood.likelihoods for m in _members(lk))
    return (likelihood,)


def _routes_data(member, obs_attr="obs", err_attr="err"):
    """Whether the member reads the Fitter's ``data``/``noise`` entries."""
    obs_key = getattr(member, f"{obs_attr}_key", None)
    err_key = getattr(member, f"{err_attr}_key", None)
    return obs_key == "data" and (err_key == "noise" or err_key is None)


def _chol_offsets(member):
    offsets = getattr(member, "chol_offsets", None)
    return None if offsets is None else tuple(int(o) for o in offsets)


def _calibration_of(member):
    return CalibrationSpec(
        wavelength=member.wavelength,
        n_poly=member.n_poly,
        prior_sigma=member.prior_sigma,
        segments=member.segments,
        n_segments=member.n_segments,
        cal_prior=member.cal_prior,
        q_window=member.q_window,
        has_q=member.has_q,
    )


def _line_spec_of(member):
    if isinstance(member, ELineFittedLikelihood):
        return LineSpec(mode="fitted", amplitude_names=tuple(member.amplitude_names))
    if isinstance(member, CloudyELineMarginalizedLikelihood):
        return LineSpec(
            mode="marginalized",
            prior_type="cloudy",
            line_wavelengths=member.line_wavelengths,
            prior_width_dex=member.prior_width_dex,
        )
    if isinstance(member, ModelELineMarginalizedLikelihood):
        return LineSpec(
            mode="marginalized",
            prior_type="model",
            line_wavelengths=member.line_wavelengths,
            prior_width_dex=member.prior_width_dex,
            match_index=member.match_index,
            matched=member.matched,
        )
    if isinstance(member, CalibrationELineMarginalizedLikelihood):
        return LineSpec(
            mode="marginalized",
            prior_type=member.eline_prior_type,
            prior_sigma=member.eline_prior_sigma,
            line_wavelengths=member.eline_line_wavelengths,
            prior_width_dex=member.eline_prior_width_dex,
            match_index=member.match_index,
            matched=member.matched,
        )
    return LineSpec(mode="marginalized", prior_type="flat", prior_variance=member.prior_variance)


def _marginal_routes_data(member):
    return (
        getattr(member, "fnu_obs_key", None) == "data"
        and getattr(member, "fnu_err_key", None) == "noise"
        and member.channel == "spec_fnu"
    )


def _flat_prior_is_supported(member):
    if not isinstance(member, ELineMarginalizedLikelihood):
        return True
    return member.prior_variance is not None and member.presence_key is None


def _split(member, fields, data_type, data_keys):
    """Fold one member into the noise fields; ``False`` when it is not a shipped term."""
    if isinstance(member, _MARGINAL_TYPES):
        if isinstance(member, ELineFittedLikelihood) and member.student_t:
            fields.update(student_t=True, dof=member.dof, f_cal_param=member.f_cal_param)
        return _fold_marginal(member, fields)
    if isinstance(member, StudentTLikelihood):
        ok = member.channel in ("phot_fnu", "spec_fnu") and _routes_data(member)
        fields.update(student_t=True, dof=member.dof, f_cal_param=member.f_cal_param)
        return ok and member.f_cal == 0.0
    if isinstance(member, CensoredLikelihood):
        return _fold_censored(member, fields)
    if isinstance(member, MultivariateGaussianLikelihood):
        if member.channel != "spec_fnu" or member.presence_key is not None:
            return False
        banded = member.chol_data is not None
        fields.update(
            covariance="banded" if banded else "dense",
            chol_offsets=_chol_offsets(member),
        )
        if member.student_t:
            fields.update(student_t=True, dof=member.dof, f_cal_param=member.f_cal_param)
        return _routes_data(member)
    if isinstance(member, GaussianLikelihood):
        if member.sigma_floor != 0.0:
            return False
        if _drops_presence(member, data_type, data_keys):
            return False
        if member.channel in _CONSTRAINT_CHANNELS:
            return True
        return member.channel in ("phot_fnu", "spec_fnu") and _routes_data(member)
    return False


def _drops_presence(member, data_type, data_keys):
    """A photometry Gaussian that ignores a ``presence`` entry the data carries."""
    return (
        member.channel == "phot_fnu"
        and data_type == "photometry"
        and member.presence_key is None
        and "presence" in data_keys
    )


def _fold_censored(member, fields):
    if member.channel == "line_fluxes":
        return member.f_cal == 0.0 and member.dof is None
    if member.channel != "phot_fnu" or not _routes_data(member):
        return False
    fields.update(dof=member.dof, f_cal_param=member.f_cal_param)
    return member.mask_key == "data_mask" and member.model_sigma_frac_key is None


def _fold_marginal(member, fields):
    if not _marginal_routes_data(member) or not _flat_prior_is_supported(member):
        return False
    if isinstance(member, CalibrationMarginalizedLikelihood):
        fields["calibration"] = _calibration_of(member)
    elif isinstance(member, CalibrationELineMarginalizedLikelihood):
        fields["calibration"] = _calibration_of(member)
        fields["lines"] = _line_spec_of(member)
    else:
        fields["lines"] = _line_spec_of(member)
    if getattr(member, "f_cal_param", None) is not None and not isinstance(
        member, ELineFittedLikelihood
    ):
        fields.update(student_t=True, dof=None, f_cal_param=member.f_cal_param)
    if getattr(member, "chol_data", None) is not None:
        if member.chol_key == "spec_cov_chol_data":
            fields.update(covariance="dense")
        else:
            fields.update(covariance="banded", chol_offsets=_chol_offsets(member))
    elif getattr(member, "cov_inv", None) is not None:
        fields.update(covariance="dense")
    return True


def _builders_of(members):
    design = prior_inputs = None
    for member in members:
        if getattr(member, "design_matrix_builder", None) is not None:
            design = member.design_matrix_builder
        elif getattr(member, "design_matrix", None) is not None:
            design = member.design_matrix
        if getattr(member, "prior_inputs_builder", None) is not None:
            prior_inputs = member.prior_inputs_builder
    return ColumnBuilders(design=design, prior_inputs=prior_inputs)


def _n_phot_of(members):
    for member in members:
        data_slice = getattr(member, "data_slice", None)
        if data_slice is None:
            continue
        if member.channel == "phot_fnu":
            return data_slice[1]
        if member.channel == "spec_fnu" or isinstance(member, _MARGINAL_TYPES):
            return data_slice[0]
    return 0


_CONSTRAINT_KEYS = {
    "line_fluxes": "line_flux_waves",
    "line_ratios": "line_ratio_obs",
    "indices": "index_obs",
}


def _channels_match_data(members, keys):
    """Whether the cohort scores exactly the constraint and censoring channels the data carries.

    The scorer selects channels from the ``data`` keys, so a cohort that omits a channel the
    data carries (a user's composite on a fit with line fluxes) cannot be read as a spec.
    """
    if not keys:
        return True
    seen = {m.channel for m in members if getattr(m, "channel", None) in _CONSTRAINT_CHANNELS}
    wanted = {name for name, key in _CONSTRAINT_KEYS.items() if key in keys}
    censored = any(isinstance(m, CensoredLikelihood) and m.channel == "phot_fnu" for m in members)
    return seen == wanted and censored == ("data_mask" in keys)


def noise_spec_of(likelihood, data_type, data_keys=()):
    """Translate a shipped likelihood cohort into ``(NoiseSpec, ColumnBuilders)``.

    Parameters
    ----------
    likelihood : Likelihood
        A cohort adapter or a :class:`CompositeLikelihood` of them.
    data_type : str
        ``"photometry"``, ``"spectroscopy"`` or ``"joint"``.
    data_keys : iterable of str, optional
        The ``data`` keys the galaxy carries.

    Returns
    -------
    tuple or None
        The noise model and the column builders, or ``None`` when any member is not a
        shipped adapter reading the Fitter's data entries.
    """
    members = _members(likelihood)
    fields: dict = {}
    for member in members:
        if not _split(member, fields, data_type, data_keys):
            return None
    if not _channels_match_data(members, set(data_keys)):
        return None
    spec = NoiseSpec(data_type=data_type, n_phot=_n_phot_of(members), **fields)
    return spec, _builders_of(members)
