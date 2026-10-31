# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic forward pass over per-galaxy observations.

:class:`~tengri.forward.sed_model.SEDModel` binds its observation at
construction: the wavelength grid, resolution matrix and redshift are closed
over, so :func:`jax.vmap` can batch parameters for one galaxy but not galaxies.
This module evaluates one galaxy from a :class:`~tengri.observation.batched.SpectroBatch`
slice and maps that over a batch.

Static and traced split
-----------------------
The :class:`~tengri.observation.batched.SpectroBatchSpec` fixes array shapes
and control flow: the padded pixel count, the banded offsets, the grid kind
and the LSF binning. The batch arrays (wavelength, mask, resolution rows,
redshift, calibration range, photometry) are traced. The resolution path is
chosen from the static grid kind, never from a traced grid: under a trace
``_is_log_uniform`` (``tengri.observation.spectrum``) answers ``False`` (unknown),
which sends the grid down the general path (#2847).

Why the redshift is free in the template
----------------------------------------
The template model must have its redshift free. A fixed redshift is baked
into the model's luminosity distance and into the stellar redshift
interpolation, so a per-galaxy ``obs.z`` could not reach them. The per-galaxy
redshift enters through the parameter dictionary instead, and any ``redshift``
key in ``params`` is ignored in favor of ``obs.z``.

Per-galaxy SFH bin edges
------------------------
A non-parametric SFH's bin ladder is part of the model, so a template carries
one ladder for every galaxy. A ladder that follows each galaxy's redshift (for
example ``make_agebins_from_zred(zred=z)``) enters the same way the redshift
does: :class:`~tengri.observation.batched.GalaxySpectrum` takes it as
``sfh_bin_edges_gyr``, the batch carries it as a traced leaf, and this module
passes it to the stellar component as the ``sfh_bin_edges_gyr`` value of the
parameter dictionary, in place of the template's ladder. The template fixes
the edge count, since its declared ratio parameters fix the bin count; the
stellar component refuses a ladder of any other length, and any SFH type
outside ``RUNTIME_BIN_EDGE_TYPES`` (``tengri.components.stellar.sfh.registry``).

Flux-conserving mode
--------------------
The resample mode is not a caller choice. :func:`~tengri.observation.batched.build_spectro_batches`
resolves it per galaxy from the template's ``resample`` setting, stores it in
``spec.conserving``, and every entry point here reads it from there.

JIT note
--------
:func:`make_batched_predict` returns a jitted, vmapped function. The template
model's spectrum LUT must be off (:func:`check_template_model` refuses it), and
the template's structure, not its grid, sets the compile. The per-call arrays
are the batch and the parameters, so one compile serves every batch that shares
a :class:`~tengri.observation.batched.SpectroBatchSpec`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.component import RUNTIME_SFH_BIN_EDGES_KEY
from tengri.observation.banded import BandedMatrix
from tengri.observation.batched import SpectroBatch, SpectroBatchSpec, require_spec
from tengri.observation.eline_marginalization import (
    apply_doublet_constraints,
    build_eline_design_matrix,
    eline_amplitude_names,
)

# Prefix of the per-line amplitude parameters of the fitted emission-line mode.
_ELINE_AMP_PREFIX = "eline_amp_"

# Observation channels the batched Gaussian likelihood does not evaluate. A template
# that configures one would be fitted without it, silently, so it is refused.
_UNSUPPORTED_CHANNELS = ("line_fluxes", "line_ratios", "spectral_indices", "lines")


def _is_zero_floor(floor) -> bool:
    """Whether a noise model's calibration floor is the scalar zero default."""
    return isinstance(floor, (int, float)) and float(floor) == 0.0


def check_template_model(model) -> None:
    """Raise if ``model`` cannot serve as a batched template.

    Parameters
    ----------
    model : SEDModel
        Template model. Its redshift must be free, its spectroscopy must be
        configured, and it must not use the spectrum LUT, a ``catalog_z_range``,
        line fluxes, line ratios, spectral indices, marginalized emission lines,
        broad emission lines (``eline_broad``), a spectral covariance or a
        Student-t or calibration-floor noise model. Emission lines with
        ``eline_mode='fitted'`` and ``eline_broad=False`` are accepted.

    Raises
    ------
    ValueError
        If the model is refused, naming the setting that refuses it.

    Notes
    -----
    Build-time Python check, not JIT-compatible. Call once before
    :func:`make_batched_predict`. The batched likelihood is the Gaussian term on
    the spectrum and the photometry only, so each refused setting would be dropped
    without a trace.
    """
    observation = getattr(model, "observation", None)
    if observation is None or not observation.can_do_spectroscopy:
        raise ValueError("template model has no spectroscopy configured")
    if model.z_fixed is not None:
        raise ValueError(
            "template model has a fixed redshift; build it with redshift free "
            "so the per-galaxy obs.z is the only redshift the batch uses"
        )
    if getattr(model, "_catalog_z_range", None) is not None:
        raise ValueError(
            "template model has a catalog_z_range (a per-galaxy redshift window in "
            "WavePrecomp); build it with redshift free and without catalog_z_range"
        )
    if bool(model._approx.get("spectrum_precomp")):
        raise ValueError(
            "template model uses the SpectrumPrecomp spectrum LUT, which bakes the "
            "observation grid; build it with spectrum_precomp off"
        )
    spectroscopy = observation.spectroscopy
    if spectroscopy.covariance is not None:
        raise ValueError(
            "template model has a spectral covariance (correlated noise), which the "
            "batched Gaussian likelihood does not evaluate; use diagonal noise"
        )
    if spectroscopy.eline_mode == "marginalized":
        raise ValueError(
            "template model marginalizes emission lines (eline_mode='marginalized'), which "
            "the batched likelihood does not evaluate yet; use eline_mode='fitted' or 'off'"
        )
    if spectroscopy.eline_mode != "off" and spectroscopy.eline_broad:
        raise ValueError(
            "template model fits emission lines with eline_broad=True, which the batched "
            "likelihood does not evaluate; use eline_broad=False"
        )
    for name in _UNSUPPORTED_CHANNELS:
        if getattr(observation, name, None) is not None:
            raise ValueError(
                f"template model configures observation.{name}, which the batched "
                "likelihood does not evaluate"
            )
    noise = observation.noise
    if noise is not None and (
        noise.student_t_dof is not None or not _is_zero_floor(noise.calibration_floor)
    ):
        raise ValueError(
            "template model has a Student-t or calibration-floor noise model, which the "
            "batched Gaussian likelihood does not evaluate"
        )


def require_model_spec(model, spec) -> SpectroBatchSpec:
    """Check that a bucket spec was built for ``model`` and can be evaluated.

    Parameters
    ----------
    model : SEDModel
        Template model.
    spec : SpectroBatchSpec
        Bucket key from :func:`~tengri.observation.batched.build_spectro_batches`.

    Returns
    -------
    SpectroBatchSpec
        The same object.

    Raises
    ------
    ValueError
        If ``spec`` is not a spec, was built without the template (its
        ``conserving`` is unresolved), disagrees with the template's ``lsf_n_bins``,
        or carries photometry whose filter count differs from the template's.

    Notes
    -----
    Build-time Python check, called from every batched entry point before tracing.
    """
    require_spec(spec)
    if spec.conserving is None:
        raise ValueError(
            "this bucket's resample mode is unresolved (it was built without the "
            "template); rebuild it with build_spectro_batches(..., model=template)"
        )
    expected_lines = _template_line_count(model)
    if spec.n_lines != expected_lines:
        raise ValueError(
            f"bucket n_lines={spec.n_lines} disagrees with the template model "
            f"({expected_lines}); rebuild the batch with model=template"
        )
    template_bins = int(model._lsf_n_bins)
    if spec.lsf_n_bins != template_bins:
        raise ValueError(
            f"bucket lsf_n_bins={spec.lsf_n_bins} disagrees with the template model "
            f"({template_bins}); rebuild the batch with model=template"
        )
    if spec.has_phot:
        photometry = model.observation.photometry
        if photometry is None:
            raise ValueError("bucket carries photometry but the template has none")
        if spec.n_filt != int(photometry.n_filters):
            raise ValueError(
                f"bucket has {spec.n_filt} photometric filters but the template has "
                f"{int(photometry.n_filters)}"
            )
    return spec


@dataclass(frozen=True)
class BatchedLineSet:
    """Emission-line data of a template, read once outside the trace.

    Attributes
    ----------
    amplitude_names : tuple of str
        ``eline_amp_<name>`` keys of the independent lines, in column order.
    wavelengths : ndarray, shape (n_lines,)
        Rest-frame vacuum line centers [Angstrom] of the full catalog.
    constraint_matrix : ndarray, shape (n_lines, n_independent)
        Doublet constraint matrix; maps independent amplitudes to all lines.

    Notes
    -----
    Frozen arrays are closed over by the compiled function; they carry no
    per-galaxy data.
    """

    amplitude_names: tuple[str, ...]
    wavelengths: jax.Array
    constraint_matrix: jax.Array


def _template_line_count(model) -> int:
    """Return the number of catalog lines the template fits, or 0 when lines are off."""
    spectroscopy = model.observation.spectroscopy
    if spectroscopy.eline_mode == "off":
        return 0
    return int(spectroscopy.effective_catalog.n_lines)


def batched_line_set(model) -> BatchedLineSet | None:
    """Return the template's :class:`BatchedLineSet`, or ``None`` when lines are off.

    Parameters
    ----------
    model : SEDModel
        Template model (see :func:`check_template_model`).

    Returns
    -------
    BatchedLineSet or None
        ``None`` when ``eline_mode == "off"``.

    Notes
    -----
    Build-time Python function, not JIT-compatible: it reads the line catalog.
    Mirrors ``Fitter._init_eline_arrays`` for the line centers and the constraint
    matrix.
    """
    spectroscopy = model.observation.spectroscopy
    if spectroscopy.eline_mode == "off":
        return None
    catalog = spectroscopy.effective_catalog
    constraint = (
        catalog.build_constraint_matrix()
        if spectroscopy.eline_fix_doublets
        else jnp.eye(catalog.n_lines)
    )
    return BatchedLineSet(
        amplitude_names=tuple(eline_amplitude_names(catalog)),
        wavelengths=jnp.asarray(catalog.wavelengths),
        constraint_matrix=jnp.asarray(constraint),
    )


def batched_eline_design(line_set: BatchedLineSet, obs: SpectroBatch, params: dict) -> jax.Array:
    r"""Return the doublet-constrained line design matrix of one galaxy.

    Parameters
    ----------
    line_set : BatchedLineSet
        Template line data from :func:`batched_line_set`.
    obs : SpectroBatch
        One galaxy, without the batch axis. Supplies the observed grid ``wave``,
        the redshift ``z``, the resolution ``eline_resolution``, the pixel mask
        ``pix_mask`` and the line mask ``line_mask``.
    params : dict
        Parameter values; ``eline_sigma_kms`` and ``eline_delta_v_kms`` default
        to zero when absent.

    Returns
    -------
    ndarray, shape (n_max, n_independent)
        Design matrix :math:`\mathbf{G}_\mathrm{eff} = \mathbf{G}\,\mathbf{C}` with
        padded pixels and unused lines set to zero.

    Notes
    -----
    Each column of :math:`\mathbf{G}` is a unit-area Gaussian profile,

    .. math::

        G_{ij} = m_i\, p_j\, \frac{\exp\left[-\tfrac{1}{2}
        \left(\frac{\lambda_i - \mu_j}{\sigma_j}\right)^2\right]}
        {\sqrt{2\pi}\,\sigma_j}, \qquad
        \mu_j = \lambda_j^{\rm rest}(1+z)\left(1+\frac{\Delta v}{c}\right),
        \qquad
        \sigma_j = \sqrt{\left(\frac{\lambda_j^{\rm obs}}{2.355\,R}\right)^2
        + \left(\frac{\lambda_j^{\rm obs}\,\sigma_v}{c}\right)^2},
        \qquad \lambda_j^{\rm obs} = \lambda_j^{\rm rest}(1+z),

    where :math:`\lambda_i` is the observed pixel wavelength [Angstrom],
    :math:`\lambda_j^{\rm rest}` the rest vacuum line center [Angstrom],
    :math:`z` the redshift [dimensionless], :math:`R` the spectral resolution
    [dimensionless], :math:`\sigma_v` the intrinsic broadening
    ``eline_sigma_kms`` [km/s], :math:`\Delta v` the velocity offset
    ``eline_delta_v_kms`` [km/s] and :math:`c` the speed of light [km/s].
    :math:`m_i` is ``pix_mask`` (1 on a good real pixel) and :math:`p_j` is
    ``line_mask`` (1 when line :math:`j` is redshifted onto the real grid).
    :math:`\mathbf{C}` is the doublet constraint matrix, so each column of
    :math:`\mathbf{G}_\mathrm{eff}` is one independent amplitude's profile.
    The profile is the one of
    ``tengri.observation.eline_marginalization.build_eline_design_matrix``.

    **JIT-compatible**: yes. The catalog and constraint matrix come from
    ``line_set`` and are closed over; the per-galaxy arrays are traced.
    **Gradient-safe**: yes, with respect to ``sigma`` and the redshift. The masks
    are constants, so a zero mask gives an exactly zero column.
    **vmap-compatible**: yes, over the galaxy axis.
    """
    G = build_eline_design_matrix(
        obs.wave,
        line_set.wavelengths,
        obs.eline_resolution,
        obs.z,
        params.get("eline_sigma_kms", 0.0),
        params.get("eline_delta_v_kms", 0.0),
    )
    G = G * obs.line_mask[None, :] * obs.pix_mask[:, None]
    return apply_doublet_constraints(G, line_set.constraint_matrix)


def _resolution_for(model, spec: SpectroBatchSpec, obs: SpectroBatch):
    """Return the ``resolution`` argument for the spectral projection.

    A per-pixel array selects the variable-R path on a nonuniform grid, and a
    scalar is used only on a log-uniform grid, where the single-FFT path is exact.
    Banded resolution ignores this argument.
    """
    if spec.grid_kind == "banded":
        return model._lsf_resolution
    if spec.grid_kind == "log_uniform":
        return obs.resolution_pp[0]
    return obs.resolution_pp


def predict_batched_observables(
    model,
    params: dict,
    obs: SpectroBatch,
    spec: SpectroBatchSpec,
    *,
    threaded: tuple[Any, Any, Any] | None = None,
    line_set: BatchedLineSet | None = None,
) -> dict[str, jax.Array]:
    """Forward spectrum (and photometry) for one galaxy from its batch slice.

    Parameters
    ----------
    model : SEDModel
        Template model with a free redshift (see :func:`check_template_model`).
    params : dict
        Free parameter values for this galaxy. A ``redshift`` key is ignored;
        the redshift is ``obs.z``.
    obs : SpectroBatch
        One galaxy, without the batch axis (use :meth:`SpectroBatch.galaxy`).
    spec : SpectroBatchSpec
        Static bucket key of the batch ``obs`` belongs to. Its ``conserving``
        flag selects the pixel integral, and its ``lsf_n_bins`` and photometry
        width must match the template (see :func:`require_model_spec`).
    threaded : tuple, optional
        ``(ssp_data, template_data, ztable_data)`` from
        ``model._resolve_threaded_data``. Resolved from the model when ``None``.
    line_set : BatchedLineSet, optional
        Template line data from :func:`batched_line_set`. Built from the model
        when ``None`` and the template fits lines.

    Returns
    -------
    dict
        ``"spec_fnu"`` of shape ``(n_max,)``, F_nu [erg/s/cm^2/Hz], zero on padded
        and masked pixels. With ``eline_mode="fitted"`` the line model
        ``G_eff @ a`` is added to the calibrated continuum before the pixel mask,
        with ``a`` the ``eline_amp_*`` entries of ``params``. ``"phot_fnu"`` of
        shape ``(n_filt,)`` when ``spec.has_phot``, otherwise shape ``(0,)``.

    Raises
    ------
    ValueError
        If ``spec`` is not a :class:`SpectroBatchSpec`, was built for a different
        template (see :func:`require_model_spec`), or carries photometry the
        template does not have. Raised at trace time if ``obs`` carries SFH bin
        edges and the template's SFH type does not take a per-galaxy ladder, or
        the ladder's length differs from the template's.

    Notes
    -----
    **JIT-compatible**: yes, with ``spec`` closed over or static (its ``conserving``
    flag and shapes select the trace).
    **Gradient-compatible**: yes, with respect to ``params`` and the real pixel data.
    Mirrors the spectral block of :meth:`Observation.predict` and the
    closed-over path of ``predict_observables_jit``; the only substitutions are the
    wavelength grid, the resolution operator, the redshift, the calibration range
    and, when ``spec.n_sfh_edges`` is nonzero, the SFH bin edges, which come from
    ``obs``.
    """
    from tengri.cosmology import luminosity_distance
    from tengri.observation.observation import project_spectrum_kernel_split
    from tengri.observation.photometry import project_photometry
    from tengri.observation.spectrum import resolve_sigma_lib_kms

    require_model_spec(model, spec)
    conserving = spec.conserving
    observation = model.observation

    if threaded is None:
        threaded = model._resolve_threaded_data(None, None, None)
    ssp_data, template_data, ztable_data = threaded

    if line_set is None:
        line_set = batched_line_set(model)
    free = {
        k: v
        for k, v in params.items()
        if k != "redshift" and not k.startswith(_ELINE_AMP_PREFIX)
    }
    fixed = {**dict(model.spec.get_fixed_values()), "redshift": obs.z}
    if spec.n_sfh_edges:
        fixed[RUNTIME_SFH_BIN_EDGES_KEY] = obs.sfh_bin_edges_gyr
    state = model.predict_state(
        free,
        fixed_values=fixed,
        ssp_data=ssp_data,
        template_data=template_data,
        ztable_data=ztable_data,
        observables_only=True,
    )
    full = {**fixed, **free}

    z = obs.z
    dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
    sed_rest = state.sed_intrinsic
    wave_rest = state.wave
    igm_trans = state.derived.get("igm_transmission", None) if state.derived is not None else None

    out: dict[str, jax.Array] = {}
    if spec.has_phot:
        out["phot_fnu"] = project_photometry(state, full, observation.photometry, dl_cm=dl_cm)
    else:
        out["phot_fnu"] = jnp.zeros((0,))

    spectroscopy = observation.spectroscopy
    sigma_lib = resolve_sigma_lib_kms(obs.wave, z, model._sigma_lib_kms, _curve(ssp_data))
    spec_fnu = project_spectrum_kernel_split(
        state,
        sed_rest,
        igm_trans,
        wave_rest,
        obs.wave,
        z,
        dl_cm,
        resolution=_resolution_for(model, spec, obs),
        sigma_lib_kms=sigma_lib,
        sigma_v_kms=model._get_sigma_v_kms(full),
        lsf_scale=model._get_lsf_scale(full),
        n_bins=spec.lsf_n_bins,
        cal_coeffs=spectroscopy.calibration_coeffs(full),
        cal_wave_range=(obs.cal_lo, obs.cal_hi),
        conserving=conserving,
        resolution_matrix=_banded_operator(spec, obs),
        lsf_pad_pixels=spec.lsf_pad_pixels,
    )
    if line_set is not None:
        amps = jnp.asarray([params[nm] for nm in line_set.amplitude_names])
        spec_fnu = spec_fnu + batched_eline_design(line_set, obs, full) @ amps
    out["spec_fnu"] = spec_fnu * obs.pix_mask
    return out


def _curve(ssp_data):
    """SSP library resolution curve as in ``predict_observables_jit``, or ``None``."""
    curve_kms = ssp_data.ssp_resolution_kms
    return (ssp_data.ssp_wave, curve_kms) if curve_kms is not None else None


def _banded_operator(spec: SpectroBatchSpec, obs: SpectroBatch):
    """The banded resolution operator for this galaxy, or ``None`` off the banded path."""
    if spec.grid_kind != "banded":
        return None
    return BandedMatrix(offsets=np.asarray(spec.offsets), data=obs.r_data)


def make_batched_predict(
    model, spec: SpectroBatchSpec
) -> Callable[[dict, SpectroBatch], dict[str, jax.Array]]:
    """Return a jitted function mapping a batch of parameters and observations.

    Parameters
    ----------
    model : SEDModel
        Template model (see :func:`check_template_model`).
    spec : SpectroBatchSpec
        Static bucket key shared by every galaxy the function will receive. Its
        ``conserving`` flag is used for every galaxy.

    Returns
    -------
    callable
        ``f(params, batch)``: ``params`` is a dict whose leaves have a leading
        batch axis ``B``, ``batch`` is a :class:`SpectroBatch` with that axis.
        Returns the dict of :func:`predict_batched_observables`, with a leading ``B``.

    Notes
    -----
    **JIT-compatible**: the returned function is jitted. One compile per
    ``spec``; the batch and parameter arrays are traced.
    The template's component chain is built here, outside the trace, so that
    the first call does not build it inside ``jit``.
    """
    check_template_model(model)
    require_model_spec(model, spec)
    threaded = model.threaded_template_inputs()
    line_set = batched_line_set(model)

    def one(p, o):
        return predict_batched_observables(
            model, p, o, spec, threaded=threaded, line_set=line_set
        )

    return jax.jit(jax.vmap(one, in_axes=(0, 0)))
