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
:func:`~tengri.observation.spectrum._is_log_uniform` answers ``True``, which
would send a linear grid down the single-FFT constant-R path.

Why the redshift is free in the template
----------------------------------------
The template model must have its redshift free. A fixed redshift is baked
into the model's luminosity distance and into the stellar redshift
interpolation, so a per-galaxy ``obs.z`` could not reach them. The per-galaxy
redshift enters through the parameter dictionary instead, and any ``redshift``
key in ``params`` is ignored in favour of ``obs.z``.

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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.observation.banded import BandedMatrix
from tengri.observation.batched import SpectroBatch, SpectroBatchSpec, require_spec


def check_template_model(model) -> None:
    """Raise if ``model`` cannot serve as a batched template.

    Parameters
    ----------
    model : SEDModel
        Template model. Its redshift must be free, its spectroscopy must be
        configured, and it must not use the spectrum LUT.

    Raises
    ------
    ValueError
        If the model has a fixed redshift, uses the ``SpectrumPrecomp`` spectrum
        LUT, or has no spectroscopy.

    Notes
    -----
    Build-time Python check, not JIT-compatible. Call once before
    :func:`make_batched_predict`.
    """
    observation = getattr(model, "observation", None)
    if observation is None or not observation.can_do_spectroscopy:
        raise ValueError("template model has no spectroscopy configured")
    if model.z_fixed is not None:
        raise ValueError(
            "template model has a fixed redshift; build it with redshift free "
            "so the per-galaxy obs.z is the only redshift the batch uses"
        )
    if bool(model._approx.get("spectrum_precomp")):
        raise ValueError(
            "template model uses the SpectrumPrecomp spectrum LUT, which bakes the "
            "observation grid; build it with spectrum_precomp off"
        )


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
    conserving: bool,
    threaded: tuple[Any, Any, Any] | None = None,
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
        Static bucket key of the batch ``obs`` belongs to.
    conserving : bool
        Flux-conserving pixel integral, static. The caller resolves it.
    threaded : tuple, optional
        ``(ssp_data, template_data, ztable_data)`` from
        ``model._resolve_threaded_data``. Resolved from the model when ``None``.

    Returns
    -------
    dict
        ``"spec_fnu"`` of shape ``(n_max,)``, F_nu [erg/s/cm^2/Hz], zero on padded
        and masked pixels. ``"phot_fnu"`` of shape ``(n_filt,)`` when ``spec.has_phot``,
        otherwise shape ``(0,)``.

    Raises
    ------
    ValueError
        If ``spec`` is not a :class:`SpectroBatchSpec`, or the bucket has photometry
        and the template has none.

    Notes
    -----
    **JIT-compatible**: yes, with ``spec`` and ``conserving`` closed over or static.
    **Gradient-compatible**: yes, with respect to ``params`` and the real pixel data.
    Mirrors the spectral block of :meth:`Observation.predict` and the
    closed-over path of ``predict_observables_jit``; the only substitutions are the
    wavelength grid, the resolution operator, the redshift and the calibration range,
    which come from ``obs``.
    """
    from tengri.cosmology import luminosity_distance
    from tengri.observation.observation import project_spectrum_kernel_split
    from tengri.observation.photometry import project_photometry
    from tengri.observation.spectrum import resolve_sigma_lib_kms

    require_spec(spec)
    observation = model.observation
    if spec.has_phot and not observation.can_do_photometry:
        raise ValueError("bucket carries photometry but the template has none")

    if threaded is None:
        threaded = model._resolve_threaded_data(None, None, None)
    ssp_data, template_data, ztable_data = threaded

    free = {k: v for k, v in params.items() if k != "redshift"}
    fixed = {**dict(model.spec.get_fixed_values()), "redshift": obs.z}
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
    )
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
    model, spec: SpectroBatchSpec, *, conserving: bool
) -> Callable[[dict, SpectroBatch], dict[str, jax.Array]]:
    """Return a jitted function mapping a batch of parameters and observations.

    Parameters
    ----------
    model : SEDModel
        Template model (see :func:`check_template_model`).
    spec : SpectroBatchSpec
        Static bucket key shared by every galaxy the function will receive.
    conserving : bool
        Flux-conserving pixel integral, passed to every galaxy.

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
    require_spec(spec)
    if getattr(model, "_cached_component_chain", None) is None:
        model._cached_component_chain = model._build_component_chain()
    threaded = model._resolve_threaded_data(None, None, None)

    def one(p, o):
        return predict_batched_observables(
            model, p, o, spec, conserving=conserving, threaded=threaded
        )

    return jax.jit(jax.vmap(one, in_axes=(0, 0)))
