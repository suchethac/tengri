# SPDX-License-Identifier: BSD-3-Clause
"""PhotometryLikelihood: convenience subclass of :class:`GaussianLikelihood`.

Pinned to the ``"phot_fnu"`` prediction-dict key with the legacy
``fnu_obs`` / ``fnu_err`` constructor argument names. Identical
math to :class:`tengri.inference.likelihoods.GaussianLikelihood`;
this class exists for ergonomic discovery (autocomplete, docs,
:func:`isinstance`-style channel detection in
:meth:`Fitter._maybe_build_default_likelihood`) and for the legacy
constructor signature.

Equivalent to:

>>> GaussianLikelihood(channel="phot_fnu", obs=fnu, err=err)

Applies to broadband and narrowband photometry, the χ² math is
identical regardless of filter bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from tengri.inference.likelihoods.protocol import GaussianLikelihood

__all__ = ["PhotometryLikelihood"]


@dataclass(frozen=True, init=False)
class PhotometryLikelihood(GaussianLikelihood):
    r"""``GaussianLikelihood`` pinned to ``channel="phot_fnu"``.

    Parameters
    ----------
    fnu_obs : ndarray, shape (n_filters,)
        Observed flux densities [erg/s/cm²/Hz].
    fnu_err : ndarray, shape (n_filters,)
        1-σ uncertainties [erg/s/cm²/Hz]. Must be > 0.
    sigma_floor : float, optional
        Fractional systematic floor added in quadrature.

    Notes
    -----
    Backward-compat names ``fnu_obs`` / ``fnu_err`` work as
    constructor kwargs and as attribute aliases.
    """

    def __init__(
        self,
        fnu_obs: jnp.ndarray,
        fnu_err: jnp.ndarray,
        sigma_floor: float = 0.0,
        data_slice: tuple[int, int] | None = None,
        presence_key: str | None = None,
    ) -> None:
        # frozen=True forbids ordinary __setattr__; route through the
        # superclass __init__ which uses object.__setattr__ internally.
        # obs_key/err_key default to the Fitter data_args entries so a
        # shared compiled loss reads the current galaxy's data (see
        # ``resolve_channel_data``); ``data_slice`` selects the photometry
        # segment of a joint data vector. ``presence_key`` names the data_args
        # entry carrying the per-band presence mask (heterogeneous catalogs,
        # #1317); a no-op when unset or absent from data_args.
        super().__init__(
            obs=fnu_obs,
            err=fnu_err,
            channel="phot_fnu",
            sigma_floor=sigma_floor,
            name="photometry_gaussian",
            obs_key="data",
            err_key="noise",
            data_slice=data_slice,
            presence_key=presence_key,
        )

    @property
    def fnu_obs(self) -> jnp.ndarray:
        return self.obs

    @property
    def fnu_err(self) -> jnp.ndarray:
        return self.err
