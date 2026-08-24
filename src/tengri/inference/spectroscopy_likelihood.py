# SPDX-License-Identifier: BSD-3-Clause
"""SpectroscopyLikelihood: convenience subclass of :class:`GaussianLikelihood`.

Pinned to the ``"spec_fnu"`` prediction-dict key with the legacy
``fnu_obs`` / ``fnu_err`` constructor argument names. Identical
math to :class:`tengri.inference.likelihoods.GaussianLikelihood`,
 this class exists for ergonomic discovery (autocomplete, docs,
``isinstance``-style channel detection in the Fitter auto-build
path) and for the legacy constructor signature.

Equivalent to:

>>> GaussianLikelihood(channel="spec_fnu", obs=fnu, err=err)

For correlated noise, use
:class:`tengri.inference.likelihoods.MultivariateGaussianLikelihood`
with ``channel="spec_fnu"`` and a pre-inverted covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from tengri.inference.likelihoods.protocol import GaussianLikelihood

__all__ = ["SpectroscopyLikelihood"]


@dataclass(frozen=True, init=False)
class SpectroscopyLikelihood(GaussianLikelihood):
    r"""``GaussianLikelihood`` pinned to ``channel="spec_fnu"``."""

    def __init__(
        self,
        fnu_obs: jnp.ndarray,
        fnu_err: jnp.ndarray,
        sigma_floor: float = 0.0,
        data_slice: tuple[int, int] | None = None,
    ) -> None:
        # obs_key/err_key default to the Fitter data_args entries so a
        # shared compiled loss reads the current galaxy's data (see
        # ``resolve_channel_data``); ``data_slice`` selects the
        # spectroscopy segment of a joint data vector.
        super().__init__(
            obs=fnu_obs,
            err=fnu_err,
            channel="spec_fnu",
            sigma_floor=sigma_floor,
            name="spectroscopy_gaussian",
            obs_key="data",
            err_key="noise",
            data_slice=data_slice,
        )

    @property
    def fnu_obs(self) -> jnp.ndarray:
        return self.obs

    @property
    def fnu_err(self) -> jnp.ndarray:
        return self.err
