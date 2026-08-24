# SPDX-License-Identifier: BSD-3-Clause
"""Stand-in runtime histories for build-time precomputation (#1718).

A precompute builder evaluates the forward model at reference parameters it gets
from ``spec.sample(...)``. That works for every model whose inputs are declared
parameters, and fails for exactly one class: a tabulated SFH declares **zero**
parameters, because the table *is* the SFH, so no prior can produce
``sfh_t_gyr`` / ``sfh_sfr`` and the stellar component raises the #996 guard
before the builder gets its first row.

The fix is not to teach the sampler about tables. It is that these builders do
not need the caller's SFH at all: they only need *an* ionizing spectrum. Both
nebular tables store luminosity **per ionizing photon** and divide Q_H back out,
which ``line_precompute`` states outright: the table is "a property of the gas,
independent of the reference SFH". Q_H itself is recomputed at every evaluation
from whatever SFH is live then.

Measured on the PRSC/MILES grid with Cue, at fixed
``met_logzsol=-0.3, logU=-2.5, logZ_gas=-0.2``, across delayed-exponential SFHs
from ``tau=0.1`` to ``tau=10`` Gyr; Q_H itself spanning 1.3e50 to 4.6e53, four
orders of magnitude:

===============  ==================
line             per-Q_H spread
===============  ==================
Halpha           1.0001x
Hbeta            1.0002x
NII_6584         1.0019x
OIII_5007        1.0232x
===============  ==================

So the invariance is essentially exact for recombination lines and good to 2.3%
for the most hardness-sensitive collisionally-excited line, that residual being
the ionizing-spectrum shape which Q_H normalization does not remove. This is not
a new approximation introduced for tabulated SFHs: the LUT is built at **one**
``spec.sample`` draw and reused for every SFH a fit visits, so a parametric model
has been relying on the same invariance all along.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "reference_history_for_config",
    "reference_history_params",
    "stellar_config_of",
]

#: Nodes in the stand-in history. Enough to resolve the young ages that carry the
#: ionizing output without making the build-time forward pass expensive; the
#: table is per-Q_H, so this count sets resolution, not normalization.
_N_NODES = 64

#: Metallicity of the stand-in Z(t), log10(Z/Zsun). Solar, matching the
#: ``met_logzsol`` default. Only reached when ``met_mode='table'``, where the
#: metallicity axis has no declared parameter for the grid to vary.
_REF_LOGZSOL = 0.0


def stellar_config_of(model):
    """The StellarSEDComponent's config on an SEDModel, or None.

    Parameters
    ----------
    model: SEDModel
        A built model.

    Returns
    -------
    StellarSEDComponentConfig or None
        ``None`` when the chain cannot be built or carries no stellar component.

    Notes
    -----
    **JIT-compatible**: no; eager introspection at build time.
    """
    from tengri.components.stellar.component import StellarSEDComponent

    try:
        chain = model._build_component_chain()
    except (AttributeError, IndexError):
        return None
    stellar = next((c for c in chain if isinstance(c, StellarSEDComponent)), None)
    return None if stellar is None else stellar.config


def reference_history_params(model, *, redshift=0.0, n_nodes=_N_NODES):
    """Runtime arrays a tabulated model needs before any table exists.

    Returns the ``{}`` no-op for every model whose SFH is parametric, so a caller
    can merge it unconditionally.

    Parameters
    ----------
    model: SEDModel
        The model whose reference evaluation is about to run.
    redshift: float, optional
        Reference redshift, used only to end the time axis at the right cosmic
        age so no star formation falls after the epoch of observation. [dimensionless]
    n_nodes: int, optional
        Nodes in the stand-in history.

    Returns
    -------
    dict
        ``{}``, or ``sfh_t_gyr`` [Gyr] + ``sfh_sfr`` [Msun/yr], plus
        ``met_history`` [log10(Z/Zsun)] when the metallicity is tabulated too.

    Notes
    -----
    **JIT-compatible**: no, numpy, once per build.

    The stand-in is a **constant SFR** over cosmic time, anchored to zero at
    ``t=0`` so nothing extrapolates past the Big Bang. Constant is the deliberate
    choice: of the SFHs measured in this module's table it sits at the center of
    the (tiny) per-Q_H spread, where a burst sits at the edge. Its amplitude is
    arbitrary and divides out with Q_H.
    """
    return reference_history_for_config(
        stellar_config_of(model), redshift=redshift, n_nodes=n_nodes
    )


def reference_history_for_config(config, *, redshift=0.0, n_nodes=_N_NODES):
    """The stand-in for a resolved stellar config: the model-free half.

    Split out from :func:`reference_history_params` so the decision can be tested
    without standing up a model: the config is the only thing it reads, and a
    stub of one is not an instance of ``StellarSEDComponent``, so a test that
    fakes the *model* silently exercises the ``None`` branch instead.

    Parameters
    ----------
    config: StellarSEDComponentConfig or None
        ``None`` yields ``{}``.
    redshift: float, optional
        Reference redshift, setting where the time axis ends. [dimensionless]
    n_nodes: int, optional
        Nodes in the stand-in history.

    Returns
    -------
    dict
        See :func:`reference_history_params`.

    Notes
    -----
    **JIT-compatible**: no, numpy, once per build.
    """
    if config is None or getattr(config, "sfh_model", None) != "table":
        return {}

    from tengri.utils.cosmology import age_at_z

    t_obs_gyr = float(age_at_z(float(redshift)))
    t = np.linspace(0.0, t_obs_gyr, n_nodes)
    sfr = np.ones(n_nodes)
    sfr[0] = 0.0  # anchor at the Big Bang; the component warns on mass before it

    out = {"sfh_t_gyr": jnp.asarray(t), "sfh_sfr": jnp.asarray(sfr)}
    if getattr(config, "metallicity_model", None) == "table":
        # Z(t) shares the SFH's time axis by contract (#996), so it is only
        # meaningful alongside a tabulated SFH: which the branch above assures.
        out["met_history"] = jnp.full(n_nodes, _REF_LOGZSOL)
    return out
