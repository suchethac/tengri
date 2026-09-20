# SPDX-License-Identifier: BSD-3-Clause
"""Shared utilities for persisting posterior draws and diagnostics.

This module contains helpers used by fit_one.py and fit_mock_joint.py to
serialize MCMC posteriors consistently across both drivers.
"""

import numpy as np


def build_npz_payload(samples_thin: dict, *extras: dict) -> dict:
    """Merge sampled draws and every derived array into one NPZ payload.

    The single place the NPZ's keys are assembled, so a derived name can never
    silently shadow -- or, under ``np.savez(**a, **b)``, collide with -- a
    sampled parameter's name (#2089). A collision raises here, before any
    expensive save work, and names the offending key.
    """
    payload = dict(samples_thin)
    for extra in extras:
        for key, value in extra.items():
            if key in payload:
                raise ValueError(f"derived quantity {key!r} collides with a sampled parameter")
            payload[key] = value
    return payload


def divergent_draw_payload(posterior) -> dict:
    """The parameter values AT the divergent transitions, unthinned.

    A divergence count cannot tell 22 divergences spread over the posterior
    from 22 in one corner of it, and those have different causes: the first is
    an integrator that is marginally too coarse everywhere, the second is a
    region of the density the sampler cannot follow. Only the second is fixed
    by changing the model. Nothing on disk could distinguish them, so the
    retune ladder was the only available response and it is the wrong one --
    on galaxy 79 configuration I it bought 3 of 14 divergences for half the
    effective sample size (ESS 273 -> 131).

    Saved UNTHINNED and separately from the thinned draws on purpose.
    Divergences are sparse -- tens out of a thousand-odd draws -- so
    ``thin_samples``' ``[::step]`` would discard most of exactly the draws
    being kept for diagnosis. The full record is a few tens of floats per
    parameter; the thinning it bypasses exists to bound a much larger array.

    Returns an empty dict when the sampler published no mask, so a backend
    that does not report one (or an older tengri) still saves.
    """
    mask = (posterior.diagnostics or {}).get("divergent_mask")
    if mask is None:
        return {}
    mask = np.asarray(mask, dtype=bool)
    n_draws = int(next(iter(posterior.samples.values())).shape[0])
    if mask.shape != (n_draws,):
        raise ValueError(
            f"divergent_mask has shape {mask.shape} against {n_draws} flattened draws. "
            "The mask must be the burn-in-sliced, chain-flattened draw axis, or every "
            "parameter value selected by it belongs to a different transition."
        )
    n_div = (posterior.diagnostics or {}).get("n_divergent")
    if n_div is not None and int(mask.sum()) != int(n_div):
        raise ValueError(
            f"divergent_mask sums to {int(mask.sum())} but n_divergent is {int(n_div)}. "
            "One of them is counting a different set of draws -- most likely the mask "
            "was published before the burn-in slice."
        )
    payload = {"divergent_mask": mask}
    for name, values in posterior.samples.items():
        payload[f"divergent_{name}"] = np.asarray(values)[mask]
    return payload


def thin_samples(samples: dict, max_draws: int) -> dict:
    """Thin flattened ``(n_chains * n_samples,)`` draws to at most ``max_draws``.

    tengri returns every chain's kept draws concatenated into one 1-D array per
    parameter; the previous save path indexed them as ``(n_chains, n_samples)``
    and raised ``IndexError`` (#2089).
    """
    n_total = int(next(iter(samples.values())).shape[0])
    step = max(1, -(-n_total // max_draws))  # ceiling division: result <= max_draws
    return {k: np.asarray(v)[::step] for k, v in samples.items()}
