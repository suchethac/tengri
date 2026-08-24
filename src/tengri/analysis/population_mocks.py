# SPDX-License-Identifier: BSD-3-Clause
"""Mock galaxy population generator for hierarchical PSD recovery studies.

Produces populations with injected truths and realistic measurement noise,
checking that injected values are discriminable from prior returns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from tengri.components.stellar.component import SFHBeforeBigBangWarning

__all__ = [
    "MockPopulation",
    "assert_truth_against_model",
    "assert_truth_is_discriminating",
    "make_population",
]


def assert_truth_is_discriminating(value, bounds, *, name, rel_tol=0.08):
    r"""Reject an injected truth that a prior-returning estimator could fake.

    Two distinct points in a bounded prior are indistinguishable from "the
    estimator returned its prior", depending on the standardization in force:

    - **Arithmetic midpoint** ``0.5(lo + hi)``: where a uniform prior or
      sigmoid-standardized logit-normal returns nothing (prior expectation).
    - **Geometric mean** ``sqrt(lo * hi)``; where a log-uniform prior returns
      nothing (same value as the lognormal median by mathematical identity).

    Which applies depends on tengri's standardization, which has changed
    historically, so both points are excluded rather than guessing which is
    current.

    Note: The geometric mean and lognormal median ``exp(0.5*(log(lo)+log(hi)))``
    are mathematically identical.

    Parameters
    ----------
    value : float
        The truth to inject [units vary by parameter].
    bounds : tuple of float
        ``(lo, hi)`` prior support, in the same units as ``value``.
    name : str
        Parameter name, for the error message.
    rel_tol : float, optional
        Fractional distance from a characteristic point that counts as too
        close [dimensionless]. Default 0.08.
        Note: For very wide bounds (e.g. (1e6, 3e8)), the span is dominated
        by the upper end; ``rel_tol * span`` may not scale intuitively.
        This guard is designed for bounded priors like (10, 500) Myr.

    Raises
    ------
    ValueError
        If ``value`` lies within ``rel_tol`` of any characteristic point.
    """
    lo, hi = float(bounds[0]), float(bounds[1])

    # Inside the support first. A truth outside the prior is not merely
    # indistinguishable from the prior -- it is UNREACHABLE, and every fit will
    # pin against the nearest bound while looking like ordinary shrinkage.
    # This is not hypothetical: a run injected sigma = 1.3 against a real prior
    # of U(0.01, 1.0), every per-galaxy estimate pinned just under 1.0, and
    # three rounds read that as "compression toward the prior" before the cause
    # was found. Check support before checking discriminability.
    if not lo < value < hi:
        raise ValueError(
            f"Injected truth {name}={value:g} is OUTSIDE the prior bounds "
            f"({lo:g}, {hi:g}), so no fit can reach it: estimates will pin at "
            f"the nearest bound and resemble shrinkage. If these bounds look "
            f"wrong, read them off the model rather than hardcoding them: "
            f"model.spec.get_distribution({name!r}).bounds."
        )

    characteristic = {
        "arithmetic midpoint": 0.5 * (lo + hi),
        "geometric mean": float(np.sqrt(lo * hi)),
    }
    span = hi - lo
    for label, point in characteristic.items():
        if abs(value - point) < rel_tol * span:
            raise ValueError(
                f"Injected truth {name}={value:g} is within {rel_tol:.0%} of the "
                f"prior's {label} ({point:g}) on bounds ({lo:g}, {hi:g}). A "
                f"recovered value there is indistinguishable from the prior: an "
                f"estimator that learned nothing returns the same number. Choose "
                f"a truth away from {sorted(set(round(p, 4) for p in characteristic.values()))}."
            )


def assert_truth_against_model(model, name, value, *, rel_tol=0.08):
    """Validate an injected truth against the model's OWN prior for that parameter.

    Prefer this over :func:`assert_truth_is_discriminating` with hand-written
    bounds. Passing bounds by hand validates a belief about the model rather
    than the model: a run once injected ``sigma = 1.3`` while checking it
    against ``(0.1, 4.0)``, when the model's actual prior was
    ``Uniform(0.01, 1.0)``. The guard passed, the truth was unreachable, and
    every fit pinned just under 1.0 for three rounds.

    Parameters
    ----------
    model : SEDModel
        The model the mock will be generated from and fitted with. Its
        ``spec.get_distribution(name).bounds`` is the authority.
    name : str
        Full parameter name, e.g. ``"sfh_field_psd_sigma"``.
    value : float
        The truth to inject, in that parameter's own units.
    rel_tol : float, optional
        Fractional distance from a characteristic point that counts as too
        close. Default 0.08.

    Returns
    -------
    bounds : tuple of float
        The ``(lo, hi)`` read off the model, so callers can reuse it for grids
        without re-deriving it.

    Raises
    ------
    KeyError
        If ``name`` is not a free parameter of this model.
    ValueError
        If the truth is outside the prior, or too close to a point where a
        prior-returning estimator would land.
    """
    spec = model.spec
    if name not in spec.free_params:
        raise KeyError(
            f"{name!r} is not a free parameter of this model. Free parameters: "
            f"{sorted(spec.free_params)}. A truth cannot be injected for a Fixed "
            f"parameter."
        )
    bounds = spec.get_distribution(name).bounds
    assert_truth_is_discriminating(value, bounds, name=name, rel_tol=rel_tol)
    return bounds


def _max_truncated_fraction(caught):
    """Worst pre-Big-Bang truncation among captured warnings [dimensionless].

    Reads the exact value off the warning instance rather than parsing the
    message, which renders it as ``{:.0%}``.

    Parameters
    ----------
    caught : list of warnings.WarningMessage
        Records from a ``warnings.catch_warnings(record=True)`` block.

    Returns
    -------
    fraction : float
        Largest fraction seen, or ``0.0`` if no such warning carried one. The
        maximum rather than the sum: repeated predictions of the same galaxy
        each describe the whole SFH, so summing would climb past 1.0.
    """
    fractions = [
        w.message.truncated_fraction
        for w in caught
        if isinstance(w.message, SFHBeforeBigBangWarning)
        and getattr(w.message, "truncated_fraction", None) is not None
    ]
    return float(max(fractions)) if fractions else 0.0


def _resample_until_realizable(draw, *, max_fraction, max_attempts):
    """Redraw a galaxy until its SFH fits inside cosmic time (#1645).

    Rejection sampling, made explicit. The accepted draw is from the prior
    **conditioned on** truncating no more than ``max_fraction``; a different
    distribution from the prior, which is why the caller opts in and why the
    count of redraws is reported back.

    Parameters
    ----------
    draw : callable
        ``draw(attempt) -> float``, producing a galaxy and returning the
        fraction of its stellar mass placed before the Big Bang
        [dimensionless]. Called with a 0-based attempt index so the caller can
        vary its PRNG key.
    max_fraction : float
        Largest acceptable truncated fraction [dimensionless]. Inclusive, to
        match the refuse path, which raises only on ``>``.
    max_attempts : int
        Redraw budget for this galaxy [count].

    Returns
    -------
    fraction : float
        Truncated fraction of the accepted draw [dimensionless].
    attempts : int
        Draws taken, including the accepted one [count]. 1 means no redraw.

    Raises
    ------
    ValueError
        If no draw satisfies the threshold within ``max_attempts``. Looping
        forever would hang a mock builder, and keeping the last draw would
        defeat the point of having asked.
    """
    best = None
    for attempt in range(max_attempts):
        fraction = float(draw(attempt))
        if fraction <= max_fraction:
            return fraction, attempt + 1
        best = fraction if best is None else min(best, fraction)
    raise ValueError(
        f"No draw truncated at most {max_fraction:g} of its stellar mass within "
        f"max_attempts={max_attempts}; the best of {max_attempts} attempts still "
        f"truncated {best:g}. Redshift and the SFH age parameters are drawn "
        f"independently, so if the redshift prior reaches high z while the SFH "
        f"timescale stays long, few or no draws are realizable and rejection "
        f"sampling cannot rescue it. Narrow the redshift range, bound the SFH age "
        f"parameters, or raise max_truncated_fraction to accept what the prior "
        f"actually produces (issue #1645)."
    )


@dataclass
class MockPopulation:
    """Mock galaxy population with truths and realistic noise.

    Attributes
    ----------
    table : ndarray
        Per-galaxy record array with keys for photometry fluxes, errors,
        line fluxes, and line errors [various units].
    truth_params : list[dict[str, ndarray]]
        List of per-galaxy parameter dicts sampled from the prior
        with injected ``sigma`` and ``tau_myr`` values.
    n_halpha_absorption : int
        Count of galaxies whose Halpha is predicted in absorption
        (non-positive flux). Never dropped; reported to avoid selection bias.
    truncated_fraction : ndarray, shape (N,), optional
        Per-galaxy fraction of formed stellar mass placed before the Big Bang
        and therefore truncated [dimensionless]. Same policy as
        ``n_halpha_absorption``: measured and reported, never silently dropped,
        so a caller can exclude or flag affected galaxies instead of learning
        about them from stderr (#1645).

        A galaxy whose SFH does not fit inside cosmic time at its own redshift
        is not a faithful fixture: its forward model does not represent the
        requested SFH. This happens because redshift and the SFH age parameters
        are drawn independently, so nothing couples the SFH timescale to the
        time actually available.

        **0.0 means "at most 1%", not "none":** the forward path only warns
        above ``frac > 0.01``, so smaller truncations are invisible here.
    n_resampled : int, optional
        Draws discarded and redrawn under ``resample_truncated`` [count]. Zero
        unless that option was used.

        **Non-zero means the population is not a draw from the prior**, but from
        the prior conditioned on being physically realizable. Report it wherever
        the prior is described; a conditioned population with the conditioning
        left unstated is the same class of error as the truncation it fixes.
    """

    table: np.ndarray
    truth_params: list[dict[str, Any]]
    n_halpha_absorption: int
    line_names: tuple[str, ...] = ()
    line_wavelengths: np.ndarray | None = None
    truncated_fraction: np.ndarray | None = None
    n_resampled: int = 0

    def line_flux_data(self, galaxy=0):
        """Build a :class:`LineFluxData` template matching these mock lines.

        The template an ``Observation`` needs so the fit scores the SAME lines
        the mock measured. Per-galaxy values ride through ``data_args``; this is
        only the declaration, so which galaxy supplies it is immaterial.

        Without this, a caller must re-derive the line names and wavelengths by
        hand, and any drift makes the fit silently photometry-only.

        Parameters
        ----------
        galaxy : int, optional
            Row whose fluxes and errors seed the template. Default 0.

        Returns
        -------
        LineFluxData
        """
        from tengri.observation.line_flux_data import LineFluxData

        if not self.line_names:
            raise ValueError(
                "This MockPopulation carries no line metadata, so no template "
                "can be built. It was generated before line_names was recorded, "
                "or with a model declaring no lines."
            )
        return LineFluxData(
            fluxes=np.asarray(self.table[galaxy]["line_flux_obs"], dtype=float),
            errors=np.asarray(self.table[galaxy]["line_flux_err"], dtype=float),
            wavelengths=np.asarray(self.line_wavelengths, dtype=float),
            names=tuple(self.line_names),
        )


def make_population(
    model,
    *,
    n_galaxies: int,
    sigma_true: float,
    tau_true_myr: float,
    key: jax.Array,
    snr_phot: float,
    snr_line: float,
    max_truncated_fraction: float | None = None,
    resample_truncated: bool = False,
    max_resample_attempts: int = 20,
) -> MockPopulation:
    """Generate a mock galaxy population with injected PSD parameters.

    Draws each galaxy by sampling its parameters from the prior, overriding
    the PSD burstiness ``sigma`` and timescale ``tau_myr`` with injected
    truths, then predicting photometry and emission lines with realistic
    noise. Halpha absorption events are counted but never dropped, as removal
    would bias survivors toward line-bright cases.

    Parameters
    ----------
    model : SEDModel
        The parametrized SED model with an established prior.
    n_galaxies : int
        Number of mock galaxies to generate [count].
    sigma_true : float
        Injected PSD burstiness parameter [dex].
    tau_true_myr : float
        Injected PSD timescale [Myr].
    key : jax.Array
        PRNGKey for galaxy sampling and noise realization.
    snr_phot : float
        Signal-to-noise ratio for photometry [dimensionless].
    snr_line : float
        Signal-to-noise ratio for emission lines [dimensionless].
    max_truncated_fraction : float, optional
        Reject the population if any galaxy places more than this fraction of
        its stellar mass before the Big Bang [dimensionless]. Default ``None``
        (accept anything), which preserves the historical behavior: the
        fraction is always measured and returned on
        :attr:`MockPopulation.truncated_fraction` regardless.

        Off by default deliberately. The fixtures this repository already ships
        contain such galaxies; the four-galaxy ESS-sweep mock truncates 3%, 5%,
        9% and 69%: so defaulting to a limit would reject the project's own
        populations rather than fix them (#1645).
    resample_truncated : bool, optional
        Redraw a galaxy that exceeds ``max_truncated_fraction`` instead of
        raising. Default ``False``. Requires ``max_truncated_fraction``, since
        without a threshold there is nothing to reject against.

        **This changes the distribution.** The result is a draw from the prior
        *conditioned on* being physically realizable, not from the prior; the
        conditioning is reported as :attr:`MockPopulation.n_resampled`.

        Off by default for the same reason as above, plus one more: redrawing
        would silently change every existing mock built from a fixed key,
        including the population behind the banked ESS-vs-breadth curve in
        ``docs/dev/hierarchical-psd-handoff.md`` §4j.

        The whole galaxy is redrawn, not its redshift alone: truncation is a
        property of the sampled redshift *and* SFH age together, so redrawing
        one without the other need not change whether the history fits.
    max_resample_attempts : int, optional
        Redraw budget per galaxy [count]. Default 20. Exhausting it raises
        rather than looping or quietly keeping the last draw: if no draw is
        realizable, the prior and the threshold are incompatible and rejection
        sampling cannot rescue it.

    Returns
    -------
    MockPopulation
        A population record with observed photometry and lines, their errors,
        and per-galaxy truth parameters.

    Notes
    -----
    Uses ``model.measure_line_fluxes(params, line_defs, approx=False)`` to
    measure emission lines, the same operator the likelihood uses, ensuring
    mock self-consistency. Halpha in absorption is a legitimate outcome of
    burstiness during a lull and is always reported via ``n_halpha_absorption``.
    """
    from tengri.observation.line_measurement import default_line_defs

    # Validate the injected truths against the model's OWN priors, read through
    # the public accessor. assert_truth_against_model also rejects an
    # out-of-support truth, which hand-written bounds cannot catch.
    assert_truth_against_model(model, "sfh_field_psd_sigma", sigma_true)
    assert_truth_against_model(model, "sfh_field_psd_tau_myr", tau_true_myr)

    # Generate keys for each galaxy
    keys = jax.random.split(key, n_galaxies)

    # Sample parameters and override PSD terms
    truth_params = []
    truncated_fractions = []
    n_resampled = 0
    phot_true_list = []
    phot_noise_list = []
    phot_obs_list = []
    line_flux_obs_list = []
    line_flux_noise_list = []
    halpha_absorption_count = 0

    # Emission lines to measure.
    #
    # READ THEM FROM THE MODEL when it declares any. The mock must measure the
    # same lines the likelihood will score, or the two disagree silently: a
    # model whose Observation carries no line_fluxes builds no line likelihood
    # at all, so mock lines generated here are simply discarded and the fit is
    # photometry-only while appearing to use lines. That happened: every
    # recovery run in this study was photometry-only for exactly this reason.
    #
    # Falling back to a default set is only correct for generating data that
    # will be wired into an Observation afterwards; the returned MockPopulation
    # carries `line_names` and `line_wavelengths` so a caller can do that
    # without re-deriving them.
    from tengri.observation.line_list import LineList

    obs_lines = getattr(getattr(model, "observation", None), "line_fluxes", None)
    if obs_lines is not None and getattr(obs_lines, "n_lines", 0) > 0:
        line_names = list(obs_lines.names)
        wavelengths = np.asarray(obs_lines.wavelengths, dtype=float)
    else:
        # Strong star-forming set: drops Hgamma and [NII]_6548 (near-zero fluxes
        # let SNR-scaled errors dominate chi-squared). Halpha/Hbeta enable the
        # Balmer decrement, hence the dust constraint the dust-SFR degeneracy
        # needs. Wavelengths come from the canonical catalog, never hardcoded.
        line_names = [
            "OII_3726",
            "OII_3729",
            "OIII_4959",
            "OIII_5007",
            "Halpha",
            "Hbeta",
            "SII_6717",
            "NII_6584",
        ]
        # select() returns wavelength-sorted, so map back to line_names order.
        canonical_lines = LineList.default_optical().select(names=line_names)
        canonical_dict = dict(
            zip(canonical_lines.names, canonical_lines.wavelengths, strict=False)
        )
        wavelengths = np.array([canonical_dict[name] for name in line_names])

    line_defs = default_line_defs(
        wavelengths=wavelengths,
        names=line_names,
    )

    for i, k_i in enumerate(keys):
        # One galaxy = sample the prior, override the injected truths, predict.
        # Bundled into a closure so `resample_truncated` can repeat the whole
        # unit with a fresh key: the truncation is a property of the SAMPLED
        # redshift and SFH age together, so redrawing one without the other
        # would not change whether the history fits in cosmic time (#1645).
        drawn = {}

        def _draw_galaxy(attempt, _k=k_i, _i=i, _out=drawn):
            key_attempt = _k if attempt == 0 else jax.random.fold_in(_k, 10_000 + attempt)
            params = model.spec.sample(key_attempt)
            params["sfh_field_psd_sigma"] = sigma_true
            params["sfh_field_psd_tau_myr"] = tau_true_myr
            # Both prediction calls are inside the capture, not just the first.
            # `predict_photometry` runs the traced path, where this check is
            # deliberately skipped; the warning comes from the EAGER path, which
            # here is `measure_line_fluxes(approx=False)`. Wrapping only the
            # photometry call recorded 0.0 for every galaxy while the warnings
            # still reached stderr.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SFHBeforeBigBangWarning)
                phot = np.asarray(model.predict_photometry(params))
                lines = np.asarray(
                    model.measure_line_fluxes(params, line_defs=line_defs, approx=False)
                )
            _out["params"] = params
            _out["phot"] = phot
            _out["lines"] = lines
            _out["caught"] = caught
            return _max_truncated_fraction(caught)

        if resample_truncated:
            if max_truncated_fraction is None:
                raise ValueError(
                    "resample_truncated=True needs max_truncated_fraction to say what "
                    "counts as acceptable; without a threshold there is nothing to "
                    "reject against. Pass e.g. max_truncated_fraction=0.1."
                )
            fraction, attempts = _resample_until_realizable(
                _draw_galaxy,
                max_fraction=max_truncated_fraction,
                max_attempts=max_resample_attempts,
            )
            n_resampled += attempts - 1
        else:
            fraction = _draw_galaxy(0)

        params = drawn["params"]
        phot_true = drawn["phot"]
        line_flux_true = drawn["lines"]
        caught = drawn["caught"]
        truth_params.append(params)
        truncated_fractions.append(fraction)

        # Re-emit everything the accepted draw produced: capturing is for
        # measurement, not suppression, and a caller who watches stderr today
        # must keep seeing what they see now. Rejected draws stay silent: a
        # galaxy that was redrawn is not one the caller has to reason about.
        for record in caught:
            warnings.warn_explicit(record.message, record.category, record.filename, record.lineno)
        phot_noise = phot_true / snr_phot

        # Add photometric noise
        key_phot = jax.random.fold_in(k_i, i * 2)
        phot_obs = phot_true + phot_noise * np.asarray(
            jax.random.normal(key_phot, shape=phot_true.shape)
        )

        phot_true_list.append(phot_true)
        phot_noise_list.append(phot_noise)
        phot_obs_list.append(phot_obs)

        line_flux_noise = np.abs(line_flux_true) / snr_line

        # Add line-flux noise (absolute value to handle zero/negative fluxes)
        key_line = jax.random.fold_in(k_i, i * 2 + 1)
        line_flux_obs = line_flux_true + line_flux_noise * np.asarray(
            jax.random.normal(key_line, shape=line_flux_true.shape)
        )

        line_flux_obs_list.append(line_flux_obs)
        line_flux_noise_list.append(line_flux_noise)

        # Count Halpha absorption (non-positive predicted flux)
        halpha_idx = line_names.index("Halpha")
        if line_flux_true[halpha_idx] <= 0:
            halpha_absorption_count += 1

    # Stack into record array
    n_bands = len(phot_true_list[0])
    n_lines = len(line_flux_obs_list[0])

    # Create structured array with photometry and line data
    dtype = [
        ("phot_flux_obs", f"({n_bands},)f8"),
        ("phot_flux_err", f"({n_bands},)f8"),
        ("line_flux_obs", f"({n_lines},)f8"),
        ("line_flux_err", f"({n_lines},)f8"),
    ]
    table = np.zeros(n_galaxies, dtype=dtype)

    for i in range(n_galaxies):
        table[i]["phot_flux_obs"] = phot_obs_list[i]
        table[i]["phot_flux_err"] = phot_noise_list[i]
        table[i]["line_flux_obs"] = line_flux_obs_list[i]
        table[i]["line_flux_err"] = line_flux_noise_list[i]

    truncated = np.asarray(truncated_fractions, dtype=float)
    if max_truncated_fraction is not None:
        over = np.flatnonzero(truncated > float(max_truncated_fraction))
        if over.size:
            worst = ", ".join(f"galaxy {int(g)}: {truncated[g]:.0%}" for g in over)
            raise ValueError(
                f"{over.size} of {n_galaxies} mock galaxies place more than "
                f"{float(max_truncated_fraction):.0%} of their stellar mass before the "
                f"Big Bang at their own redshift ({worst}). That mass is truncated, so "
                f"those galaxies' photometry does not represent the SFH recorded in "
                f"truth_params, and any recovery statistic computed on them measures "
                f"something other than what was injected. Redshift and the SFH age "
                f"parameters are drawn independently, so nothing couples the SFH "
                f"timescale to the cosmic time available -- narrow the redshift range, "
                f"bound the SFH age parameters, or raise max_truncated_fraction to "
                f"accept it deliberately (issue #1645)."
            )

    return MockPopulation(
        table=table,
        truth_params=truth_params,
        n_halpha_absorption=halpha_absorption_count,
        line_names=tuple(line_names),
        line_wavelengths=np.asarray(wavelengths, dtype=float),
        truncated_fraction=truncated,
        n_resampled=n_resampled,
    )
