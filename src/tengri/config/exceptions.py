# SPDX-License-Identifier: BSD-3-Clause
"""Exception hierarchy for Tengri (Naming Contract §8).

All Tengri exceptions inherit from ``TengriError``.  Domain-specific
exceptions also inherit from the closest stdlib type so that existing
``except ValueError`` / ``except OSError`` handlers still work.

Also hosts :func:`warn_measured` / :func:`measurements_of`, the mechanism that
lets a warning carry the exact quantity it reports instead of only rendering a
rounded copy of it into prose (#1645).
"""

import warnings


class TengriError(Exception):
    """Base exception for all Tengri errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class ParameterError(TengriError, ValueError):
    """Invalid parameter names, values, or conflicts.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class ParameterMapError(ParameterError):
    """Parameter map validation failure during SEDModel construction.

    Raised when:

    - A free parameter in spec has no entry in the parameter map
    - Multiple components claim conflicting (scale, offset) for the same parameter
    - Other parameter map consistency violations

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class UnknownParameterError(ParameterError):
    """Unknown parameter name passed to a ``predict_*`` method.

    Raised by :meth:`SEDModel.predict_photometry`, ``predict_spectrum``,
    ``predict_rest_sed``, ``predict_emission_lines``, etc., when the user
    passes an override key that doesn't match any free or fixed parameter
    of the assembled model (typo, deleted param, or param from a component
    the spec doesn't use).

    Silent no-ops on overrides are the worst class of bug — they make
    plausibly-correct downstream plots/fits encode stale defaults. We raise
    instead, and include a "did you mean…" suggestion built from the live
    param-map.

    Parameters
    ----------
    message : str
        Human-readable error description, listing unknown keys and
        suggested matches.
    """


class MissingParameterError(ParameterError):
    """A free parameter was given no value in a ``predict_*`` call.

    Every non-``Fixed`` parameter needs a value at predict time. Without
    this check the missing key surfaces deep inside a component as a bare
    ``KeyError`` (e.g. ``'dust_tau_bc'``) with no hint about the cause —
    commonly hit by ``model.mock({})`` / ``model.predict_photometry({})``
    on a model whose default dust group carries free optical depths.

    Parameters
    ----------
    message : str
        Human-readable error description listing the missing names and how
        to supply or fix them.
    """


class ParameterDefaultMissingError(ParameterError):
    """A parameter was marked FIXED without a physically-motivated default.

    Raised by :func:`tengri.parameters.groups.parse_groups` when the
    ``'all_params': FIXED`` wildcard (or an explicit short-form ``FIXED``) is applied
    to a registry entry whose ``Distribution`` carries no ``default=``. Prior
    behavior silently fell back to the midpoint of the prior support, which
    was an implicit and often physically wrong choice (e.g. ``Uniform(0, 5)``
    for ``log10(n_H/cm^-3)`` collapsed to 2.5 — 316 cm^-3 — when the
    CIGALE-faithful value is 2.0).

    Fix: register a default at the declaration site, e.g.
    ``Uniform(0, 5, default=2.0)``.

    Parameters
    ----------
    message : str
        Human-readable error description naming the offending parameter and
        suggesting where to add the default.
    """


class ConfigError(TengriError, ValueError):
    """Invalid Config construction or missing fields.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class BackendError(TengriError, RuntimeError):
    """Backend initialization or computation failure.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class InferenceError(TengriError, RuntimeError):
    """Sampler/optimizer failures (convergence, NaN, etc.).

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class TengriIOError(TengriError, OSError):
    """File I/O, missing data files, format mismatch.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """


class AGNDustDoubleCountWarning(UserWarning):
    """Composable AGN and Dale2014 ``dust_frac_agn`` both inject AGN IR.

    The composable AGN (``agn_ir_frac > 0``) and Dale2014's embedded quasar
    template (``dust_frac_agn > 0``) are two distinct AGN surfaces, both keyed
    off the same stellar ``L_absorbed``. Using both with positive values
    double-counts AGN mid/far-IR (ADR-0018 §5, issue #721). Pick one surface;
    filter this category if the overlap is deliberate.
    """


class WildcardPartialFreeWarning(UserWarning):
    """``all_params: FREE`` freed only some of the parameters it covered.

    ``FREE`` resolves each parameter to its declared ``free_prior``. A parameter
    with no ``free_prior`` falls back to its ``prior`` — a ``Fixed`` scalar — and
    stays pinned. When a group holds both kinds, the wildcard frees one subset
    and silently leaves the rest frozen, and the fit reports a posterior with
    that physics held constant (issue #1474).

    A partial result is not always a defect: ``dust_Rv`` is fixed by definition
    for a Calzetti law, and ``dust_delta`` applies only to the Noll-modified
    variant. Whether a parameter *should* be freeable is a per-parameter physics
    question — so this warns rather than raising. Filter this category when the
    partial free is deliberate.

    See Also
    --------
    tengri.config.exceptions.ParameterError
        Raised instead when the wildcard frees *nothing*, which is never
        intended.
    """


class DeadGradientParameterWarning(UserWarning):
    """A freed parameter whose gradient is identically zero (#1206).

    The parameter reaches the SED only through a kernel whose derivative rule
    does not differentiate it, so every gradient-based backend (MAP, NUTS, VI)
    sees exactly ``0.0``. The sampler never moves it, and the reported posterior
    is the prior — which is indistinguishable, in the output, from a parameter
    the data genuinely failed to constrain.

    That ambiguity is the reason this warns at build time rather than being left
    to inspection: "the posterior equals the prior" is a result a user can
    reasonably expect to see, so nothing downstream can flag it for them.

    Warns rather than raises: pinning the parameter is a legitimate
    configuration, the forward model is correct either way, and a gradient-free
    sampler can still fit it. Filter this category if you are sampling without
    gradients.

    See Also
    --------
    tengri.components.agn.component.Float32UnsafeAGNWarning
        The same discipline for a block that is numerically unsafe rather than
        gradient-dead.
    """


class CorruptEnergyBalanceWarning(UserWarning):
    """The dust energy-balance integrand was non-finite (#1527).

    ``L_absorbed`` came back ``+inf`` because the intrinsic or attenuated SED
    reaching the energy balance contained ``Inf`` or ``NaN`` — most often the
    ``Inf * 0`` product of an extreme-metallicity SSP flux (the BUG-NSS-02
    artifact class), or an attenuation curve whose far-UV extrapolation
    amplified without bound.

    The ``+inf`` is deliberate and replaces a silent ``0.0``: before #1527 a
    single corrupt pixel made the whole IR budget vanish and the model emitted
    a plausible dust-free galaxy, which is a wrong answer that looks like a
    right one. ``+inf`` instead propagates to ``L_ir`` and shows up as a NaN
    fit — loud, but on its own it says nothing about *where* the corruption
    entered, which is what this warning supplies.

    Fires only on the eager forward path. Under ``jit``/``grad``/``vmap`` there
    is no concrete value to inspect, so nothing is emitted and the ``+inf``
    travels on its own — exploring corrupt draws during inference is expected
    and warning per-sample would be unusable.

    See Also
    --------
    tengri.forward.energy_balance.bolometric_absorbed_log10
        The strict producer that emits the ``+inf``.
    tengri.forward.energy_balance.bolometric_absorbed
        The linear form, which deliberately keeps the old clamp.
    """


class AdvisoryWarning(UserWarning):
    """Base for construction-time advisories about a model the user is building.

    An advisory says "this model will run, but probably not do what you meant".
    That is only meaningful for a model someone intends to *fit*, so the
    introspection paths that build a throwaway ``Parameters`` purely to
    enumerate names (``recipe_parameters`` and the builder factory discovery it
    feeds) silence this category wholesale. Inheriting from it is what keeps a
    new advisory from firing on ``import tengri``.

    Subclass this rather than :class:`UserWarning` for any new
    construction-time advisory. Existing ``UserWarning`` filters keep matching.
    """


class GridSupportWarning(AdvisoryWarning):
    """A parameter's reachable range overhangs the grid that consumes it.

    A template-backed component interpolates over grid axes and clips values
    onto the edge node. Outside the axes ``jnp.clip`` is flat, so the SED is
    bit-identical and the gradient is *exactly* zero: a fit gets no signal and
    cannot move the parameter, with nothing raised, warned or NaN (issue
    #1586). The declared prior records one support; the grid is a second,
    implicit one that no declaration can express, because the same parameter is
    often shared with grid-free analytic models that legitimately want the
    wider range.

    This warns rather than raising: overhanging a grid is wasteful but not
    ill-posed, and a fit whose posterior stays inside the grid is unaffected.
    Narrow the parameter to the quoted extent, or select a component with no
    template grid. Filter this category if the overhang is deliberate.

    See Also
    --------
    tengri.components.grid_support
        The registry of ``(component, parameter)`` grid extents, and why the
        constraint cannot live on the parameter declaration.
    """


class OutOfSSPGridWarning(UserWarning):
    """An ingested metallicity history reaches past the SSP's metallicity grid.

    The metallicity lookup ``jnp.clip``s onto ``ssp_lgmet``, so a node outside
    the grid returns the edge template: ``logzsol = -6`` was measured to give
    byte-identical photometry to the grid edge at ``-2.152`` (issue #1677). The
    SED stays smooth and plausible, which is what makes it hard to notice.

    :func:`~tengri.inference.history_ingest.ingest_histories` raises on this by
    default; this category is what ``on_out_of_grid='warn'`` emits instead. It is
    deliberately **not** an :class:`AdvisoryWarning`: advisories describe a model
    being constructed and are silenced wholesale by the introspection paths,
    whereas this describes *data* arriving at ingest and must survive them.

    The payload carries ``mass_fraction_outside`` — the share of stellar mass
    formed on clamped nodes — because node counts alone do not say whether the
    clamp matters. Read it back with
    :func:`~tengri.config.exceptions.measurements_of`.
    """


class MetallicityUnitWarning(UserWarning):
    """A metallicity history looks like a metal mass fraction read as log10(Z/Zsun).

    ``met_unit=`` exists so a caller can declare which convention a history
    arrives in, but declaring it is optional and the default is ``'logzsol'``.
    That default cannot be checked against the SSP grid, because a mass fraction
    is a small *positive* number and small positive log10(Z/Zsun) values are
    legal — ``Z = 0.011`` read as log10(Z/Zsun) is 1.03 :math:`Z_\\odot`, well
    inside every grid. The out-of-range guard is structurally unable to see it;
    the result is a plausible all-solar SED (issue #1677).

    What separates the two is dynamic range, not magnitude. Chemical enrichment
    moves Z(t) by orders of magnitude across cosmic time, so a history whose
    every node sits inside a ~0.1 dex band immediately above solar is a mass
    fraction essentially every time.

    The test runs on the **converted** values, so a history correctly declared
    ``met_unit='z_mass_fraction'`` lands near :math:`-2 \\ldots 0` and never
    trips it. Only the ambiguous case is reachable.
    """


class GasStellarMetallicityWarning(UserWarning):
    """Enriched stars sitting in gas that never enriched with them.

    Stellar metallicity (which SSP templates the population is drawn from) and
    gas-phase metallicity (which drives nebular emission) are separate
    parameters, and correctly so — inflow of pristine gas genuinely decouples
    them. But ``neb_logZ_gas`` has a *declared default* of -0.3, and the build
    grammar always supplies it, so the ``if neb_logZ_gas is None: neb_logZ_gas
    = log_z`` inheritance inside the CLOUDY / Cue / MAPPINGS backends never
    runs. Measured: leaving it unset is bit-identical to setting -0.3 (#1677).

    A tabulated stellar history therefore enriches the stars while the gas stays
    pinned at 0.5 :math:`Z_\\odot`, silently. This warns only when the value was
    left at its declaration and the caller passed neither ``met_gas=`` nor a
    ``neb_logZ_gas`` column — a deliberate offset never trips it.
    """


class PrecompBiasWarning(AdvisoryWarning):
    """The precompute LUT's forward bias, amplified by this fit's SNR, is material.

    ``WavePrecomp`` / ``SpectrumPrecomp`` carry a small forward bias (measured
    0.13-0.26 % on photometry) that is constant in SNR — so no forward-model
    check can see it — but enters the posterior gradient as ``bias x SNR``:
    ~5 % wrong at SNR 30, ~50 % at SNR 300, rotated as well as rescaled at the
    high end (issue #1671). It is a bias, not noise: it does not average out
    over MCMC draws, it moves the mode, and better data makes it worse. #1688
    measured the spectroscopy sibling as a ~1-sigma posterior shift on a
    50-pixel, 5 %-noise fixture. Since #1641 the LUT is the resolved default
    for every fit, so this is the default gradient, not an opt-in trade.

    This warning is the measurement made operational: at fit construction one
    exact-vs-LUT forward call prices the bias on the actual model, and the
    estimate ``max_i(bias_i x SNR_i)`` above threshold warns with the number.
    Warns rather than raising: the LUT posterior is still useful, and the
    speedup (measured 7-10x) is the reason it is the default. For final
    inference at high SNR, rerun with ``approx=None`` (the exact path) or
    compare the two posteriors. Filter this category if the trade is
    deliberate.
    """


class ComponentDataNotAvailableWarning(AdvisoryWarning):
    """A component declared outputs but load() returned None and
    requires_template_data=True.

    The component's precompute() method was called and returned a state,
    but load() returned None, meaning no cached template data was prepared.
    The component advertises outputs that should carry the loaded data, so the
    absence suggests either:

    1. A required data file (template grid, precomputed table) is missing
       from the file system.
    2. The component's requires_template_data flag is misconfigured — it
       should be False for closed-form analytic models that intentionally
       load nothing.

    Remedy: check that required data files are present, or update the
    component's requires_template_data = False if the no-op is intentional
    (see issue #1738).

    Warns rather than raises: optional data file absences are often legitimate
    and should not break a fit.
    """


class DeadPrecomputeAxisWarning(AdvisoryWarning):
    """A precompute module's ``AXIS_PARAMS`` names no parameter that can exist.

    Every ``*_precompute.py`` module declares ``AXIS_PARAMS``: the parameter
    name governing each grid axis, in axis order. ``collapse_fixed_axes`` matches
    those names against ``Parameters.get_fixed_values()`` and collapses the axes
    whose parameter is ``Fixed``, which is what the modules' docstrings advertise
    ("auto-collapses Fixed axes", "to avoid memory explosion ... this function
    auto-detects and collapses them").

    The two sides are plain strings that nothing forces to agree. When a
    component's declared parameter names and its precompute module's
    ``AXIS_PARAMS`` drift apart, ``pname in fixed_values`` is simply always
    ``False``: no axis is ever collapsed, no error is raised, and the promised
    grid reduction silently never happens. ``cat3d_precompute`` declares
    ``cat3d_cos_inc`` while ``Cat3DTorus`` declares ``parameter_prefix = "agn_"``
    and ``cos_inc``, so the live name is ``agn_cos_inc`` (issue #1738).

    This fires only when *no* declared axis name is a valid parameter for the
    model being built — a name set that cannot resolve under any assignment,
    which is a declaration defect rather than a configuration. A model that
    simply leaves every axis parameter free resolves its names fine and is
    silent.

    Warns rather than raising: an uncollapsed grid is larger and slower to
    interpolate but still numerically correct, so this degrades performance
    rather than results. Remedy: align ``AXIS_PARAMS`` with the names the
    component actually declares (``spec.valid_param_names``).
    """


class LaplaceVarianceCeilingWarning(UserWarning):
    """Eigenvalue clipping assigned a variance ceiling to unconstrained directions.

    ``regularize=True`` floors the Hessian spectrum at ``min_eigenvalue`` to
    force positive-definiteness, then takes ``cov = H^-1``. Because the
    covariance is the *inverse*, the floor does not damp the clipped
    directions — it **assigns** each of them variance ``1 / min_eigenvalue``.
    At the ``1e-6`` default that is a variance of ``1e6``, i.e. a standard
    deviation of 1000 in the unconstrained parameterization (issue #1515).

    So the directions the data determine *least* well come back with the
    *widest* draws, and the number is an artifact of the floor rather than a
    measurement. The fit still returns a full sample set with finite marginals
    and nothing fails.

    A clipped direction is usually one of three things, and the remedy differs:

    - **an exact degeneracy** — two parameters that enter the model only in
      one combination, so the likelihood is flat along a ridge (``met_alpha_fe``
      and ``met_logzsol`` are exactly this, see
      :class:`DegenerateParameterPairWarning` and issue #1095). Fix the model,
      not the floor: hold one of the pair fixed.
    - **a genuinely unconstrained parameter** — the data carry no information
      about it. The prior, not ``1 / min_eigenvalue``, is the honest answer;
      consider fixing it or reporting it as prior-dominated.
    - **numerical noise** in a finite-difference Hessian near a flat direction,
      in which case a tighter ``eps`` or an analytic Hessian is the fix.

    Reported as the count of clipped directions, the implied standard
    deviation, and the smallest unclipped eigenvalue, so the severity is
    visible rather than inferred. Filter this category when the ceiling is
    deliberate.
    """


class LaplaceNotAtModeWarning(UserWarning):
    """The Laplace expansion point is not a stationary point of the loss.

    ``cov = H^-1`` is a covariance only at a mode. Away from one the Hessian
    describes the curvature of a *slope*: still symmetric, still positive
    definite, still invertible, so the fit returns a full sample set with
    plausible marginals and nothing fails (issue #1537).

    ``run_map`` runs a fixed number of Adam steps with no convergence test, so
    an under-converged expansion point is the ordinary way this happens. The
    resulting posterior is typically far too *narrow*, because a point on a
    steep slope has much higher curvature than the mode it is sliding toward.

    No between-chain statistic can catch it: Laplace draws are i.i.d. from the
    fitted Gaussian, so R-hat is ~1 and the divergence count is 0 however wrong
    the Gaussian is. Only the shape is wrong.

    Measured severity is reported as the Newton decrement
    ``d = 0.5 g^T H^-1 g`` [nats], the loss drop a quadratic model predicts
    between the expansion point and the true mode. An offset of ``delta``
    standard deviations along one direction gives ``d = delta^2 / 2``.

    Raise ``n_map_steps``, or pass an already-converged ``init_from``. Filter
    this category when an off-mode expansion is deliberate.

    See Also
    --------
    tengri.inference.backends.laplace.run_laplace
        Emits this warning; reports ``newton_decrement`` in ``diagnostics``.
    """


#: Attribute names a measurement may not use: they belong to ``BaseException``
#: or to this mechanism itself, and shadowing them breaks ``str(w)``, pickling,
#: or the uniform accessor.
_RESERVED_MEASUREMENT_NAMES = frozenset({"args", "with_traceback", "add_note", "measurements"})


def warn_measured(message, category=UserWarning, *, stacklevel=2, **measurements):
    """Emit a warning that carries the quantities it reports (#1645).

    A warn site that renders a computed number into prose and discards the value
    forces consumers to regex-parse the message, and to accept whatever the
    format spec rounded it to: ``{frac:.0%}`` turns 0.6916830115613221 into
    "69%", which is anything in 0.685-0.695. The message keeps its rounding for
    humans; the exact values ride on the instance for code.

    Parameters
    ----------
    message : str
        Human-readable text, formatted as usual. Round freely here.
    category : type, optional
        Warning class. Default ``UserWarning``. Unchanged by this helper, so
        existing ``warnings.filterwarnings`` entries keep working.
    stacklevel : int, optional
        Frames to skip, counted from the *caller* exactly as ``warnings.warn``
        counts them. Default 2. This helper's own frame is added internally, so
        a site migrating from ``warnings.warn(..., stacklevel=2)`` keeps the
        same number and the same attribution.
    **measurements
        Named numeric quantities the message reports. Each is attached as an
        attribute and collected into ``measurements``.

    Raises
    ------
    ValueError
        If a measurement name is reserved.
    TypeError
        If a measurement is not numeric. Prose belongs in ``message``.

    Notes
    -----
    Read the values back with :func:`measurements_of`, which returns ``{}`` for
    warnings that predate this mechanism, so a consumer never needs to know
    whether a given site has been migrated.

    ``tools/check_warning_payloads.py`` fails when a warn site renders a rounded
    number without carrying it.

    Examples
    --------
    >>> import warnings
    >>> from tengri.config.exceptions import measurements_of, warn_measured
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_measured(f"lost {0.6917:.0%} of the mass", UserWarning, truncated_fraction=0.6917)
    >>> measurements_of(caught[0].message)
    {'truncated_fraction': 0.6917}
    """
    bad_names = sorted(set(measurements) & _RESERVED_MEASUREMENT_NAMES)
    if bad_names:
        raise ValueError(
            f"Measurement name(s) {bad_names} are reserved by BaseException or by "
            f"warn_measured itself; shadowing them breaks str(warning), pickling, or "
            f"measurements_of(). Rename the measurement."
        )
    for name, value in measurements.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Measurement {name!r} = {value!r} is not numeric. This payload exists "
                f"so a consumer can compare or threshold the value; descriptive text "
                f"belongs in the message, which already carries it."
            )

    warning = category(message)
    values = {name: float(value) for name, value in measurements.items()}
    warning.measurements = values
    for name, value in values.items():
        setattr(warning, name, value)
    # +1 for this frame, so the report blames whoever called us -- matching what
    # a bare warnings.warn(..., stacklevel=N) would have done at the same site.
    warnings.warn(warning, stacklevel=stacklevel + 1)


def measurements_of(warning):
    """Exact quantities carried by a warning, or ``{}``.

    Safe on any object, so a consumer never needs to know whether a particular
    warn site has been migrated to :func:`warn_measured`.

    Parameters
    ----------
    warning : Warning or object
        Typically ``record.message`` from
        ``warnings.catch_warnings(record=True)``.

    Returns
    -------
    values : dict
        Mapping of name to exact value [units vary by measurement]. Empty for
        warnings that carry none.
    """
    if not isinstance(warning, Warning):
        return {}
    return dict(getattr(warning, "measurements", {}) or {})
