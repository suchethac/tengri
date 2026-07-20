# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for NIFTy variational inference (geoVI/MGVI/EVI).

Default values follow Philipp Frank's recommendations (2025 discussion).
EVI = Expansion-point Variational Inference: runs MGVI (linear) for
early iterations then switches to geoVI (nonlinear) for accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Philipp Frank's recommended kwargs for jft.optimize_kl
_DEFAULT_DRAW_LINEAR_KWARGS = {
    "cg_name": "SL",
    "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30},
}

_DEFAULT_NONLINEARLY_UPDATE_KWARGS = {
    "minimize_kwargs": {
        "name": "SN",
        "xtol": 1e-3,
        "cg_kwargs": {"name": None},
        "maxiter": 3,
    },
}

_DEFAULT_KL_KWARGS = {
    "minimize_kwargs": {
        "name": "M",
        "absdelta": 1e-3,
        "cg_kwargs": {"name": "MCG"},
        "maxiter": 10,
    },
}


@dataclass(frozen=True)
class VIConfig:
    """Configuration for geoVI/MGVI/EVI optimize_kl calls.

    Parameters
    ----------
    n_samples : int or callable
        Samples per KL iteration. ``mirror_samples=True`` (default in NIFTy)
        doubles this internally, so 3 → 6 effective samples.
    n_iterations : int
        Number of KL minimization iterations.
    use_vmap : bool
        Use ``jax.vmap`` for ``residual_map`` (faster, slightly more memory).
    evi_linear_fraction : float
        Fraction of iterations using ``linear_resample`` before switching to
        ``nonlinear_resample`` in EVI mode.
    draw_linear_kwargs : dict
        Kwargs for the CG solver that generates each sample.
    nonlinearly_update_kwargs : dict
        Kwargs for the Newton-CG that inverts the coordinate transform.
    kl_kwargs : dict
        Kwargs for the outer KL minimization.

    Attributes
    ----------
    n_samples : int or callable
        Samples per KL iteration (doubled by NIFTy's ``mirror_samples``).
    n_iterations : int
        Number of KL minimization iterations.
    use_vmap : bool
        Whether to use ``jax.vmap`` for residual mapping.
    evi_linear_fraction : float
        Fraction of iterations to use linear (MGVI) before nonlinear (geoVI).
    draw_linear_kwargs : dict
        Conjugate gradient solver configuration for sampling.
    nonlinearly_update_kwargs : dict
        Newton-CG configuration for coordinate transform inversion.
    kl_kwargs : dict
        Outer KL minimization configuration.

    Notes
    -----
    Frozen dataclass configuring the variational inference backend. Key fields:
    ``method`` selects the VI algorithm. ``'vi'`` (NIFTy geoVI) and
    ``'vi_linear'`` (NIFTy MGVI) are the working paths.

    The pure-JAX ``native_vi_nonlinear`` / ``native_vi_linear`` backends are
    registered ``tier="broken"`` (#1287) — they segfault on DPL/dense_basis
    photometry mocks — and ``'vi_native'`` was never a registered name at all;
    it raises ``KeyError``. The "~19x faster" figure once quoted here was
    withdrawn: it compared MGVI against geoVI, not native against NIFTy.

    Examples
    --------
    >>> from tengri import VIConfig
    >>> cfg = VIConfig(n_samples=4, n_iterations=50)
    >>> cfg.n_samples
    4
    """

    n_samples: int | Callable = 3
    n_iterations: int = 10
    use_vmap: bool = True
    evi_linear_fraction: float = 0.5
    draw_linear_kwargs: dict = field(
        default_factory=lambda: {
            "cg_name": "SL",
            "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30},
        }
    )
    nonlinearly_update_kwargs: dict = field(
        default_factory=lambda: {
            "minimize_kwargs": {
                "name": "SN",
                "xtol": 1e-3,
                "cg_kwargs": {"name": None},
                "maxiter": 3,
            },
        }
    )
    kl_kwargs: dict = field(
        default_factory=lambda: {
            "minimize_kwargs": {
                "name": "M",
                "absdelta": 1e-3,
                "cg_kwargs": {"name": "MCG"},
                "maxiter": 10,
            },
        }
    )


@dataclass(frozen=True)
class BlockStep:
    """One step in a block Gibbs schedule.

    Parameters
    ----------
    sample_mode : str
        ``"linear_resample"``, ``"nonlinear_resample"``, or
        ``"nonlinear_update"``.
    constants : tuple of str
        Parameter names frozen during KL minimization (their mean
        doesn't move, but they are still sampled for uncertainty
        propagation).
    point_estimates : tuple of str
        Parameter names excluded from sampling (residual zeroed).
        Faster than ``constants`` but ignores their uncertainty.
    n_samples : int or None
        Override the default n_samples for this block. ``None`` uses
        the fitter's n_samples.

    Notes
    -----
    Immutable step descriptor for one block in a multi-block VI schedule.
    Consumed by :class:`BlockSchedule`.
    """

    sample_mode: str = "nonlinear_resample"
    constants: tuple[str, ...] = ()
    point_estimates: tuple[str, ...] = ()
    n_samples: int | None = None


@dataclass(frozen=True)
class BlockSchedule:
    """Block Gibbs schedule for structured variational inference.

    Each optimization iteration cycles through all blocks in order.
    Different blocks can use different sample modes, freeze different
    parameters, and draw different numbers of samples.

    Parameters
    ----------
    blocks : tuple of BlockStep
        The blocks to cycle through per iteration.

    Notes
    -----
    Frozen sequence of :class:`BlockStep` objects controlling parameter-group
    update ordering. Enables block coordinate descent with different optimizers
    per group.
    """

    blocks: tuple[BlockStep, ...]

    @staticmethod
    def individual_geovi() -> BlockSchedule:
        """Default schedule for individual galaxy geoVI.

        Block A: Update physical params (geoVI), SFH ξ frozen.
        Block B: Update SFH ξ (MGVI), physical params frozen.
        Alternates A-B every iteration.

        Returns
        -------
        BlockSchedule
            Two-block schedule for alternating physical and SFH parameter updates.
        """
        return BlockSchedule(
            blocks=(
                BlockStep(
                    sample_mode="nonlinear_resample",
                    constants=("sfh_field_xi",),
                ),
                BlockStep(
                    sample_mode="linear_resample",
                    constants=(),  # joint update for cross-correlations
                ),
            )
        )

    @staticmethod
    def hierarchical() -> BlockSchedule:
        """Default schedule for hierarchical PSD inference.

        Block 1: Shared PSD params (geoVI), per-galaxy frozen.
        Block 2: Per-galaxy physical (geoVI), shared+ξ frozen.
        Block 3: Per-galaxy ξ (MGVI), shared+physical frozen.

        Returns
        -------
        BlockSchedule
            Three-block schedule for hierarchical parameter inference.

        Examples
        --------
        >>> from tengri.inference.vi_config import BlockSchedule
        >>> sched = BlockSchedule.hierarchical()
        >>> len(sched.blocks)
        3
        """
        return BlockSchedule(
            blocks=(
                BlockStep(
                    sample_mode="nonlinear_resample",
                    point_estimates=("gal",),
                    n_samples=6,
                ),
                BlockStep(
                    sample_mode="nonlinear_resample",
                    constants=("psd_sigma_u", "psd_tau_u"),
                    n_samples=3,
                ),
                BlockStep(
                    sample_mode="linear_resample",
                    constants=("psd_sigma_u", "psd_tau_u"),
                    n_samples=2,
                ),
            )
        )


@dataclass(frozen=True)
class OptimizationSchedule:
    """Unified optimization schedule for variational inference.

    Controls what happens at each iteration: which sample mode, which
    parameters to update, how many samples.  Passed to ``fitter.run()``
    via the ``schedule`` kwarg.

    The schedule is a callable ``f(iteration: int) -> BlockStep`` that
    returns the configuration for each iteration.  Factory methods
    provide common strategies.

    Parameters
    ----------
    get_step : callable
        ``get_step(i: int) -> BlockStep`` returning the configuration
        for iteration ``i``.
    n_iterations : int
        Total number of iterations.
    description : str
        Human-readable description of the strategy.

    Notes
    -----
    Frozen schedule of ``(n_steps, lr)`` tuples for multi-phase learning rate
    annealing. Used internally by the VI optimizer.

    Examples
    --------
    >>> sched = OptimizationSchedule.geovi()  # recommended default
    >>> sched = OptimizationSchedule.evi(n_iterations=20, transition=10)
    >>> sched = OptimizationSchedule.custom(lambda i: ...)
    """

    get_step: Callable[[int], BlockStep]
    n_iterations: int = 15
    description: str = ""

    def __call__(self, i: int) -> BlockStep:
        """Get BlockStep configuration for iteration i."""
        return self.get_step(i)

    def sample_mode_at(self, i: int) -> str:
        """NIFTy-compatible sample_mode callable for ``jft.optimize_kl``.

        Parameters
        ----------
        i : int
            Iteration index.

        Returns
        -------
        str
            Sample mode at iteration ``i``.
        """
        return self.get_step(i).sample_mode

    @staticmethod
    def geovi(
        n_iterations: int = 15,
        resample_every: int = 5,
        n_samples: int = 3,
    ) -> OptimizationSchedule:
        """Recommended geoVI schedule.

        Iteration 0: ``nonlinear_resample`` (establish samples).
        Every ``resample_every`` iterations: ``nonlinear_resample`` (refresh).
        All other iterations: ``nonlinear_update`` (deterministic refinement).

        This prevents sample staleness while maintaining stable convergence.

        Parameters
        ----------
        n_iterations : int
            Total number of iterations. Default: 15.
        resample_every : int
            Resample every this many iterations. Default: 5.
        n_samples : int
            Number of samples per iteration. Default: 3.

        Returns
        -------
        OptimizationSchedule
            Configured geoVI schedule.
        """

        def _get_step(i: int) -> BlockStep:
            """Resample at iteration 0 and every resample_every, update otherwise."""
            if i == 0 or i % resample_every == 0:
                return BlockStep(sample_mode="nonlinear_resample", n_samples=n_samples)
            return BlockStep(sample_mode="nonlinear_update", n_samples=n_samples)

        return OptimizationSchedule(
            get_step=_get_step,
            n_iterations=n_iterations,
            description=(
                f"geoVI: resample at iter 0 then every {resample_every}, "
                f"update between ({n_samples} samples)"
            ),
        )

    @staticmethod
    def evi(
        n_iterations: int = 20,
        transition: int = 10,
        resample_every: int = 5,
        n_samples: int = 3,
    ) -> OptimizationSchedule:
        """EVI schedule: MGVI first, then geoVI.

        Iterations 0..transition-1: ``linear_resample`` (cheap MGVI).
        Iteration transition: ``nonlinear_resample`` (establish geoVI samples).
        Every ``resample_every`` after: ``nonlinear_resample`` (refresh).
        All other iterations: ``nonlinear_update`` (deterministic).

        Parameters
        ----------
        n_iterations : int
            Total number of iterations. Default: 20.
        transition : int
            Iteration at which to switch from MGVI to geoVI. Default: 10.
        resample_every : int
            Resample every this many iterations after transition. Default: 5.
        n_samples : int
            Number of samples per iteration. Default: 3.

        Returns
        -------
        OptimizationSchedule
            Configured EVI schedule.
        """

        def _get_step(i: int) -> BlockStep:
            """MGVI before transition, then geoVI with periodic resampling."""
            if i < transition:
                return BlockStep(sample_mode="linear_resample", n_samples=n_samples)
            if i == transition or (i > transition and (i - transition) % resample_every == 0):
                return BlockStep(sample_mode="nonlinear_resample", n_samples=n_samples)
            return BlockStep(sample_mode="nonlinear_update", n_samples=n_samples)

        return OptimizationSchedule(
            get_step=_get_step,
            n_iterations=n_iterations,
            description=(
                f"EVI: linear_resample for {transition} iters, then "
                f"nonlinear resample+update ({n_samples} samples)"
            ),
        )

    @staticmethod
    def mgvi(n_iterations: int = 15, n_samples: int = 3) -> OptimizationSchedule:
        """Pure MGVI (linear only).

        Parameters
        ----------
        n_iterations : int
            Total number of iterations. Default: 15.
        n_samples : int
            Number of samples per iteration. Default: 3.

        Returns
        -------
        OptimizationSchedule
            Configured MGVI schedule.
        """
        return OptimizationSchedule(
            get_step=lambda i: BlockStep(sample_mode="linear_resample", n_samples=n_samples),
            n_iterations=n_iterations,
            description=f"MGVI: linear_resample ({n_samples} samples)",
        )

    @staticmethod
    def gibbs(
        blocks: tuple[BlockStep, ...],
        n_iterations: int = 15,
        resample_every: int = 5,
    ) -> OptimizationSchedule:
        """Block Gibbs schedule cycling through parameter blocks.

        Each iteration cycles through all blocks in order.
        Blocks with ``nonlinear_resample`` are switched to
        ``nonlinear_update`` except every ``resample_every`` iterations.

        Parameters
        ----------
        blocks : tuple of BlockStep
            Blocks to cycle. Each block specifies which params to
            freeze and which sample mode to use.
        n_iterations : int
            Number of full cycles through all blocks. Default: 15.
        resample_every : int
            Resample every this many full cycles. Default: 5.

        Returns
        -------
        OptimizationSchedule
            Configured block Gibbs schedule.
        """
        n_blocks = len(blocks)

        def _get_step(i: int) -> BlockStep:
            """Cycle through blocks, switching resample to update except at resample points."""
            block_idx = i % n_blocks
            block = blocks[block_idx]
            outer_iter = i // n_blocks
            # Switch nonlinear_resample to nonlinear_update except at
            # resample points
            if (
                block.sample_mode == "nonlinear_resample"
                and outer_iter > 0
                and outer_iter % resample_every != 0
            ):
                return BlockStep(
                    sample_mode="nonlinear_update",
                    constants=block.constants,
                    point_estimates=block.point_estimates,
                    n_samples=block.n_samples,
                )
            return block

        return OptimizationSchedule(
            get_step=_get_step,
            n_iterations=n_iterations * n_blocks,
            description=f"Gibbs: {n_blocks} blocks x {n_iterations} cycles",
        )

    @staticmethod
    def custom(
        get_step: Callable[[int], BlockStep],
        n_iterations: int = 15,
        description: str = "custom",
    ) -> OptimizationSchedule:
        """Fully custom schedule.

        Parameters
        ----------
        get_step : callable
            ``get_step(i: int) -> BlockStep`` returning the configuration
            for iteration ``i``.
        n_iterations : int
            Total number of iterations. Default: 15.
        description : str
            Human-readable description. Default: ``"custom"``.

        Returns
        -------
        OptimizationSchedule
            Configured custom schedule.
        """
        return OptimizationSchedule(
            get_step=get_step,
            n_iterations=n_iterations,
            description=description,
        )


def evi_sample_mode(n_iterations: int, linear_fraction: float = 0.5):
    """Return a callable sample_mode for EVI: MGVI first, then geoVI.

    Early iterations use ``linear_resample`` (cheap, expansion point far from
    optimum). Later iterations use ``nonlinear_resample`` (accurate near the
    converged expansion point).

    Parameters
    ----------
    n_iterations : int
        Total number of KL iterations.
    linear_fraction : float
        Fraction of iterations to run as MGVI (linear_resample).
        Default: 0.5.

    Returns
    -------
    callable
        ``sample_mode(i: int) -> str`` for ``jft.optimize_kl``.
    """
    transition = int(n_iterations * linear_fraction)

    def _mode(i: int) -> str:
        """Return sample mode: linear before transition, nonlinear after."""
        return "linear_resample" if i < transition else "nonlinear_resample"

    return _mode
