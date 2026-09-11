# SPDX-License-Identifier: BSD-3-Clause
"""A shock build in unpopulated MAPPINGS V grid territory used to predict an
exactly-zero shock spectrum with no error or warning (#2065).

The MAPPINGS V grid is sparse -- 75 of 210 (density, B) cells populated at
solar abundance, and only 8 of 210 for the other four shipped abundances
(SMC, LMC, Dopita2005, TwiceSolar), whose density coverage collapses to a
single node. Querying outside the populated region does not raise: the ratio
grid is zero-filled at unpopulated cells, so the model compiles and runs, it
just silently contributes zero shock line flux.

``SEDModel._validate_shock_coverage`` (called from ``_init_multiwavelength``
during construction) closes this: it is the honest place to raise, since
``shock_abundance`` is a concrete Python string at build time and the JIT'd
predict path cannot raise on a traced value. Guarded by
:func:`tengri.components.nebular.shock.population_envelope`, a cheap
numpy-only summary of the sparse grid's per-axis coverage.

Data-gated: ``data/mappings_templates.h5`` must be reachable (else the
fallback Allen+2008 Table 5 path is used, which has no sparsity to guard
against and the module skips outright, mirroring ``test_shock.py``'s
``h5_only`` marker).
"""

from __future__ import annotations

import warnings

import pytest

from tengri._data_setup import find_data
from tengri.components.nebular.shock import population_envelope

pytestmark = [
    pytest.mark.regression_bug,
    pytest.mark.skipif(
        find_data("mappings_templates.h5") is None,
        reason="data/mappings_templates.h5 not found; run scripts/download_mappings_templates.py",
    ),
]


def _build(ssp, obs, shock_kwargs, **extra):
    """Model with everything else fixed; ``shock_kwargs`` is the group under test."""
    from tengri import DEFAULT, Fixed, SEDModel

    kwargs = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.0,
            "tau_diff": 0.0,
        },
        redshift=Fixed(0.1),
        shock=shock_kwargs,
    )
    kwargs.update(extra)
    from tengri.components.nebular.baked_in import BakedInNebularWarning
    from tengri.config.exceptions import DefaultFixedParametersWarning

    with warnings.catch_warnings():
        # Noise unrelated to the #2065 guard under test. Filtered by exact
        # category (not a blanket "ignore") so the guard's own UserWarning
        # still reaches an enclosing ``catch_warnings(record=True)``.
        warnings.filterwarnings("ignore", category=DefaultFixedParametersWarning)
        warnings.filterwarnings("ignore", category=BakedInNebularWarning)
        return SEDModel.build(**kwargs)


# ── population_envelope: the numpy-only coverage summary ──────────


def test_population_envelope_matches_measured_solar_range():
    """Solar's every density node has *some* populated B; the marginal envelope
    is therefore the full declared range, unlike the four sparse abundances."""
    envelope = population_envelope("solar", "combined")
    assert envelope is not None
    dens_lo, dens_hi, b_lo, b_hi = envelope
    assert dens_lo == pytest.approx(-2.0)
    assert dens_hi == pytest.approx(3.0)
    assert b_lo == pytest.approx(1e-4)
    assert b_hi == pytest.approx(1000.0)


@pytest.mark.parametrize("abundance", ["lmc", "smc", "dopita2005", "2xsolar"])
def test_population_envelope_collapses_to_one_density_node_for_sparse_abundances(abundance):
    """The four non-solar shipped abundances have data at exactly one density
    node (log_density=0.0): their marginal density envelope is a single point."""
    dens_lo, dens_hi, _b_lo, _b_hi = population_envelope(abundance, "combined")
    assert dens_lo == pytest.approx(0.0)
    assert dens_hi == pytest.approx(0.0)


# ── build-time guard: Fixed values ─────────────────────────────────


def test_default_shock_build_does_not_raise(ssp_data_wne, synthetic_tophat_obs):
    """Setup guard: the declared Fixed defaults (density=0.0, B=1.0, solar)
    sit inside the populated envelope, so an ordinary shock build is unaffected."""
    _build(ssp_data_wne, synthetic_tophat_obs, {"frac": 0.5})


def test_fixed_density_outside_envelope_raises(ssp_data_wne, synthetic_tophat_obs):
    """A Fixed shock_log_density outside a sparse abundance's single-node
    envelope raises ParameterError naming the value and the populated range."""
    from tengri.config.exceptions import ParameterError

    with pytest.raises(ParameterError) as exc_info:
        _build(
            ssp_data_wne,
            synthetic_tophat_obs,
            {"frac": 0.5, "abundance": "lmc", "log_density": 2.9},
        )
    message = str(exc_info.value)
    assert "shock_log_density=2.9" in message
    assert "[0, 0]" in message
    assert "#2065" in message


def test_fixed_density_inside_solar_envelope_does_not_raise(ssp_data_wne, synthetic_tophat_obs):
    """The same value that raises for a sparse abundance is fine for solar,
    whose envelope spans the full declared range."""
    _build(ssp_data_wne, synthetic_tophat_obs, {"frac": 0.5, "log_density": 2.9})


# ── build-time guard: free priors ──────────────────────────────────


def test_free_density_entirely_outside_envelope_raises(ssp_data_wne, synthetic_tophat_obs):
    """A free prior with no overlap with the populated envelope raises: every
    draw would predict an exactly-zero shock spectrum."""
    from tengri import Uniform
    from tengri.config.exceptions import ParameterError

    with pytest.raises(ParameterError, match="#2065"):
        _build(
            ssp_data_wne,
            synthetic_tophat_obs,
            {"frac": 0.5, "log_density": Uniform(10.0, 11.0)},
        )


def test_free_density_degenerate_envelope_containing_point_warns(
    ssp_data_wne, synthetic_tophat_obs
):
    """A sparse abundance's envelope collapses to the single point 0.0 (a
    degenerate interval). A continuous free prior containing that point has
    zero *interval-overlap width* against it, which must not be confused with
    "entirely outside": it warns (100% dead, but reachable), not raises."""
    from tengri import Uniform
    from tengri.config.exceptions import measurements_of

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(
            ssp_data_wne,
            synthetic_tophat_obs,
            {"frac": 0.5, "abundance": "lmc", "log_density": Uniform(-2.0, 3.0)},
        )

    matches = [
        w for w in caught if "shock_log_density" in str(w.message) and "#2065" in str(w.message)
    ]
    assert len(matches) == 1, f"expected exactly one #2065 warning, got {len(matches)}"
    payload = measurements_of(matches[0].message)
    assert payload["dead_fraction"] == pytest.approx(1.0)
    assert payload["envelope_lo"] == pytest.approx(0.0)
    assert payload["envelope_hi"] == pytest.approx(0.0)


def test_free_density_degenerate_envelope_missing_point_raises(ssp_data_wne, synthetic_tophat_obs):
    """The same sparse abundance's degenerate envelope, but the free prior
    does not even contain the one populated value: this must raise."""
    from tengri import Uniform
    from tengri.config.exceptions import ParameterError

    with pytest.raises(ParameterError, match="#2065"):
        _build(
            ssp_data_wne,
            synthetic_tophat_obs,
            {"frac": 0.5, "abundance": "lmc", "log_density": Uniform(1.0, 2.0)},
        )


def test_free_density_partial_overlap_warns_with_measured_dead_fraction(
    ssp_data_wne, synthetic_tophat_obs
):
    """A free prior straddling the envelope boundary warns, carrying the exact
    dead fraction (not just a rounded string) via ``warn_measured`` (#1645)."""
    from tengri import Uniform
    from tengri.config.exceptions import measurements_of

    # Solar envelope is [-2, 3]; Uniform(1, 5) has half its mass (2 of 4 dex)
    # above 3 -- exactly 50% dead.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(ssp_data_wne, synthetic_tophat_obs, {"frac": 0.5, "log_density": Uniform(1.0, 5.0)})

    matches = [
        w for w in caught if "shock_log_density" in str(w.message) and "#2065" in str(w.message)
    ]
    assert len(matches) == 1, f"expected exactly one #2065 warning, got {len(matches)}"
    payload = measurements_of(matches[0].message)
    assert payload["dead_fraction"] == pytest.approx(0.5)
    assert payload["prior_lo"] == pytest.approx(1.0)
    assert payload["prior_hi"] == pytest.approx(5.0)
    assert payload["envelope_lo"] == pytest.approx(-2.0)
    assert payload["envelope_hi"] == pytest.approx(3.0)


def test_free_density_fully_inside_envelope_does_not_warn(ssp_data_wne, synthetic_tophat_obs):
    """A free prior fully contained in the populated envelope is silent on
    this guard (it may still warn for unrelated reasons, e.g. other groups)."""
    from tengri import Uniform

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(
            ssp_data_wne,
            synthetic_tophat_obs,
            {"frac": 0.5, "log_density": Uniform(-1.0, 1.0)},
        )

    matches = [w for w in caught if "#2065" in str(w.message)]
    assert matches == []


# ── smoke: the guard does not fire when the grid is used correctly ─


def test_free_shock_all_params_does_not_raise(ssp_data_wne, synthetic_tophat_obs):
    """``all_params: FREE`` must not trip this guard: whatever ``shock_log_density``
    resolves to under the wildcard (pinned, or freed at its declared interval --
    see ``tests/contract/test_shock_group_free_priors.py`` for that contract),
    it stays inside the populated envelope by construction."""
    from tengri import FREE

    _build(ssp_data_wne, synthetic_tophat_obs, {"norm": "frac", "all_params": FREE})
