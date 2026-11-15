# SPDX-License-Identifier: BSD-3-Clause
"""Catalog-fit `Fixed(redshift)` cross-compile reuse contract.

`WavePrecomp(catalog_z_range=(z_min, z_max))` should let a single
:class:`tengri.SEDModel` instance handle multiple per-galaxy
``Fixed(redshift)`` values **without** recompiling the JIT kernel for
each. The compile signature must collapse the per-row z value so the
structural cache key is invariant under z.

See `docs/dev/cross-compile-fixed-z-design.md` for the design.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.contract

from tengri.forward.sed_model import WavePrecomp


def test_wave_precomp_accepts_catalog_z_range():
    """The dataclass accepts the new knob without breaking the simple case."""
    plain = WavePrecomp()
    assert plain.catalog_z_range is None

    cz = WavePrecomp(catalog_z_range=(0.05, 1.5))
    assert cz.catalog_z_range == (0.05, 1.5)


def test_wave_precomp_catalog_range_is_hashable():
    """The frozen dataclass must work as a JAX static argument / cache key.

    "Hashable" is not the property that matters — the default object hash is
    always available and always self-consistent. What a cache key needs is
    that two *separately constructed* equal policies collide, so a second
    build finds the first one's entry.

    This asserted ``hash(cz) == hash(cz)`` on a single object, which is true
    of any object at all, including one whose ``__hash__`` falls back to
    ``id()`` — precisely the failure that would make every build miss the
    cache and recompile.
    """
    a = WavePrecomp(catalog_z_range=(0.05, 1.5))
    b = WavePrecomp(catalog_z_range=(0.05, 1.5))
    assert a is not b, "the two arms must be distinct objects or this proves nothing"
    assert a == b
    assert hash(a) == hash(b), "equal policies must collide, or the cache never hits"

    # And the key must still separate policies that differ.
    other = WavePrecomp(catalog_z_range=(0.05, 2.5))
    assert a != other
    assert hash(a) != hash(other), "different z ranges share a cache slot"

    # The property as the cache actually uses it.
    cache = {a: "compiled"}
    assert cache.get(b) == "compiled", "an equal policy missed its own cache entry"
    assert cache.get(other) is None, "a different policy hit the wrong cache entry"


def test_catalog_range_distinct_compile_signature_from_no_range():
    """A model built with catalog_z_range must have a different
    :meth:`compile_signature` from one without. Otherwise the two cache
    slots collide and the catalog-fit kernel reuses the fixed-z one
    (or vice versa) — which silently breaks the LUT shape.

    We construct two minimal SEDModels and only assert on the signatures.
    The actual physics paths are exercised elsewhere; here we only pin the
    cache-key invariant.
    """
    from pathlib import Path

    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, load_ssp_data

    ssp_candidates = [
        "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        "data/ssp_prsc_bc03_chabrier.h5",
    ]
    ssp_path = next((p for p in ssp_candidates if Path(p).is_file()), None)
    if ssp_path is None:
        pytest.skip("No SSP grid on disk; skipping signature test.")

    ssp = load_ssp_data(ssp_path)
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    def _build(approx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzetti",
                    "tau_v": Fixed(0.3),
                },
                redshift=Fixed(0.1),
                approx=approx,
            )

    m_plain = _build(WavePrecomp())
    m_catalog = _build(WavePrecomp(catalog_z_range=(0.05, 1.5)))

    assert m_plain.compile_signature() != m_catalog.compile_signature(), (
        "catalog_z_range must not share a cache slot with plain WavePrecomp"
    )


def test_catalog_range_shared_signature_across_fixed_z_values():
    """The headline guarantee: two SEDModels at different ``Fixed(z)`` values
    but the same ``WavePrecomp(catalog_z_range=...)`` share the same
    :meth:`compile_signature` — i.e. **one** JIT compile across the catalog.
    """
    from pathlib import Path

    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, load_ssp_data

    ssp_candidates = [
        "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        "data/ssp_prsc_bc03_chabrier.h5",
    ]
    ssp_path = next((p for p in ssp_candidates if Path(p).is_file()), None)
    if ssp_path is None:
        pytest.skip("No SSP grid on disk; skipping signature test.")

    ssp = load_ssp_data(ssp_path)
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
    cz = WavePrecomp(catalog_z_range=(0.05, 1.5))

    def _build(z):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzetti",
                    "tau_v": Fixed(0.3),
                },
                redshift=Fixed(z),
                approx=cz,
            )

    m_lo = _build(0.1)
    m_hi = _build(1.0)

    assert m_lo.compile_signature() == m_hi.compile_signature(), (
        "two Fixed(z) values under catalog_z_range must share the compile signature"
    )


def test_derived_range_covers_every_redshift_with_a_stated_margin():
    """A catalog with per-galaxy z and no declared range gets one derived from the table.

    The margin is 1% of the span, at least 1e-3, and the lower edge is clipped at z=0.
    """
    from tengri.inference.fitter import derived_catalog_z_range

    lo, hi = derived_catalog_z_range([0.1, 0.5, 0.3])
    assert lo == pytest.approx(0.1 - 0.004)
    assert hi == pytest.approx(0.5 + 0.004)

    # A tiny span is padded by the 1e-3 floor, not by 1% of nothing.
    lo, hi = derived_catalog_z_range([0.2, 0.2001])
    assert lo == pytest.approx(0.2 - 1e-3)
    assert hi == pytest.approx(0.2001 + 1e-3)

    # The lower edge never goes below zero.
    lo, _ = derived_catalog_z_range([0.0005, 0.001])
    assert lo == 0.0


def test_derived_range_is_applied_to_a_copy_and_the_user_model_is_untouched(
    synthetic_ssp, simple_observation
):
    from tengri import Fixed, Parameters, SEDModel
    from tengri.inference.fitter import derived_catalog_z_range, with_derived_catalog_z_range

    spec = Parameters(
        sfh_dpl_log_total_mass=Fixed(9.0),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    user = SEDModel(spec, synthetic_ssp, observation=simple_observation)
    assert user._catalog_z_range is None
    redshifts = [0.05, 0.2, 0.4]

    configured = with_derived_catalog_z_range(user, redshifts)

    assert configured is not user
    assert user._catalog_z_range is None, "the caller's model was mutated"
    assert configured._catalog_z_range == pytest.approx(derived_catalog_z_range(redshifts))
    lo, hi = configured._catalog_z_range
    assert lo <= min(redshifts) and max(redshifts) <= hi
    # A model that already declares its range is returned as it is.
    assert with_derived_catalog_z_range(configured, redshifts) is configured
    # The derived clone has the compile key of a model that declares the same range, so
    # it shares the one compiled program the declared-range path already uses.
    declared = user.with_approx((WavePrecomp(catalog_z_range=configured._catalog_z_range),))
    assert configured.compile_signature() == declared.compile_signature()


def test_exact_model_given_the_tabulated_path_logs_a_warning(
    synthetic_ssp, simple_observation, caplog
):
    """approx=None is exact, so turning it into a LUT must be logged, never silent.

    The exact path cannot take a per-galaxy redshift (its fixed-z tables are built at the
    spec's z), so the tabulated path is used for this model; the WARNING says so.
    """
    import logging

    from tengri import Fixed, Parameters, SEDModel
    from tengri.inference.fitter import with_derived_catalog_z_range

    spec = Parameters(
        sfh_dpl_log_total_mass=Fixed(9.0),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    exact = SEDModel(spec, synthetic_ssp, observation=simple_observation)
    assert exact.approx_configs == ()
    with caplog.at_level(logging.WARNING, logger="tengri.inference.fitter"):
        with_derived_catalog_z_range(exact, [0.05, 0.2])
    assert any("exact model (approx=None)" in r.getMessage() for r in caplog.records)


def test_non_finite_redshift_is_refused_naming_the_rows():
    """A NaN or inf redshift has no place in a range; the error names the rows."""
    from tengri.inference.fitter import derived_catalog_z_range

    with pytest.raises(ValueError, match=r"row\(s\) \[1, 3\]"):
        derived_catalog_z_range([0.1, float("nan"), 0.3, float("inf")])
