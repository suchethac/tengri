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


def test_catalog_span_covers_every_redshift_and_is_quantized_outward():
    """The span is [min z, max z] padded by max(1% of span, 1e-3), clipped at 0, on a 0.05 grid."""
    from tengri.inference.fitter import catalog_redshift_span

    lo, hi = catalog_redshift_span([0.12, 0.31, 0.58])
    assert (lo, hi) == pytest.approx((0.10, 0.60))
    # A tiny span is padded by the floor, then rounded outward.
    lo, hi = catalog_redshift_span([0.2, 0.2001])
    assert (lo, hi) == pytest.approx((0.15, 0.25))
    # The lower edge never goes below zero.
    assert catalog_redshift_span([0.0005, 0.001])[0] == 0.0
    # A redshift on a grid line is still covered with a pad.
    lo, hi = catalog_redshift_span([0.3])
    assert lo < 0.3 < hi
    # Similar catalogs share one span, which is what lets them share one compile.
    assert catalog_redshift_span([0.11, 0.33, 0.57]) == catalog_redshift_span([0.12, 0.31, 0.58])


def test_non_finite_redshift_is_refused_naming_the_rows():
    """A NaN or inf redshift has no place in a table span; the error names the rows."""
    from tengri.inference.fitter import catalog_redshift_span

    with pytest.raises(ValueError, match=r"row\(s\) \[1, 3\]"):
        catalog_redshift_span([0.1, float("nan"), 0.3, float("inf")])


def test_promotion_is_cached_per_model_and_span_and_leaves_the_user_model(
    synthetic_ssp, simple_observation
):
    """Similar catalogs get the same promoted object; the caller's model is not mutated."""
    from tengri import Fixed, SEDModel
    from tengri.inference.fitter import promote_to_runtime_redshift

    user = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl"},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    first = promote_to_runtime_redshift(user, [0.06, 0.2, 0.41])
    again = promote_to_runtime_redshift(user, [0.07, 0.21, 0.42])
    assert first is again
    assert first is not user
    assert user._catalog_z_range is None, "the caller's model was mutated"
    lo, hi = first._catalog_z_range
    assert lo <= 0.06 and hi >= 0.41
    # A declared range is used as it is.
    assert promote_to_runtime_redshift(first, [0.06, 0.2, 0.41]) is first

    exact = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl"},
        redshift=Fixed(0.1),
        approx=None,
    )
    promoted = promote_to_runtime_redshift(exact, [0.05, 0.2, 0.4])
    assert promoted._catalog_z_range is None and promoted._z_runtime
    assert not promoted._approx["wave_precomp"], "an exact model stays exact"
