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

    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, load_ssp_data

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
                sfh={"type": "dpl", "*": FIXED},
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

    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, load_ssp_data

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
                sfh={"type": "dpl", "*": FIXED},
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
