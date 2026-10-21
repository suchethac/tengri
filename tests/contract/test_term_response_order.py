# SPDX-License-Identifier: BSD-3-Clause
"""The term-response table's rows follow the emitter's ``emission_terms`` order.

``term_band_response`` documents the contract: "Term order matches the emitter's
``emission_terms`` dict order", and every consumer (the radio, X-ray and AGN
components) pairs table rows with terms by position,
``jnp.stack([t[i] for i, t in enumerate(ref.values())]) / band["S_ref"]``.

The trap this pins (#2914): a ``jax.jit``-ted function returns a dict output with
its keys **sorted**, because JAX flattens dicts in sorted key order. Building the
table from a traced ``emission_terms`` and iterating that returned dict put the
rows in sorted order (``agn, ff, sf``) while the consumer walked the emitter's
native order (``sf, ff, agn``); the radio and X-ray bands of a free-redshift model
were off by ~75 % against the exact Fixed(z) build, with every value in the table
correct to 1e-16. The builder now captures the native order inside the trace.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.contract

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"


@pytest.fixture(scope="module")
def free_z_emitters():
    """A free-redshift WavePrecomp model with the radio and X-ray additive emitters."""
    if not _SSP_PATH.is_file():
        pytest.skip(f"needs {_SSP_PATH}")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=Observation(
            photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4", "herschel_250"])
        ),
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        },
        xray={"type": "simple", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Uniform(0.1, 1.5),
        approx=WavePrecomp(n_z=12, z_min=0.1, z_max=1.5),
    )


@pytest.mark.parametrize("name", ["radio", "xray"])
def test_table_rows_are_in_the_emitters_term_order(free_z_emitters, name):
    """``S_ref`` at the first node equals the terms at ``lam_ref``, paired by position."""
    model = free_z_emitters
    comp = next(c for c in model._cached_component_chain if getattr(c, "name", "") == name)
    table = getattr(model, f"_{name}_term_response_cache", None)
    assert isinstance(table, dict), f"the {name} term-response table was not built"

    fixed = {k: jnp.asarray(v) for k, v in model.spec.get_fixed_values().items()}
    z0 = float(jnp.expm1(table["ln1pz"][0]))
    ref = comp.emission_terms(
        {**fixed, "redshift": jnp.asarray(z0)}, table["lam_ref"], **comp.EMITTER_PROBE_INPUTS[0]
    )
    # The consumer's own pairing: row i of the table against the i-th term of the dict.
    by_position = np.asarray(jnp.stack([t[i] for i, t in enumerate(ref.values())]))
    s_ref = np.asarray(table["S_ref"][0])
    np.testing.assert_allclose(
        by_position,
        s_ref,
        rtol=1e-8,
        err_msg=(
            f"{name}: table rows are not in the emitter's term order {list(ref)}; "
            "a traced emission_terms returns its dict with sorted keys"
        ),
    )
