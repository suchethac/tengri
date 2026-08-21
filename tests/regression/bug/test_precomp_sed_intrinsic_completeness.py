# SPDX-License-Identifier: BSD-3-Clause
"""``sed_intrinsic`` must be COMPLETE under WavePrecomp, not just the LUT families.

``SEDModelComponent.apply`` used to return from its LUT branch without touching
``sed_intrinsic`` at all -- "The full-grid SED is intentionally NOT updated on any LUT
path" -- on the theory that keeping additive emitters off the full-grid array is what
makes the fast path fast.

That theory is wrong, and the cost of it was a silent one. ``predict_via_precomp`` never
*reads* ``sed_intrinsic``; it sums the ``*_phot_lnu_precomp`` families. So XLA
dead-code-eliminates the whole full-grid chain regardless, and *writing* an array nobody
reads is still dead code. Radio and X-ray have always added unconditionally and still
compile to ~143 us.

What the omission actually bought was a WavePrecomp model whose panchromatic SED had no
dust IR in it. ``Prediction.photometry()`` -- exact-by-default since #1097 -- projects
``sed_intrinsic`` directly, so it read 5.8x low in W3 and 6x low in W4, bit-identical to a
model built with no dust emission at all. The likelihood was fine (it reads the LUT), so
every fit was correct and every best-fit overlay, residual plot and mid-IR diagnostic
drawn from one was missing the IR bump entirely. Nothing failed. Nothing warned.

Pinned here:
  1. ``pred.photometry()`` on a WavePrecomp model equals the exact model's, and
  2. it is NOT equal to a model with the dust emission removed -- the assertion that
     would have caught this, and the one a "it looks about right" eyeball would not, and
  3. the fit path stays fast, so the fix cannot be undone in the name of speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel, WavePrecomp
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug

#: W3/W4 are where the dust IR bump lands at z=0.1; sdss_g is the control (no IR there).
BANDS = ["sdss_g", "wise_w3", "wise_w4"]

#: The LUT fit path compiled to ~3.6e5 FLOPs with this fix in place, against ~3.7e5
#: before it. If someone "restores" the no-write policy for speed, this budget shows there
#: was never any speed to restore.
MAX_LUT_FLOPS = 1_000_000


def _model(approx, *, emission=True):
    dust = {"type": "two_component", "law": "calzetti", "all_params": FIXED}
    groups = {}
    if emission:
        # A peer group now, not a sub-block. The emission=False arm omits it
        # entirely, which is what "a model with the dust emission removed" means
        # and what the vacuous-equality test below depends on.
        groups["dust_emission"] = {"type": "dale2014", "all_params": FIXED}
    return SEDModel.build(
        ssp_data=pytest.importorskip("tengri").load_ssp(),
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation=dust,
        approx=approx,
        **groups,
    )


def _params(m):
    p = {k: jnp.asarray(v) for k, v in m.spec.sample(jax.random.PRNGKey(0)).items()}
    p.update({k: jnp.asarray(float(v)) for k, v in m.spec.get_fixed_values().items()})
    return p


def test_prediction_photometry_carries_dust_ir_under_wave_precomp():
    """``pred.photometry()`` on a WavePrecomp model must match the exact model.

    The exact model is the independent reference: it never uses a LUT, so its
    ``sed_intrinsic`` has always carried the dust IR.
    """
    exact = _model(None)
    lut = _model(WavePrecomp())

    f_exact = np.asarray(exact.predict(_params(exact)).photometry())
    f_lut = np.asarray(lut.predict(_params(lut)).photometry())

    rel = np.abs(f_lut / f_exact - 1.0)
    assert rel.max() < 1e-9, (
        f"Prediction.photometry() on a WavePrecomp model differs from the exact model by "
        f"{dict(zip(BANDS, rel, strict=True))} — the dust IR emitter is being dropped from "
        "sed_intrinsic on the LUT path."
    )


def test_it_is_not_vacuously_equal_to_a_model_without_dust_emission():
    """The IR bands must MOVE when dust emission is removed.

    Without this, the test above could pass on a model where dust IR contributes nothing
    to any band — proving only that zero equals zero. The original bug was precisely that
    the precomp model's photometry was bit-identical (1.1e-05) to a no-emission model, so
    this is the assertion that distinguishes "correct" from "the emitter is absent".
    """
    lut = _model(WavePrecomp())
    off = _model(WavePrecomp(), emission=False)

    f_on = np.asarray(lut.predict(_params(lut)).photometry())
    f_off = np.asarray(off.predict(_params(off)).photometry())

    ratio = f_on / f_off
    w3, w4 = BANDS.index("wise_w3"), BANDS.index("wise_w4")
    assert ratio[w3] > 2.0 and ratio[w4] > 2.0, (
        f"removing dust emission barely changed the IR bands ({ratio=}) — this fixture "
        "cannot see the emitter, so the completeness test above proves nothing."
    )


def test_the_fit_path_stays_fast():
    """Writing sed_intrinsic on the LUT path costs nothing, so it must not be reverted.

    ``predict_via_precomp`` sums the ``*_phot_lnu_precomp`` families and never reads
    ``sed_intrinsic``, so XLA prunes the full-grid chain either way. This pins that: if the
    no-write policy is ever restored "for speed", this budget shows there was no speed in it.
    """
    lut = _model(WavePrecomp())
    flops = jax.jit(lut.predict_photometry).lower(_params(lut)).compile().cost_analysis()["flops"]
    assert flops < MAX_LUT_FLOPS, (
        f"{flops:,.0f} compiled FLOPs — writing sed_intrinsic on the LUT path was supposed "
        "to be free (it is dead code unless something reads it)."
    )
