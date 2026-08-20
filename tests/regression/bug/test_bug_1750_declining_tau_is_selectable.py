# SPDX-License-Identifier: BSD-3-Clause
"""The declining-tau SFH must be reachable by name, and not by an ambiguous one (#1750).

``declining_exponential`` is FSPS ``sfh=1`` / Bagpipes ``'exponential'`` — the
parametric SFH most often quoted in the literature. It was implemented,
exported and documented, but absent from ``SFH_REGISTRY``, so no user could
select it through ``SEDModel.build(sfh={'type': ...})``.

The omission was deliberate. #406 registered it as ``"tau"``, users read that
name as CIGALE's ``sfhdelayed`` — which is tau-*delayed* and rises from zero —
and the mix-up produced a silent wavelength-dependent residual. The remedy
taken was to delete the entry.

That fixed the confusion by removing the model, which is why this file pins
*both* halves at once:

* the model is selectable again, and
* it is still measurably distinct from the two SFHs it was confused with.

A future change that restores the name ``"tau"`` would satisfy the first and
break the intent of the second, so the naming is asserted too.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import FREE, Fixed, SEDModel
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

pytestmark = pytest.mark.regression_bug

_TYPE = "declining_exp"


# ── selectable, and under a name that cannot be misread ───────────


def test_the_declining_tau_model_is_registered():
    """It must be reachable by name — the whole defect (#1750)."""
    assert _TYPE in SFH_REGISTRY, (
        f"{_TYPE!r} absent from SFH_REGISTRY; FSPS sfh=1 is unselectable again"
    )


def test_it_is_advertised_as_production_not_unvalidated():
    """Listed and buildable, not parked in UNVALIDATED_SFH_TYPES."""
    rows = [r for r in tengri.list_sfh_models() if r["name"] == _TYPE]
    assert rows, f"{_TYPE!r} missing from tengri.list_sfh_models()"
    assert rows[0]["status"] == "production"


def test_the_ambiguous_name_is_not_reintroduced():
    """``tau`` names a timescale both models have, so it cannot distinguish them.

    #406's finding was that the *name* was wrong. Putting it back — as the
    original report for #1750 proposed — reinstates the confusion that produced
    the residual, and would also force the parameter ``sfh_tau_tau_gyr``.
    """
    assert "tau" not in SFH_REGISTRY, (
        "'tau' is back in SFH_REGISTRY; #406 removed it because it reads as "
        "CIGALE sfhdelayed. Use 'declining_exp'."
    )


def test_the_parameters_carry_the_full_prefix():
    """NAMING_CONTRACT 3.2: free params use the full ``sfh_<type>_`` prefix."""
    declared = set(SFH_REGISTRY[_TYPE].params)
    assert declared == {
        "sfh_declining_exp_log_total_mass",
        "sfh_declining_exp_tau_gyr",
        "sfh_declining_exp_age_gyr",
    }, declared


# ── the two gates that drifted must agree ─────────────────────────


def test_no_allowlist_entry_names_an_unregistered_sfh():
    """The forward allowlist may not admit a name the registry cannot resolve.

    This is the drift that hid the defect. #406 deleted ``"tau"`` from
    ``SFH_REGISTRY`` but left it in ``_SUPPORTED_SFH`` in
    ``components/stellar/component.py``, so the two gates disagreed for as long
    as nobody re-read both: the allowlist said the model was validated, and the
    registry lookup one line later would have raised ``KeyError``.

    Generalized deliberately — a dead entry for any SFH is the same defect.
    """
    import re
    from pathlib import Path

    src = Path(tengri.__file__).parent / "components" / "stellar" / "component.py"
    text = src.read_text()
    block = re.search(r"_SUPPORTED_SFH = \((.*?)\)", text, re.S)
    assert block, "_SUPPORTED_SFH not found — did it move?"

    # Drop comment lines before reading the names. The comment recording this
    # very fix quotes "tau", and scraping it made the guard report the defect it
    # had just verified fixed — a source-text guard must read code, not prose.
    code = "\n".join(
        line for line in block.group(1).splitlines() if not line.strip().startswith("#")
    )
    names = set(re.findall(r'"([^"]+)"', code))

    dead = sorted(names - set(SFH_REGISTRY))
    assert not dead, (
        f"_SUPPORTED_SFH admits {dead}, absent from SFH_REGISTRY. The lookup "
        f"raises KeyError first, so the allowlist entry is dead either way."
    )


# ── still distinct from the two it was confused with ──────────────


def _sed(ssp_data, sfh_type, tau_gyr=2.0, age_gyr=5.0):
    """Rest-frame L_nu for one SFH family, normalized to unit peak."""
    from tengri.observation import Photometry

    obs = Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
    model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": sfh_type, "all_params": FREE},
        dust={"type": "none"},
        redshift=Fixed(0.1),
    )
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    for key, value in (("tau_gyr", tau_gyr), ("age_gyr", age_gyr)):
        name = f"sfh_{sfh_type}_{key}"
        if name in params:
            params[name] = jnp.asarray(value)
    sed = np.asarray(model.predict(params).rest_sed())
    return sed / np.max(sed)


@pytest.mark.parametrize("other", ["exp", "delayed"])
def test_it_is_not_the_model_it_was_confused_with(ssp_data_wne, other):
    """Measured distinctness from ``exp`` (rising) and ``delayed`` (tau-delayed).

    If a future refactor points ``declining_exp`` at either callable, the SEDs
    coincide and this fails — which is the #406 residual, caught at the shape
    rather than in a wavelength-dependent flux ratio months later.
    """
    mine = _sed(ssp_data_wne, _TYPE)
    theirs = _sed(ssp_data_wne, other)

    max_diff = float(np.max(np.abs(mine - theirs)))
    assert max_diff > 1e-3, (
        f"{_TYPE!r} and {other!r} give the same normalized SED "
        f"(max |diff| = {max_diff:.3e}) — they are different SFH families"
    )


# ── physics ───────────────────────────────────────────────────────


def test_slower_decline_is_bluer(ssp_data_wne):
    """Larger tau leaves more recent star formation, so the SED is bluer.

    SFR(T) ~ exp(-T/tau) with T cosmic time from formation: small tau puts the
    star formation at formation and leaves an old, red population today.
    """
    fast = _sed(ssp_data_wne, _TYPE, tau_gyr=0.3)
    slow = _sed(ssp_data_wne, _TYPE, tau_gyr=8.0)

    # Blue/red ratio on the normalized SEDs; index 0 is the bluest sample.
    n = fast.shape[0]
    blue, red = slice(0, n // 4), slice(3 * n // 4, n)
    ratio_fast = float(np.mean(fast[blue]) / np.mean(fast[red]))
    ratio_slow = float(np.mean(slow[blue]) / np.mean(slow[red]))

    assert ratio_slow > ratio_fast, (
        f"tau=8 Gyr should be bluer than tau=0.3 Gyr; "
        f"blue/red = {ratio_slow:.4f} vs {ratio_fast:.4f}"
    )


def test_every_declared_parameter_moves_the_photometry(ssp_data_wne):
    """No declared knob may be dead — the #1764 failure mode, checked up front.

    A registered model whose parameters have identically-zero gradient reports
    the prior back as the posterior and nothing raises.
    """
    from tengri.observation import Photometry

    obs = Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
    model = SEDModel.build(
        ssp_data=ssp_data_wne,
        observation=obs,
        sfh={"type": _TYPE, "all_params": FREE},
        dust={"type": "none"},
        redshift=Fixed(0.1),
    )
    params = model.spec.sample(jax.random.PRNGKey(0))
    grad = jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))(params)

    for name in SFH_REGISTRY[_TYPE].params:
        g = float(grad[name])
        assert np.isfinite(g), f"{name}: non-finite gradient {g}"
        assert g != 0.0, f"{name}: identically-zero gradient — an inert knob (#1764)"
