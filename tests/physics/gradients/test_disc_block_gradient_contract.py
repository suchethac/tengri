# SPDX-License-Identifier: BSD-3-Clause
"""Class-wide gradient contract for every registered composable disc block (#2572).

For each disc block in ``AGN_BLOCKS["disc"]`` (bar ``"none"``), ``jax.grad`` of
``sum(log10 sed_agn)`` with respect to every free ``agn_*`` parameter must match a
central finite difference to ``rel 1e-4`` at an interior point (x64).

Why this exists
---------------
``kubota_done`` returned ``d/d(agn_log_mbh) = -1190.6`` where a central difference
gave ``-293.9`` (a 305% error; ``d/d(agn_log_lbol)`` was off by 6%). Every other disc
agreed to seven digits, and the disc's own unit tests passed: the ``custom_jvp`` on
the nthcomp template interpolation had dropped the seed-temperature tangent,
justified by a measurement made where the warm-zone Planck term dominates. A
gradient that is wrong only in the UV/X-ray is invisible to any test that checks
finiteness or an optical band.

The test point is the issue's: ``agn_log_lbol=11.5``, ``agn_log_mbh=8.5``,
``agn_log_ledd=-1.0`` wherever the block exposes them, every other parameter at
its deterministic prior draw (``PRNGKey(0)``).

Zero derivatives
----------------
FD == AD == 0 is not evidence (a saturated ``jnp.clip`` returns it legitimately,
so does a constant). A parameter's agreement is therefore judged against a floor
of ``1e-3`` of the largest finite-difference derivative *of the same block*, and
each block must show a non-trivial derivative in at least one parameter so the check
cannot pass vacuously.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tests._data_skip import DATA_DIR

jax.config.update("jax_enable_x64", True)

import tengri
from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.agn.blocks import AGN_BLOCKS

_SSP = DATA_DIR / "bc03_pdva_stelib_chabrier.h5"
_NONE = {"type": "none"}
_REL_TOL = 1e-4
_H = 1e-4
_ZERO_FLOOR = 1e-3
# ``agn_kt_warm = 0.13`` keV sits mid-cell of the nthcomp template's kTe axis (nodes
# 0.1143 and 0.1464), so the central difference does not straddle a slope kink.
_POINT = {
    "agn_log_lbol": 11.5,
    "agn_log_mbh": 8.5,
    "agn_log_ledd": -1.0,
    "agn_kt_warm": 0.13,
    "agn_a_spin": 0.5,
}
# ``agn_a_spin = 0.5`` (a moderately spinning hole) replaces the prior draw 0.4874, which sits
# 1.8e-3 below a genuine slope kink: warm ring 31 of 50 crosses the nthcomp template's kTbb
# node 0.00044122 keV at a = 0.48916 (d sum(log10 L)/da drops 13.22 -> 13.06 across it, while
# AD = FD = 12.9466 for h <= 1e-3 with the template interpolated in float64). At 0.5 no ring
# crosses a template node anywhere in a +- 0.006.
# The nthcomp kernel quantises kTe to float32 (relative 6e-8) on the forward pass, so a
# central difference in ``agn_kt_warm`` (likewise ``agn_gamma_warm``) is noise below
# h ~ 1e-3 (measured at 0.1166:
# FD = -5.84 at h=1e-5, -12.2 at 1e-6, -6.52 at 1e-3 against AD -6.52). The default
# step is far below that.
_STEP = {"agn_kt_warm": 2e-3, "agn_gamma_warm": 1e-3, "agn_a_spin": 3e-3}
# ``agn_kt_warm`` and ``agn_a_spin`` are held to 1e-3, not 1e-4: through 50 warm rings the
# float32-quantised, piecewise-linear template makes the central difference itself scatter by
# ~5e-4 of its value across steps. agn_kt_warm, h = 1e-3..5e-3: -5.029, -5.032, -5.033, -5.035
# against AD -5.0323. agn_a_spin at the Page-Thorne point, h = 1e-5..3e-3: 4.96, 5.07, 5.02,
# 4.82, 4.92, 4.91 against AD 4.915, so it is stepped at 3e-3 on the plateau. The kernel-level
# test below pins the slope itself at 1e-4.
_TOL = {"agn_kt_warm": 1e-3, "agn_a_spin": 1e-3}
_DISC_BLOCKS = sorted(name for name in AGN_BLOCKS["disc"] if name != "none")

pytestmark = pytest.mark.skipif(not _SSP.is_file(), reason=f"BC03 SSP not found at {_SSP}")


@pytest.fixture(scope="module")
def ssp():
    """The bare BC03 SSP the AGN-only build needs."""
    return tengri.load_ssp(str(_SSP))


def _build_agn_only(ssp, disc_spec: dict):
    """AGN-only composable model: the disc alone, ``sed_agn`` is the disc SED."""
    agn = {
        "type": "composable",
        "disc": disc_spec,
        "torus": _NONE,
        "nlr": _NONE,
        "blr": _NONE,
        "atten": _NONE,
        "agn_log_lbol": FREE,
        "all_params": Fixed(DEFAULT),
        "norm": "independent",
    }
    sfh = {"type": "const", "log_total_mass": -10.0, "all_params": Fixed(DEFAULT)}
    return SEDModel.build(ssp, sfh=sfh, agn=agn, redshift=Fixed(0.5))


def _grad_and_fd(model, overrides=None):
    """Return ``{param: (ad, fd)}`` of sum(log10 sed_agn) over the free agn_* params."""
    base = {
        k: jnp.asarray(v, dtype=jnp.float64)
        for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()
    }
    for name, value in {**_POINT, **(overrides or {})}.items():
        if name in base:
            base[name] = jnp.asarray(value, dtype=jnp.float64)
    free = [k for k in base if k.startswith("agn_")]

    def objective(sub):
        p = {**base, **sub}
        sed = model.predict(p).sed.components["sed_agn"]
        return jnp.sum(jnp.log10(jnp.maximum(sed, 1e-300)))

    sub = {k: base[k] for k in free}
    ad = jax.grad(objective)(sub)
    out = {}
    for k in free:
        h = _STEP.get(k, _H * max(1.0, abs(float(base[k]))))
        fd = (objective({**sub, k: sub[k] + h}) - objective({**sub, k: sub[k] - h})) / (2.0 * h)
        out[k] = (float(ad[k]), float(fd))
    return out


# Extra kubota_done point (#2572). ``agn_f_hard=0.005`` puts the hot-flow target at
# ~0.0376 L0, inside the 0.33 L0 ceiling, so R_hot is solved in the interior of its
# bracket (x_hot 1.6). Before the derivative fix the default f_hard pinned R_hot at the
# old ceiling at the issue point; this point is interior either way.
_CASES = [(name, "", {}) for name in _DISC_BLOCKS] + [
    ("kubota_done", "-r_hot_unclipped", {"agn_f_hard": 0.005}),
]


@pytest.mark.parametrize(
    ("disc_type", "overrides"),
    [pytest.param(n, o, id=n + tag) for n, tag, o in _CASES],
)
def test_disc_block_gradient_matches_central_fd(ssp, disc_type, overrides):
    """AD and central FD of sum(log10 sed_agn) agree to 1e-4 for every free parameter."""
    model = _build_agn_only(ssp, {"type": disc_type, "all_params": FREE})
    grads = _grad_and_fd(model, overrides)

    scale = max(abs(fd) for _, fd in grads.values())
    assert scale > 0.0, "vacuous check: the SED does not respond to any free parameter"
    failures = []
    for name, (ad, fd) in grads.items():
        denom = max(abs(fd), _ZERO_FLOOR * scale)
        rel = abs(ad - fd) / denom
        if not rel < _TOL.get(name, _REL_TOL):
            failures.append(f"{name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2e}")
    assert not failures, (
        f"{disc_type}: gradient != central FD (rel tol {_REL_TOL}):\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("kw", [{}, {"agn_f_hard": 0.005}], ids=["default", "r_hot_unclipped"])
def test_kubota_done_full_agn_gradient_matches_central_fd(kw):
    """``kubota_done_full_agn`` shares the K&D disc path, so it shares the defect.

    Monolithic (not a registered disc block, so outside the parametrised contract
    above). Objective: ``sum(log10 L_lambda)`` over ``lambda < 4000 A``, at the issue's
    point. Measured before the fix: ``d/d(agn_log_lbol)`` 236.3 vs FD 244.2 (3%) and
    ``d/d(agn_log_mbh)`` **+4.39 vs FD -11.31: the wrong sign**.

    ``agn_log_lbol`` is held to the contract's 1e-4. ``agn_log_mbh`` is held to 1e-2
    only: its derivative is a cancellation of large per-wavelength terms, and the
    float32-quantised template makes the central difference itself scatter by ~0.4%
    across steps 1e-4..1e-2 (-11.36, -11.31, -11.29), while the AD value is exact.
    The pre-fix error is 139%, so 1e-2 still fails the defect.
    """
    from tengri.components.agn.unified import kubota_done_full_agn

    wl = jnp.logspace(2.0, 6.0, 600)
    uv = wl < 4000.0

    def objective(lbol, mbh):
        sed = kubota_done_full_agn(wl, lbol, agn_log_mbh=mbh, **kw)
        return jnp.sum(jnp.log10(jnp.maximum(sed, 1e-300))[uv])

    x = (jnp.float64(11.5), jnp.float64(8.5))
    ad = jax.grad(objective, argnums=(0, 1))(*x)
    h = 1e-4
    fd_lbol = (objective(x[0] + h, x[1]) - objective(x[0] - h, x[1])) / (2 * h)
    fd_mbh = (objective(x[0], x[1] + h) - objective(x[0], x[1] - h)) / (2 * h)
    # 1e-4, except at the interior-R_hot point: there the central difference at h=1e-4 straddles
    # a template slope kink 3e-5 away (FD 318.46 for h = 1e-4..3e-3), while with the template
    # interpolated in float64 FD(h=1e-5) = 318.55017 equals AD = 318.55017 to 1e-8.
    tol_lbol = 1e-3 if kw else _REL_TOL
    assert abs(float(ad[0]) - float(fd_lbol)) / abs(float(fd_lbol)) < tol_lbol
    assert abs(float(ad[1]) - float(fd_mbh)) / abs(float(fd_mbh)) < 1e-2


def test_nthcomp_kernel_tangents_are_the_exact_cell_slopes():
    """Kernel-level: every operand of the nthcomp shape carries its exact slope (#2572).

    The kernel's ``custom_jvp`` dropped the ``kTbb`` tangent and took finite
    differences spanning template cells for ``gamma``/``kTe``. The interpolant is
    ``exp`` of a function linear in each operand within a cell, and evaluated at
    template ``nu`` nodes the resample onto ``nu`` is exact. So ``d log(shape)/dx`` is
    a constant across the cell and a *wide* central difference (0.4 of a cell, well
    above the kernel's float32 quantisation noise) measures it exactly; the AD
    tangent must equal ``shape * that``.
    """
    import numpy as np

    from tengri.components.agn._nthcomp import load_nthcomp_table, nthcomp_lnu_interp

    table = load_nthcomp_table()
    assert table is not None, "nthcomp templates are packaged with the source tree"
    axes = (table.gamma, table.kte, table.ktbb)
    cells = (6, 4, 20)
    x0 = jnp.asarray(
        [0.5 * (float(a[i]) + float(a[i + 1])) for a, i in zip(axes, cells)], dtype=jnp.float64
    )
    steps = [0.4 * (float(a[i + 1]) - float(a[i])) for a, i in zip(axes, cells)]
    nu = jnp.asarray(np.asarray(table.nu)[[100, 200, 300]], dtype=jnp.float64)

    def shape(x):
        return nthcomp_lnu_interp(nu, x[0], x[1], x[2])

    base = shape(x0)
    assert bool(jnp.all(base > 0.0)), "vacuous: the shape underflows at the probe frequencies"
    jac = jax.jacfwd(shape)(x0)  # (n_nu, 3): tangent of each nu sample
    for i, axis in enumerate(("gamma", "kTe", "kTbb")):
        e = jnp.zeros(3).at[i].set(steps[i])
        dlog = (jnp.log(shape(x0 + e)) - jnp.log(shape(x0 - e))) / (2.0 * steps[i])
        assert bool(jnp.all(dlog != 0.0)), f"vacuous: the shape does not move with {axis}"
        np.testing.assert_allclose(
            np.asarray(jac[:, i]), np.asarray(base * dlog), rtol=1e-4, err_msg=axis
        )
