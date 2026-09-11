# SPDX-License-Identifier: BSD-3-Clause
"""The two shipped bugs, transcribed as they stood before their fixes.

A guard is verified against history, not against its author's intuition. These
are not paraphrases: each assertion below is the wording that was in the tree
while the corresponding defect shipped, so a future edit that stops the guard
firing on them is a regression in the guard rather than a matter of taste.

``pre_2100`` — ``tests/regression/precision/test_inference_grad_float32.py`` as
it stood on ``main`` through 2026-09-05. The bare ``sum(predict_photometry)``
float32 gradient was identically zero on CPU and GPU and this pinned it finite,
so the coverage that was supposed to catch #2100 never could.

``pre_2178`` — ``tests/regression/precision/test_float32_fitting_path_seams.py``
at ``ce5c4908c^``. A NaN gradient satisfied the non-zero assertion (``nan != 0.0``
is ``True``), XPASSed a strict xfail, and read as "the underflow was fixed at
source". It shipped a float32 NaN on the default spectroscopy path.
"""

import jax
import numpy as np


def pre_2100(ctx, da, jax_random_key):
    """Verbatim: the finite-only assertion on ``grad(neg_log_posterior)``."""
    p = ctx.initial_params(jax_random_key)
    g = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, da))(p)
    leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(g)]
    assert all(np.all(np.isfinite(v)) for v in leaves), (
        "grad(nlp) is non-finite in pure float32 — a reverse-pass "
        "overflow in the stellar mass scaling or the sub-band node ratio"
    )


def pre_2178(unweighted):
    """Verbatim: the non-zero-only assertion on the unweighted float32 gradient."""
    seam, out = unweighted
    g32 = out["f32"]["grad"]
    assert np.any(g32 != 0.0), (
        f"unweighted float32 gradient is IDENTICALLY ZERO on {seam} ({g32}) — this is "
        f"the #2100 defect, and a finite-only guard cannot see it. "
        f"approx={out['f32']['approx_state']}"
    )
