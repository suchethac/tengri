# Not a posterior. A memory measurement that happened to write one.

`fit_mock_joint --n-chains 1 --n-samples 10 --dense-mass` on 2026-09-28, run to
measure peak RSS of dense mass-matrix adaptation at D=36, not to sample.

It did not converge: chi2/dof 3.3392 against truth's 1.0738, 3350.6 worse in
chi2, cold start, one chain, ten draws. The recovery deltas in its log measure
the optimizer, not the model.

It is here rather than deleted because the measurement it carries is the point:
**peak RSS 8.13 GB for a single chain** over a full 1000-step warmup, which is
what sent the cap back to 30 (see nuts.py DENSE_MASS_MAX_DIM).

Never point `fig01_mock_joint_infer` or `derive_mock_properties` at these files.
