# Superseded: Configuration VI before the AGN floor fix (#2495)

These two cells were fit with `agn_log_lbol ~ Uniform(9.42, 13.42)`, whose
floor of ~1e43 erg/s cannot express "no AGN". Configuration VI now frees
`agn_lum_ratio ~ Uniform(0.0, 5.0)` instead (commit a7f73a362), so these
are a different model and must not be mixed with the new row.

Both were unadopted under the old parametrization:

    267_VI   divergences  72   rhat_max 1.0152   ess_min 139.6
    79_VI    divergences 107   rhat_max 1.0168   ess_min  75.3

Kept rather than deleted because they are the control arm of the A/B
recorded in #2495 and in `config_VI`'s docstring.
