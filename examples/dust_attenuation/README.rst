Dust Attenuation
================

Two-component Charlot & Fall geometry: ``dust_tau_bc`` on the birth clouds,
``dust_tau_diff`` on the diffuse ISM. ``dust_slope`` defaults to -0.7, the
diffuse-ISM value; -1.3 is the birth-cloud one. The 2175 Å bump is a separate
always-on modifier, ``dust_bump_strength``, defaulting to 0.0 — Calzetti
carries no bump unless you ask for one.

Dust emission templates load from ``data/``. There is no analytic fallback: a
missing template raises ``FileNotFoundError`` rather than quietly substituting
a worse model.
