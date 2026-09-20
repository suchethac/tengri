# SPDX-License-Identifier: BSD-3-Clause
"""The age-0 SSP anchor convention shared by the stellar and nebular paths.

Some SSP libraries ship an age-0 anchor template (BC03 STELIB:
``ssp_lg_age_gyr[0] = -inf``). Every consumer that turns the SSP age grid into
a ``log10(age/yr)`` axis floors that anchor at the one value below, so the
surviving-mass fraction (#1016) and the photoionized backends' Q_H tables
(#2418) agree on where the anchor's stars and ionizing photons are counted.
This module imports nothing from ``tengri``, so both sides can read it without
a package cycle.
"""

# Floor of an SSP age axis in log10(age/yr): 0.1 Myr. No star has died and no
# tabulated Q_H differs from its zero-age value below it, and it is a no-op for
# every grid whose youngest template is already >= 0.1 Myr (all shipped SSPs
# except BC03 STELIB). Consumers: ``components/stellar/sps/dsps_wrapper.py``
# (surviving mass, #1016) and ``components/nebular/_shared.py`` (Q_H axes,
# #2418), the latter through ``components/nebular/_constants.py``.
ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR: float = 5.0
