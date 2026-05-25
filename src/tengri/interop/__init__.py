# SPDX-License-Identifier: BSD-3-Clause
"""Interop helpers for matching tengri inputs to other SED codes.

Currently exports CIGALE-equivalence conversions used by the
reproduction notebook (``reproduction/cigale/01_cigale.py``) and the
audit in #357:

- :func:`tengri.interop.cigale.log_peak_sfr_for_mass_formed` — back out
  the ``log_peak_sfr`` value that matches a target total stellar mass
  formed (CIGALE's ``sfhdelayed(..., normalise=True)`` convention).
- :func:`tengri.interop.cigale.cigale_ebv_lines_to_tau` — map
  CIGALE ``dustatt_modified_starburst(E_BV_lines=...)`` onto the
  Charlot-Fall ``two_component(tau_bc, tau_diff)`` pair.

These are not load-bearing for any tengri-only workflow; they exist
so cross-code comparison runs aren't booby-trapped by silent
convention mismatches.
"""

from __future__ import annotations

from tengri.interop import cigale

__all__ = ["cigale"]
