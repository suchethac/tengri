# SPDX-License-Identifier: BSD-3-Clause
"""Driver shims for the Synthesizer reproduction notebook.

``synthesizer_driver`` wraps the Synthesizer forward model (stellar grids,
parametric SFHs, attenuation/dust/IGM laws, and the ``UnifiedAGN`` black-hole
emission tree); ``units`` adapts Synthesizer's ``unyt``-tagged output to the
notebook's plain-NumPy erg/s/Hz convention and carries the shared plotting
helpers.
"""
