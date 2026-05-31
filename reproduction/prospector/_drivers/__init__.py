"""Reproduction-notebook drivers for Prospector (Johnson+2021).

Thin wrappers around Prospector's forward engine — ``python-fsps`` (the
FSPS stellar population synthesis library Prospector wraps) plus
``sedpy.attenuation`` (the dust-law source Prospector imports) — that
expose their outputs in tengri's unit convention (erg/s/Hz on an
Angstrom rest-frame grid).
"""
