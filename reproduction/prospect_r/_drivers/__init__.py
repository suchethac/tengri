"""Driver glue for the ProSpect ↔ tengri reproduction notebook.

``units`` converts ProSpect's L⊙/Å spectra into tengri's erg/s/Hz convention
and provides the shared plotting helpers; ``prospect_driver`` wraps the R
package ProSpect through ``rpy2`` so the notebook can call ProSpect's forward
model live and read the results back as NumPy arrays.
"""
