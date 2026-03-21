# Photometry

The `Photometry` class is a frozen dataclass that holds filter transmission curves.
Filters are loaded once and remain immutable throughout the fitting run.

```python
from tengri import Photometry
```

## Loading filters by name

The simplest way to create a `Photometry` object: pass short names from the built-in
filter registry.

```python
phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
```

Filters are downloaded from the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/)
on first use and cached locally in `data/filters/`.

## Available filters

The built-in registry covers the most common photometric surveys:

| Prefix | Instrument | Example names |
|--------|------------|---------------|
| `sdss_` | SDSS ugriz | `sdss_u`, `sdss_g`, `sdss_r`, `sdss_i`, `sdss_z` |
| `lsst_` | Rubin/LSST | `lsst_u`, `lsst_g`, `lsst_r`, `lsst_i`, `lsst_z`, `lsst_y` |
| `hst_` | HST ACS/WFC + WFC3/IR | `hst_f435w`, `hst_f606w`, `hst_f814w`, `hst_f160w` |
| `jwst_` | JWST NIRCam | `jwst_f090w`, `jwst_f115w`, `jwst_f150w`, `jwst_f200w`, `jwst_f277w`, `jwst_f356w`, `jwst_f410m`, `jwst_f444w` |
| `roman_` | Roman/WFI | `roman_f062`, `roman_f087`, `roman_f106`, `roman_f129`, `roman_f158`, `roman_f184` |

To see the full list programmatically:

```python
from tengri.models.observation.filters import list_available_filters
list_available_filters()
```

## Custom filters

For filters not in the registry, load from a two-column text file
(wavelength in Angstrom, transmission):

```python
from tengri.models.observation.filters import load_custom_filter

my_filter = load_custom_filter("path/to/my_filter.dat")
phot = Photometry(filters=(my_filter,))
```

## Backward compatibility with load_filter_set

If you have code that uses the older `load_filter_set()` function, you can wrap its
output directly:

```python
from tengri import load_filter_set

filter_data = load_filter_set(["sdss_r", "sdss_i", "sdss_z"])
phot = Photometry.from_filter_set(filter_data)
```

`from_filter_set()` accepts either the 3-tuple returned by `load_filter_set()`
(waves, transmissions, curves) or a plain list of `FilterCurve` objects.

## Accessing filter data

Once created, the `Photometry` object exposes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `phot.filters` | `tuple[FilterCurve, ...]` | Raw filter curve objects |
| `phot.names` | `tuple[str, ...]` | Human-readable names |
| `phot.filter_waves` | `tuple[jnp.ndarray, ...]` | Wavelength arrays per filter |
| `phot.filter_trans` | `tuple[jnp.ndarray, ...]` | Transmission arrays per filter |
| `phot.n_filters` | `int` | Number of filters |

```python
>>> phot = Photometry.from_names(["jwst_f200w", "jwst_f356w"])
>>> phot.n_filters
2
>>> phot.names
('jwst_f200w', 'jwst_f356w')
>>> phot.summary()
'2 filters: jwst_f200w, jwst_f356w'
```
