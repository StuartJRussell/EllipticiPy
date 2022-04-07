## About

A python package for the calculation of ellipticity corrections for seismic phases in elliptical planetary models.

Authors:
- Stuart Russell
- Jessica C. E. Irving
- John F. Rudge
- Sanne Cottaar

The workings of this package are described in:

insertpaperhere

Please cit this publication when publishing work that has made use of this package.

## LICENSE

License for this package - standard LGPL license.

## Installation

The package can be installed using pip the same as many python packages:

```
pip install ellipticity
```

This package depends on ObsPy. For information regarding ObsPy please see the relevant documentation: https://docs.obspy.org/

## Usage

```
>>> from obspy.taup import TauPyModel
>>> from ellipticity import ellipticity\_correction
>>> model = TauPyModel('prem')
>>> arrivals = model.get\_ray_paths(source\_depth\_in\_km = 124,
    distance\_in\_degree = 65, phase\_list = ['pPKiKP'])
>>> ellipticity\_correction(arrivals, azimuth = 39, source\_latitude = 45)
[-0.7761560510457043]
```

## Examples

Further examples of code usage in Jupyter Notebook format can be found in src/

- example\_get\_corrections.ipynb
- example\_Mars.ipynb

An example velocity model of Mars, as used by example\_mars.ipynb, is in the mars\_model directory.
