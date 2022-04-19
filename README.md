## About

A python package for the calculation of ellipticity corrections for seismic phases in elliptical planetary models.

Authors:
- Stuart Russell
- Jessica C. E. Irving
- John F. Rudge
- Sanne Cottaar

The workings of this package are described in:

insertpaperhere

Please cite this publication when publishing work that has made use of this package.

## LICENSE

License for this package - standard LGPL license.

## Installation

The package can be installed using pip, the same as many python packages:

```
pip install ellipticity
```

This package depends on ObsPy. For information regarding ObsPy please see the relevant documentation: https://docs.obspy.org/

## Usage

This package is intended to be used in Python

```
>>> from obspy.taup import TauPyModel
>>> from ellipticity import ellipticity_correction
>>> model = TauPyModel('prem')
>>> arrivals = model.get_ray_paths(source_depth_in_km = 124,
    distance_in_degree = 65, phase_list = ['pPKiKP'])
>>> ellipticity_correction(arrivals, azimuth = 39, source_latitude = 45)
[-0.7761560510457043]
```

For users that do not wish to directly interact with Python, there is a command line wrapper although this is substantially less efficient.

```
>>> ./ellipticity_correction.py -d 134 -deg 64 -az 15 -sl 23 -ph P,PcP,PKiKP -mod ak135
Model: ak135
 Distance    Depth     Phase         Ray Param     Spherical    Ellipticity      Elliptical
   (deg)      (km)     Name          p (s/deg)     Travel       Correction       Travel
                                                   Time (s)         (s)          Time (s)
-------------------------------------------------------------------------------------------
     64.0    134.0     P                 6.536       619.05         -0.45          618.61
     64.0    134.0     PcP                4.11       653.31         -0.48          652.83
     64.0    134.0     PKiKP             1.307      1020.55         -0.75          1019.8
```

## Examples

Further examples of code usage in Jupyter Notebook format can be found in src/

- example\_get\_corrections.ipynb
- example\_Mars.ipynb

An example velocity model of Mars, as used by example\_mars.ipynb, is in the mars\_model directory.
