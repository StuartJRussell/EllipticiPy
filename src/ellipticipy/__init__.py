"""
EllipticiPy
===================

:copyright:
    Stuart Russell
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

A Python package for the calculation of ellipticity corrections for seismic phases in planetary models.

When publishing work produced with this package please cite the following publication:
https://doi.org/10.1093/gji/ggac315

Functions:
    ellipticity_correction - Calculates ellipticity correction for a given ray path and velocity model

"""

__version__ = "1.0.0"
__author__ = "Stuart Russell, John F. Rudge, Jessica C. E. Irving, Sanne Cottaar"

# Import functions
from .main import ellipticity_correction
