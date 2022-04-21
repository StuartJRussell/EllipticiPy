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
INSERT PAPER HERE


Functions:
    ellipticity_correction - Calculates ellipticity correction for a given ray path and velocity model

"""

__version__ = "0.9.0"
__author__ = (
    "Stuart Russell, John F. Rudge, Jessica C. E. Irving, Sanne Cottaar"
)

# Import functions
from .main import ellipticity_correction
