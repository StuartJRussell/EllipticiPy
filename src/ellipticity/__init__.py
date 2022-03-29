# Copyright (C) 2022 Stuart Russell
"""
Ellipticity toolkit
===================
A Python package for the calculation of ellipticity corrections for seismic phases in planetary models

When publishing work produced with this package please cite the following publication:
INSERT PAPER HERE


Functions:
    ellipticity_correction - Calculates ellipticity correction for a given ray path and velocity model

"""

__version__ = "1.1"
__author__ = "Stuart Russell"

# Import functions
from .main import ellipticity_correction, ellip_corr
