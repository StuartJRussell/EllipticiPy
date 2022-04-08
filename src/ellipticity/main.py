# Copyright (C) 2022 Stuart Russell
"""
This file contains the necessary function for a user to calculate an
ellipticity correction for a seismic ray path in a given model.
"""

from obspy.taup import TauPyModel
from .tools import (
    ellipticity_coefficients,
    correction_from_coefficients,
    EARTH_LOD,
)


def ellipticity_correction(arrivals, azimuth, source_latitude, lod=EARTH_LOD):
    """
    Ellipticity correction to be added to a travel time.

    :param arrivals: TauP Arrivals object with ray paths calculated
    :type arrivals: :class:`obspy.taup.tau.Arrivals`
    :param azimuth: azimuth from source to receiver in degrees from N
    :type azimuth: float
    :param source_latitude: source latitude in degrees
    :type source_latitude: float
    :param lod: optional, length of day in seconds. Defaults to Earth value
    :type lod: float
    :returns: ellipticity correction in seconds for each arrival
    :rtype: list[float]

    Usage:

    >>> from obspy.taup import TauPyModel
    >>> from ellipticity import ellipticity_correction
    >>> model = TauPyModel('prem')
    >>> arrivals = model.get_ray_paths(source_depth_in_km = 124,
        distance_in_degree = 65, phase_list = ['pPKiKP'])
    >>> ellipticity_correction(arrivals, azimuth = 39, source_latitude = 45)
    [-0.7761560510457043]
    """

    # Enforce that event latitude must be in range -90 to 90 degrees
    if not -90 <= source_latitude <= 90:
        raise ValueError("Source latitude must be in range -90 to 90 degrees")

    # Enforce that azimuth must be in range 0 to 360 degrees
    if not 0 <= azimuth <= 360:
        raise ValueError("Azimuth must be in range 0 to 360 degrees")

    # Calculate the ellipticity coefficients
    sigma = ellipticity_coefficients(arrivals, lod)

    # Calculate time
    dt = [
        correction_from_coefficients(sig, azimuth, source_latitude)
        for sig in sigma
    ]

    return dt
