# Copyright (C) 2022 Stuart Russell
"""
This module contains the necessary function for a user to calculate an
ellipticity correction for a seismic ray path in a given model.
"""

from obspy.taup import TauPyModel
from obspy.taup.helper_classes import Arrival
from .tools import (
    ellipticity_coefficients,
    correction_from_coefficients,
    azimuth_source_latitude_from_geo_arrival,
    EARTH_LOD,
)


def ellipticity_correction(arrivals, azimuth=None, source_latitude=None, lod=EARTH_LOD):
    """
    Ellipticity correction to be added to a travel time.

    :param arrivals: TauP Arrival or Arrivals object with ray paths calculated.
    :type arrivals: :class:`obspy.taup.tau.Arrivals` or
                    :class:`obspy.taup.tau.Arrival`
    :param azimuth: azimuth from source to receiver in degrees from N
    :type azimuth: float
    :param source_latitude: source latitude in degrees
    :type source_latitude: float
    :param lod: optional, length of day in seconds. Defaults to Earth value
    :type lod: float
    :returns: ellipticity correction in seconds for each arrival
    :rtype: list[float] for Arrivals, float for Arrival

    For arrivals generated by TauPyModel.get_ray_paths_geo the azimuth and source
    latitude do not need to be specified as they are inferred from the geographic path.

    Usage:

    >>> from obspy.taup import TauPyModel
    >>> from ellipticipy import ellipticity_correction
    >>> model = TauPyModel('prem')
    >>> arrivals = model.get_ray_paths(source_depth_in_km = 124,
        distance_in_degree = 65, phase_list = ['pPKiKP'])
    >>> ellipticity_correction(arrivals, azimuth = 39, source_latitude = 45)
    [-0.7731978967098823]
    """
    # Calculate the ellipticity coefficients
    sigma = ellipticity_coefficients(arrivals, lod=lod)

    if azimuth is None or source_latitude is None:
        if isinstance(arrivals, Arrival):
            azimuth, source_latitude = azimuth_source_latitude_from_geo_arrival(
                arrivals
            )
        else:
            azimuth, source_latitude = azimuth_source_latitude_from_geo_arrival(
                arrivals[0]
            )

    # Calculate time
    if isinstance(arrivals, Arrival):
        return correction_from_coefficients(sigma, azimuth, source_latitude)
    return [
        correction_from_coefficients(sig, azimuth, source_latitude) for sig in sigma
    ]
