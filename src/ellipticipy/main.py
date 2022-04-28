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


def ellipticity_correction(arrivals, azimuth, source_latitude, model=None, lod=EARTH_LOD):
    """
    Ellipticity correction to be added to a travel time.

    :param arrivals: TauP Arrivals object with ray paths calculated
    :type arrivals: :class:`obspy.taup.tau.Arrivals`
    :param azimuth: azimuth from source to receiver in degrees from N
    :type azimuth: float
    :param source_latitude: source latitude in degrees
    :type source_latitude: float
    :param model: optional, model used to calculate the arrivals
    :type model: :class:`obspy.taup.tau.TauPyModel` or
                 :class:`obspy.taup.tau_model.TauModel`
    :param lod: optional, length of day in seconds. Defaults to Earth value
    :type lod: float
    :returns: ellipticity correction in seconds for each arrival
    :rtype: list[float]

    Usage:

    >>> from obspy.taup import TauPyModel
    >>> from ellipticipy import ellipticity_correction
    >>> model = TauPyModel('prem')
    >>> arrivals = model.get_ray_paths(source_depth_in_km = 124,
        distance_in_degree = 65, phase_list = ['pPKiKP'])
    >>> ellipticity_correction(arrivals, azimuth = 39, source_latitude = 45)
    [-0.7731978967098823]
    """
    
    # Get model from arrivals object if model is None
    if model is None:
        model = arrivals.model
    if isinstance(model, TauPyModel):
        model = model.model

    # Enforce that event latitude must be in range -90 to 90 degrees
    if not -90 <= source_latitude <= 90:
        raise ValueError("Source latitude must be in range -90 to 90 degrees")

    # Enforce that azimuth must be in range 0 to 360 degrees
    if not 0 <= azimuth <= 360:
        raise ValueError("Azimuth must be in range 0 to 360 degrees")

    # Calculate the ellipticity coefficients
    sigma = ellipticity_coefficients(arrivals, model, lod)

    # Calculate time
    dt = [
        correction_from_coefficients(sig, azimuth, source_latitude)
        for sig in sigma
    ]

    return dt
