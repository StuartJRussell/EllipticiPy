# Copyright (C) 2022 Stuart Russell
"""
This file contains the necessary function for a user to calculate an
ellipticity correction for a seismic ray path in a given model.
"""

# Import modules
import numpy as np
from obspy.taup import TauPyModel
from .tools import ellipticity_coefficients, correction_from_coefficients, EARTH_LOD


def ellipticity_correction(
    arrivals, azimuth, source_latitude, model=None, lod=EARTH_LOD
):
    """
    Returns the ellipticity correction to a 1D traveltime for a ray path in a 1D velocity model.

    Inputs:
        arrivals - a TauP Arrivals object
        azimuth - float, azimuth from source to receiver in degrees from N
        source_latitude - float, source latitude in degrees

    Optional inputs:
        model - obspy.taup.tau.TauPyModel or obspy.taup.tau_model.TauModel object
        lod - float, length of day of the model. Defaults to Earth value

    Output:
        float, ellipticity correction in seconds

    Example:
        >>> from obspy.taup import TauPyModel
        >>> model = TauPyModel('prem')
        >>> arrivals = model.get_ray_paths(source_depth_in_km = 124, distance_in_degree = 65,
                phase_list = ['pPKiKP'])
        >>> ellipticity_correction(arrivals, azimuth = 39, source_latitude = 45)
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
    dt = np.array(
        [correction_from_coefficients(sig, azimuth, source_latitude) for sig in sigma]
    )

    return dt


def ellip_corr(
    model_name,
    phase,
    source_depth_in_km,
    distance_in_degree,
    azimuth,
    source_latitude,
    lod=EARTH_LOD,
):
    """Simple wrapper of ellipticity_correction for those who don't want to directly call obspy."""
    model = TauPyModel(model_name)
    arrivals = model.get_ray_paths(
        source_depth_in_km, distance_in_degree, phase_list=[phase]
    )
    dt = ellipticity_correction(arrivals, azimuth, source_latitude, lod=EARTH_LOD)
    return dt[0]
