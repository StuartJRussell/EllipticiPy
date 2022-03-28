# Copyright (C) 2022 Stuart Russell
"""
This file contains the necessary function for a user to calculate an
ellipticity correction for a seismic ray path in a given model.
"""
# ---------------------------------------------------------------------------
# Import modules
import obspy
import numpy as np
from .tools import ellipticity_coefficients, weighted_alp2, EARTH_LOD


def ellipticity_correction(arrival, azimuth, source_latitude, model, lod=EARTH_LOD):
    """
    Returns the ellipticity correction to a 1D traveltime for a ray path in a 1D velocity model.

    Inputs:
        arrival - EITHER a TauP arrival object
            OR a list containing [phase, distance, source_depth, index] where:
                phase - string, TauP phase name
                distance - float, epicentral distance in degrees
                source_depth - float, source depth in km
                index - int, the index of the desired arrival, starting from 0
        azimuth - float, azimuth from source to receiver in degrees from N
        source_latitude - float, source latitude in degrees
        model - TauPyModel object OR string defining the path to a velocity model usable by TauP

    Optional inputs:
        lod - float, length of day of the model. Defaults to Earth value

    Output:
        float, ellipticity correction in seconds

    Example:
        >>> from obspy.taup import TauPyModel
        >>> model = TauPyModel('prem')
        >>> arrival = model.get_ray_paths(source_depth_in_km = 124, distance_in_degree = 65,
                phase_list = ['pPKiKP'])
        >>> ellipticity_correction(arrival, azimuth = 39, source_latitude = 45, model = model)
    """

    # Enforce that event latitude must be in range -90 to 90 degrees
    if not -90 <= source_latitude <= 90:
        raise ValueError("Source latitude must be in range -90 to 90 degrees")

    # Enforce that azimuth must be in range 0 to 360 degrees
    if not 0 <= azimuth <= 360:
        raise ValueError("Azimuth must be in range 0 to 360 degrees")

    # Assess whether input is arrival or coefficients
    if isinstance(arrival, obspy.taup.helper_classes.Arrival) or (
        isinstance(arrival, list) and len(arrival) == 4 and isinstance(arrival[0], str)
    ):

        # Get the coefficients
        sigma = [ellipticity_coefficients(arrival, model, lod)]

    # If a list of arrivals then deal with that
    elif isinstance(arrival, obspy.taup.tau.Arrivals) or (
        isinstance(arrival, list)
        and isinstance(arrival[0], obspy.taup.helper_classes.Arrival)
    ):

        # Get coefficients for each entry in the list
        sigma = [ellipticity_coefficients(arr, model, lod) for arr in arrival]

    # Deal with the case where the inputs are coefficients
    elif (
        isinstance(arrival, list)
        and len(arrival) == 3
        and float(arrival[0]) == arrival[0]
    ):

        # Assign coefficients
        sigma = [arrival]

    else:
        raise TypeError("Arrival/Coefficients not correctly defined")

    # Convert azimuth to radians
    az = np.radians(azimuth)

    # Convert latitude to colatitude
    evcla = np.radians(90 - source_latitude)

    # Calculate time
    dt = np.array(
        [
            sum(sig[m] * weighted_alp2(m, evcla) * np.cos(m * az) for m in [0, 1, 2])
            for sig in sigma
        ]
    )

    # Return value or list
    if len(dt) == 1:
        return dt[0]
    return dt
