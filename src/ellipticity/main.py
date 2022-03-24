#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Stuart Russell
# Created Date: March 2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
This file contains the necessary function for a user to calculate an
ellipticity correction for a seismic ray path in a given model.
"""
# ---------------------------------------------------------------------------
# Import modules
import obspy
import warnings
import numpy as np
from obspy.taup import TauPyModel
from .tools import calculate_coefficients, list_coefficients, alp2, factor
# ---------------------------------------------------------------------------
# Suppress warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="Resizing a TauP array inplace failed due to the existence of other references to the array, creating a new array. See Obspy #2280.",
)
# ---------------------------------------------------------------------------

def calculate_correction(arrival, azimuth, source_latitude, model, lod = 86164.0905):
    """
    Returns the ellipticity correction to be added to a 1D traveltime for a given ray path in a 1D velocity model.

    Inputs:
        arrival - EITHER a TauP arrival object OR a list containing [phase, distance, source_depth, index] where:
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
    """

    # Assess whether input is arrival or coefficients
    if type(arrival) == obspy.taup.helper_classes.Arrival or (
        type(arrival) == list and len(arrival) == 4 and type(arrival[0]) == str
    ):

        # Get the coefficients
        sigma = [calculate_coefficients(arrival, model, lod)]

    # If a list of arrivals then deal with that
    elif type(arrival) == obspy.taup.tau.Arrivals or (
        type(arrival) == list and type(arrival[0]) == obspy.taup.helper_classes.Arrival
    ):

        # Get coefficients for each entry in the list
        sigma = list_coefficients(arrival, model, lod)

    # Deal with the case where the inputs are coefficients
    elif (
        type(arrival) == list and len(arrival) == 3 and float(arrival[0]) == arrival[0]
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
            sum(sig[m] * factor(m) * alp2(m, evcla) * np.cos(m * az) for m in [0, 1, 2])
            for sig in sigma
        ]
    )

    # Return value or list
    if len(dt) == 1:
        return dt[0]
    else:
        return dt
