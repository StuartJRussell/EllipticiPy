#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Stuart Russell
# Created Date: March 2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
This file contains the functions to support the calculation of ellipticity
corrections. All functions in this file are called by the main functions
in the main file.
"""
# ---------------------------------------------------------------------------
# Import modules
import os
import obspy
import warnings
import numpy as np
from obspy.taup import TauPyModel
from scipy.integrate import cumtrapz

# ---------------------------------------------------------------------------
# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="Resizing a TauP array inplace failed due to the existence of other references to the array, creating a new array. See Obspy #2280.",
)
# ---------------------------------------------------------------------------

earth_lod = 86164.0905 # s, length of day

# Define Exceptions
class PhaseError(Exception):
    """
    Class for handing exception of when there is no phase arrival for the inputted geometry
    """

    def __init__(self, phase, vel_model):
        self.message = (
            "Phase "
            + phase
            + " does not arrive at specified distance in model "
            + vel_model
        )

    def __str__(self):
        return self.message


def kron0(j):
    """
    Kronecker delta in the case of i = 0
    """

    if j == 0:
        return 1
    else:
        return 0


def weighted_alp2(m, theta):
    """
    Returns the weighted degree 2 associated Legendre polynomial for a given order and value.

    Inputs:
        m - int, order of polynomial (0, 1, or 2)
        theta - float

    Output:
        out - float, value of weighted associated Legendre polynomial of degree 2 and order m
              at x = cos(theta)
    """

    norm = np.sqrt(
        (2 - kron0(m)) * (np.math.factorial(2 - m) / np.math.factorial(2 + m))
    )

    if m == 0:
        return norm * 0.5 * (3.0 * np.cos(theta) ** 2.0 - 1.0)
    elif m == 1:
        return norm * 3.0 * np.cos(theta) * np.sin(theta)
    elif m == 2:
        return norm * 3.0 * np.sin(theta) ** 2.0


def model_epsilon(model, lod=earth_lod, taper=True, dr=100):
    """
    Calculates a profile of ellipticity of figure (epsilon) through a planetary model.

    Inputs:
        model - TauPyModel
        lod - float, length of day in the model in seconds
        taper - bool, whether to taper below ICB or not. Causes problems if False
            (and True is consistent with previous works, e.g. Bullen & Haddon (1973))
        dr - float, step length in metres for discretization

    Output:
        Adds arrays of epsilon and radius to the model instance as attributes
        model.model.s_mod.v_mod.epsilon_r and model.model.s_mod.v_mod.epsilon
    """

    # Angular velocity of model
    Omega = 2 * np.pi / lod  # s^-1

    # Universal gravitational constant
    G = 6.67408e-11  # m^3 kg^-1 s^-2

    # Radius of planet in m
    a = model.model.radius_of_planet * 1e3

    # Radii to evaluate integrals
    r = np.arange(0, a + dr, dr)

    # Get the density (in kg m^-3) at these radii
    rho = np.append(
        model.model.s_mod.v_mod.evaluate_above((a - r[:-1]) / 1000.0, "d") * 1000.0,
        model.model.s_mod.v_mod.evaluate_below(0.0, "d")[0] * 1000.0,
    )

    # Mass within each spherical shell
    Mr = 4 * np.pi * cumtrapz(rho * (r**2) * dr)

    # Total mass of body
    M = Mr[-1]

    # Moment of inertia of each spherical shell
    Ir = (8.0 / 3.0) * np.pi * cumtrapz(rho * (r**4) * dr)

    # Calculate y (moment of inertia factor) for surfaces within the body
    y = Ir / (Mr * r[1:] ** 2)

    # Taper if required
    # Have maximum y of 0.4, this is where eta is 0
    # Otherwise epsilon tends to infinity at the centre of the planet
    if taper:
        y[y > 0.4] = 0.4

    # Calculate Radau's parameter
    eta = 6.25 * (1 - 3 * y / 2) ** 2 - 1

    # Calculate h
    # Ratio of centrifugal force and gravity for a particle on the equator at the surface
    ha = (a**3 * Omega**2) / (G * M)

    # epsilon at surface
    epsilona = (5 * ha) / (2 * eta[-1] + 4)

    # Solve the differential equation
    epsilon = np.exp(cumtrapz(dr * eta / r[1:], initial=0.0))
    epsilon = epsilona * epsilon / epsilon[-1]
    epsilon = np.insert(epsilon, 0, epsilon[0])  # add a centre of planet value

    # Output as model attributes
    model.model.s_mod.v_mod.epsilon_r = r
    model.model.s_mod.v_mod.epsilon = epsilon


def get_taup_arrival(phase, distance, source_depth, arrival_index, model):
    """
    Returns a TauP arrival object for the given phase, distance, depth and velocity model

    Inputs:
        phase - string, TauP phase name
        distance - float, epicentral distance in degrees
        source_depth  - float, source depth in km
        arrival_index - int, the index of the desired arrival, starting from 0
        model - TauPyModel object

    Output:
        TauP arrival object
    """

    # Get the taup arrival for this phase
    arrivals = model.get_ray_paths(
        source_depth_in_km=source_depth,
        distance_in_degree=distance,
        phase_list=[phase],
        receiver_depth_in_km=0.0,
    )
    arrivals = [x for x in arrivals if abs(x.purist_distance - distance) < 0.0001]
    if len(arrivals) == 0:
        vel_model_name = str(model.model.s_mod.v_mod.model_name)
        if "'" in vel_model_name:
            vel_model = vel_model_name.split("'")[1]
        else:
            vel_model = vel_model_name
        raise PhaseError(phase, vel_model)

    return arrivals[arrival_index]


def get_correct_taup_arrival(arrival, model, extra_distance=0.0):
    """
    Returns a TauP arrival object in the correct form if the original is not

    Inputs:
        arrival - TauP arrival object
        model - TauPyModel object
        extra_distance - float, any further distance than the inputted arrival to obtain the new arrival

    Output:
        TauP arrival object
    """

    # Get arrival with the same ray parameter as the input arrival
    new_arrivals = model.get_ray_paths(
        source_depth_in_km=arrival.source_depth,
        distance_in_degree=arrival.distance + extra_distance,
        phase_list=[arrival.name],
        receiver_depth_in_km=0.0,
    )
    index = np.array(
        [abs(x.ray_param - arrival.ray_param) for x in new_arrivals]
    ).argmin()
    new_arrival = new_arrivals[index]
    return new_arrival


def centre_of_planet_coefficients(arrival, model):
    """
    Returns coefficients when a array passes too close to the centre of the Earth.
    When a ray passes very close to the centre of the Earth there is a step in distance which is problematic.
    In this case then interpolate the coefficients for two nearby arrivals.

    Inputs:
        arrival - TauP arrival object
        model - TauPyModel object

    Output:
        List of three floats, approximate ellipticity coefficients for the inputted Arrival
    """

    # Get two arrivals that do not go so close to the centre of the planet
    arrival1 = get_correct_taup_arrival(arrival, model, extra_distance=-0.05)
    arrival2 = get_correct_taup_arrival(arrival, model, extra_distance=-0.10)

    # Get the corrections for these arrivals
    coeffs1 = calculate_coefficients(arrival1, model)
    coeffs2 = calculate_coefficients(arrival2, model)

    # Linearly interpolate each coefficient to get final coefficients
    coeffs = [
        (
            coeffs1[i]
            + (
                (arrival.distance - arrival1.distance)
                / (arrival2.distance - arrival1.distance)
            )
            * (coeffs2[i] - coeffs1[i])
        )
        for i in range(len(coeffs1))
    ]

    return coeffs


def get_epsilon(model, radius):
    """
    Gets the value of epsilon for that model at a specified radius

    Inputs:
        model - TauPyModel object
        radius - float, radius  in m

    Output:
        float, value of epsilon
    """

    # Epsilon and radii arrays
    epsilon = model.model.s_mod.v_mod.epsilon
    radii = model.model.s_mod.v_mod.epsilon_r

    # Get the nearest value of epsilon to the given radius
    idx = np.searchsorted(radii, radius, side="left")
    if idx > 0 and (
        idx == len(radii)
        or np.math.fabs(radius - radii[idx - 1]) < np.math.fabs(radius - radii[idx])
    ):
        return epsilon[idx - 1]
    else:
        return epsilon[idx]


def get_dvdr_below(model, radius, wave):
    """
    Gets the value of dv/dr for that model immediately below a specified radius

    Inputs:
        model - TauPyModel object
        radius - float, radius in m
        wave - str, wave type: 'p' or 's'

    Output:
        float, value of dv/dr
    """

    Re = model.model.radius_of_planet
    v_mod = model.model.s_mod.v_mod
    return -evaluate_derivative_below(v_mod, Re - radius / 1e3, wave)[0]


def get_dvdr_above(model, radius, wave):
    """
    Gets the value of dv/dr for that model immediately above a specified radius

    Inputs:
        model - TauPyModel object
        radius - float, radius in m
        wave - str, wave type: 'p' or 's'

    Output:
        float, value of dv/dr
    """

    Re = model.model.radius_of_planet
    v_mod = model.model.s_mod.v_mod
    return -evaluate_derivative_above(v_mod, Re - radius / 1e3, wave)[0]


def evaluate_derivative_below(model, depth, prop):
    """Evaluate depth derivative of material property at bottom of a velocity layer."""
    layer = model.layers[model.layer_number_below(depth)]
    return evaluate_derivative_at(layer, depth, prop)


def evaluate_derivative_above(model, depth, prop):
    """Evaluate depth derivative of material property at top of a velocity layer."""
    layer = model.layers[model.layer_number_above(depth)]
    return evaluate_derivative_at(layer, depth, prop)


def evaluate_derivative_at(layer, depth, prop):
    """Evaluate depth derivative of material property in a velocity layer."""

    thick = layer["bot_depth"] - layer["top_depth"]
    prop = prop.lower()
    if prop == "p":
        slope = (layer["bot_p_velocity"] - layer["top_p_velocity"]) / thick
        return slope
    elif prop == "s":
        slope = (layer["bot_s_velocity"] - layer["top_s_velocity"]) / thick
        return slope
    elif prop in "rd":
        slope = (layer["bot_density"] - layer["top_density"]) / thick
        return slope
    raise ValueError("Unknown material property, use p, s, or d.")


def calculate_coefficients(arrival, model, lod=earth_lod):
    """
    Returns ellipticity coefficients for a given ray path

    Inputs:
        arrival - EITHER a TauP arrival object OR a list containing [phase, distance, source_depth, index] where:
                  phase - string, TauP phase name
                  distance - float, epicentral distance in degrees
                  source_depth - float, source depth in km
                  index - int, the index of the desired arrival, starting from 0
        model - TauPyModel object
        lod - float, length of day of the model in seconds, only needed if calculating coefficients for a new model

    Output:
        list of three floats, ellipticity coefficients
    """

    ########################
    ##### Open 1D model ####
    ########################

    # If model is not initialised then do so
    if type(model) == str:
        model = TauPyModel(model=model)
    elif type(model) != obspy.taup.tau.TauPyModel:
        raise TypeError("Velocity model not correct type")

    # Name of model
    vel_model_name = str(model.model.s_mod.v_mod.model_name)
    if "'" in vel_model_name:
        vel_model = vel_model_name.split("'")[1]
    else:
        vel_model = vel_model_name

    # Radius of Earth
    Re = model.model.radius_of_planet

    # Get discontinuities in the model
    discs = model.model.s_mod.v_mod.get_discontinuity_depths()[:-1]
    discsr = [Re - x for x in discs]

    # Calculate epsilon values if they don't already exist
    if not hasattr(model.model.s_mod.v_mod, "epsilon"):
        model_epsilon(model, lod)

    #########################
    ##### Get an arrival ####
    #########################

    # Check if arrival is a TauP object or a list and get arrival if needed
    if type(arrival) == list:

        # Call an arrival, this will error if it the phase input is unrealistic
        # Ideally users should use TauP Arrivals as inputs but some may not
        arrival = get_taup_arrival(
            arrival[0], arrival[1], arrival[2], arrival[3], model
        )

    elif type(arrival) == obspy.taup.helper_classes.Arrival and type(
        arrival.path
    ) == type(None):

        # Call an arrival that has a ray path
        # Ideally users should use the ObsPy TauP get_ray_paths() to get their Arrivals, but if they haven't then this will fix it
        warnings.warn(
            "Arrival does not have ray path, in future please input the correct arrival for greater efficiency"
        )
        arrival = get_correct_taup_arrival(arrival, model)

    # If ray parameter is zero then this is problematic, so adjust the distance slightly
    if arrival.distance == 0.0:

        # Call an arrival that has non-zero ray parameter
        # We can't integrate the ray when the ray parameter is zero, but the integral does converge when the distance is zero so just add a tiny bit of distance
        arrival = get_correct_taup_arrival(arrival, model, extra_distance=1e-10)

    # Bottoming depth of ray
    bot_dep = max([x[3] for x in arrival.path])

    # When the ray goes close to the centre of the Earth, the distance function has a step in it
    # This is problematic to integrate along
    # Instead, if the ray goes within 50m of the centre of the planet, calculate for nearby two values and interpolate
    # This produces a satisfactory approximation
    if (model.model.radius_of_planet - bot_dep) * 1e3 < 50:
        sigma = centre_of_planet_coefficients(arrival, model)

    else:

        ##########################
        ##### Set up ray path ####
        ##########################

        # Get wave types for each segment of ray path
        phase_name = arrival.name
        if "diff" in phase_name:
            segments = phase_name.replace(
                "diff", phase_name[phase_name.index("diff") - 1]
            )
        else:
            segments = phase_name
        segments = (
            segments.replace("c", "")
            .replace("i", "")
            .replace("K", "P")
            .replace("I", "P")
            .replace("J", "S")
        )
        letter = 0
        waves = []
        for i in range(len(arrival.path)):
            if (
                i != 0
                and i != len(arrival.path) - 1
                and arrival.path[i][3]
                in [0.0, model.model.cmb_depth, model.model.iocb_depth]
                and arrival.path[i][3] != arrival.path[i - 1][3]
            ):
                letter = letter + 1
            waves.append(segments[letter].lower())

        # Split the path at discontinuities and bottoming depth
        wave = {}
        paths = {}
        count = -1
        for i in range(len(arrival.path)):
            if (
                i == 0 or arrival.path[i][3] in discs or arrival.path[i][3] == bot_dep
            ) and i != len(arrival.path) - 1:
                count = count + 1
                paths[count] = []
                wave[count] = []
                if count != 0:
                    paths[count - 1].append(list(arrival.path[i]))
                    wave[count - 1].append(waves[i - 1])
            paths[count].append(list(arrival.path[i]))
            wave[count].append(waves[i])
        paths = {x: np.array(paths[x]) for x in paths}
        wave_paths = {x: np.array(wave[x]) for x in paths}

        #############################
        ##### Integrate ray path ####
        #############################

        # Ray parameter in sec/rad
        p = arrival.ray_param

        # Loop through path segments
        seg_ray_sigma = {}
        for x in paths:
            path = paths[x]

            # Depth in m
            dep = path[:, 3] * 1e3
            # Remove centre of the Earth so that TauP doesn't error
            dep[dep == Re * 1e3] = Re * 1e3 - 1
            # Radius in m
            r = Re * 1e3 - dep
            # Velocity in m/s
            v = np.array(
                [
                    model.model.s_mod.v_mod.evaluate_below(
                        dep[i] / 1e3, wave_paths[x][i]
                    )[0]
                    * 1e3
                    if dep[i] != max(dep)
                    else model.model.s_mod.v_mod.evaluate_above(
                        dep[i] / 1e3, wave_paths[x][i]
                    )[0]
                    * 1e3
                    for i in range(len(path))
                ]
            )
            # Gradient of v wrt r
            dvdr = np.array(
                [
                    get_dvdr_below(model, r[i], wave_paths[x][0])
                    if dep[i] != max(dep)
                    else get_dvdr_above(model, r[i], wave_paths[x][0])
                    for i in range(len(path))
                ]
            )
            # eta in s
            eta = r / v
            # epsilon
            epsilon = np.array(
                [
                    get_epsilon(model, r[i])
                    if dep[i] != max(dep)
                    else get_epsilon(model, r[i])
                    for i in range(len(path))
                ]
            )
            # Distance in radians
            dist = np.array([x[2] for x in path])
            # lambda
            lamda = {
                x: (-1.0) * (2.0 / 3.0) * weighted_alp2(x, dist) for x in [0, 1, 2]
            }
            # Do the integration
            seg_ray_sigma[x] = {
                m: np.sum(
                    np.trapz((eta**3.0) * dvdr * epsilon * lamda[m], x=dist) / p
                )
                for m in [0, 1, 2]
            }

        # Sum coefficients for each segment to get total ray path contribution
        ray_sigma = {m: np.sum([seg_ray_sigma[x][m] for x in paths]) for m in [0, 1, 2]}

        #####################################
        ##### Effects of discontinuities ####
        #####################################

        # Including the bottoming depth allows cross indexing with the paths variable when the start point is not the lowest point on the ray path
        if bot_dep == arrival.path[0][3]:
            assess_discs = discs
        else:
            assess_discs = np.append(discs, bot_dep)

        # Get which discontinuities the phase interacts with, include bottoming depth to allow cross indexing with the paths variable
        ids = [
            (i, arrival.path[i][3], arrival.path[i][2])
            for i in range(len(arrival.path))
            if arrival.path[i][3] in assess_discs
        ]
        if arrival.source_depth != 0 and arrival.source_depth != ids[0][1]:
            ids = [(0, arrival.source_depth, 0)] + ids
        idiscs = {
            i: {
                "ind": ids[i][0],
                "order": i,
                "dep": ids[i][1] * 1e3,
                "r": (Re - ids[i][1]) * 1e3,
                "dist": ids[i][2],
                "p": arrival.path[0][0],
            }
            for i in range(len(ids))
        }

        # Loop through discontinuities and assess what is occurring
        for d in idiscs:

            # Do not sum if diffracted and this is the CMB
            if (
                "diff" in arrival.name
                and idiscs[d]["dep"] == model.model.cmb_depth * 1e3
            ):
                idiscs[d]["yn"] = False

            # Do not calculate for bottoming depth if this is not a discontinuity
            elif round(idiscs[d]["dep"] * 1e-3, 5) in discs or d == 0:
                idiscs[d]["yn"] = True

            # Do not sum if this is the bottoming depth
            else:
                idiscs[d]["yn"] = False

            # Proceed if summing this discontinuity
            if idiscs[d]["yn"]:

                # epsilon at this depth
                idiscs[d]["epsilon"] = get_epsilon(model, idiscs[d]["r"])

                # lambda at this distance
                idiscs[d]["lambda"] = {
                    x: (-1.0) * (2.0 / 3.0) * weighted_alp2(x, idiscs[d]["dist"])
                    for x in [0, 1, 2]
                }

                # Calculate the factor
                extra = {
                    x: idiscs[d]["epsilon"] * idiscs[d]["lambda"][x] for x in [0, 1, 2]
                }

                # The surface must be treated differently due to TauP indexing constraints
                if idiscs[d]["dep"] != 0.0 and idiscs[d]["ind"] != 0:

                    # Depths before and after interactions
                    dep0 = arrival.path[idiscs[d]["ind"] - 1][3]
                    dep1 = arrival.path[idiscs[d]["ind"]][3]
                    dep2 = arrival.path[idiscs[d]["ind"] + 1][3]

                    # Direction before interaction
                    if dep0 < dep1:
                        idiscs[d]["pre"] = "down"
                    elif dep0 == dep1:
                        idiscs[d]["pre"] = "diff"
                    else:
                        idiscs[d]["pre"] = "up"

                    # Direction after interaction
                    if dep1 < dep2:
                        idiscs[d]["post"] = "down"
                    elif dep1 == dep2:
                        idiscs[d]["post"] = "diff"
                    else:
                        idiscs[d]["post"] = "up"

                    # Reflection or transmission
                    if idiscs[d]["pre"] == idiscs[d]["post"]:
                        idiscs[d]["type"] = "trans"
                    elif "diff" in [idiscs[d]["pre"], idiscs[d]["post"]]:
                        idiscs[d]["type"] = "diff"
                    else:
                        idiscs[d]["type"] = "refl"

                    # Phase before and after
                    idiscs[d]["ph_pre"] = wave_paths[d - 1][-1]
                    idiscs[d]["ph_post"] = wave_paths[d][0]

                    # Deal with a transmission case
                    if idiscs[d]["type"] == "trans":

                        # Phase above
                        if idiscs[d]["pre"] == "down":
                            idiscs[d]["ph_above"] = idiscs[d]["ph_pre"]
                            idiscs[d]["ph_below"] = idiscs[d]["ph_post"]
                        elif idiscs[d]["pre"] == "up":
                            idiscs[d]["ph_above"] = idiscs[d]["ph_post"]
                            idiscs[d]["ph_below"] = idiscs[d]["ph_pre"]

                        # Velocity above and below discontinuity
                        idiscs[d]["v0"] = (
                            model.model.s_mod.v_mod.evaluate_above(
                                idiscs[d]["dep"] / 1e3, idiscs[d]["ph_above"]
                            )[0]
                            * 1e3
                        )
                        idiscs[d]["v1"] = (
                            model.model.s_mod.v_mod.evaluate_below(
                                idiscs[d]["dep"] / 1e3, idiscs[d]["ph_below"]
                            )[0]
                            * 1e3
                        )

                        # eta above and below discontinuity
                        idiscs[d]["eta0"] = idiscs[d]["r"] / idiscs[d]["v0"]
                        idiscs[d]["eta1"] = idiscs[d]["r"] / idiscs[d]["v1"]

                        # Evaluate the time difference
                        eva = (-1.0) * (
                            np.sqrt(idiscs[d]["eta0"] ** 2 - idiscs[d]["p"] ** 2)
                            - np.sqrt(idiscs[d]["eta1"] ** 2 - idiscs[d]["p"] ** 2)
                        )

                    # Deal with an underside reflection case
                    if idiscs[d]["type"] == "refl" and idiscs[d]["pre"] == "up":

                        # Velocity below discontinuity
                        idiscs[d]["v0"] = (
                            model.model.s_mod.v_mod.evaluate_below(
                                idiscs[d]["dep"] / 1e3, idiscs[d]["ph_pre"]
                            )[0]
                            * 1e3
                        )
                        idiscs[d]["v1"] = (
                            model.model.s_mod.v_mod.evaluate_below(
                                idiscs[d]["dep"] / 1e3, idiscs[d]["ph_post"]
                            )[0]
                            * 1e3
                        )

                        # eta below discontinuity
                        idiscs[d]["eta0"] = idiscs[d]["r"] / idiscs[d]["v0"]
                        idiscs[d]["eta1"] = idiscs[d]["r"] / idiscs[d]["v1"]

                        # Evaluate the time difference
                        eva = np.sqrt(
                            idiscs[d]["eta0"] ** 2 - idiscs[d]["p"] ** 2
                        ) + np.sqrt(idiscs[d]["eta1"] ** 2 - idiscs[d]["p"] ** 2)

                    # Deal with a topside reflection case
                    if idiscs[d]["type"] == "refl" and idiscs[d]["pre"] == "down":

                        # Velocity above discontinuity
                        idiscs[d]["v0"] = (
                            model.model.s_mod.v_mod.evaluate_above(
                                idiscs[d]["dep"] / 1e3, idiscs[d]["ph_pre"]
                            )[0]
                            * 1e3
                        )
                        idiscs[d]["v1"] = (
                            model.model.s_mod.v_mod.evaluate_above(
                                idiscs[d]["dep"] / 1e3, idiscs[d]["ph_post"]
                            )[0]
                            * 1e3
                        )

                        # eta above discontinuity
                        idiscs[d]["eta0"] = idiscs[d]["r"] / idiscs[d]["v0"]
                        idiscs[d]["eta1"] = idiscs[d]["r"] / idiscs[d]["v1"]

                        # Evaluate the time difference
                        eva = (-1) * (
                            np.sqrt(idiscs[d]["eta0"] ** 2 - idiscs[d]["p"] ** 2)
                            + np.sqrt(idiscs[d]["eta1"] ** 2 - idiscs[d]["p"] ** 2)
                        )

                # Deal with source depth and also end point
                elif idiscs[d]["ind"] == 0 or idiscs[d]["ind"] == len(arrival.path) - 1:

                    # Assign wave type
                    if idiscs[d]["ind"] == 0:
                        wave = wave_paths[0][0]
                    elif idiscs[d]["ind"] == len(arrival.path) - 1:
                        wave = wave_paths[max(list(paths.keys())) - 1][-1]

                    # Deal with phases that start with an upgoing segment
                    if arrival.name[0] in ["p", "s"] and idiscs[d]["ind"] == 0:

                        # Velocity above source
                        idiscs[d]["v1"] = (
                            model.model.s_mod.v_mod.evaluate_above(
                                idiscs[d]["dep"] / 1e3, wave
                            )[0]
                            * 1e3
                        )

                        # eta above source
                        idiscs[d]["eta1"] = idiscs[d]["r"] / idiscs[d]["v1"]

                        # Evaluate the time difference
                        eva = (-1.0) * np.sqrt(
                            idiscs[d]["eta1"] ** 2 - idiscs[d]["p"] ** 2
                        )

                    # Deal with ending the ray path at the surface
                    else:
                        # Velocity below surface
                        idiscs[d]["v1"] = (
                            model.model.s_mod.v_mod.evaluate_below(
                                idiscs[d]["dep"] / 1e3, wave
                            )[0]
                            * 1e3
                        )

                        # eta below surface
                        idiscs[d]["eta1"] = idiscs[d]["r"] / idiscs[d]["v1"]

                        # Evaluate the time difference
                        eva = (-1.0) * (
                            0 - np.sqrt(idiscs[d]["eta1"] ** 2 - idiscs[d]["p"] ** 2)
                        )

                # Deal with surface reflection
                elif idiscs[d]["dep"] == 0.0:

                    # Assign type of interaction
                    idiscs[d]["type"] = "refl"

                    # Phase before and after
                    idiscs[d]["ph_pre"] = wave_paths[d - 1][-1]
                    idiscs[d]["ph_post"] = wave_paths[d][0]

                    # Velocity below surface
                    idiscs[d]["v0"] = (
                        model.model.s_mod.v_mod.evaluate_below(
                            idiscs[d]["dep"] / 1e3, idiscs[d]["ph_pre"]
                        )[0]
                        * 1e3
                    )
                    idiscs[d]["v1"] = (
                        model.model.s_mod.v_mod.evaluate_below(
                            idiscs[d]["dep"] / 1e3, idiscs[d]["ph_post"]
                        )[0]
                        * 1e3
                    )

                    # eta below surface
                    idiscs[d]["eta0"] = idiscs[d]["r"] / idiscs[d]["v0"]
                    idiscs[d]["eta1"] = idiscs[d]["r"] / idiscs[d]["v1"]

                    # Evaluate time difference
                    eva = np.sqrt(
                        idiscs[d]["eta0"] ** 2 - idiscs[d]["p"] ** 2
                    ) + np.sqrt(idiscs[d]["eta1"] ** 2 - idiscs[d]["p"] ** 2)

                # Output coefficients for this discontinuity
                idiscs[d]["sigma"] = {x: extra[x] * eva for x in [0, 1, 2]}

        # Sum the contribution to the contribution to the coefficients from discontinuities
        disc_sigma = {
            x: np.sum([idiscs[i]["sigma"][x] for i in idiscs if idiscs[i]["yn"]])
            for x in [0, 1, 2]
        }

        ################################
        ##### Sum the contributions ####
        ################################

        # Sum the contribution from the ray path and the discontinuities to get final coefficients
        sigma = [ray_sigma[x] + disc_sigma[x] for x in [0, 1, 2]]

    return sigma
