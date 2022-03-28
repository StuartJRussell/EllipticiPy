# Copyright (C) 2022 Stuart Russell
"""
This file contains the functions to support the calculation of ellipticity
corrections. All functions in this file are called by the main functions
in the main file.
"""

import warnings
import obspy
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

EARTH_LOD = 86164.0905  # s, length of day
G = 6.67408e-11  # m^3 kg^-1 s^-2, universal gravitational constant


def model_epsilon(model, lod=EARTH_LOD, taper=True, dr=100):
    """
    Calculates a profile of ellipticity of figure (epsilon) through a planetary model.

    Inputs:
        model - obspy.taup.tau_model.TauModel object
        lod - float, length of day in seconds. Defaults to Earth value
        taper - bool, whether to taper below ICB or not. Causes problems if False
            (and True is consistent with previous works, e.g. Bullen & Haddon (1973))
        dr - float, step length in metres for discretization

    Output:
        Adds arrays of epsilon and radius to the model instance as attributes
        model.s_mod.v_mod.epsilon and model.s_mod.v_mod.epsilon_r
    """

    # Angular velocity of model
    Omega = 2 * np.pi / lod  # s^-1

    # Radius of planet in m
    a = model.radius_of_planet * 1e3

    # Radii to evaluate integrals
    r = np.linspace(0, a, 1 + int(a / dr))

    # Get the density (in kg m^-3) at these radii
    v_mod = model.s_mod.v_mod
    rho = np.append(
        v_mod.evaluate_above((a - r[:-1]) / 1e3, "d") * 1e3,
        v_mod.evaluate_below(0.0, "d")[0] * 1e3,
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
    v_mod.epsilon_r = r
    v_mod.epsilon = epsilon


def get_epsilon(model, radius):
    """
    Gets the value of epsilon for a model at a specified radius

    Inputs:
        model - obspy.taup.tau_model.TauModel object
        radius - float, radius in m

    Output:
        float, value of epsilon
    """

    # Epsilon and radii arrays
    epsilon = model.s_mod.v_mod.epsilon
    radii = model.s_mod.v_mod.epsilon_r

    # Get the nearest value of epsilon to the given radius
    idx = np.searchsorted(radii, radius, side="left")
    if idx > 0 and (
        idx == len(radii)
        or np.math.fabs(radius - radii[idx - 1]) < np.math.fabs(radius - radii[idx])
    ):
        return epsilon[idx - 1]
    return epsilon[idx]


def get_dvdr_below(model, radius, wave):
    """
    Gets the value of dv/dr for a model immediately below a specified radius

    Inputs:
        model - obspy.taup.tau_model.TauModel object
        radius - float, radius in m
        wave - str, wave type: 'p' or 's'

    Output:
        float, value of dv/dr
    """

    Re = model.radius_of_planet
    v_mod = model.s_mod.v_mod
    return -evaluate_derivative_below(v_mod, Re - radius / 1e3, wave)[0]


def get_dvdr_above(model, radius, wave):
    """
    Gets the value of dv/dr for a model immediately above a specified radius

    Inputs:
        model - obspy.taup.tau_model.TauModel object
        radius - float, radius in m
        wave - str, wave type: 'p' or 's'

    Output:
        float, value of dv/dr
    """

    Re = model.radius_of_planet
    v_mod = model.s_mod.v_mod
    return -evaluate_derivative_above(v_mod, Re - radius / 1e3, wave)[0]


def evaluate_derivative_below(model, depth, prop):
    """Evaluate depth derivative of material property at bottom of a velocity layer."""
    layer = model.layers[model.layer_number_below(depth)]
    return evaluate_derivative_at(layer, prop)


def evaluate_derivative_above(model, depth, prop):
    """Evaluate depth derivative of material property at top of a velocity layer."""
    layer = model.layers[model.layer_number_above(depth)]
    return evaluate_derivative_at(layer, prop)


def evaluate_derivative_at(layer, prop):
    """Evaluate depth derivative of material property in a velocity layer."""

    thick = layer["bot_depth"] - layer["top_depth"]
    prop = prop.lower()
    if prop == "p":
        slope = (layer["bot_p_velocity"] - layer["top_p_velocity"]) / thick
        return slope
    if prop == "s":
        slope = (layer["bot_s_velocity"] - layer["top_s_velocity"]) / thick
        return slope
    if prop in "rd":
        slope = (layer["bot_density"] - layer["top_density"]) / thick
        return slope
    raise ValueError("Unknown material property, use p, s, or d.")


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

    kronecker_0m = 1 if m == 0 else 0
    norm = np.sqrt(
        (2 - kronecker_0m) * (np.math.factorial(2 - m) / np.math.factorial(2 + m))
    )

    if m == 0:
        return norm * 0.5 * (3.0 * np.cos(theta) ** 2.0 - 1.0)
    if m == 1:
        return norm * 3.0 * np.cos(theta) * np.sin(theta)
    if m == 2:
        return norm * 3.0 * np.sin(theta) ** 2.0
    raise ValueError("Invalid value of m")


def ellipticity_coefficients(arrivals, model=None, lod=EARTH_LOD):
    """
    Returns ellipticity coefficients for a set of arrivals

    Inputs:
        arrivals - EITHER a TauP Arrivals object
            OR a list containing [phase, distance, source_depth, index] where:
                phase - string, TauP phase name
                distance - float, epicentral distance in degrees
                source_depth - float, source depth in km
                index - int, the index of the desired arrival, starting from 0
        model - obspy.taup.tau.TauPyModel or obspy.taup.tau_model.TauModel object
        lod - float, length of day of the model in seconds, only needed if calculating
            coefficients for a new model

    Output:
        list of lists of three floats, ellipticity coefficients
    """

    if model is None:
        model = arrivals.model

    # Assess whether input is arrivals or coefficients
    if isinstance(arrivals, obspy.taup.helper_classes.Arrival) or (
        isinstance(arrivals, list)
        and len(arrivals) == 4
        and isinstance(arrivals[0], str)
    ):
        return [individual_ellipticity_coefficients(arrivals, model, lod)]

    # If a list of arrivals then deal with that
    if isinstance(arrivals, obspy.taup.tau.Arrivals) or (
        isinstance(arrivals, list)
        and isinstance(arrivals[0], obspy.taup.helper_classes.Arrival)
    ):
        return [
            individual_ellipticity_coefficients(arr, model, lod) for arr in arrivals
        ]

    else:
        raise TypeError("Arrival not correctly defined")


def individual_ellipticity_coefficients(arrival, model, lod=EARTH_LOD):
    """
    Returns ellipticity coefficients for a given ray path

    Inputs:
        arrival - EITHER a TauP Arrival object
            OR a list containing [phase, distance, source_depth, index] where:
                phase - string, TauP phase name
                distance - float, epicentral distance in degrees
                source_depth - float, source depth in km
                index - int, the index of the desired arrival, starting from 0
        model - obspy.taup.tau.TauPyModel or obspy.taup.tau_model.TauModel object
        lod - float, length of day of the model in seconds, only needed if calculating
            coefficients for a new model

    Output:
        list of three floats, ellipticity coefficients
    """

    # If model is not initialised then do so
    if isinstance(model, str):
        model = TauPyModel(model=model).model
    if isinstance(model, obspy.taup.tau.TauPyModel):
        model = model.model
    if not isinstance(model, obspy.taup.tau_model.TauModel):
        raise TypeError("Velocity model not correct type")

    # Calculate epsilon values if they don't already exist
    if not hasattr(model.s_mod.v_mod, "epsilon"):
        model_epsilon(model, lod)

    # Check if arrival is a TauP object or a list and get arrival if needed
    if isinstance(arrival, list):

        # Call an arrival, this will error if the phase input is unrealistic
        # Ideally users should use TauP Arrivals as inputs but some may not
        arrival = get_taup_arrival(
            arrival[0], arrival[1], arrival[2], arrival[3], model
        )

    elif (
        isinstance(arrival, obspy.taup.helper_classes.Arrival) and arrival.path is None
    ):

        # Call an arrival that has a ray path
        # Ideally users should use the ObsPy TauP get_ray_paths() to get their arrivals,
        # but if they haven't then this will fix it
        warnings.warn(
            "Arrival does not have ray path, in future please input the correct arrival for greater efficiency"
        )
        arrival = get_correct_taup_arrival(arrival, model)

    # If ray parameter is zero then this is problematic, so adjust the distance slightly
    if arrival.distance == 0.0:

        # Call an arrival that has non-zero ray parameter
        # We can't integrate the ray when the ray parameter is zero, but the integral does converge
        # when the distance is zero so just add a tiny bit of distance
        arrival = get_correct_taup_arrival(arrival, model, extra_distance=1e-10)

    # Bottoming depth of ray
    bot_dep = max([point[3] for point in arrival.path])

    # When the ray goes close to the centre of the Earth, the distance function has a step in it
    # This is problematic to integrate along
    # Instead, if the ray goes within 50m of the centre of the planet, calculate for nearby two
    # values and interpolate. This produces a satisfactory approximation
    if (model.radius_of_planet - bot_dep) * 1e3 < 50:
        sigma = centre_of_planet_coefficients(arrival, model)
    else:
        ray_sigma = integral_coefficients(arrival, model)
        disc_sigma = discontinuity_coefficients(arrival, model)

        # Sum the contribution from the ray path and the discontinuities to get final coefficients
        sigma = [ray_sigma[m] + disc_sigma[m] for m in [0, 1, 2]]

    return sigma


def split_ray_path(arrival, model):
    """Split and label ray path according to type of wave."""

    # Bottoming depth of ray
    bot_dep = max([x[3] for x in arrival.path])

    # Get discontinuity depths in the model in km
    discs = model.s_mod.v_mod.get_discontinuity_depths()[:-1]

    # Get wave types for each segment of ray path
    segments = arrival.name
    if "diff" in segments:
        segments = segments.replace("diff", segments[segments.index("diff") - 1])
    segments = (
        segments.replace("c", "")
        .replace("i", "")
        .replace("K", "P")
        .replace("I", "P")
        .replace("J", "S")
    )

    # Label path with wave types
    letter = 0
    wave_labels = []
    for i, point in enumerate(arrival.path):
        depth = point[3]
        if (
            i != 0  # not beginning
            and i != len(arrival.path) - 1  # not end
            and depth
            in [
                0.0,
                model.cmb_depth,
                model.iocb_depth,
            ]  # surface, CMB, or IOCB
            and depth != arrival.path[i - 1][3]  # checking against previous depth
        ):
            letter = letter + 1
        wave_labels.append(segments[letter].lower())

    # Split the path at discontinuities and bottoming depth
    waves = []
    paths = []
    count = -1
    for i, point in enumerate(arrival.path):
        depth = point[3]
        if (
            i == 0  # is beginning
            or depth in discs  # or at a discontinuity
            or depth == bot_dep  # or at the bottoming depth
        ) and i != len(
            arrival.path
        ) - 1:  # not at the end
            count = count + 1
            paths.append([])
            waves.append([])
            if count != 0:
                paths[count - 1].append(list(point))
                waves[count - 1].append(wave_labels[i - 1])
        paths[count].append(list(point))
        waves[count].append(wave_labels[i])

    paths = [np.array(path) for path in paths]
    waves = [np.array(wave) for wave in waves]

    return paths, waves


def integral_coefficients(arrival, model):
    """Calculate correction coefficients due to integral along ray path"""

    # Radius of Earth
    Re = model.radius_of_planet

    # Loop through path segments
    seg_ray_sigma = []
    paths, waves = split_ray_path(arrival, model)
    for path, wave in zip(paths, waves):

        # Depth in m
        depth = path[:, 3] * 1e3
        # Remove centre of the Earth so that TauP doesn't error
        depth[depth == Re * 1e3] = Re * 1e3 - 1

        # Radius in m
        radius = Re * 1e3 - depth

        # Velocity in m/s
        v_mod = model.s_mod.v_mod
        v = np.array(
            [
                v_mod.evaluate_below(d / 1e3, w)[0] * 1e3
                if d != max(depth)
                else v_mod.evaluate_above(d / 1e3, w)[0] * 1e3
                for d, w in zip(depth, wave)
            ]
        )

        # Gradient of v wrt r in s^-1
        dvdr = np.array(
            [
                get_dvdr_below(model, r, wave[0])
                if r != min(radius)
                else get_dvdr_above(model, r, wave[0])
                for r in radius
            ]
        )

        # eta in s
        eta = radius / v

        # epsilon
        epsilon = np.array([get_epsilon(model, r) for r in radius])

        # Epicentral distance in radians
        distance = path[:, 2]

        # lambda
        lamda = [-(2.0 / 3.0) * weighted_alp2(m, distance) for m in [0, 1, 2]]

        # Do the integration
        seg_ray_sigma.append(
            [
                np.trapz((eta**3.0) * dvdr * epsilon * lamda[m], x=distance)
                / arrival.ray_param
                for m in [0, 1, 2]
            ]
        )

    # Sum coefficients for each segment to get total ray path contribution
    return [np.sum([s[m] for s in seg_ray_sigma]) for m in [0, 1, 2]]


def discontinuity_coefficients(arrival, model):
    """Calculate correction coefficients due to discontinuities"""

    paths, waves = split_ray_path(arrival, model)

    # Radius of Earth
    Re = model.radius_of_planet

    # Bottoming depth of ray
    bot_dep = max([point[3] for point in arrival.path])

    # Get discontinuities in the model
    v_mod = model.s_mod.v_mod  # velocity model
    discs = v_mod.get_discontinuity_depths()[:-1]

    # Including the bottoming depth allows cross indexing with the paths variable when
    # the start point is not the lowest point on the ray path
    if bot_dep == arrival.path[0][3]:
        assess_discs = discs
    else:
        assess_discs = np.append(discs, bot_dep)

    # Get which discontinuities the phase interacts with, include bottoming depth to allow
    # cross indexing with the paths variable
    ids = [
        (i, point[3], point[2])  # index, depth, distance
        for i, point in enumerate(arrival.path)
        if point[3] in assess_discs
    ]
    if arrival.source_depth not in (0, ids[0][1]):
        ids = [(0, arrival.source_depth, 0.0)] + ids
    idiscs = [
        {
            "ind": x[0],
            "order": i,
            "d": x[1] * 1e3,
            "r": (Re - x[1]) * 1e3,
            "dist": x[2],
            "p": arrival.path[0][0],
        }
        for i, x in enumerate(ids)
    ]

    # Loop through discontinuities and assess what is occurring
    for i, idisc in enumerate(idiscs):

        # Do not sum if diffracted and this is the CMB
        if "diff" in arrival.name and idisc["d"] == model.cmb_depth * 1e3:
            idisc["yn"] = False

        # Do not calculate for bottoming depth if this is not a discontinuity
        elif round(idisc["d"] * 1e-3, ndigits=5) in discs or i == 0:
            idisc["yn"] = True  # ALERT does rounding assume something about discs?

        # Do not sum if this is the bottoming depth
        else:
            idisc["yn"] = False

        # Proceed if summing this discontinuity
        if idisc["yn"]:

            # epsilon at this depth
            idisc["epsilon"] = get_epsilon(model, idisc["r"])

            # lambda at this distance
            idisc["lambda"] = [
                -(2.0 / 3.0) * weighted_alp2(m, idisc["dist"]) for m in [0, 1, 2]
            ]

            # Calculate the factor
            extra = [idisc["epsilon"] * idisc["lambda"][m] for m in [0, 1, 2]]

            # The surface must be treated differently due to TauP indexing constraints
            if idisc["d"] != 0.0 and idisc["ind"] != 0:

                # Depths before and after interactions
                dep0 = arrival.path[idisc["ind"] - 1][3]
                dep1 = arrival.path[idisc["ind"]][3]
                dep2 = arrival.path[idisc["ind"] + 1][3]

                # Direction before interaction
                if dep0 < dep1:
                    idisc["pre"] = "down"
                elif dep0 == dep1:
                    idisc["pre"] = "diff"
                else:
                    idisc["pre"] = "up"

                # Direction after interaction
                if dep1 < dep2:
                    idisc["post"] = "down"
                elif dep1 == dep2:
                    idisc["post"] = "diff"
                else:
                    idisc["post"] = "up"

                # Reflection or transmission
                if idisc["pre"] == idisc["post"]:
                    idisc["type"] = "trans"
                elif "diff" in [idisc["pre"], idisc["post"]]:
                    idisc["type"] = "diff"
                else:
                    idisc["type"] = "refl"

                # Phase before and after
                idisc["ph_pre"] = waves[i - 1][-1]
                idisc["ph_post"] = waves[i][0]

                # Deal with a transmission case
                if idisc["type"] == "trans":

                    # Phase above
                    if idisc["pre"] == "down":
                        idisc["ph_above"] = idisc["ph_pre"]
                        idisc["ph_below"] = idisc["ph_post"]
                    elif idisc["pre"] == "up":
                        idisc["ph_above"] = idisc["ph_post"]
                        idisc["ph_below"] = idisc["ph_pre"]

                    # Velocity above and below discontinuity
                    idisc["v0"] = (
                        v_mod.evaluate_above(idisc["d"] / 1e3, idisc["ph_above"])[0]
                        * 1e3
                    )
                    idisc["v1"] = (
                        v_mod.evaluate_below(idisc["d"] / 1e3, idisc["ph_below"])[0]
                        * 1e3
                    )

                    # eta above and below discontinuity
                    idisc["eta0"] = idisc["r"] / idisc["v0"]
                    idisc["eta1"] = idisc["r"] / idisc["v1"]

                    # Evaluate the time difference
                    eva = -(
                        np.sqrt(idisc["eta0"] ** 2 - idisc["p"] ** 2)
                        - np.sqrt(idisc["eta1"] ** 2 - idisc["p"] ** 2)
                    )

                # Deal with an underside reflection case
                if idisc["type"] == "refl" and idisc["pre"] == "up":

                    # Velocity below discontinuity
                    idisc["v0"] = (
                        v_mod.evaluate_below(idisc["d"] / 1e3, idisc["ph_pre"])[0] * 1e3
                    )
                    idisc["v1"] = (
                        v_mod.evaluate_below(idisc["d"] / 1e3, idisc["ph_post"])[0]
                        * 1e3
                    )

                    # eta below discontinuity
                    idisc["eta0"] = idisc["r"] / idisc["v0"]
                    idisc["eta1"] = idisc["r"] / idisc["v1"]

                    # Evaluate the time difference
                    eva = np.sqrt(idisc["eta0"] ** 2 - idisc["p"] ** 2) + np.sqrt(
                        idisc["eta1"] ** 2 - idisc["p"] ** 2
                    )

                # Deal with a topside reflection case
                if idisc["type"] == "refl" and idisc["pre"] == "down":

                    # Velocity above discontinuity
                    idisc["v0"] = (
                        v_mod.evaluate_above(idisc["d"] / 1e3, idisc["ph_pre"])[0] * 1e3
                    )
                    idisc["v1"] = (
                        v_mod.evaluate_above(idisc["d"] / 1e3, idisc["ph_post"])[0]
                        * 1e3
                    )

                    # eta above discontinuity
                    idisc["eta0"] = idisc["r"] / idisc["v0"]
                    idisc["eta1"] = idisc["r"] / idisc["v1"]

                    # Evaluate the time difference
                    eva = -(
                        np.sqrt(idisc["eta0"] ** 2 - idisc["p"] ** 2)
                        + np.sqrt(idisc["eta1"] ** 2 - idisc["p"] ** 2)
                    )

            # Deal with source depth and also end point
            elif idisc["ind"] == 0 or idisc["ind"] == len(arrival.path) - 1:

                # Assign wave type
                if idisc["ind"] == 0:
                    wave = waves[0][0]
                elif idisc["ind"] == len(arrival.path) - 1:
                    # wave = waves[max(list(paths.keys())) - 1][-1]
                    ## ALERT - may have a broken something here in change from dict to list?
                    ## Stuart, check this
                    wave = waves[-1][-1]

                # Deal with phases that start with an upgoing segment
                if arrival.name[0] in ["p", "s"] and idisc["ind"] == 0:

                    # Velocity above source
                    idisc["v1"] = v_mod.evaluate_above(idisc["d"] / 1e3, wave)[0] * 1e3

                    # eta above source
                    idisc["eta1"] = idisc["r"] / idisc["v1"]

                    # Evaluate the time difference
                    eva = -np.sqrt(idisc["eta1"] ** 2 - idisc["p"] ** 2)

                # Deal with ending the ray path at the surface
                else:
                    # Velocity below surface
                    idisc["v1"] = v_mod.evaluate_below(idisc["d"] / 1e3, wave)[0] * 1e3

                    # eta below surface
                    idisc["eta1"] = idisc["r"] / idisc["v1"]

                    # Evaluate the time difference
                    eva = (-1.0) * (
                        0 - np.sqrt(idisc["eta1"] ** 2 - idisc["p"] ** 2)
                    )  ## ALERT -- looks odd, just write + ?

            # Deal with surface reflection
            elif idisc["d"] == 0.0:

                # Assign type of interaction
                idisc["type"] = "refl"

                # Phase before and after
                idisc["ph_pre"] = waves[i - 1][-1]
                idisc["ph_post"] = waves[i][0]

                # Velocity below surface
                idisc["v0"] = (
                    v_mod.evaluate_below(idisc["d"] / 1e3, idisc["ph_pre"])[0] * 1e3
                )
                idisc["v1"] = (
                    v_mod.evaluate_below(idisc["d"] / 1e3, idisc["ph_post"])[0] * 1e3
                )

                # eta below surface
                idisc["eta0"] = idisc["r"] / idisc["v0"]
                idisc["eta1"] = idisc["r"] / idisc["v1"]

                # Evaluate time difference
                eva = np.sqrt(idisc["eta0"] ** 2 - idisc["p"] ** 2) + np.sqrt(
                    idisc["eta1"] ** 2 - idisc["p"] ** 2
                )

            # Output coefficients for this discontinuity
            idisc["sigma"] = [extra[m] * eva for m in [0, 1, 2]]

    # Sum the contribution to the coefficients from discontinuities
    disc_sigma = [
        np.sum([idisc["sigma"][m] for idisc in idiscs if idisc["yn"]])
        for m in [0, 1, 2]
    ]

    return disc_sigma


def centre_of_planet_coefficients(arrival, model):
    """
    Returns coefficients when an arrival passes too close to the centre of the Earth.
    When a ray passes very close to the centre of the Earth there is a step in distance which is problematic.
    In this case then interpolate the coefficients for two nearby arrivals.

    Inputs:
        arrival - TauP arrival object
        model - obspy.taup.tau_model.TauModel object

    Output:
        List of three floats, approximate ellipticity coefficients for the arrival
    """

    # ALERT -- feels like there should be a better way of doing this, without recalculating arrivals

    # Get two arrivals that do not go so close to the centre of the planet
    arrival1 = get_correct_taup_arrival(arrival, model, extra_distance=-0.05)
    arrival2 = get_correct_taup_arrival(arrival, model, extra_distance=-0.10)

    # Get the corrections for these arrivals
    coeffs1 = individual_ellipticity_coefficients(arrival1, model)
    coeffs2 = individual_ellipticity_coefficients(arrival2, model)

    # Linearly interpolate each coefficient to get final coefficients
    coeffs = [
        (
            c1
            + (
                (arrival.distance - arrival1.distance)
                / (arrival2.distance - arrival1.distance)
            )
            * (c2 - c1)
        )
        for c1, c2 in zip(coeffs1, coeffs2)
    ]

    return coeffs


## ALERT -- I don't think we should calculate taup arrivals in this code. Can code below be removed somehow?

# Define Exception
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
        super().__init__(self.message)

    def __str__(self):
        return self.message


def get_taup_arrival(phase, distance, source_depth, arrival_index, model):
    """
    Returns a TauP arrival object for the given phase, distance, depth and velocity model

    Inputs:
        phase - string, TauP phase name
        distance - float, epicentral distance in degrees
        source_depth  - float, source depth in km
        arrival_index - int, the index of the desired arrival, starting from 0
        model - obspy.taup.tau_model.TauModel object

    Output:
        TauP arrival object
    """

    # Get the taup arrival for this phase
    # arrivals = model.get_ray_paths(
    # source_depth_in_km=source_depth,
    # distance_in_degree=distance,
    # phase_list=[phase],
    # receiver_depth_in_km=0.0,
    # )
    from obspy.taup.taup_path import TauPPath
    from obspy.taup.tau import Arrivals

    rp = TauPPath(model, [phase], source_depth, distance, 0.0)
    rp.run()
    arrivals = Arrivals(sorted(rp.arrivals, key=lambda x: x.time), model=model)

    arrivals = [x for x in arrivals if abs(x.purist_distance - distance) < 0.0001]
    if len(arrivals) == 0:
        vel_model_name = str(model.s_mod.v_mod.model_name)
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
        model - obspy.taup.tau_model.TauModel object
        extra_distance - float, any further distance than the inputted arrival
            to obtain the new arrival

    Output:
        TauP arrival object
    """

    # Get arrival with the same ray parameter as the input arrival
    # new_arrivals = model.get_ray_paths(
    # source_depth_in_km=arrival.source_depth,
    # distance_in_degree=arrival.distance + extra_distance,
    # phase_list=[arrival.name],
    # receiver_depth_in_km=0.0,
    # )
    from obspy.taup.taup_path import TauPPath
    from obspy.taup.tau import Arrivals

    rp = TauPPath(
        model,
        [arrival.name],
        arrival.source_depth,
        arrival.distance + extra_distance,
        0.0,
    )
    rp.run()
    new_arrivals = Arrivals(sorted(rp.arrivals, key=lambda x: x.time), model=model)

    index = np.array(
        [abs(x.ray_param - arrival.ray_param) for x in new_arrivals]
    ).argmin()
    new_arrival = new_arrivals[index]
    return new_arrival
