# Copyright (C) 2022 Stuart Russell
"""
This file contains the functions to support the calculation of ellipticity
corrections. All functions in this file are called by the main functions
in the main file.
"""

# Import modules
import warnings
import obspy
import numpy as np
from scipy.integrate import cumtrapz

# Constants
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
        return norm * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    if m == 1:
        return norm * 3.0 * np.cos(theta) * np.sin(theta)
    if m == 2:
        return norm * 3.0 * np.sin(theta) ** 2
    raise ValueError("Invalid value of m")


def ellipticity_coefficients(arrivals, model=None, lod=EARTH_LOD):
    """
    Returns ellipticity coefficients for a set of arrivals

    Inputs:
        arrivals - a TauP Arrivals object
        model - obspy.taup.tau.TauPyModel or obspy.taup.tau_model.TauModel object
        lod - float, length of day of the model in seconds, only needed if calculating
            coefficients for a new model

    Output:
        list of lists of three floats, ellipticity coefficients
    """

    if model is None:
        model = arrivals.model

    return [individual_ellipticity_coefficients(arr, model, lod) for arr in arrivals]


def individual_ellipticity_coefficients(arrival, model, lod=EARTH_LOD):
    """
    Returns ellipticity coefficients for a given ray path

    Inputs:
        arrival - a TauP Arrival object
        model - obspy.taup.tau.TauPyModel or obspy.taup.tau_model.TauModel object
        lod - float, length of day of the model in seconds, only needed if calculating
            coefficients for a new model

    Output:
        list of three floats, ellipticity coefficients
    """

    if isinstance(model, obspy.taup.tau.TauPyModel):
        model = model.model
    if not isinstance(model, obspy.taup.tau_model.TauModel):
        raise TypeError("Velocity model not correct type")

    # Calculate epsilon values if they don't already exist
    if not hasattr(model.s_mod.v_mod, "epsilon"):
        model_epsilon(model, lod)

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

    # Split the path at discontinuities, bottoming depth and source depth
    waves = []
    paths = []
    count = -1
    for i, point in enumerate(arrival.path):
        depth = point[3]
        if (
            depth in discs  # Is at a discontinuity
            or depth == bot_dep  # or at the bottoming depth
            or depth == arrival.source_depth # or the source depth (beginning)
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
        p = arrival.ray_param
        if abs(p) > 1e-8:  # some tolerance, not sure what to set
            integral = [
                np.trapz((eta**3.0) * dvdr * epsilon * lamda[m], x=distance)
                / arrival.ray_param
                for m in [0, 1, 2]
            ]
        else:
            # When ray parameter close to vertical, switch to doing a radial integral
            sign = np.sign(radius[-1] - radius[0])
            prefactor = (
                1 + 0.5 * (p / eta) ** 2
            )  # binomial approximation of eta / np.sqrt(eta**2 - p**2)
            integral = [
                np.trapz(
                    sign * prefactor * (v**-2) * dvdr * radius * epsilon * lamda[m],
                    x=radius,
                )
                for m in [0, 1, 2]
            ]

        seg_ray_sigma.append(integral)

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

    # Including the bottoming and source depths allows cross indexing with the paths variable when
    # the start point is not the lowest point on the ray path
    assess_discs = np.sort(np.append(discs, (bot_dep, arrival.source_depth)))

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
            
        elif (idisc["d"] == bot_dep * 1e3 or idisc["d"] == arrival.source_depth * 1e3) and idisc["d"] * 1e-3 not in discs and idisc["ind"] != 0:
            idisc["yn"] = False

        # Calculate if this is a true discontinuity
        else:
            idisc["yn"] = True

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


def correction_from_coefficients(coefficients, azimuth, source_latitude):
    """Obtain an ellipticity correction given the coefficients."""

    # Convert latitude to colatitude
    colatitude = np.radians(90 - source_latitude)

    # Convert azimuth to radians
    azimuth = np.radians(azimuth)

    return sum(
        coefficients[m] * weighted_alp2(m, colatitude) * np.cos(m * azimuth)
        for m in [0, 1, 2]
    )
