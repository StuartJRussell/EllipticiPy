# Copyright (C) 2022 Stuart Russell
"""
This file contains the functions to support the calculation of ellipticity
corrections. All functions in this file are called by the main functions
in the main file.
"""

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
        dr - float, step length in m for discretization

    Output:
        Adds arrays of epsilon and radius to the model instance as attributes
        model.s_mod.v_mod.epsilon and model.s_mod.v_mod.epsilon_r
    """

    # Angular velocity of model
    Omega = 2 * np.pi / lod  # s^-1

    # Radius of planet in m
    a = model.radius_of_planet * 1e3

    # Radii in m to evaluate integrals
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
    v_mod.epsilon_r = r / 1e3  # convert to km
    v_mod.epsilon = epsilon


def get_epsilon(model, radius):
    """
    Gets the value of epsilon for a model at a specified radius

    Inputs:
        model - obspy.taup.tau_model.TauModel object
        radius - float, radius in km

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
        radius - float, radius in km
        wave - str, wave type: 'p' or 's'

    Output:
        float, value of dv/dr
    """

    Re = model.radius_of_planet
    v_mod = model.s_mod.v_mod
    return -evaluate_derivative_below(v_mod, Re - radius, wave)[0]


def get_dvdr_above(model, radius, wave):
    """
    Gets the value of dv/dr for a model immediately above a specified radius

    Inputs:
        model - obspy.taup.tau_model.TauModel object
        radius - float, radius in km
        wave - str, wave type: 'p' or 's'

    Output:
        float, value of dv/dr
    """

    Re = model.radius_of_planet
    v_mod = model.s_mod.v_mod
    return -evaluate_derivative_above(v_mod, Re - radius, wave)[0]


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

    # Split the path at discontinuities, bottoming depth and source depth
    paths = []
    count = -1
    for i, point in enumerate(arrival.path):
        depth = point[3]
        if (
            depth in discs  # Is at a discontinuity
            or depth == bot_dep  # or at the bottoming depth
            or depth == arrival.source_depth  # or the source depth (beginning)
        ) and i != len(
            arrival.path
        ) - 1:  # not at the end
            count = count + 1
            paths.append([])
            if count != 0:
                paths[count - 1].append(list(point))
        paths[count].append(list(point))
    paths = [np.array(path) for path in paths]

    # Remove path segments with zero change in distance or zero change in depth
    # Except if the ray parameter is zero as these phases are vertical
    paths = [
        x
        for x in paths
        if len(x) > 2
        or not ((x[0][2] == x[1][2] and arrival.ray_param != 0.0) or x[0][3] == x[1][3])
    ]

    # Get SeismicPhase object
    ph = arrival.phase

    # Get the branches of the TauModel
    branches = [
        (x.top_depth, x.bot_depth)
        for x in model.depth_correct(arrival.source_depth).tau_branches[0]
    ]

    # Get which branch is the the outer core branch
    OC_branch = branches.index((model.cmb_depth, model.iocb_depth))

    # Wave for each branch in sequence
    # ObsPy doesnt' always assign the correct wave type for the outer core, so enforce P wave
    branch_seq = ph.branch_seq
    wave_type = [
        ph.wave_type[i] if branch_seq[i] != OC_branch else True
        for i in range(len(ph.wave_type))
    ]

    # Make arrays of wave types
    waves = ["p" if wave_type[i] else "s" for i in range(len(paths))]

    return paths, waves


def integral_coefficients(arrival, model):
    """Calculate correction coefficients due to integral along ray path"""

    # Radius of Earth
    Re = model.radius_of_planet

    # Split the ray path
    paths, waves = split_ray_path(arrival, model)

    # Loop through path segments
    seg_ray_sigma = []
    for path, wave in zip(paths, waves):

        # Depth in km
        depth = path[:, 3]

        # Radius in km
        radius = Re - depth

        # Velocity in m/s
        v_mod = model.s_mod.v_mod
        v = np.array(
            [
                v_mod.evaluate_below(d, wave)[0]
                if d != max(depth)
                else v_mod.evaluate_above(d, wave)[0]
                for d in depth
            ]
        )

        # Gradient of v wrt r in s^-1
        dvdr = np.array(
            [
                get_dvdr_below(model, r, wave)
                if r != min(radius)
                else get_dvdr_above(model, r, wave)
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

        # Calculate vertical slowness
        y = eta**2 - arrival.ray_param**2
        sign = np.sign(y[-1] - y[0])  # figure out if going up or down
        vertical_slowness = np.sqrt(y * (y > 0))  # in s

        # Do the integration
        s = v**-1 * dvdr * radius
        integral = [
            np.trapz((s / (1.0 - s)) * epsilon * sign * lamda[m], x=vertical_slowness)
            for m in [0, 1, 2]
        ]

        seg_ray_sigma.append(integral)

    # Sum coefficients for each segment to get total ray path contribution
    return [np.sum([s[m] for s in seg_ray_sigma]) for m in [0, 1, 2]]


def discontinuity_coefficients(arrival, model):
    """Calculate correction coefficients due to discontinuities"""

    # Split the ray path
    paths, waves = split_ray_path(arrival, model)

    # Continuous ray path
    arrival_path = np.vstack(
        [paths[i][:-1] if i != len(paths) - 1 else paths[i] for i in range(len(paths))]
    )

    # Bottoming depth of ray
    bot_dep = max([point[3] for point in arrival.path])

    # Get discontinuities in the model
    v_mod = model.s_mod.v_mod  # velocity model
    discs = v_mod.get_discontinuity_depths()[:-1]

    # Including the bottoming and source depths allows cross indexing with the paths variable when
    # the start point is not the lowest point on the ray path
    assess_discs = np.append(discs, (bot_dep, arrival.source_depth, arrival.receiver_depth))

    # Get which discontinuities the phase interacts with, include bottoming depth to allow
    # cross indexing with the paths variable
    ids = [
        (i, point[3], point[2])  # index, depth, distance
        for i, point in enumerate(arrival_path)
        if point[3] in assess_discs
    ]

    ray_param = arrival_path[0][0]
    sigmas = []
    for i, x in enumerate(ids):
        ind = x[0]
        depth = x[1]
        distance = x[2]
        radius = model.radius_of_planet - depth

        # Depth and phases pre, source is a special case
        if ind == 0:
            depth_pre = depth
            phase_pre = waves[i]
        else:
            depth_pre = arrival_path[ind - 1][3]
            phase_pre = waves[i - 1]

        # Depth and phases post, receiver is a special case
        if ind == len(arrival_path) - 1:
            depth_post = depth
            phase_post = waves[i - 1]
        else:
            depth_post = arrival_path[ind + 1][3]
            phase_post = waves[i]

        def f(depth_p, phase_p):
            if depth_p >= depth:
                v = v_mod.evaluate_below(depth, phase_p)[0]
            else:
                v = v_mod.evaluate_above(depth, phase_p)[0]

            eta = radius / v
            y = eta**2 - ray_param**2
            vertical_slowness = np.sqrt(y * (y > 0))

            if depth_p == depth:
                vertical_slowness = 0.0

            # above/below signs, positive if above
            sign = np.sign(depth - depth_p)
            return -sign * vertical_slowness

        pre = f(depth_pre, phase_pre)
        post = f(depth_post, phase_post)

        # epsilon at this depth
        epsilon = get_epsilon(model, radius)

        # lambda at this distance
        lamda = [-(2.0 / 3.0) * weighted_alp2(m, distance) for m in [0, 1, 2]]

        # Output coefficients for this discontinuity
        sigmas.append([epsilon * lamda[m] * (pre + post) for m in [0, 1, 2]])

    # Sum the contribution to the coefficients from discontinuities
    disc_sigma = [np.sum([s[m] for s in sigmas]) for m in [0, 1, 2]]

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
