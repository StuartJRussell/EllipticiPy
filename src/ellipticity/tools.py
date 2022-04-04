# Copyright (C) 2022 Stuart Russell
"""
This file contains the functions to support the calculation of ellipticity
corrections. All functions in this file are called by the main functions
in the main file.
"""

import numpy as np
from scipy.integrate import cumtrapz

import obspy

# Constants
EARTH_LOD = 86164.0905  # s, length of day
G = 6.67408e-11  # m^3 kg^-1 s^-2, universal gravitational constant


def model_epsilon(model, lod=EARTH_LOD):
    """
    Calculates a profile of ellipticity of figure (epsilon) through a planetary model.

    :param model: The tau model object
    :type model: :class:`obspy.taup.tau_model.TauModel`
    :param lod: length of day in seconds. Defaults to Earth value
    :type lod: float
    :returns: Adds arrays of epsilon at top and bottom of each velocity layer as
        attributes model.s_mod.v_mod.top_epsilon and model.s_mod.v_mod.bot_epsilon
    """

    # Angular velocity of model
    omega = 2 * np.pi / lod  # s^-1

    # Radius of planet in m
    a = model.radius_of_planet * 1e3

    # Depth and density information of velocity layers
    v_mod = model.s_mod.v_mod  # velocity_model
    top_depth = v_mod.layers["top_depth"][::-1] * 1e3  # in m
    bot_depth = v_mod.layers["bot_depth"][::-1] * 1e3  # in m
    top_density = v_mod.layers["top_density"][::-1] * 1e3  # in kg m^-3
    bot_density = v_mod.layers["bot_density"][::-1] * 1e3  # in kg m^-3
    top_radius = a - top_depth
    bot_radius = a - bot_depth

    # Mass within each spherical shell by trapezoidal rule
    top_volume = (4.0 / 3.0) * np.pi * top_radius**3
    volume = np.zeros(len(top_depth) + 1)
    volume[1:] = top_volume
    d_volume = volume[1:] - volume[:-1]
    d_mass = 0.5 * (bot_density + top_density) * d_volume
    mass = np.cumsum(d_mass)

    total_mass = mass[-1]

    # Moment of inertia of each spherical shell by trapezoidal rule
    j_top = (8.0 / 15.0) * np.pi * top_radius**5
    j = np.zeros(len(top_depth) + 1)
    j[1:] = j_top
    d_j = j[1:] - j[:-1]
    d_inertia = 0.5 * (bot_density + top_density) * d_j

    moment_of_inertia = np.cumsum(d_inertia)

    # Calculate y (moment of inertia factor) for surfaces within the body
    y = moment_of_inertia / (mass * top_radius**2)

    # Calculate Radau's parameter
    radau = 6.25 * (1 - 3 * y / 2) ** 2 - 1

    # Calculate h
    # Ratio of centrifugal force and gravity for a particle on the equator at the surface
    ha = (a**3 * omega**2) / (G * total_mass)

    # epsilon at surface
    epsilona = (5 * ha) / (2 * radau[-1] + 4)

    # Solve the differential equation
    epsilon = np.exp(cumtrapz(radau / top_radius, x=top_radius, initial=0.0))
    epsilon = epsilona * epsilon / epsilon[-1]

    # Output as model attributes
    r = np.zeros(len(top_radius) + 1)
    r[1:] = top_radius
    epsilon = np.insert(epsilon, 0, epsilon[0])  # add a centre of planet value
    v_mod.top_epsilon = epsilon[::-1][:-1]
    v_mod.bot_epsilon = epsilon[::-1][1:]


def get_epsilon(model, depth):
    """
    Gets ellipticity of figure for a model at a specified depth.

    :param model: The tau model object
    :type model: :class:`obspy.taup.tau_model.TauModel`
    :param depth: depth in km
    :type depth: float
    :returns: value of epsilon, ellipticity of figure
    :rtype: float
    """

    v_mod = model.s_mod.v_mod
    if depth > 0.0:
        layer_idx = v_mod.layer_number_above(depth)[0]
    else:
        layer_idx = v_mod.layer_number_below(depth)[0]

    layer = v_mod.layers[layer_idx]
    thick = layer["bot_depth"] - layer["top_depth"]
    bot_eps = v_mod.bot_epsilon[layer_idx]
    top_eps = v_mod.top_epsilon[layer_idx]
    slope = (bot_eps - top_eps) / thick
    return slope * (depth - layer["top_depth"]) + top_eps


def evaluate_derivative_below(model, depth, prop):
    """
    Evaluate depth derivative of material property at bottom of a velocity layer.
    """

    layer = model.layers[model.layer_number_below(depth)]
    return evaluate_derivative_at(layer, prop)


def evaluate_derivative_above(model, depth, prop):
    """
    Evaluate depth derivative of material property at top of a velocity layer.
    """

    layer = model.layers[model.layer_number_above(depth)]
    return evaluate_derivative_at(layer, prop)


def evaluate_derivative_at(layer, prop):
    """
    Evaluate depth derivative of material property in a velocity layer.
    """

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
    The weighted degree 2 associated Legendre polynomial for a given order and value.

    :param m: order of polynomial (0, 1, or 2)
    :type m: int
    :param theta: angle
    :type theta: float
    :returns: value of weighted associated Legendre polynomial of degree 2
        and order m at x = cos(theta)
    :rtype: float
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
    Ellipticity coefficients for a set of arrivals.

    :param arrivals: TauP Arrivals object with ray paths calculated.
    :type arrivals: :class:`obspy.taup.tau.Arrivals`
    :param model: optional, model used to calculate the arrivals
    :type model: :class:`obspy.taup.tau.TauPyModel` or
                 :class:`obspy.taup.tau_model.TauModel`
    :param lod: optional, length of day in seconds. Defaults to Earth value
    :type lod: float
    :returns: list of lists of three floats, ellipticity coefficients
    :rtype: list[list]

    Usage:

    >>> from obspy.taup import TauPyModel
    >>> from ellipticity.tools import ellipticity_coefficients
    >>> model = TauPyModel('prem')
    >>> arrivals = model.get_ray_paths(source_depth_in_km = 124,
        distance_in_degree = 65, phase_list = ['pPKiKP'])
    >>> ellipticity_coefficients(arrivals)
    [[-0.9323682254592675, -0.6888598392172868, -0.8824096866702915]]
    """

    if model is None:
        model = arrivals.model

    return [individual_ellipticity_coefficients(arr, model, lod) for arr in arrivals]


def individual_ellipticity_coefficients(arrival, model, lod=EARTH_LOD):
    """
    Ellipticity coefficients for a single ray path.

    :param arrivals: TauP Arrival object with ray path calculated.
    :type arrivals: :class:`obspy.taup.helper_classes.Arrival`
    :param model: Tau model used to calculate the arrival
    :type model: :class:`obspy.taup.tau_model.TauModel`
    :param lod: optional, length of day in seconds. Defaults to Earth value
    :type lod: float
    :returns: list of three floats, ellipticity coefficients
    :rtype: list
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
    """
    Split and label ray path according to type of wave.
    """

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

    # Label the paths that are diffracted rays
    diffracted = [np.all(p[:, 3] == p[0, 3]) for p in paths]

    # Get SeismicPhase object
    ph = arrival.phase

    # Get the branches of the TauModel
    branches = [
        (x.top_depth, x.bot_depth)
        for x in model.depth_correct(arrival.source_depth).tau_branches[0]
    ]

    # Get which branch is the the outer core branch
    oc_branch = branches.index((model.cmb_depth, model.iocb_depth))

    # Wave for each branch in sequence
    # ObsPy doesnt' always assign the correct wave type for the outer core, so enforce P wave
    wave_type = [
        wt if ph.branch_seq[i] != oc_branch else True
        for i, wt in enumerate(ph.wave_type)
    ]
    wave_type = ["p" if wt else "s" for wt in wave_type]

    waves = []
    j = 0
    for i, path in enumerate(paths):
        if diffracted[i]:
            # Set diffracted segments to p, but they won't make a contribution.
            waves.append("p")
        else:
            waves.append(wave_type[j])
            j = j + 1

    return paths, waves


def integral_coefficients(arrival, model):
    """
    Ellipticity coefficients due to integral along ray path.
    """

    # Split the ray path
    paths, waves = split_ray_path(arrival, model)

    # Loop through path segments
    sigmas = []
    for path, wave in zip(paths, waves):

        # Depth in km
        depth = path[:, 3]

        # Radius in km
        radius = model.radius_of_planet - depth

        # Velocity in km/s
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
                -evaluate_derivative_below(v_mod, d, wave)[0]
                if d != max(depth)
                else -evaluate_derivative_above(v_mod, d, wave)[0]
                for d in depth
            ]
        )

        # eta in s
        eta = radius / v

        # epsilon
        epsilon = np.array([get_epsilon(model, d) for d in depth])

        # Epicentral distance in radians
        distance = path[:, 2]

        # lambda
        lam = [-(2.0 / 3.0) * weighted_alp2(m, distance) for m in [0, 1, 2]]

        # vertical slowness
        y = eta**2 - arrival.ray_param**2
        sign = np.sign(y[-1] - y[0])  # figure out if going up or down
        vertical_slowness = np.sqrt(y * (y > 0))  # in s

        # Do the integration
        s = v**-1 * dvdr * radius
        integral = [
            np.trapz((s / (1.0 - s)) * epsilon * sign * lam[m], x=vertical_slowness)
            for m in [0, 1, 2]
        ]

        sigmas.append(integral)

    # Sum coefficients for each segment to get total ray path contribution
    return [np.sum([s[m] for s in sigmas]) for m in [0, 1, 2]]


def discontinuity_coefficients(arrival, model):
    """
    Ellipticity coefficients due to discontinuities.
    """

    # Split the ray path
    paths, waves = split_ray_path(arrival, model)

    # Velocity model
    v_mod = model.s_mod.v_mod

    # Points at which to evaluate discontinuity coefficients
    pierce_points = [p[0] for p in paths]
    pierce_points.append(paths[-1][-1, :])  # the receiver point

    sigmas = []
    for i, point in enumerate(pierce_points):
        ray_param = point[0]
        distance = point[2]
        depth = point[3]
        radius = model.radius_of_planet - depth

        # Depth and phases pre, source is a special case
        if i == 0:
            # source
            depth_pre = depth
            phase_pre = waves[i]
        else:
            depth_pre = paths[i - 1][-2, 3]
            phase_pre = waves[i - 1]

        # Depth and phases post, receiver is a special case
        if i == len(pierce_points) - 1:
            # receiver
            depth_post = depth
            phase_post = waves[i - 1]
        else:
            depth_post = paths[i][1, 3]
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

            # above/below sign, positive if above
            sign = np.sign(depth - depth_p)

            return -sign * vertical_slowness

        pre = f(depth_pre, phase_pre)
        post = f(depth_post, phase_post)

        # epsilon at this depth
        epsilon = get_epsilon(model, depth)

        # lambda at this distance
        lam = [-(2.0 / 3.0) * weighted_alp2(m, distance) for m in [0, 1, 2]]

        # coefficients for this discontinuity
        sigmas.append([epsilon * lam[m] * (pre + post) for m in [0, 1, 2]])

    # Sum the coefficients from all discontinuities
    disc_sigma = [np.sum([s[m] for s in sigmas]) for m in [0, 1, 2]]

    return disc_sigma


def correction_from_coefficients(coefficients, azimuth, source_latitude):
    """
    Ellipticity correction given the ellipticity coefficients.
    """

    # Convert latitude to colatitude
    colatitude = np.radians(90 - source_latitude)

    # Convert azimuth to radians
    azimuth = np.radians(azimuth)

    return sum(
        coefficients[m] * weighted_alp2(m, colatitude) * np.cos(m * azimuth)
        for m in [0, 1, 2]
    )
