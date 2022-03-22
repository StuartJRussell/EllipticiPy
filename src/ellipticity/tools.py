#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Stuart Russell
# Created Date: March 2022
# version ='1.0'
# ---------------------------------------------------------------------------
'''
This file contains the functions to support the calculation of ellipticity
corrections. All functions in this file are called by the main functions
in the main file.
'''
# ---------------------------------------------------------------------------
#Import modules
import os
import obspy
import scipy
import warnings
import numpy as np
from obspy.taup import TauPyModel
# ---------------------------------------------------------------------------
#Suppress warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', message = 'Resizing a TaP array inplace failed due to the existence of other references to the array, creating a new array. See Obspy #2280.')
# ---------------------------------------------------------------------------

#Define Exceptions       
class PhaseError(Exception):
    '''
    Class for handing exception of when there is no phase arrival for the inputted geometry.
    '''
    def __init__(self, phase, vel_model):
        self.message = 'Phase ' + phase + ' does not arrive at specified distance in model ' + vel_model
    def __str__(self):
        return self.message
        
def kron0(j):
    '''
    Kronecker delta in the case of i = 0
    '''
    
    if j == 0:
        out = 1
    else:
        out = 0
    
    return out

def factor(m):

    '''
    Returns the weighting for the associated legendre polynomial
    
    Inputs:
        m - int, order of polynomial
        
    Output:
        out - float, value of the weighting
    '''
    
    fact = np.sqrt((2 - kron0(m))*(np.math.factorial(2 - m)/np.math.factorial(2 + m)))
    
    return fact

def alp2(m, x):

    '''
    Returns the degree 2 associated legendre polynomial for a given order and value.
    
    Inputs:
        m - int, order of polynomial
        x - float, value to calculate at
        
    Output:
        out - float, value of associated legendre polynomial of degree 2 and order m at value x
    '''
    
    if m == 0:
        out = 0.5*(3.*np.cos(x)**2. - 1.)
    elif m == 1:
        out = (3.)*np.cos(x)*np.sin(x)
    elif m ==2:
        out = 3.*np.sin(x)**2.
    
    return out

def calculate_model_dvdr(model):

    '''
    Calculates the rate of change of velocity with radius for a given velocity model.
    
    Inputs:
        model - TauPyModel
        
    Output:
        adds arrays of dv/dr and radius to the model instance as attributes.
    '''
    
    #Radius of planet
    Re = model.model.radius_of_planet
    
    #Discontinuities in the model
    discs = model.model.s_mod.v_mod.get_discontinuity_depths()
    
    #Depths between discontinuities
    deps = [np.linspace(discs[i]*1e3, discs[i+1]*1e3, int(np.floor((discs[i+1]*1e3 - discs[i]*1e3)/1000))) for i in range(len(discs)-1)]
    if deps[-1][-1] != model.model.radius_of_planet*1e3:
        deps[-1] = np.append(deps[-1], model.model.radius_of_planet*1e3)
    #Radii
    r = [Re*1e3 - np.array(deps[i]) for i in range(len(deps))]
    #Velocities
    v = {ph:[np.append(model.model.s_mod.v_mod.evaluate_below(deps[i][:-1]/1e3, ph)*1e3, model.model.s_mod.v_mod.evaluate_above(deps[i][-1]/1e3, ph)*1e3)
            for i in range(len(deps))] for ph in ['p','s']}
    #Steps in velocity
    dv = {ph:[np.gradient(v[ph][i]) for i in range(len(deps))] for ph in ['p','s']}
    #Steps in radius
    dr = [np.gradient(r[i]) for i in range(len(deps))]
    
    #Calculate and output dvdr and corresponding radii
    model.model.s_mod.v_mod.dvdr = {ph:np.flip(np.array(sum([list(dv[ph][i]/dr[i]) for i in range(len(deps))],[])), axis = 0) for ph in ['p','s']}
    model.model.s_mod.v_mod.dvdr_r = np.flip(np.array(sum([list(x) for x in r],[])), axis = 0)

def get_model_epsilon(model, lod):

    '''
    Get the values of epsilon with radius from a pre-written file, or calculates if the file doesn't exist.
    
    Inputs:
        model - TauPyModel
        lod - float, length of day of the model in seconds. Only needed if this the first time that a model is used.
        
    Output:
        adds arrays of epsilon and radius to the model instance as attributes.
    '''
    
    #Get velocity model name
    if "'" in model.model.s_mod.v_mod.model_name:
        vel_model = model.model.s_mod.v_mod.model_name.split("'")[1]
    else:
        vel_model = model.model.s_mod.v_mod.model_name
    
    #See if epsilon values are already calculated and if not then inform the user
    e_file = '/'.join(__file__.split('/')[:-1]) + '/epsilon/epsilon_' + vel_model.split('/')[-1].split('.')[0] + '.txt'
    if not os.path.exists(e_file):
        print('Epsilon not calculated for this model. Calculating...')
        #Warn user if assuming Earth length of day
        if lod == 86164.0905:
            warnings.warn('Assuming Earth value of length of day for epsilon calculation.')
        #Calculate epsilon
        calculate_model_epsilon(model, e_file, lod)
    
    #Get the values of epsilon for this model from the file
    f = open(e_file, 'r')
    epsilon_list = [[float(y) for y in x.strip().split(',')] for x in f.readlines()]
    epsilon_r = np.array([x[0] for x in epsilon_list])
    epsilon_list = np.array([x[1] for x in epsilon_list])
    f.close()
    
    model.model.s_mod.v_mod.epsilon_r = epsilon_r
    model.model.s_mod.v_mod.epsilon = epsilon_list

def get_taup_arrival(phase, distance, source_depth, arrival_index, model):

    '''
    Returns a TauP arrival object for the given phase, distance, depth and velocity model.
    
    Inputs:
        phase - string, TauP phase name
        distance - float, epicentral distance in degrees
        source_depth  - float, source depth in km
        arrival_index - int, the index of the desired arrival, starting from 0
        model - TauPyModel object
        
    Output:
        TauP arrival object
    '''
    
    #Get the taup arrival for this phase
    arrivals = model.get_ray_paths(source_depth_in_km = source_depth, distance_in_degree = distance, phase_list = [phase], receiver_depth_in_km = 0.)
    arrivals = [x for x in arrivals if abs(x.purist_distance - distance) < 0.0001]
    if len(arrivals) == 0:
        if "'" in model.model.s_mod.v_mod.model_name:
            vel_model = model.model.s_mod.v_mod.model_name.split("'")[1]
        else:
            vel_model = model.model.s_mod.v_mod.model_name
        raise PhaseError(phase, vel_model)
    
    return arrivals[arrival_index]
    
def get_correct_taup_arrival(arrival, model, extra_distance = 0.):

    '''
    Returns a TauP arrival object in the correct form if the original is not
    
    Inputs:
        arrival - TauP arrival object
        model - TauPyModel object
        extra_distance - float, any further distance than the inputted arrival to obtain the new arrival
        
    Output:
        TauP arrival object
    '''
    
    #Get arrival with the same ray parameter as the input arrival
    new_arrivals = model.get_ray_paths(source_depth_in_km = arrival.source_depth, distance_in_degree = arrival.distance + extra_distance, 
                    phase_list = [arrival.name], receiver_depth_in_km = 0.)
    index = np.array([abs(x.ray_param - arrival.ray_param) for x in new_arrivals]).argmin()
    new_arrival = new_arrivals[index]
    return new_arrival

def centre_of_planet_coefficients(arrival, model):
    
    '''
    Returns coefficients when a aray passes too close to the centre of the Earth.
    When a ray passes very close to the centre of the Earth there is a step in distance which is problematic.
    In this case then interpolate the coefficients for two nearby arrivals.
    
    Inputs:
        arrival - TauP arrival object
        model - TauPyModel object
        
    Output:
        List fo three floats, aprroximate ellipticity coefficients for the inputted Arrival
    '''
    
    #Get two arrivals that do not go so close to the centre of the planet
    arrival1 = get_correct_taup_arrival(arrival, model, extra_distance = -0.05)
    arrival2 = get_correct_taup_arrival(arrival, model, extra_distance = -0.10)
    
    #Get the corrections for these arrivals
    coeffs1 = calculate_coefficients(arrival1, model)
    coeffs2 = calculate_coefficients(arrival2, model)
    
    #Linearly interpolate each coefficient to get final coefficients
    coeffs = [(coeffs1[i] + ((arrival.distance - arrival1.distance) / (arrival2.distance - arrival1.distance)) * (coeffs2[i] - coeffs1[i])) 
                    for i in range(len(coeffs1))]
    
    return coeffs
    
def list_coefficients(arrival, model, lod):
    
    #Get a correction for each arrival
    coeffs = [calculate_coefficients(arr, model, lod = lod) for arr in arrival]
    
    return coeffs

def get_epsilon(model, radius):

    '''
    Gets the value of epsilon for that model at a specified radius
    
    Inputs:
        model - TauPyModel object
        radius - float, radius  in m
        
    Output:
        float, value of epsilon
    '''
    
    #Epsilon and radii arrays
    epsilon = model.model.s_mod.v_mod.epsilon
    radii = model.model.s_mod.v_mod.epsilon_r
    
    #Get the nearest value of epsilon to the given radius
    idx = np.searchsorted(radii, radius, side = "left")
    if idx > 0 and (idx == len(radii) or np.math.fabs(radius - radii[idx - 1]) < np.math.fabs(radius - radii[idx])):
        return epsilon[idx - 1]
    else:
        return epsilon[idx]
        
def get_epsilon_above(model, radius):

    '''
    Gets the value of epsilon for that model immediately above a specified radius
    
    Inputs:
        model - TauPyModel object
        radius - float, radius  in m
        
    Output:
        float, value of epsilon
    '''
    
    out = get_epsilon(model, radius + 1)
    
    return out
    
def get_epsilon_below(model, radius):

    '''
    Gets the value of epsilon for that model immediately below a specified radius
    
    Inputs:
        model - TauPyModel object
        radius - float, radius  in m
        
    Output:
        float, value of epsilon
    '''
    
    out = get_epsilon(model, radius - 1)
    
    return out

def get_dvdr(model, radius, wave):

    '''
    Gets the value of dv/dr for that model at a specified radius
    
    Inputs:
        model - TauPyModel object
        radius - float, radius  in m
        wave - str, wave type: 'p' or 's'
        
    Output:
        float, value of dv/dr
    '''
    
    #dv/dr and radii
    dvdr = model.model.s_mod.v_mod.dvdr[wave]
    radii = model.model.s_mod.v_mod.dvdr_r
    
    #Get the nearest value of dv/dr to the given radius
    idx = np.searchsorted(radii, radius, side = "left")
    if idx > 0 and (idx == len(radii) or np.math.fabs(radius - radii[idx - 1]) < np.math.fabs(radius - radii[idx])):
        return dvdr[idx - 1]
    else:
        return dvdr[idx]
        
def get_dvdr_above(model, radius, wave):

    '''
    Gets the value of dv/dr for that model immediately above a specified radius
    
    Inputs:
        model - TauPyModel object
        radius - float, radius  in m
        wave - str, wave type: 'p' or 's'
        
    Output:
        float, value of dv/dr
    '''
    
    out = get_dvdr(model, radius + 1, wave)
    
    return out
    
def get_dvdr_below(model, radius, wave):

    '''
    Gets the value of dv/dr for that model immediately below a specified radius
    
    Inputs:
        model - TauPyModel object
        radius - float, radius  in m
        wave - str, wave type: 'p' or 's'
        
    Output:
        float, value of dv/dr
    '''
    
    out = get_dvdr(model, radius - 1, wave)
    
    return out

def calculate_coefficients(arrival, model, lod):

    '''
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
    '''
    
    ########################
    ##### Open 1D model ####
    ########################

    #If model is not initialised then do so
    if type(model) == str:
        model = TauPyModel(model = model)
    elif type(model) != obspy.taup.tau.TauPyModel:
        raise TypeError("Velocity model not correct type")
        
    #Name of model
    if "'" in model.model.s_mod.v_mod.model_name:
        vel_model = model.model.s_mod.v_mod.model_name.split("'")[1]
    else:
        vel_model = model.model.s_mod.v_mod.model_name
    
    #Caluclate dv/dr if it doesn't already exist
    if not hasattr(model.model.s_mod.v_mod, 'dvdr'):
        calculate_model_dvdr(model)

    #Radius of Earth
    Re = model.model.radius_of_planet

    #Get discontinuities in the model
    discs = model.model.s_mod.v_mod.get_discontinuity_depths()[:-1]
    discsr = [Re - x for x in discs]
    
    #Calculate epsilon values if they don't already exist
    if not hasattr(model.model.s_mod.v_mod, 'epsilon'):
        get_model_epsilon(model, lod)

    #########################
    ##### Get an arrival ####
    #########################
    
    #Check if arrival is a TauP object or a list and get arrival if needed
    if type(arrival) == list:
       
        #Call an arrival, this will error if it the phase input is unrealistic
        #Ideally users should use TauP Arrivals as inputs but some may not
        arrival = get_taup_arrival(arrival[0], arrival[1], arrival[2], arrival[3], model)
        
    elif type(arrival) == obspy.taup.helper_classes.Arrival and type(arrival.path) == type(None):
        
        #Call an arrival that has a ray path
        #Ideally users should use the ObsPy TauP get_ray_paths() to get their Arrivals, but if they ahven't then this will fix it
        warnings.warn('Arrival does not have ray path, in future please input the correct arrival for greater efficiency')
        arrival = get_correct_taup_arrival(arrival, model)
        
    #If ray parameter is zero then this is problematic, so adjust the ditance slightly
    if arrival.distance == 0.:
        
        #Call an arrival that has non-zero ray parameter
        #We can't integrate the ray when the ray parameter is zero, but the integral does converge when the distance is zero so just add a tiny bit of distance
        arrival = get_correct_taup_arrival(arrival, model, extra_distance = 1e-10)

    #Bottoming depth of ray
    bot_dep = max([x[3] for x in arrival.path])
        
    #When the ray goes close to the centre of the Earth, the distance function has a step in it
    #This is problmeatic to integrate along
    #Instead, if the ray goes within 50m of the centre of the planet, calculate for nearby two values and interpolate
    #This produces a satisfactory approximation
    if (model.model.radius_of_planet - bot_dep)*1e3 < 50:
        sigma = centre_of_planet_coefficients(arrival, model)
        
    else:
    
        ##########################
        ##### Set up ray path ####
        ##########################

        #Get wave types for each segment of ray path
        phase_name = arrival.name
        if 'diff' in phase_name:
            segments = phase_name.replace('diff',phase_name[phase_name.index('diff')-1])
        else:
            segments = phase_name
        segments = segments.replace('c','').replace('i','').replace('K','P').replace('I','P').replace('J','S')
        letter = 0
        waves = []
        for i in range(len(arrival.path)):
            if (i != 0 and i != len(arrival.path)-1 and arrival.path[i][3] in [0., model.model.cmb_depth, model.model.iocb_depth]
                and arrival.path[i][3] != arrival.path[i-1][3]):
                letter = letter + 1
            waves.append(segments[letter].lower())

        #Split the path at discontinuities and bottoming depth
        wave = {}
        paths = {}
        count = -1
        for i in range(len(arrival.path)):
            if (i == 0 or arrival.path[i][3] in discs or arrival.path[i][3] == bot_dep) and i != len(arrival.path)-1:
                count = count + 1
                paths[count] = []
                wave[count] = []
                if count != 0:
                    paths[count-1].append(list(arrival.path[i]))
                    wave[count-1].append(waves[i-1])
            paths[count].append(list(arrival.path[i]))
            wave[count].append(waves[i])
        paths = {x:np.array(paths[x]) for x in paths}
        wave_paths = {x:np.array(wave[x]) for x in paths}

        #############################
        ##### Integrate ray path ####
        #############################
        
        #Ray parmaeter in sec/rad
        p = arrival.ray_param

        #Loop through path segments
        seg_ray_sigma = {}
        for x in paths:
            path = paths[x]

            #Depth in m
            dep = path[:,3]*1e3
            #Remove centre of the Earth so that TauP doesn't error
            dep[dep == Re*1e3] = Re*1e3 - 1
            #Radius in m
            r = Re*1e3 - dep
            #Velocity in m/s
            v = np.array([model.model.s_mod.v_mod.evaluate_below(dep[i]/1e3, wave_paths[x][i])[0]*1e3 if dep[i] != max(dep) 
                    else model.model.s_mod.v_mod.evaluate_above(dep[i]/1e3, wave_paths[x][i])[0]*1e3 for i in range(len(path))])
            #Gradient of v wrt r
            dvdr = np.array([get_dvdr_below(model, r[i], wave_paths[x][0]) if dep[i] != max(dep) 
                    else get_dvdr_above(model, r[i], wave_paths[x][0]) for i in range(len(path))])
            #eta in s
            eta = r/v
            #epsilon
            epsilon = np.array([get_epsilon_below(model, r[i]) if dep[i] != max(dep) 
                    else get_epsilon_above(model, r[i]) for i in range(len(path))])
            #Distance in radians
            dist = np.array([x[2] for x in path])
            #lambda
            lamda = {x:factor(x)*(-1.)*(2./3.)*alp2(x, dist) for x in [0, 1, 2]}
            #Do the integration
            seg_ray_sigma[x] = {m:np.sum(scipy.integrate.trapz((eta**3.)*dvdr*epsilon*lamda[m], x = dist)/p) for m in [0, 1, 2]}
            
        #Sum coefficients for each segment to get total ray path contribution
        ray_sigma = {m:np.sum([seg_ray_sigma[x][m] for x in paths]) for m in [0, 1, 2]}
        
        #####################################
        ##### Effects of discontinuities ####
        #####################################

        #Including the bottoming depth allows cross indexing with the paths variable when the start point is not the lowest point on the ray path
        if bot_dep == arrival.path[0][3]:
            assess_discs = discs
        else:
            assess_discs = np.append(discs, bot_dep)   
            
        #Get which discontinuities the phase interacts with, include bottoming depth to allow cross indexing with the paths variable
        ids = [(i, arrival.path[i][3], arrival.path[i][2]) for i in range(len(arrival.path)) if arrival.path[i][3] in assess_discs]
        if arrival.source_depth != 0 and arrival.source_depth != ids[0][1]:
            ids = [(0, arrival.source_depth, 0)] + ids
        idiscs = {i:{'ind':ids[i][0], 'order':i, 'dep':ids[i][1]*1e3, 'r':(Re - ids[i][1])*1e3, 'dist':ids[i][2], 'p':arrival.path[0][0]} 
                        for i in range(len(ids))}

        #Loop through discontinuities and assess what is occurring
        for d in idiscs:
            
            #Do not sum if diffracted and this is the CMB
            if 'diff' in arrival.name and idiscs[d]['dep'] == model.model.cmb_depth*1e3:
                idiscs[d]['yn'] = False
            
            #Do not calculate for bottoming depth if this is not a discontinuity
            elif (round(idiscs[d]['dep']*1e-3,5) in discs or d == 0):
                idiscs[d]['yn'] = True
                
            #Do not sum if this is the bottoming depth
            else:
                idiscs[d]['yn'] = False
                
            #Proceed if summing this discontinuity
            if idiscs[d]['yn']:

                #epsilon at this depth
                idiscs[d]['epsilon'] = get_epsilon(model, idiscs[d]['r'])
                
                #lambda at this distance
                idiscs[d]['lambda'] = {x:factor(x)*(-1.)*(2./3.)*alp2(x, idiscs[d]['dist']) for x in [0, 1, 2]}
                
                #Calculate the factor
                extra = {x:idiscs[d]['epsilon']*idiscs[d]['lambda'][x] for x in [0, 1, 2]}

                #The surface must be treated differently due to TauP indexing constraints
                if idiscs[d]['dep'] != 0. and idiscs[d]['ind'] != 0:
                    
                    #Depths before and after interactions
                    dep0 = arrival.path[idiscs[d]['ind']-1][3]
                    dep1 = arrival.path[idiscs[d]['ind']][3]
                    dep2 = arrival.path[idiscs[d]['ind']+1][3]
                    
                    #Direction before interaction
                    if dep0 < dep1:
                        idiscs[d]['pre'] = 'down'
                    elif dep0 == dep1:
                        idiscs[d]['pre'] = 'diff'
                    else:
                        idiscs[d]['pre'] = 'up'
                        
                    #Direction after interaction
                    if dep1 < dep2:
                        idiscs[d]['post'] = 'down'
                    elif dep1 == dep2:
                        idiscs[d]['post'] = 'diff'
                    else:
                        idiscs[d]['post'] = 'up'
                        
                    #Reflection or transmission
                    if idiscs[d]['pre'] == idiscs[d]['post']:
                        idiscs[d]['type'] = 'trans'
                    elif 'diff' in [idiscs[d]['pre'],idiscs[d]['post']]:
                        idiscs[d]['type'] = 'diff'
                    else:
                        idiscs[d]['type'] = 'refl'

                    #Phase before and after
                    idiscs[d]['ph_pre'] =  wave_paths[d-1][-1]
                    idiscs[d]['ph_post'] =  wave_paths[d][0]
                                    
                    #Deal with a transmission case
                    if idiscs[d]['type'] == 'trans':
                        
                        #Phase above
                        if idiscs[d]['pre'] == 'down':
                            idiscs[d]['ph_above'] = idiscs[d]['ph_pre']
                            idiscs[d]['ph_below'] = idiscs[d]['ph_post']
                        elif idiscs[d]['pre'] == 'up':
                            idiscs[d]['ph_above'] = idiscs[d]['ph_post']
                            idiscs[d]['ph_below'] = idiscs[d]['ph_pre']

                        #Velocity above and below discontinuity
                        idiscs[d]['v0'] = model.model.s_mod.v_mod.evaluate_above(idiscs[d]['dep']/1e3, idiscs[d]['ph_above'])[0]*1e3
                        idiscs[d]['v1'] = model.model.s_mod.v_mod.evaluate_below(idiscs[d]['dep']/1e3, idiscs[d]['ph_below'])[0]*1e3
                        
                        #eta above and below discontinuity
                        idiscs[d]['eta0'] = idiscs[d]['r']/idiscs[d]['v0']
                        idiscs[d]['eta1'] = idiscs[d]['r']/idiscs[d]['v1']
                    
                        #Evaluate the time difference
                        eva = (-1.)*(np.sqrt(idiscs[d]['eta0']**2 - idiscs[d]['p']**2) - np.sqrt(idiscs[d]['eta1']**2 - idiscs[d]['p']**2))

                    #Deal with an underside reflection case
                    if idiscs[d]['type'] == 'refl' and idiscs[d]['pre'] == 'up':
                        
                        #Velocity below discontinuity
                        idiscs[d]['v0'] = model.model.s_mod.v_mod.evaluate_below(idiscs[d]['dep']/1e3, idiscs[d]['ph_pre'])[0]*1e3
                        idiscs[d]['v1'] = model.model.s_mod.v_mod.evaluate_below(idiscs[d]['dep']/1e3, idiscs[d]['ph_post'])[0]*1e3
                        
                        #eta below discontinuity
                        idiscs[d]['eta0'] = idiscs[d]['r']/idiscs[d]['v0']
                        idiscs[d]['eta1'] = idiscs[d]['r']/idiscs[d]['v1']

                        #Evaluate the time difference
                        eva = np.sqrt(idiscs[d]['eta0']**2 - idiscs[d]['p']**2) + np.sqrt(idiscs[d]['eta1']**2 - idiscs[d]['p']**2)
                        
                    #Deal with a topside reflection case
                    if idiscs[d]['type'] == 'refl' and idiscs[d]['pre'] == 'down':
                        
                        #Velocity above discontinuity
                        idiscs[d]['v0'] = model.model.s_mod.v_mod.evaluate_above(idiscs[d]['dep']/1e3, idiscs[d]['ph_pre'])[0]*1e3
                        idiscs[d]['v1'] = model.model.s_mod.v_mod.evaluate_above(idiscs[d]['dep']/1e3, idiscs[d]['ph_post'])[0]*1e3
                        
                        #eta above discontinuity
                        idiscs[d]['eta0'] = idiscs[d]['r']/idiscs[d]['v0']
                        idiscs[d]['eta1'] = idiscs[d]['r']/idiscs[d]['v1']
                        
                        #Evaluate the time difference
                        eva = (-1)*(np.sqrt(idiscs[d]['eta0']**2 - idiscs[d]['p']**2) + np.sqrt(idiscs[d]['eta1']**2 - idiscs[d]['p']**2))

                #Deal with source depth and also end point
                elif idiscs[d]['ind'] == 0 or idiscs[d]['ind'] == len(arrival.path)-1:
                    
                    #Assign wave type
                    if idiscs[d]['ind'] == 0:
                        wave = wave_paths[0][0]
                    elif idiscs[d]['ind'] == len(arrival.path)-1:
                        wave = wave_paths[max(list(paths.keys()))-1][-1]
                    
                    #Deal with phases that start with an upgoing segment
                    if arrival.name[0] in ['p', 's'] and idiscs[d]['ind'] == 0:
                        
                        #Velocity above source
                        idiscs[d]['v1'] = model.model.s_mod.v_mod.evaluate_above(idiscs[d]['dep']/1e3, wave)[0]*1e3
                        
                        #eta above source
                        idiscs[d]['eta1'] = idiscs[d]['r']/idiscs[d]['v1']

                        #Evaluate the time difference
                        eva = (-1.)*np.sqrt(idiscs[d]['eta1']**2 - idiscs[d]['p']**2)
                    
                    #Deal with ending the ray path at the surface
                    else:
                        #Velocity below surface
                        idiscs[d]['v1'] = model.model.s_mod.v_mod.evaluate_below(idiscs[d]['dep']/1e3, wave)[0]*1e3
                        
                        #eta below surface
                        idiscs[d]['eta1'] = idiscs[d]['r']/idiscs[d]['v1']

                        #Evaluate the time difference
                        eva = (-1.)*(0 - np.sqrt(idiscs[d]['eta1']**2 - idiscs[d]['p']**2))
                    
                #Deal with surface reflection
                elif idiscs[d]['dep'] == 0.:
                    
                    #Assign type of interaction
                    idiscs[d]['type'] = 'refl'

                    #Phase before and after
                    idiscs[d]['ph_pre'] =  wave_paths[d-1][-1]
                    idiscs[d]['ph_post'] =  wave_paths[d][0]
                    
                    #Velocity below surface
                    idiscs[d]['v0'] = model.model.s_mod.v_mod.evaluate_below(idiscs[d]['dep']/1e3, idiscs[d]['ph_pre'])[0]*1e3
                    idiscs[d]['v1'] = model.model.s_mod.v_mod.evaluate_below(idiscs[d]['dep']/1e3, idiscs[d]['ph_post'])[0]*1e3
                    
                    #eta below surface
                    idiscs[d]['eta0'] = idiscs[d]['r']/idiscs[d]['v0']
                    idiscs[d]['eta1'] = idiscs[d]['r']/idiscs[d]['v1']

                    #Evaluate time difference
                    eva = np.sqrt(idiscs[d]['eta0']**2 - idiscs[d]['p']**2) + np.sqrt(idiscs[d]['eta1']**2 - idiscs[d]['p']**2)
                        
                #Output coefficients for this discontinuity
                idiscs[d]['sigma'] = {x:extra[x]*eva for x in[0, 1, 2]}
            
        #Sum the contribution to the contribution to the coefficients from discontinuities
        disc_sigma = {x:np.sum([idiscs[i]['sigma'][x] for i in idiscs if idiscs[i]['yn']]) for x in [0, 1, 2]}

        ##############################
        ##### Sum to coefficients ####
        ##############################

        #Sum the contribution from the ray path and the discontinuities to get final coefficients
        sigma = [ray_sigma[x] + disc_sigma[x] for x in [0, 1, 2]]
    
    return sigma

def calculate_model_epsilon(model, filename, lod, taper = True):

    '''
    Calculates a profile of elliptiicty (epsilon) through an inputted planetary mdoel.
    
    Inputs:
        model - TauPyModel
        filename - string, output file name
        lod - float, length of day in the model in seconds
        taper - bool, whether to taper below ICB or not. Causes problems if False (and True is consistent with previous works, e.g. Bullen & Haddon (1973))
        
    Output:
        text file of epsilon values with radius
    '''
    
    from numpy import inf
    
    #Angular velocity of model
    Omega = 2*np.pi/lod

    #Universal gravitational constant
    G = 6.67408*10**(-11)

    #Radius of Earth
    a = model.model.radius_of_planet*1000

    #Loop over r and calculate the total mass of the body in kg and moment of inertia
    #Radius steps in m
    dr = 100
    r = np.arange(0, a + dr, dr)

    #Get the density at these depths
    rho = np.append(model.model.s_mod.v_mod.evaluate_above((a - r[:-1])/1000., 'd')*1000., model.model.s_mod.v_mod.evaluate_below(0., 'd')[0]*1000.)

    #Mass within each spherical shell
    Mr = np.cumsum(4*np.pi*rho*(r**2)*dr)

    #Total mass of body
    M = Mr[-1]

    #Moment of inertia of each spherical shell
    Ir = (8./3.)*np.pi*np.cumsum(rho*(r**4)*dr)

    #Moment of inertia at surface
    I = Ir[-1]

    #Calculate y (moment of inertia factor) for surfaces within the body
    y = Ir/(Mr*r**2)

    #Taper if required
    #Taper at closest point to 0.4, this is where eta is 0
    #Otherwise epsilon tends to infinity at the centre of the planet
    if taper:
        y = np.array([x if x < 0.4 else 0.4 for x in y])

    #Calculate Radau's parameter
    eta = 6.25*(1 - 3*(y)/2)**2 - 1

    #Calculate h
    #Ratio of centrifugal force and gravity for a particle on the equator at the surface
    ha = (a**3 * Omega**2) / (G * M)

    #epsilon at surface
    epsilona = (5*ha)/(2*eta[-1] + 4)

    #Solve the differential equation
    LHS = dr*eta/r
    if -inf in LHS:
        LHS[LHS == -inf] = 0
    LHS = np.cumsum(np.nan_to_num(LHS))
    LHS = np.exp(LHS)
    c = epsilona/LHS[-1]
    epsilon = c*LHS

    #Write out epsilon to text file for future usage
    string = ''.join([str(r[i]) + ', ' + str(epsilon[i]) + '\n' for i in range(len(epsilon))])
    f = open(filename, 'w')
    f.write(string)
    f.close()

