#!/usr/bin/python3

"""
Python wrapper that runs the ellipticity correction package based on a command line input.
"""

# Import modules and the main package
import sys
import argparse
import warnings
from obspy.taup import TauPyModel
from src.ellipticity import ellipticity_correction
from src.ellipticity.tools import EARTH_LOD

# Ignore TauP warning
warnings.filterwarnings("ignore", message="Resizing a TauP array inplace failed due to the existence of other references to the array, creating a new array. See Obspy #2280.")

# Call ArgumentParser
parser = argparse.ArgumentParser()

# Float arguments - source depth, epicentral distance, azimuth, source latitude and length of day
parser.add_argument("-d", "--depth", type=float, required=True,
                     help="Source depth in km")
parser.add_argument("-deg", "--distance", type=float, required=True,
                     help="Epicentral distance in degrees")
parser.add_argument("-az", "--azimuth", type=float, required=True,
                     help="Azimuth from source to receiver in degrees from N")
parser.add_argument("-sl", "--latitude", type=float, required=True,
                     help="Source latitude in degrees")
parser.add_argument("-lod", "--period", type=float, default=EARTH_LOD,
                     help="Length of day in seconds")

#String arguments
parser.add_argument("-ph", "--phase", type=str, required=True,
                     help="TauP phase names - if several then seperate by commas")
parser.add_argument("-mod", "--model", type=str, required=True,
                     help="Velocity model")

# Pass arguments
args = parser.parse_args()

# Initialse TauPyModel
model = TauPyModel(model = args.model)

# Get list of phases
phases = args.phase.split(',')

# Get Arrivals object from ObsPy TauP
arrivals = model.get_ray_paths(source_depth_in_km = args.depth, distance_in_degree = args.distance, phase_list = phases)

# Calculate ellipticity corrections
corrections = ellipticity_correction(arrivals, args.azimuth, args.latitude, lod=args.period)

# Final travel times
times = [arrivals[i].time + corrections[i] for i in range(len(arrivals))]

# Create an output message
line0 = "Model: " + args.model
line1 = " Distance    Depth     Phase         Ray Param     Spherical    Ellipticity      Elliptical"
line2 = "   (deg)      (km)     Name          p (s/deg)     Travel       Correction       Travel    "
line3 = "                                                   Time (s)         (s)          Time (s)  "
line4 = "------------------------------------------------------------------------------------------"
lines = [str(round(arrivals[i].purist_distance, 2)).rjust(9, " ") + str(round(arrivals[i].source_depth, 2)).rjust(9, " ") 
          + "     " + arrivals[i].name.ljust(14, " ") + str(round(arrivals[i].ray_param_sec_degree, 3)).rjust(9, " ") + 
          str(round(arrivals[i].time, 2)).rjust(13, " ") +  str(round(corrections[i], 2)).rjust(14, " ") + 
          str(round(times[i], 2)).rjust(16, " ")
          for i in range(len(arrivals))]
message0 = "\n".join([line0, line1, line2, line3, line4])
message1 = "\n".join(lines)
message = message0 + "\n" + message1

# Output
print(message)




