#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:07:44 2024

@author: anikamondal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('/Users/anikamondal/Desktop/combined_data_distant.csv')
r = df['Radius'].values
v = df['RadialVelocity'].values

# Initialize arrays to store the radial velocity dispersion and the number of data points at each radius
dispersions = []
counts = []

# Calculate radial velocity dispersion for each unique radius
unique_radii = np.unique(r)
for radius in unique_radii:
    mask = (r == radius)
    velocities_at_radius = v[mask]
    dispersion = np.std(velocities_at_radius)  # Calculate standard deviation as radial velocity dispersion
    dispersions.append(dispersion)
    counts.append(len(velocities_at_radius))

# Convert lists to arrays
dispersions = np.array(dispersions)
counts = np.array(counts)

# Plot the radial velocity dispersion as a function of radius
plt.figure(figsize=(8, 6))
plt.errorbar(unique_radii, dispersions, yerr=dispersions/np.sqrt(counts), fmt='o', capsize=5)
plt.xlabel('Radius')
plt.ylabel('Radial Velocity Dispersion')
plt.title('Radial Velocity Dispersion of Distant Galaxies')
plt.show()

#calculating mass from virial theorem using radii
df = pd.read_csv('/Users/anikamondal/Desktop/combined_data_distant.csv')
v = df['RadialVelocity'].values
average_velocity = np.mean(v)
velocity_dispersion = np.sqrt(np.mean((v - average_velocity)**2))

print(f"Velocity Dispersion: {velocity_dispersion} km/s")


R = df['Radius'].values  # Radius in kpc


# Constants
G = 4.302e-3  # Gravitational constant in pc * (km/s)^2 / Msun


# Calculate the mass
mass = 5 * velocity_dispersion**2 * R / G  # in solar masses

print(f"Mass of the Galaxy: {mass} Solar Masses")

plt.figure(figsize=(8, 6))
plt.scatter(R, mass, marker='o')
plt.xlabel('Radius of galaxy (kpc)')
plt.ylabel('Mass (Solar Masses)')
plt.title('Mass of Galaxies vs. Radius of galaxy')
plt.show()

std_deviations = []
ranges = []

# Calculate the standard deviation and range for each unique radius
unique_radii = np.unique(r)
for radius in unique_radii:
    mask = (r == radius)
    masses_at_radius = mass[mask]
    std_deviation = np.std(masses_at_radius)
    range_value = np.max(masses_at_radius) - np.min(masses_at_radius)
    std_deviations.append(std_deviation)
    ranges.append(range_value)

# Convert lists to arrays
std_deviations = np.array(std_deviations)
ranges = np.array(ranges)

# Print the standard deviation and range of mass values at each radius
for i, radius in enumerate(unique_radii):
    print(f"Radius: {radius}, Std Deviation: {std_deviations[i]}, Range: {ranges[i]}")


