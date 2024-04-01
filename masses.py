# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:35:13 2024

@author: enidh
"""

import os 
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import glob

ddir_stars = 'C:/Users/enidh/OneDrive/2024/PHYS3080/DL Project/'
distant=glob.glob(ddir_stars +'XrayGalaxyDistances.csv')
DistantGalaxies = pd.concat([pd.read_csv(table) for table in distant])
G=6.67*pow(10,-11)

DistantGalaxies['Luminosity']=4*np.pi*DistantGalaxies['Distance']*(DistantGalaxies['RedF']+DistantGalaxies['GreenF']+DistantGalaxies['BlueF'])
DistantGalaxies['Mass']=DistantGalaxies['Radius']*pow(DistantGalaxies['RadVel'].astype(float),2)/G

A = np.vander(DistantGalaxies['Mass'].astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,DistantGalaxies['Radius'].astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(DistantGalaxies.Mass[len(DistantGalaxies.Mass)-1]-DistantGalaxies.Mass[0])
inter=reconstructed[0]-grad*DistantGalaxies.Mass[0]

fig=plt.figure()
plt.plot(DistantGalaxies['Mass'].astype(float),DistantGalaxies['Radius'].astype(float),'.')
plt.plot(DistantGalaxies['Mass'].astype(float),reconstructed)
plt.ylabel('Radius (m)')
plt.xlabel('Mass (kg)')
plt.title('Mass vs Radius of Galaxies')

A = np.vander(DistantGalaxies['Mass'].astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,DistantGalaxies['RadVel'].astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(DistantGalaxies.Mass[len(DistantGalaxies.Mass)-1]-DistantGalaxies.Mass[0])
inter=reconstructed[0]-grad*DistantGalaxies.Mass[0]

fig=plt.figure()
plt.plot(DistantGalaxies['Mass'].astype(float),DistantGalaxies['RadVel'].astype(float),'.')
plt.plot(DistantGalaxies['Mass'].astype(float),reconstructed)
plt.ylabel('Radial Velocity (km/sec-squared)')
plt.xlabel('Mass (kg)')
plt.title('Mass vs Radial Velocity of Galaxies')

A = np.vander(DistantGalaxies['Luminosity'].astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,DistantGalaxies['Mass'].astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(DistantGalaxies.Luminosity[len(DistantGalaxies.Luminosity)-1]-DistantGalaxies.Luminosity[0])
inter=reconstructed[0]-grad*DistantGalaxies.Luminosity[0]

fig=plt.figure()
plt.plot(DistantGalaxies['Luminosity'].astype(float),DistantGalaxies['Mass'].astype(float),'.')
plt.plot(DistantGalaxies['Luminosity'].astype(float),reconstructed)
plt.xlabel('Luminosity (W/nm)')
plt.ylabel('Mass (kg)')
plt.title('Luminosity vs Mass of Galaxies')