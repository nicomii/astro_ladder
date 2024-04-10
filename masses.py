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

ddir_stars = 'C:/Users/enidh/OneDrive/2024/PHYS3080/DL Project'
distant=glob.glob(ddir_stars +'/XrayDistances.csv')
DistantGalaxies = pd.concat([pd.read_csv(table) for table in distant])
G=6.67*pow(10,-11)
#listy=glob.glob(ddir_stars+'/Universe_8/Distant/')
#checker=glob.glob(ddir_stars+'/Universe_8/Distant/*.csv')
#RadiiGalaxies=pd.DataFrame()
#for name in checker:
#    galaxdata = pd.read_csv(f'{name}', delimiter=' ')
#    RadiiGalaxies = pd.concat([RadiiGalaxies,galaxdata])
#DistantGalaxies.Radius=''
#for i in range(len(DistantGalaxies)):
#    for k in range(len(RadiiGalaxies)):
#        if DistantGalaxies.iat[i,1]==RadiiGalaxies.iat[k,1]:
#            DistantGalaxies.iat[i,5]=RadiiGalaxies.iat[k,9]
    

DistantGalaxies['Luminosity']=4*np.pi*DistantGalaxies['Distance']*(DistantGalaxies['RedF']+DistantGalaxies['GreenF']+DistantGalaxies['BlueF'])
DistantGalaxies['Mass']=DistantGalaxies['Radius']*pow(DistantGalaxies['RadialVelocity'].astype(float),2)/G

A = np.vander(DistantGalaxies['Mass'].astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,DistantGalaxies['Radius'].astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(DistantGalaxies.Mass[len(DistantGalaxies.Mass)-1]-DistantGalaxies.Mass[0])
inter=reconstructed[0]-grad*DistantGalaxies.Mass[0]
print("Mass Radius")
print("Grad:", grad)
print("Inter:", inter)

fig=plt.figure()
plt.plot(np.log10(DistantGalaxies['Mass'].astype(float)),np.log10(DistantGalaxies['Radius'].astype(float)),'.')
#plt.plot(np.log10(DistantGalaxies['Mass'].astype(float)),np.log10(reconstructed))
plt.ylabel('Log10 of Radius (kpc)')
plt.xlabel('Log10 of Mass (kg)')
plt.title('Mass vs Radius of Galaxies')

A = np.vander(DistantGalaxies['Mass'].astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,DistantGalaxies['RadialVelocity'].astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(DistantGalaxies.Mass[len(DistantGalaxies.Mass)-1]-DistantGalaxies.Mass[0])
inter=reconstructed[0]-grad*DistantGalaxies.Mass[0]
print("Mass Velocity")
print("Grad:", grad)
print("Inter:", inter)

fig=plt.figure()
plt.plot(np.log10(DistantGalaxies['Mass'].astype(float)),np.log10(abs(DistantGalaxies['RadialVelocity'].astype(float))),'.')
#plt.plot(np.log10(DistantGalaxies['Mass'].astype(float)),np.log10(abs(reconstructed)))
plt.ylabel('Log10 of Radial Velocity (km/sec)')
plt.xlabel('Log10 of Mass (kg)')
plt.title('Mass vs Radial Velocity of Galaxies')

A = np.vander(DistantGalaxies['Luminosity'].astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,DistantGalaxies['Mass'].astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(DistantGalaxies.Luminosity[len(DistantGalaxies.Luminosity)-1]-DistantGalaxies.Luminosity[0])
inter=reconstructed[0]-grad*DistantGalaxies.Luminosity[0]
print("Mass Luminosity")
print("Grad:", grad)
print("Inter:", inter)

fig=plt.figure()
plt.plot(np.log10(DistantGalaxies['Luminosity'].astype(float)),np.log10(DistantGalaxies['Mass'].astype(float)),'.')
#plt.plot(DistantGalaxies['Luminosity'].astype(float),reconstructed)
plt.xlabel('Log 10 of V-band Luminosity (W/nm)')
plt.ylabel('Log 10 of Mass (kg)')
plt.title('Luminosity vs Mass of Galaxies')
