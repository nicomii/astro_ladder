import numpy as np # for maths 
import matplotlib # for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

from tqdm import tqdm # tqdm is a package that lets you make progress bars to see how a loop is going

import os 

import pandas as pd # pandas is a popular library in industry for manipulating large data tables

from astropy.timeseries import LombScargle

# configure notebook for plotting
%matplotlib inline

mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme

# subsequent lines default plot settings
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)   
matplotlib.rcParams['font.size']=16              
matplotlib.rcParams['savefig.dpi']= 300             

import warnings
warnings.filterwarnings('ignore')
    
ddir_stars = 'C:/Users/enidh/OneDrive/2024/PHYS3080/DL Project/universe_8/' # point this to where you unzip your data!

ddir = ddir_stars + '/Variable_Star_Data/'

fname = 'BackS016039.csv' # put your filename here

data = pd.read_csv(ddir+fname) # load in CSV data as a Pandas object
print(data.keys()) # see what's in it
time, flux = data.Time, data.NormalisedFlux # just extract the columns as variables
dt = np.median(np.diff(time))
print('Nyquist Limit',0.5/dt,'cycles per hour') # can't get frequencies higher than the Nyquist limit

LS = LombScargle(time,flux) # initialize a Lomb-Scargle algorithm from Astropy
freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
power = LS.power(freqs) # calculate LS power

print('Best period: %.2f h' % (1/freqs[np.argmax(power)]))

import glob # this package lets you search for filenames

fnames = glob.glob(ddir+'*.csv')

freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
periods = [] # start an empty list to hold the period 

freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
periods = [] # start an empty list to hold the period 
names = []

for fname in tqdm(fnames): # tqdm is a package that gives you a progress bar - neat! 
    data = pd.read_csv(fname) # load in CSV data as a Pandas object

    time, flux = data.Time, data.NormalisedFlux # just extract the columns as variables

    LS = LombScargle(time,flux) # initialize a Lomb-Scargle
    power = LS.power(freqs) # calculate LS power 
    bestfreq = freqs[np.argmax(power)] # which frequency has the highest Lomb-Scargle power?
    
    pred = LS.model(time,bestfreq) # make a sine wave prediction at the best frequency
    
    periods.append(1/bestfreq) # add each period to the list
    names.append(os.path.basename(fname).strip('.csv')) # os.path.basename gets rid of directories and gives you the filename; then we strip '.csv'

periods = np.array(periods) # turn it from a list to an array

import glob # this package lets you search for filenames
import os

variables = pd.DataFrame({'Name':names,
              'Period':periods}) # you can turn a dictionary into a dataframe like this
variables.Name = variables.Name.astype('|S') # have to do this so that it knows the names are strings

all_star_files = glob.glob(ddir_stars+'*/Star_Data.csv')

all_stars = pd.concat([pd.read_csv(table) for table in all_star_files]) # we are concatenating a list of dataframes; 
#we generate this list with a "list comprehension", a loop you write inside a list bracket 

all_stars.Name = all_stars.Name.astype('|S') # have to do this so that it knows the names are strings
all_stars_1 = all_stars[all_stars.Parallax > 0.01] # 10 mas parallax cut
print(len(all_stars_1),'stars above 10 mas parallax') # check how many stars there are total with good parallax

variables_1 = pd.merge(all_stars_1,variables,on='Name') # merge these two arrays according to the keyword 'name'
print('Of which',len(variables_1),'variables') # cut down to a small list

m0, m1, m2 = np.log10(all_stars_1['BlueF']), np.log10(all_stars_1['GreenF']), np.log10(all_stars_1['RedF']) 
colour = m2-m0
abs_mag = m1 + 2*np.log10(1./all_stars_1.Parallax) 

v0, v1, v2 = np.log10(variables_1['BlueF']), np.log10(variables_1['GreenF']), np.log10(variables_1['RedF']) 
variable_colour = v2-v0
abs_mag_v = v1 + 2*np.log10(1./variables_1.Parallax)

variables_1['AbsMag']=abs_mag_v
fig=plt.figure()
plt.plot(variables_1.Period,variables_1.AbsMag,'.',color='C2')
plt.title("Magnitude-Period relation")
plt.xlabel('Period (h)')
plt.ylabel('Absolute Magnitude in G band (mag)')

variableslong=variables_1.query('Period > 45')
variableslong=variableslong.reset_index()

variablesshort=variables_1.query('Period > 10 and Period < 30')
variablesshort=variablesshort.reset_index()

periodlarr=np.array(variableslong.Period)
luminlarr=np.array(variableslong.AbsMag)
C = np.vander(periodlarr,2) # the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
d, residuals, rank, s = np.linalg.lstsq(C,luminlarr)
reconstructed2 = C @ d
grad2=(reconstructed2[len(reconstructed2)-1]-reconstructed2[0])/(periodlarr[len(periodlarr)-1]-periodlarr[0])
int2=reconstructed2[0]-grad2*periodlarr[0]

fig=plt.figure()
plt.plot(variableslong.Period,variableslong.AbsMag,'.',color='C2')
plt.plot(variableslong.Period,grad2*variableslong.Period+int2)
plt.title("Magnitude-Period relation for long period")
plt.xlabel('Period (h)')
plt.ylabel('Absolute Magnitude in G band (mag)')
print("Long Gradient:", grad2, "Long Intercept:", int2)

periodsarr=np.array(variablesshort.Period)
luminsarr=np.array(variablesshort.AbsMag)
A = np.vander(periodsarr,2) # the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b, residuals, rank, s = np.linalg.lstsq(A,luminsarr)
reconstructed1 = A @ b
grad1=(reconstructed1[len(reconstructed1)-1]-reconstructed1[0])/(periodsarr[len(periodsarr)-1]-periodsarr[0])
int1=reconstructed1[0]-grad1*periodsarr[0]

fig=plt.figure()
plt.plot(variablesshort.Period,variablesshort.AbsMag,'.',color='C2')
plt.plot(variablesshort.Period,grad1*variablesshort.Period+int1)
plt.title("Magnitude-Period relation for short period")
plt.xlabel('Period (h)')
plt.ylabel('Absolute Magnitude in G band (mag)')
print("Short Gradient:", grad1, "Short Intercept:", int1)

variable_all=all_stars[all_stars.Name.isin(variables.Name)]
variable_all=variable_all.reset_index()
variable_all=pd.concat([variables,variable_all],axis=1)
variable_all['TotalF']=variable_all.BlueF+variable_all.GreenF+variable_all.RedF

newvariableslong=variable_all.drop(variable_all[variable_all.Period < 45].index)
newvariablesshort1=variable_all.drop(variable_all[variable_all.Period > 30].index)
newvariablesshort=newvariablesshort1.drop(newvariablesshort1[newvariablesshort1.Period < 10].index)

newvariableslong['AbsMag']=newvariableslong.Period*grad2+int2
newvariablesshort['AbsMag']=newvariablesshort.Period*grad1+int1

newvariableslong.RadialVelocity=-(newvariableslong.RadialVelocity)
newvariablesshort.RadialVelocity=-(newvariablesshort.RadialVelocity)

newvariablesshort['Distance']=pow(10,(newvariablesshort.AbsMag-np.log10(newvariablesshort.GreenF))/2)/1000
newvariableslong['Distance']=pow(10,(newvariableslong.AbsMag-np.log10(newvariableslong.GreenF))/2)/1000
newvariablesshort=newvariablesshort.drop(newvariablesshort[newvariablesshort.Distance > 8].index)
newvariablesshort=newvariablesshort.drop(newvariablesshort[(newvariablesshort.Distance > 2) &(newvariablesshort.RadialVelocity > 5)].index)
newvariableslong=newvariableslong.reset_index()
newvariablesshort=newvariablesshort.reset_index()
newvariablesshort.to_csv('Short Variables.csv')
newvariableslong.to_csv('Long Variables.csv')

#shortdist=np.array(newvariablesshort.Distance.astype(float))
#shortvelo=np.array(newvariablesshort.RadialVelocity.astype(float))
#q = np.vander(shortdist,2) # the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
#p, residuals, rank, s = np.linalg.lstsq(q,shortvelo)
#reconstructedq = q @ p
#gradq=(reconstructedq[len(reconstructedq)-1]-reconstructedq[0])/(shortdist[len(shortdist)-1]-shortdist[0])
#intq=reconstructedq[0]-gradq*shortdist[0]

#fig=plt.figure()
#plt.plot(newvariablesshort.Distance,newvariablesshort.RadialVelocity,'.',color='C2')
#plt.xlabel('Distance (Mpc)')
#plt.ylabel('Radial Velocity (km/sec)')
#plt.plot(shortdist,reconstructedq)
#print("Short grad",round(gradq,2),"Short int", round(intq,2))

#longdist=np.array(newvariableslong.Distance.astype(float))
#longvelo=np.array(newvariableslong.RadialVelocity.astype(float))
#m = np.vander(longdist,2) # the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
#n, residuals, rank, s = np.linalg.lstsq(m,longvelo)
#reconstructedm = m @ n
#gradm=(reconstructedm[len(reconstructedm)-1]-reconstructedm[0])/(longdist[len(longdist)-1]-longdist[0])
#intm=reconstructedm[0]-gradm*longdist[0]

#fig=plt.figure()
#plt.plot(newvariableslong.Distance,newvariableslong.RadialVelocity,'.',color='C2')
#plt.xlabel('Distance (Mpc)')
#plt.ylabel('Radial Velocity (km/sec)')
#plt.plot(longdist,reconstructedm)
#print("Long grad",round(gradm,2),"Long int", round(intm,2))