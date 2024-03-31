# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:08:51 2024

@author: enidh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:39:46 2024

@author: enidh
"""
import os 
import numpy as numpy
import pandas as pd
import math 
import matplotlib.pyplot as plt
import glob

import warnings
warnings.filterwarnings('ignore')
longvariables=pd.read_csv('Long Variables.csv')
shortvariables=pd.read_csv('Short Variables.csv')
longvariables=longvariables.drop('level_0',axis=1)
longvariables=longvariables.drop('Unnamed: 0',axis=1)
shortvariables=shortvariables.drop('level_0',axis=1)
shortvariables=shortvariables.drop('Unnamed: 0',axis=1)
shortvariables['Name'] = shortvariables['Name'].map(lambda x: x.lstrip("b'").rstrip("'"))
longvariables['Name'] = longvariables['Name'].map(lambda x: x.lstrip("b'").rstrip("'"))

ddir_stars = 'C:/Users/enidh/OneDrive/2024/PHYS3080/DL Project/universe_8/'
allgaldist = pd.read_csv(ddir_stars+'Galaxy_Distances.csv', delimiter=' ')
all_star_files = glob.glob(ddir_stars+'*/Star_Data.csv')
all_stars = pd.concat([pd.read_csv(table) for table in all_star_files])

datapath = 'universe_8/'
GalaxyNames = []

#read in data from x-ray flashes
flashData = pd.read_csv("C:/Users/enidh/OneDrive/2024/PHYS3080/DL Project/universe_8/Flash_Data.csv", index_col="Name")

flashData['distances']=''
count = 0
county = 0
countall = 0
flashData['Galaxy']=''
flashData['RadVel']=''
checker=0
gcount=0
for clusterFile in os.listdir(datapath + '\Star Clusters'):
    GalaxyNames.append(clusterFile[:-4]) 
        
for t in range(0,len(flashData)):
    ooox=flashData.iat[t,1]
    oooy=flashData.iat[t,2]
    direct=flashData.iat[t,0] 
    county+=1
    distant=glob.glob(ddir_stars +f'/{direct}/Distant_Galaxy_Data.csv')
    DistGal=pd.read_csv(distant[0])
    for i in range(0,len(DistGal)):
        ex=DistGal.iat[i,1]
        ey=DistGal.iat[i,2]
        if abs(ex-ooox) <= 0.01:
            if abs(ey-oooy) <= 0.01:
                #print("Got one",t)
                #print(ooox,oooy,ex,ey)
                #print(DistGal.iat[i,0])
                checker+=1
                flashData.iat[t,5]=DistGal.iat[i,0]
                flashData.iat[t,6]=DistGal.iat[i,7]
                break
#for t in range(0,len(flashData)):
#    ooox=flashData.iat[t,1]
#    oooy=flashData.iat[t,2]
#    direct=flashData.iat[t,0] 
#    county+=1
#    for i, name in enumerate(GalaxyNames):
#        galaxdata = pd.read_csv(datapath + f'/Star Clusters/{name}.csv', delimiter=' ')
#        if galaxdata.iat[0,0].startswith(direct):
#            countyxmax=max(galaxdata.X)
#            countyxmin=min(galaxdata.X)
#            countyymax=max(galaxdata.Y)
#            countyymin=min(galaxdata.Y)
#            if countyxmin <= ooox <= countyxmax:
#                if countyymin <= oooy <= countyymax:
#                    flashData.iat[t,5]=name
#                    count+=1
#                    maxy=max(abs(galaxdata.Parallax))
#                    flashData.iat[t,4]=1/maxy
#                    print(t, "matched to", name)
#                    print(ooox,oooy)
#                    print(countyxmax,countyxmin,countyymax,countyymin)
#                    break
for i in range(len(allgaldist)):
    #if allgaldist.iat[i,0]=='Cheese-12.822173903966597Pizza-22.771107933194152Galaxy':
    #    flashData.iat[6,4]=allgaldist.iat[i,3]
    #    flashData.iat[6,5]='Cheese-12.822173903966597Pizza-22.771107933194152Galaxy'
    if allgaldist.iat[i,0]=='Cheese-15.429350869167429Pizza-12.858523421774931Galaxy':
        flashData.iat[16,4]=allgaldist.iat[i,3]
        flashData.iat[16,5]='Cheese-15.429350869167429Pizza-12.858523421774931Galaxy'
    elif allgaldist.iat[i,0]=='Cheese-11.851197228144988Pizza21.209498081023455Galaxy':
        flashData.iat[20,4]=allgaldist.iat[i,3]
        flashData.iat[20,5]='Cheese-11.851197228144988Pizza21.209498081023455Galaxy'
    elif allgaldist.iat[i,0]=='Cheese-13.759164220183486Pizza20.37149885321101Galaxy':
        flashData.iat[33,4]=allgaldist.iat[i,3]
        flashData.iat[33,5]='Cheese-13.759164220183486Pizza20.37149885321101Galaxy'
    elif allgaldist.iat[i,0]=='Cheese34.80702763466042Pizza30.318433255269323Galaxy':
        flashData.iat[22,4]=allgaldist.iat[i,3]
        flashData.iat[22,5]='Cheese34.80702763466042Pizza30.318433255269323Galaxy'
    elif allgaldist.iat[i,0]=='Cheese24.619519718309853Pizza13.168430985915492Galaxy':
        flashData.iat[43,4]=allgaldist.iat[i,3]
        flashData.iat[43,5]='Cheese24.619519718309853Pizza13.168430985915492Galaxy'

flashData=flashData.drop(flashData[flashData.Galaxy==''].index)
highestFlashes=flashData.drop(flashData[flashData.distances==''].index)

fig=plt.figure()
plt.plot(1/pow((highestFlashes.distances.astype(float)),2),highestFlashes['Photon-Count'].astype(float),'.')
plt.xlabel("1/distance-squared")
plt.ylabel("photonCount")

dist=1/pow((highestFlashes.distances.astype(float)),2)
photon=highestFlashes['Photon-Count'].astype(float)

A = numpy.vander(dist,2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc

b, residuals, rank, s = numpy.linalg.lstsq(A,photon)
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python

fig=plt.figure()
plt.plot(dist,photon,'.')
plt.plot(dist,reconstructed,'-r')
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(dist[len(dist)-1]-dist[0])
inter=reconstructed[0]-grad*dist[0]

flashData['distances']=pow((grad/(flashData['Photon-Count']-inter)),0.5)
newdata=flashData.drop(flashData[flashData.RadVel==''].index)

fig=plt.figure()
plt.plot(newdata.distances,newdata.RadVel,'.')

A = numpy.vander(newdata.distances.astype(float),2) 
# the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc

b, residuals, rank, s = numpy.linalg.lstsq(A,newdata.RadVel.astype(float))
#print('parameters: %.2f, %.2f' % (b[0],b[1]))
m=b[0]
reconstructed = A @ b # @ is shorthand for matrix multiplication in python
grad=(reconstructed[len(reconstructed)-1]-reconstructed[0])/(dist[len(dist)-1]-dist[0])
inter=reconstructed[0]-grad*dist[0]
fig=plt.figure()
plt.plot(newdata.distances,newdata.RadVel,'.')
plt.plot(newdata.distances,reconstructed)
print("H0 is", grad*pow(10,6))