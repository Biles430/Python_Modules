import h5py
import numpy as np
import pandas as pd

T_given = 19.37 #C
data = pd.read_csv('air_prop.txt', sep = '\t', skiprows=1)
data.columns = ['T', 'rho', 'cp', 'k', 'nu', 'b', 'Pr']

##    Write out data

readme = ('This file contians thermal properties of air for given temps')

units = {'T': 'C',
'rho': 'kg/m^3',
'cp': 'KJ/kgK',
'k': 'W/mK',
'nu': '*10^-6 m^2/sec',
'b': '*10^-3 1/K',
'Pr': 'unitless'}

#create df
hdf = pd.HDFStore('data/airprop1.h5')
hdf.put('air_prop', data)
hdf.put('units', pd.Series(units))
hdf.put('readme', pd.Series(readme))

#read in
#data = pd.read_hdf('/home/drummmond/biles430@gmail.com/Documents/Gradschool/Python/Modules/air_prop/air_prop.h5', 'air_prop')

# for j in range(0, len(data['T'])):
#     if data['T'][j] < T_given:
#         if data['T'][j+1] > T_given:
#             delta_T= data['T'][j] - data['T'][j-1]
#             delta_given = T_given - data['T'][j]
#             rho = (data['rho'][j] - data['rho'][j-1])/delta_T * delta_given + data['rho'][j]
#             k = (data['k'][j] - data['k'][j-1])/delta_T * delta_given + data['k'][j]
#             cp = (data['cp'][j] - data['cp'][j-1])/delta_T * delta_given + data['cp'][j]
#             nu = (data['nu'][j] - data['nu'][j-1])/delta_T * delta_given + data['nu'][j]
#             Pr = (data['Pr'][j] - data['Pr'][j-1])/delta_T * delta_given + data['Pr'][j]
