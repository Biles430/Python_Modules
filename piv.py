#!/usr/bin/python
# Filename: PIV.py

###################     IMPORTS #######################
import pandas as pd
from pandas import DataFrame
import numpy as np



###################################################
############# READ IN PIV DATA #####################
#Function
#this function will read in PIV data output from LaVision software
#The form of the input data is .txt files with 4 columns
#column 1 = X location (mm)
#column 2 = Y Location (mm)
#column 3 = U velocity (m/sec)
#column 4 = V velocity (m/sec)

#UPDATED: 1-20-2016
#INPUTS, base_name_input = Name of Folder with PIV data inside it
#        data_sets = number of data sets (txt files) in folder
#        size = pixel size (32 x 32)
#OUTPUTS,
#                             col 1 = time
#                             col 2 = test data
#                             col 3 = delta t
def piv_readin(base_name_input, data_sets, size):
    #initalize data
    temp_u = np.ndarray([data_sets-1, size, size])
    temp_v = np.ndarray([data_sets-1, size, size])
    count = 0
    x_range = np.arange(1, data_sets)
    for i in x_range:
        #create file name for each txt file
        loc = base_name_input + '/B' + str('{0:05}'.format(i)) + '.txt'
        #read in txt file but skip first row
        temp = pd.read_csv(loc, sep='\t', skiprows=1, header=None)
        #rename columns to designated davis output
        temp.columns = ['Xlocation (mm)', 'Ylocation (mm)', 'U (m/sec)', 'V (m/sec)']
        #reorganize into seperate arrays
        temp_x = np.array(np.reshape(temp['Xlocation (mm)'], (size, -1)))
        temp_y = np.array(np.reshape(temp['Ylocation (mm)'], (size, -1)))
        temp_u[count] = np.array(np.reshape(temp['U (m/sec)'], (size, -1)))
        temp_v[count] = np.array(np.reshape(temp['V (m/sec)'], (size, -1)))
        count+=1
        #print(i/data_sets*100), end='\r' ),
    x_axis = temp_x[0]
    y_axis = temp_y[:,0]
    print('Done Read in!')
    return(x_axis, y_axis, temp_u, temp_v)


###################################################
#############                 #####################
