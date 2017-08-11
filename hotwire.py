#!/usr/bin/python
# Filename: HotWire.py

################### IMPORTS #######################
import pandas as pd
from pandas import DataFrame
import numpy as np
import h5py
#######################################################
    #LAST UPDATED : 8-16-16
#######################################################

#############################################################################
####################  Read in Hot Wire text files  #####################
#   FUNCTION
#This function will take a series of text files and arrange them into a
#dataframe for further analysis
#changed to put first value as zero
#   UPDATED: 4-22-2016
#   INPUTS
#name = name of file
#num_files = number of files
#delim = deliminter between values
#file_format is the format of the data file (.csv, .txt ...)
#   OUTPUTS,
#returndata = np.array containing all imported data sets
#   [j] = individual data sets
#   NOTES
#Updated to put ouput as np.array for easier vector manipulation

def readin(name, numfiles, delim, file_format):

    #determine length
    N = len(np.loadtxt((name+str(0)+file_format), dtype=float, delimiter=delim, skiprows=1))
    #initalize size of return data set
    returndata = np.zeros([numfiles, N])
    #step through name of file
    for j in range(0,numfiles):
        file_name = name + str(j) + file_format
        #open data file (could use data path)
        #with open(file_name,'r') as datafile:
        #skip first row and load all data as float
        data = np.loadtxt(file_name, dtype=float, delimiter=delim, skiprows=1)
        #place data in overall array
        returndata[j] = data[0:N]
    return(returndata)

#############################################################################
####################  Determine Wall Norm Position  #####################
#   FUNCTION
#This function creates a set of log spaced points to be used as the wall
#normal positions from a profiling experiment
#Changed to put first position as zero
#   UPDATED: 4-22-2016
#   INPUTS
#L = Lowest Position
#H = Highest Position
#N = Number of steps
#   OUTPUTS,
#x = np.array containing N number of heights

def probe_height(L, H, N):
    #initalize variables
    x = np.zeros(N)
    position = (np.log10(H)-np.log10(L))/(N-1)
    for i in range(0,N):
        x[i] = L*10**((i)*position)
    return(x)

#############################################################################
####################  Calculate BL thickness  #####################
#   FUNCTION
#This function will compute the boundary layer thickness using the 99% method
#Note it is for Twall > Tair
#   UPDATED: 4-06-2017
#   INPUTS
#data = set of mean data values at each position x
#x = wall normal positions
#   OUTPUTS,
#delta = boundary layer thickness
#   UPDATES
#Need to change such that it can caluculate Twall>Tair or for Twall<Tair

def delta(data, x, U_inf, threshold, profile):
    #create place holder
    temp_delta = 0
    N = len(data)
    temp = data/U_inf
    ##0 -> 1
    if profile == 0:
        for j in range(1,N):
            if temp[j] >= threshold:
                if temp[j-1] <= threshold:
                    #interpolate to find true value
                    temp_delta = (threshold-temp[j-1])/(temp[j]-temp[j-1])*(x[j]-x[j-1])+x[j-1]
                    break
        delta_pos = j+1
    ##1 -> 0
    if profile == 1:
        for j in range(1,N):
            if temp[j] <= threshold:
                if temp[j-1] >= threshold:
                    #interpolate to find true value
                    temp_delta = (threshold-temp[j-1])/(temp[j]-temp[j-1])*(x[j]-x[j-1])+x[j-1]
                    break
        delta_pos = j+1
    return(temp_delta, delta_pos)

#############################################################################
####################  Normalize Temp Profile  #####################
#   FUNCTION
#This function normalizes the thermal BL profile and computes
#the analytical profile
#   UPDATED: 2-24-2016
#   INPUTS
#data = set of mean data values at each position x
#x = wall normal positions
#   OUTPUTS,
#delta = boundary layer thickness
#   UPDATES

def T_norm(data, y, delta):
    #theta = pd.DataFrame()
    N = len(data)
    #normalize data
    temp1 = (data-data[N-1])/(data[0]-data[N-1])
    #calculate eta
    temp2 = y/delta
    #calc analytical profile
    temp3 = 1-1.5*temp2+.5*temp2**3
    d = {'Eta': temp2, 'Theta_T': temp1, 'Theta_A': temp3}
    theta = pd.DataFrame(d)
    return(theta)

#############################################################################
####################  DETERMINE AIR PROP  #####################
#   FUNCTION
#This function determines the thermal propeties of air from a ref table
#   UPDATED: 4-19-2016
#   INPUTS
#T_given = input temperature
#x = wall normal positions
#   OUTPUTS,
#k
#rho#nu
#c

#   UPDATES
def air_prop(T_given):
    air_prop_data = pd.read_hdf('data/airprop1.h5', 'air_prop')
    for j in range(0, len(air_prop_data['T'])):
        if T_given < air_prop_data['T'][j]:
            #if air_prop_data['T'][j+1] > T_given:
            delta_T= air_prop_data['T'][j] - air_prop_data['T'][j-1]
            delta_given = T_given - air_prop_data['T'][j]
            rho = (air_prop_data['rho'][j] - air_prop_data['rho'][j-1])/delta_T * delta_given + air_prop_data['rho'][j]
            k = (air_prop_data['k'][j] - air_prop_data['k'][j-1])/delta_T * delta_given + air_prop_data['k'][j]
            cp = (air_prop_data['cp'][j] - air_prop_data['cp'][j-1])/delta_T * delta_given + air_prop_data['cp'][j]
            nu = ((air_prop_data['nu'][j] - air_prop_data['nu'][j-1])/delta_T * delta_given + air_prop_data['nu'][j])*(10**(-6))
            Pr = (air_prop_data['Pr'][j] - air_prop_data['Pr'][j-1])/delta_T * delta_given + air_prop_data['Pr'][j]
            air_prop = {'k':k, 'nu':nu, 'rho':rho, 'cp':cp, 'Pr':Pr}
            return(air_prop)

#############################################################################
####################  Read in Hot Wire text files  #####################
#   FUNCTION
#This function performs a spatial average on a dataset. It was written for
# thermocouple profiles so account for the diam of the TC bulb
#   UPDATED: 8-16-16
#   INPUTS
#data = name of file
#y_pos = number of files
#probe_diam = deliminter between values
#file_format is the format of the data file (.csv, .txt ...)
#   OUTPUTS,
#returndata = np.array containing all imported data sets
#   [j] = individual data sets
#   NOTES
#Updated to put ouput as np.array for easier vector manipulation
def spatial_avg(data, y_pos, probe_diam, walloffset):
    #probe_diam = .00106 #m
    probe_pos = np.zeros(1) + probe_diam/2 + walloffset
    pos = int(y_pos[0])
    data_avg = np.array(data[0])
    while (probe_pos[-1] + probe_diam/2) < y_pos[-1]:
        count  = 0
        sum_data = 0
        for j in range(pos, len(y_pos)):
            if y_pos[j] < (probe_pos[-1] + probe_diam/2):
                count+=1
                sum_data = sum_data + data[j]
            else:
                #if y_pos > then average above summed data and set new y probe pos
                #avg sum datasets
                sum_data = sum_data/count
                #append avg onto new dataset
                data_avg = np.append(data_avg, sum_data)
                #append y pos onto new dataset
                probe_pos = np.append(probe_pos, y_pos[j]+probe_diam/2)
                #save last position so for loop can begin there
                pos  = j
                break
    return(probe_pos, data_avg)

#############################################################################
####################  Perform differentiation on dataset  #####################
#   FUNCTION
#This function performs a differentiation based on the provided inputs. It utilizes a
# richardson nonunifrom approach as recommended by Dr. Ebadi
#   UPDATED: 05-02-17
#   INPUTS
#x= x of dataset
#y = y of dataset
#   OUTPUTS,
#dydx = differentiated dataset
#   NOTES

def richardson(x,y):
    y = np.array(y)
    x = np.array(x)
    m = np.size(y)
    dydx = np.zeros(m)
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    if m == 1:
        dydx[-1] = dydx[0];
    elif m < 5:
        for i in range(1, len(y) - 2):
            dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    else:
        for i in range(1,3):
            dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        for i in range(3, m - 2):
            dydx[i] = (-y[i + 2] + 8*y[i + 1] - 8*y[i - 1] + y[i - 2]) / (6*(x[i + 1] - x[i - 1]));
        i = m - 2
        dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]);

    return dydx

#############################################################################
####################  Perform shear based PST correction #####################
#   FUNCTION
#This function takes a correction factor from S.C.C. Bailey et al and applies it
# to experimental PST datasets
#   UPDATED: 05-18-17
#   INPUTS
#u= u of dataset
#y = y of dataset
#uinf = free stream velocity
#Dp = diameter of PST
#   OUTPUTS,
#ynew = new y positions based on delta_y correction
#   NOTES

def pst_shear_correction(u, y, uinf, Dp):
	#define richardson #
    def richardson(x,y):
        y = np.array(y)
        x = np.array(x)
        m = np.size(y)
        dydx = np.zeros(m)
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        if m == 1:
            dydx[-1] = dydx[0];
        elif m < 5:
            for i in range(1, len(y) - 2):
                dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
            dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            for i in range(1,3):
                dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
            for i in range(3, m - 2):
                dydx[i] = (-y[i + 2] + 8*y[i + 1] - 8*y[i - 1] + y[i - 2]) / (6*(x[i + 1] - x[i - 1]));
            i = m - 2
            dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
            dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]);

        return dydx

    ynew = np.zeros(len(y))
    dudy = ((richardson(y, u)**2)**(1/2))
    alpha = (Dp/ (2*uinf) ) * dudy
    delta_y = (.15 * np.tanh(4 * (alpha)**(1/2) ) ) * Dp
    ynew = y + delta_y
    ynew[0] = 0
    return(ynew)

#############################################################################
####################  Perform new wall based PST correction #################
#   FUNCTION
#This function takes a correction factor from S.C.C. Bailey et al and applies it
# to experimental PST datasets
#   UPDATED: 05-18-17
#   INPUTS
#u= u of dataset
#y = y of dataset
#utau = friction velocity determined from dataset
#Dp = diameter of PST
#   OUTPUTS,
#unew = shifted velocities 
#   NOTES

def pst_wall_correction(u, y, utau, Dp, nu):
    unew = np.zeros(len(u))
    dplus = (Dp*utau) / nu
    udelta = (20*np.exp(-.1*dplus) + 1)*.015 * np.exp(-2.5* (y/Dp - .5))
    unew = (1/ (1-udelta))*u
    return(unew)


