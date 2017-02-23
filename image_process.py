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
#This function will read in a series of thermal images
#changed to put first value as zero
#   UPDATED: 2-23-2017
#   INPUTS
#name = name of file
#start = first index
#stop = last index
#file_format is the format of the data file (.csv, .txt ...)
#   OUTPUTS,
#returndata = np.array containing all imported data sets
#   [j] = individual data sets
#   NOTES
#Updated to put ouput as np.array for easier vector manipulation

def readin(name, start, stop, file_format):

    ##read in first image for estimate of size
    N = np.arange(start, stop)
    N = len(N)+1
    temp = np.array(pd.read_csv(name + str(start) + file_format))
    [j,k] = np.shape(temp)
    #initalize data with shape
    test = np.zeros([N, j, k])
    #run through data read in
    for j in range(start, stop+1):
        test[j-start] = np.array(pd.read_csv(name + str(j) + file_format))

    return(test)
