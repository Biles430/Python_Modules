#!/usr/bin/python
# Filename: PIV.py

###################     IMPORTS #######################
import pandas as pd
from pandas import DataFrame
import numpy as np
import time
import sys

###################################################
############# READ IN PIV DATA #####################
#Function
#this function will read in PIV data output from LaVision software
#The form of the input data is .txt files with 4 columns
#column 1 = X location (mm)
#column 2 = Y Location (mm)
#column 3 = U velocity (m/sec)
#column 4 = V velocity (m/sec)

#UPDATED: 05-05-2017
#INPUTS, base_name_input = Name of Folder with PIV data inside it
#        data_sets = number of data sets (txt files) in folder
#        size = pixel size (32 x 32)
#OUTPUTS,
#                             col 1 = time
#                             col 2 = test data
#                             col 3 = delta t
def piv_readin(base_name_input, data_sets, sizex, sizey):
    #initalize data
    temp_u = np.ndarray([data_sets-1, sizex, sizey])
    temp_v = np.ndarray([data_sets-1, sizex, sizey])
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
        temp_x = np.array(np.reshape(temp['Xlocation (mm)'], (sizex, sizey)))
        temp_y = np.array(np.reshape(temp['Ylocation (mm)'], (sizex, sizey)))
        temp_u[count] = np.array(np.reshape(temp['U (m/sec)'], (sizex, sizey)))
        temp_v[count] = np.array(np.reshape(temp['V (m/sec)'], (sizex, sizey)))
        count+=1
        #print(i/data_sets*100), end='\r' ),
    x_axis = temp_x[0]
    y_axis = temp_y[:,0]
    print('Done Read in!')
    return(x_axis, y_axis, temp_u, temp_v)

###################################################
############# READ IN PIV DATA #####################
#Function
#this function will read in PIV data output from LaVision software
#The form of the input data is .txt files with 4 columns
#column 1 = X location (mm)
#column 2 = Y Location (mm)
#column 3 = U velocity (m/sec)
#column 4 = V velocity (m/sec)

#UPDATED: 08-25-2017
#INPUTS, 
#base_name_input = Name of Folder with PIV data inside it
#data_sets = number of data sets (txt files) in folder
#sizex = image size in x
#sizey = image size in y
#OUTPUTS,
#x_axis =
#y_axis =
#temp_u =
#temp_v = 
def piv_readin_mod(base_name_input, data_sets, sizex, sizey):
    #Print iterations progress
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total:
            print()
    #initalize data
    temp_u = np.ndarray([data_sets-1, sizex, sizey])
    temp_v = np.ndarray([data_sets-1, sizex, sizey])
    count = 0
    x_range = np.arange(1, data_sets)
    #setup progressbar
    printProgressBar(0, len(x_range), prefix = 'Reading In:', suffix = 'Complete', length = 50)
    for i in x_range:
        #create file name for each txt file
        loc = base_name_input + '/B' + str('{0:05}'.format(i)) + '.txt'
        #read in txt file but skip first row
        temp = pd.read_csv(loc, sep='\t', skiprows=1, header=None)
        #rename columns to designated davis output
        temp.columns = ['Xlocation (mm)', 'Ylocation (mm)', 'U (m/sec)', 'V (m/sec)']
        #for j in range(0, len(temp['U (m/sec)'])):
            #temp['U (m/sec)'][j] = float(temp['U (m/sec)'][j].replace(',','.'))
            #temp['V (m/sec)'][j] = float(temp['V (m/sec)'][j].replace(',','.'))
            #temp['Xlocation (mm)'][j] = float(temp['Xlocation (mm)'][j].replace(',','.'))
            #temp['Ylocation (mm)'][j] = float(temp['Ylocation (mm)'][j].replace(',','.'))
        #reorganize into seperate arrays
        temp_x = np.array(np.reshape(temp['Xlocation (mm)'], (sizex, sizey)))
        temp_y = np.array(np.reshape(temp['Ylocation (mm)'], (sizex, sizey)))
        temp_u[count] = np.array(np.reshape(temp['U (m/sec)'], (sizex, sizey)))
        temp_v[count] = np.array(np.reshape(temp['V (m/sec)'], (sizex, sizey)))
        count+=1
        printProgressBar(i, len(x_range), prefix = 'Reading In:', suffix = 'Complete', length = 50)
        #print(i/data_sets*100), end='\r' ),
    x_axis = temp_x[0]
    y_axis = temp_y[:,0]
    print('Done Read in!')
    return(x_axis, y_axis, temp_u, temp_v)

###########################################################
############# Determine Mask Location #####################
#Function
#this function will take in an average PIV velocity field, preferiably u
# and determine the mask location by examining for zero point on the sides.

#UPDATED: 05-30-2017
#INPUTS, 
#u = mean u velocity field

#OUTPUTS,
#mask[0] = location of mask position at bottom of image
#mask[1] = location of mask position on left side of image
#mask[2] = location of mask position on right side of image

def mask_loc(u):
    #determine size in x and y
    [sizey, sizex] = np.shape(u)
    #initalize variables as the outer most boundary so that if no
    #mask is found they will include the whole field
    mask = [0, 0, 0, 0]
    mask[0] = int(0)
    mask[1] = int(sizey)
    mask[2] = int(0)
    mask[3] = int(sizex)
    ##y_mask1##
    #start looking along y axis
    # start at middle and work up to y ->[0]
    for j in range(int(sizey/2), 0, -1):
        if u[j, int(sizex/2)] == 0:
            mask[0] = int(j)
            break
    ##y_mask2##
    #start looking along x axis
    # start at middle and work down to y ->[-1]
    for j in range(int(sizey/2), int(sizey)):
        if u[j, int(sizex/2)] == 0:
            mask[1] = int(j)
            break
    ##x_mask1##
    #start looking along x axis
    # start at middle and work over in x ->[0]
    for j in range(int(sizex/2), 0, -1):
        if u[int(sizey/2), j] == 0:
            mask[2] = int(j)
            break
    ##x_mask2##
    #start looking along x axis
    # start at middle and work over to x ->[-1]
    for j in range(int(sizex/2), int(sizex)):
        if u[int(sizey/2), j] == 0:
            mask[3] = int(j)
            break
    
    print('Mask Found!')
    return(mask)

###########################################################
############# Filter PIV images #####################
#Function
#this function will a piv image dataset and compute the a mean value based on the top
# of the image. 
#	1. compute avg of top of PIV image
#	2. Remove images with ~zero mean (done so std is not skewed)
#	3. Re-compute mean and then std based on top 1/3 of image
#	4. Apply std filer
#UPDATED: 08-25-2017
#INPUTS, 
#u_vel = mean u velocity field
#v_vel = mean v velocity field
#Uinfinity = expected uinfinity of piv images
#sizey = vertical dimmension of image

#OUTPUTS,
#u_vel = mean u velocity field
#v_vel = mean v velocity field
#count = num of bad images

#This function is used to set bad and non-physical images to a nan value
def filt_images(u_vel, v_vel, Uinfinity, sizey):
	#count number of bad images
	count1 = 0
	#initalize the mean values which images will be filtered on
	Umean_top = np.zeros(len(u_vel))
	#compute means for top of images after above zero filter has been applied
	for j in range(0, len(u_vel[0,:])):
		Umean_top[j] = np.mean(np.mean(u_vel[j, int(2*(sizey/3)):-1]))
	####remove all images which have ~zero mean such that when STD filter is appled
	# the average is not skewed to towards zero
	for j in range(0, len(v_vel[0,:])):
		if Umean_top[j] < Uinfinity/10:
			u_vel[0, j] = np.nan
			v_vel[0, j] = np.nan
			count1+=1
	#compute new means for top of images after above zero filter has been applied
	for j in range(0, len(u_vel[0,:])):
		Umean_top[j] = np.mean(np.mean(u_vel[j, int(2*(sizey/3)):-1]))
	####Apply STD filter 
	#number of times to iterate through STD filter   
	num_loops = 4
	#width of filter in STD
	filter_width = 1
	for k in range(0, num_loops):
		#compute mean of top 1/3 of image for STD filtering
		for j in range(0, len(u_vel[0,:])):
			Umean_top[j] = np.mean(np.mean(u_vel[j, int(2*(sizey/3)):-1]))
		#STD filter  
		for j in range(0, len(u_vel[0,:])):
			#remove images with average value less than avg - x*STD
			if Umean_top[j] < np.nanmean(Umean_top) - filter_width * np.sqrt(np.nanvar(Umean_top)):
				u_vel[0, j] = np.nan
				v_vel[0, j] = np.nan
				count1+=1
			#remove images with average value greater than avg - x*STD
			if Umean_top[j] > np.nanmean(Umean_top) + filter_width * np.sqrt(np.nanvar(Umean_top)):
				u_vel[0, j] = np.nan
				v_vel[0, j] = np.nan
				count1+=1
	return(u_vel, v_vel, count1)
