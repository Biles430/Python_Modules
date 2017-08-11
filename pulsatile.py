

#!/usr/bin/python
# Filename: HotWire.py

################### IMPORTS #######################
import pandas as pd
from pandas import DataFrame
import numpy as np
import h5py
from pylab import *
from math import atan2
import time_series as ts
#######################################################
    #LAST UPDATED : 7-14-16
#######################################################

#############################################################################
####################  FFT Peak  #####################
#   FUNCTION
#This function takes the real portion of an FFT and determines the maximum spectral peak
#   UPDATED: 7-14-2016
#   INPUTS
#y_freq = frequency vector corresponding to fft data
#y_fft = fft data
#   OUTPUTS,
#max_fft = maximum fft value
#max_freq = frequency corresponding to maximum fft value
#max_pos = vector #position for maximum fft/freq
#   NOTES

def peak_freq(data, delta):
    N = len(data)
    # subtract mean from time series
    y = data - np.mean(data)
    #compute Fourier Transform
    y_fft = np.real((1/N)*np.fft.fft(y)) #be sure to divide by N
    #caluclate frequencies
    y_freq = np.fft.fftfreq(len(y), delta)
    #take abs of fft data to find max pos
    y_fft_abs = np.abs(y_fft)
    max_fft = max(y_fft_abs)
    temp = [i for i, j in enumerate(y_fft_abs) if j == max_fft]
    max_pos= temp[0]
    max_freq = np.abs(y_freq[int(max_pos)])
    return [max_fft, max_freq]

#############################################################################
####################  Determine Sine Function  #####################
#   FUNCTION
#This function takes in a time vector, periodic signal and frequency of signal and
#returns the amp, phase and bias of the fitted sine wave
#   Source
# http://exnumerus.blogspot.com/2010/04/how-to-fit-sine-wave-example-in-python.html
#   UPDATED: 7-14-2016
#   INPUTS
#tList = time vector (sec)
#yList = periodic signal with same len as tList
#freq = freq of periodic signal (Hz)
#   OUTPUTS,
#phase = phase of signal (radians)
#amp = amplitue of signal
#bias = bias of signal
#   NOTES
#y = amp*np.sin(t*freq*2*np.pi + phase) + bias

def fitSine(tList,yList,freq):
   '''
       freq in Hz
       tList in sec
   returns
       phase in radians
   '''
   b = matrix(yList).T
   rows = [ [sin(freq*2*pi*t), cos(freq*2*pi*t), 1] for t in tList]
   A = matrix(rows)
   (w,residuals,rank,sing_vals) = lstsq(A,b)
   phase = atan2(w[1,0],w[0,0])#*180/pi
   amplitude = norm([w[0,0],w[1,0]],2)
   bias = w[2,0]
   return (phase,amplitude,bias)

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

#############################################################################
###################  BIN PHASE AVERAGE 2 data sets  #####################
#   FUNCTION
#This function takes in a time series and performs a phase average based on
# supplied freqeucny
#   UPDATED: 08-10-2017
#   INPUTS
#time = time vector (sec)
#data_1 = trace signal (sinusoidal velocity)
#data_2 = other data signal
#avg_freq = freq of periodic signal (Hz)
#num_bins = final number of bins for phase average
#   OUTPUTS,
#phase_avg = phase avg
#   NOTES
#phase avg data will start from first upward zero crossing
# skipping all data before and then will end on last upward zero	
# crossing skipping all data afterwards


def Phase_Avg(time, data1, data2, avg_freq, sample_freq, num_bins):
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

	#check to make sure there is hihg enough data resolution for number of bins selected
	if 1/avg_freq/num_bins*sample_freq < 1:
		print('Number of bins greater than sampling resolution, reduce num_bins')
	##create ref time signal to bin data based off of
	#width of bin in seconds
	bin_width = 1.0/avg_freq/num_bins
	##create time vector based on overall time and bin width
	time_ref = np.arange(0, 1/avg_freq+bin_width, bin_width)
	##create reference trace signal which will be used to identify zero crossings
	data_ref = data1 - np.mean(data1)
	#find first derivative
	dydx_ref = ts.Richardson_nonunif(time, data1)
	#length of data set
	N = len(data1)
	#find all upward zero crossings
	zero_crossing = dict()
	count  = 0
	for j in range(0, int(N)-1):
		#find upward zero crossing
		if data_ref[j] < 0 and data_ref[j+1] > 0 and dydx_ref[j] > 0:
			zero_crossing[count] = j
			count+=1
	#setup progressbar
	printProgressBar(0, count*2, prefix = 'Finalizing Data:', suffix = 'Complete', length = 50)
	##########
	###MEAN###
	#find mean values
	#j = zero crossing position
	#jj = bin number
	#jjj = data position inbetween zero crossings (j -> j+1)
	#setup data binning
	data_bin1 = np.zeros(num_bins)
	data_bin1_count = np.zeros(num_bins)
	data_bin2 = np.zeros(num_bins)
	data_bin2_count = np.zeros(num_bins)
	#step through all zero crossings to total-1 as below utilizes j to j+1
	for j in range(0, count-1):
		#create a time vecotr based on data which can be compared to sinusoidal time
		# in order to seperate data into bins
		bin_time = time[zero_crossing[j]:zero_crossing[j+1]]-time[zero_crossing[j]]
		bin_count = 0
		#step through each bin
		for jj in range(0, num_bins):
			#step through data inbetween zero crossings
			#include first point	
			#exclude last point
			for jjj in range(int(zero_crossing[j]), int(zero_crossing[j+1])-1): 
				#determine if data time fits into ref time for chosen bin
				if bin_time[jjj - int(zero_crossing[j])] > time_ref[jj] and bin_time[int(jjj - zero_crossing[j])] < time_ref[jj+1]:
					#sum up data into appropriate bin
					data_bin1[jj] = data1[jjj] + data_bin1[jj]
					data_bin2[jj] = data2[jjj] + data_bin2[jj]
					data_bin1_count[jj]+=1
					data_bin2_count[jj]+=1
		#update progress bar
		printProgressBar(j, count*2, prefix = 'Binning Data:', suffix = 'Complete', length = 50)
	#avg to find final phase avg values
	data1_phaseavg = data_bin1/data_bin1_count
	data2_phaseavg = data_bin2/data_bin2_count
	#################
	###Fluctuating###
	#reuse same procedure but determine rms values
	#j = zero crossing position
	#jj = bin number
	#jjj = data position inbetween zero crossings (j -> j+1)
	#setup data binning
	data_bin1 = np.zeros(num_bins)
	data_bin1_count = np.zeros(num_bins)
	data_bin2 = np.zeros(num_bins)
	data_bin2_count = np.zeros(num_bins)
	#step through all zero crossings to total-1 as below utilizes j to j+1
	for j in range(0, count-1):
		#create a time vecotr based on data which can be compared to sinusoidal time
		# in order to seperate data into bins
		bin_time = time[zero_crossing[j]:zero_crossing[j+1]]-time[zero_crossing[j]]
		bin_count = 0
		#step through each bin
		for jj in range(0, num_bins):
			#step through data inbetween zero crossings
			#include first point	
			#exclude last point
			for jjj in range(int(zero_crossing[j]), int(zero_crossing[j+1])-1): 
				#determine if data time fits into ref time for chosen bin
				if bin_time[jjj - int(zero_crossing[j])] > time_ref[jj] and bin_time[int(jjj - zero_crossing[j])] < time_ref[jj+1]:
					#sum up data into appropriate bin
					data_bin1[jj] = (data1[jjj] - data1_phaseavg[jj])**2  + data_bin1[jj]
					data_bin2[jj] = (data2[jjj] - data2_phaseavg[jj])**2 + data_bin2[jj]
					data_bin1_count[jj]+=1
					data_bin2_count[jj]+=1
		#update progress bar
		printProgressBar(j+count, count*2, prefix = 'Binning Data:', suffix = 'Complete', length = 50)
	#avg to find final phase avg values
	data1_phaseavg_prime = np.sqrt(data_bin1/data_bin1_count)
	data2_phaseavg_prime = np.sqrt(data_bin2/data_bin2_count)
	#print('Done!')
	return [data1_phaseavg, data2_phaseavg, data1_phaseavg_prime, data2_phaseavg_prime]
