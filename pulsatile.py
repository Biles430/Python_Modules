

#!/usr/bin/python
# Filename: HotWire.py

################### IMPORTS #######################
import pandas as pd
from pandas import DataFrame
import numpy as np
import h5py
from pylab import *
from math import atan2
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
    max_freq = y_freq[int(max_pos)]
    return [max_fft, max_freq, max_pos]

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

   #############################################################################
   ####################  BIN PHASE AVERAGE PIV  #####################
   #   FUNCTION
   #This function takes in a time series and performs a phase average based on
   # supplied freqeucny
   #   UPDATED: 7-17-2016
   #   INPUTS
   #time = time vector (sec)
   #data = periodic signal with same len as tList
   #avg_freq = freq of periodic signal (Hz)
   #num_bins = final number of bins for phase average
   #   OUTPUTS,
  #phase_avg = phase avg
   #   NOTES
#assumes 64 by 64 pixel FOV


def Phase_Avg(time, data_u, data_v, avg_freq, num_bins):
    #width of bin in seconds
    bin_width = 1.0/avg_freq/num_bins
    #create time vector
    time2 = np.arange(0, time[-1] , bin_width/2.0)
    ###### initalize variables ###########
    u_phase_avg = np.zeros([int(1.0/avg_freq/bin_width), 64, 64])
    u_phase_avg_prime = np.zeros([int(1.0/avg_freq/bin_width), 64, 64])
    v_phase_avg = np.zeros([int(1.0/avg_freq/bin_width), 64, 64])
    v_phase_avg_prime = np.zeros([int(1.0/avg_freq/bin_width), 64, 64])
    #create new time vector which includes an end point for the last bin
    # final point = final time
    time_phaseavg = np.zeros(len(time2)+1)
    time_phaseavg[:len(time2)] = time2
    time_phaseavg[len(time2)] = time[-1]  #createfinal time point
    count1 = 0
    for q in range(0, 64):
       for i in range(0, 64):
            u_phase_sum = np.zeros(int(len(time2)/2.0))
            v_phase_sum = np.zeros(int(len(time2)/2.0))
            phase_count = np.zeros(int(len(time2)/2.0))
            ### Bin MEAN velocity ###
            for k in range(0, len(data_u[:, q, i])):
                count = -1
                #step through time based on bin size
                for j in range(1, len(time2), 2):
                    #set value to be binned
                    value_u = data_u[k, q, i]
                    value_v = data_v[k, q, i]
                    time_value = time[k]
                    count+=1
                    #step through time to find correct time/phase based bin
                    if time_phaseavg[j-1] < time_value <= time_phaseavg[j+1]:
                        u_phase_sum[count] = u_phase_sum[count] + value_u
                        v_phase_sum[count] = v_phase_sum[count] + value_v
                        phase_count[count] += 1
                        break
            ####   Bin PRIME velocity ###
            u_phase_sum_prime = np.zeros(int(len(time2)/2.0))
            v_phase_sum_prime = np.zeros(int(len(time2)/2.0))
            phase_count_prime = np.zeros(int(len(time2)/2.0))
            for k in range(0, len(data_u[:, q, i])):
                count = -1
                #step through time based on bin size
                for j in range(1, len(time2), 2):
                    #set value to be binned
                    value_u = data_u[k, q, i]
                    value_v = data_v[k, q, i]
                    time_value = time[k]
                    count+=1
                    #step through time to find correct time/phase based bin
                    if time_phaseavg[j-1] < time_value <= time_phaseavg[j+1]:
                        u_phase_sum_prime[count] = u_phase_sum_prime[count] + np.square(value_u - u_phase_sum[count]/phase_count[count])
                        v_phase_sum_prime[count] = v_phase_sum_prime[count] + np.square(value_v - v_phase_sum[count]/phase_count[count])
                        phase_count_prime[count] += 1
                        break
            #collapse all phase
            u_phase_sum2 = np.zeros(int(1.0/avg_freq/bin_width))
            v_phase_sum2 = np.zeros(int(1.0/avg_freq/bin_width))
            phase_count2 = np.zeros(int(1.0/avg_freq/bin_width))
            u_phase_sum_prime2 = np.zeros(int(1.0/avg_freq/bin_width))
            v_phase_sum_prime2 = np.zeros(int(1.0/avg_freq/bin_width))
            phase_count_prime2 = np.zeros(int(1.0/avg_freq/bin_width))
            for k in range(0, num_bins):
                for j in range(0, int(time2[-1]*avg_freq)):
                    shift = num_bins*j
                    u_phase_sum2[k] = u_phase_sum2[k] + u_phase_sum[k+shift]
                    v_phase_sum2[k] = v_phase_sum2[k] + v_phase_sum[k+shift]
                    phase_count2[k] = phase_count2[k] + phase_count[k+shift]
                    u_phase_sum_prime2[k] = u_phase_sum_prime2[k] + u_phase_sum_prime[k+shift]
                    v_phase_sum_prime2[k] = v_phase_sum_prime2[k] + v_phase_sum_prime[k+shift]
                    phase_count_prime2[k] = phase_count_prime2[k] + phase_count_prime[k+shift]
            u_phase_avg[:, q, i] = u_phase_sum2/phase_count2
            u_phase_avg_prime[:, q, i] = np.sqrt(u_phase_sum_prime2/phase_count_prime2)
            v_phase_avg[:, q, i] = v_phase_sum2/phase_count2
            v_phase_avg_prime[:, q, i] = np.sqrt(v_phase_sum_prime2/phase_count_prime2)
            count1+=1
            print(str(count1/(64*64)*100) +'%', end='\r' ),
    print('Done!')
    return [u_phase_avg, v_phase_avg, u_phase_avg_prime, v_phase_avg_prime]
