#!/usr/bin/python
# Filename: time_series.py

################### IMPORTS #######################
import pandas as pd
from pandas import DataFrame
import numpy as np

#############################################################################
#################### Perform Discrete Fourier Transform #####################
    #FUNCTION
#this function will take a time series and compute the DFT
#Note this function uses convections from OE 896 (ref. pg. 5-12 in notes)
#   UPDATED: 1-21-2016
#   INPUTS
#g = time series
#delta = sampling interval
#   OUTPUTS,
#f = computed fourier frequencies
#G = amplitude of fourier transform

def dft(g, delta):
    N1 = len(g)
    #initalize ouputs
    G = np.zeros(N1, dtype = np.complex128)
    f = np.zeros(N1)
    #caluclate half of data set (int() rounds down)
    Nhalf = int(N1/2)
    for j in range(-Nhalf,Nhalf+1):
        sum1=0
        sum_1=0
        for n in range(-Nhalf,Nhalf+1):
            #calc. fourier freq
            f_j=j/(N1*delta)
            t_n=n*delta
            #calc. FT(g) @ freq j
            sum1=g[n+Nhalf]*np.exp(-1j*2*np.pi*f_j*t_n)
            sum_1=sum_1+sum1
        f[j+Nhalf]=f_j
        G[j+Nhalf]=sum_1/N1
    return(f, G)

#############################################################################
#################### Hyperbolic Tangent Spectral Filter #####################
    #FUNCTION
#this function takes a symmetric 2-sided Fourier Transform and applies a filter
#which uses a hyperbolic tangent transition as cutoff
#Design for removing power noise
#   UPDATED: 4-18-2016
#   INPUTS
#g = time series
#delta = sampling interval
#   OUTPUTS,
#f = computed fourier frequencies
#G = amplitude of fourier transform

def tanh_filt(cutoff_freq, N, delta1):
    #find position to center filter transition about
    cutoff_pos = int(np.floor(cutoff_freq/(1.0/(N*delta1))))
    #initalize filter
    tanh_filter = np.ones([N])
    tanh_filter[cutoff_pos:] = 0.0
    ####      CREATE TANH TRANSITION     #####
    #need to go from 0 -> 2pi with sampel interval as seen in time series
    start_position = 2*np.pi*(1.0/(delta1*N))
    #Top curve
    temp1 = np.arange(start_position, 0, -delta1)
    y1 = np.tanh(temp1*(delta1*N))
    #Bottom curve
    temp2 = np.arange(0, start_position, delta1)
    y2 = (np.tanh(temp2*(delta1*N))-1)*-1.0
    #determine how much profiles should overlap
    overlap = int(np.floor(len(y2)*.1))
    #cut section from second series
    temp4 = y2[overlap:]
    #initalize length of full transition
    tanh_tran = np.zeros(int(len(y1)*2.0 - overlap*2.0))
    #input first function
    tanh_tran[0:int(len(y1))] = y1
    #input second function
    tanh_tran[int(len(y1)-overlap):] = temp4
    #reverse Tanh transition for other side
    tanh_tran_reverse = (tanh_tran-1)*-1.0
    ### put filter section into full filter ##
    tanh_filter[cutoff_pos - int(np.floor(len(tanh_tran)/2)):cutoff_pos + int(np.floor(len(tanh_tran)/2))] = tanh_tran
    # opposite side
    cutoff_pos_2 = N-cutoff_pos
    tanh_filter[cutoff_pos_2 - int(np.floor(len(tanh_tran)/2)):cutoff_pos_2 + int(np.floor(len(tanh_tran)/2))] = tanh_tran_reverse
    #set after filter to 1
    tanh_filter[cutoff_pos_2 + int(np.floor(len(tanh_tran)/2)):] = 1
    return(tanh_filter)

#####################################################################
#############    Probability Density Function   #####################
#   FUNCTION
#This function takes the input data and creates a pdf based on input bin_size
# and range1
#   NOTES
# Created From Hw#4 Time Series Analysis
# For ref. see notes pg 43-47
#   UPDATED: 2-17-2016
#   INPUTS
#data = data to create pdf from
#bin_size = bin width for pdf
#range1 = how far to examine from the mean
#OUTPUTS,
#returndata = returndata[0] = bins
#    = returndata[1] = counts

def pdf1(data, bin_size, range1):
    #data = DataFrame(data)
    mean1 = np.nanmean(data)
    var1 = np.nanvar(data)
    #set up bins
    bins = np.arange(mean1-range1+bin_size/2, mean1+range1+bin_size/2, bin_size )
    #initalize counts file
    counts = np.zeros(len(bins))
    #count number of non NAN values
    counter = np.count_nonzero(~np.isnan(data))

    #step through bins
    for k in range(0, len(bins)-1):
        #for each bin count data in bin width
        counts = len(data[np.logical_and(data > bins[k], data <= bins[k])])

    counts = counts/(counter)
    return(bins, counts)

#####################################################################
#############    Probability Density Function   #####################
#   FUNCTION
#This function takes the input data and creates a pdf based on input bin_size
# and range1
#   NOTES
# Created From Hw#4 Time Series Analysis
# For ref. see notes pg 43-47
#   UPDATED: 2-17-2016
#   INPUTS
#data = data to create pdf from
#bin_size = bin width for pdf
#range1 = how far to examine from the mean
#OUTPUTS,
#returndata = returndata[0] = bins
#    = returndata[1] = counts

def pdf1(data, bin_size, range1):
    #data = DataFrame(data)
    mean1 = np.nanmean(data)
    var1 = np.nanvar(data)
    #set up bins
    bins = np.arange(mean1-range1+bin_size/2, mean1+range1+bin_size/2, bin_size )
    #initalize counts file
    counts = np.zeros(len(bins))
    #count number of non NAN values
    counter = np.count_nonzero(~np.isnan(data))

    #step through bins
    for k in range(0, len(bins)-1):
        #for each bin count data in bin width
        counts = len(data[np.logical_and(data > bins[k], data <= bins[k])])

    counts = counts/(counter)
    return(bins, counts)

#####################################################################
#############   Compute Cross-Correlation  #####################
#   FUNCTION
#This function takes the input data and performs the autocorrelation for given # of lags
#   NOTES
# Created From Hw#5 Time Series Analysis
# For ref. see notes pg 50-51
#   UPDATED: 4-272016
##  INPUTS
# x = data set
# lags = # of times to lag datasets
# delta = sampling interval (sec)
## OUTPUTS
#temp = output of autocorrelation
##  NOTES
#Created from Hw#5 time series analysis
#For details ref notes pg50-51 coding
#x,y must be the same length

#create function to compute correlation at each time point
def autocorr(x, lags, delta):
    x_mean = x - np.nanmean(x)
    #initalize variables
    temp = np.zeros(lags)
    for j in range(0, lags):
        temp[j] = np.corrcoef(np.array([x_mean[0:len(x)-j], x_mean[j:len(x)]]))[0,1]

    return temp

#####################################################################
#############   Compute Cross-Correlation  #####################
#   FUNCTION
#This function takes the input data and creates a pdf based on input bin_size
# and range1
#   NOTES
# Created From Hw#5 Time Series Analysis
# For ref. see notes pg 50-51
#   UPDATED: 3-6-2016
##  INPUTS
# x = data set 1
# y = data set 2
# lag = amount of time to lag up to (sec)
# delta = sampling interval (sec)
## OUTPUTS
#Cor[0] = correlation from 0:lag
#Cor[1] = amount of time lagged
##  NOTES
#Created from Hw#5 time series analysis
#For details ref notes pg50-51 coding
#x,y must be the same length

def cross_cor(x, y, lag, delta):
    #determine number of lag steps based on
    #lag and delta
    lag_steps=lag/delta;
    ## compute initial mean to subtract off
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    x = x-x_mean
    y = y-y_mean
    # initalize return variable
    Cor= np.zeros([2, 2*lag_steps+1])
    ## COMPUTE CORRELATION
    count=0
    neg_lag = 0
    pos_lag = 0
    #compute negative lag correlation
    #step through given lags
    for k in range(int(lag_steps), 0, -1):
        #initalize place holders
        sx=0
        sxx=0
        sxy=0
        sylag=0
        syylag=0
        counter=0
        #step through time series
        for j in range(0, len(x)-k):
            #check to make sure not nans
            if np.isnan(x[j+k]) == False:
                if np.isnan(y[j]) == False:
                    #sum all parts
                    sx=x[j+k]+sx
                    sxx=x[j+k]*x[j+k]+sxx
                    sxy=x[j+k]*y[j]+sxy
                    sylag=y[j]+sylag
                    syylag=y[j]*y[j]+syylag
                    counter+=1
        #caluclate cross-correl for each lag
        num = (sxy/counter)-(sx/counter)*(sylag/counter)
        den = np.sqrt((sxx/counter)-(sx/counter)**2)*np.sqrt((syylag/counter)-(sylag/counter)**2)
        Cor[1][count] = num/den;
        Cor[0][count] = -k*delta;
        neg_lag+=1
        count+=1
    ##  compute positive lag correlation
    #step through given lags
    for k in range(0, int(lag_steps)+1):
        #initalize place holders
        sx=0
        sxx=0
        sxy=0
        sylag=0
        syylag=0
        counter=0
        #step through time series
        for j in range(0, len(x)-k):
            #check to make sure not nans
            if np.isnan(x[j]) == False:
                if np.isnan(y[j+k]) == False:
                    #sum all parts
                    sx=x[j]+sx
                    sxx=x[j]*x[j]+sxx
                    sxy=x[j]*y[j+k]+sxy
                    sylag=y[j+k]+sylag
                    syylag=y[j+k]*y[j+k]+syylag
                    counter+=1
        #caluclate cross-correl for each lag
        num = (sxy/counter)-(sx/counter)*(sylag/counter)
        den = np.sqrt((sxx/counter)-(sx/counter)**2)*np.sqrt((syylag/counter)-(sylag/counter)**2);
        Cor[1][count] = num/den;
        Cor[0][count] = k*delta;
        count+=1
        pos_lag+=1
    return(Cor)


#####################################################################
#############   Compute SPECTRA   #####################
#   FUNCTION
#This function takes the input data and creates a pdf based on input bin_size
# and range1
#   NOTES
# Created From Hw#4 Time Series Analysis
# For ref. see notes pg 43-47
#   UPDATED: 2-24-2016
#   INPUTS
#data = data for spectral analysis
#nensembles = # of ensembles to average
#nbands = # of adjacent frequncie bands to average
#delta = sampling interval of t.s.
#Window 1=Boxcar
#        2=Bartlet Triangle
#        3=Pre-whiten/post-color
#        4=Hanning
# OUTPUTS
#F_final = spectral frequencies
#S_final = sample spectral density
#conf2(1,1) = frequency at which to plot confidence
#conf2(2,1) = spectral for which confidence is determined
#conf2(3,1) = - conf interval
#conf2(4,1) = + conf interval
#DOF = degrees of freedom for spectra
#var2 = varaince of spectra to check and see if var is conserved
#   NOTES
#Spectral analysis comes fromm class:time series analysis
#For referance Spectra on topic see class notes pg. 73-79
#               Whiten-color pg 88-90
# To improve... increase speed
#--TO Fix, post coloring


def spectra(data, delta):
    nensembles=1
    nbands=1
    N = len(data)
    DOF=2*nbands*nensembles
    K = np.floor(N/nensembles)

    #initalize all variables
    #data1 = np.zeros(N,1);
    #data_avg = np.zeros(nensembles,1);
    #var1=np.zeros(nensembles,1);
    G=np.zeros([K])
    A=np.zeros([K])
    B=np.zeros([K])
    S=np.zeros([K])
    F = np.zeros([K])

    #REOMVE MEAN
    data = data - np.mean(data)

    #compute FFT
    G = (1/N)*np.fft.fft(data)
    #seperate into components
    A = np.real(G)
    B = np.imag(G)
    #compute spectral density
    for j in range(0,int(K)):
        S[j] = (delta*K)*(A[j]**2 + B[j]**2)
    #one sided spectrum
    S = S*2
    #create freq
    J = np.arange(0,int(K))
    F = J/(K*delta)

    return(F,S)

#####################################################################
#############   Whittacker Smoother   ###############################
#   FUNCTION
#This function takes in a vector and a smoothing parameter and produces a 
#smoothed vector
#   NOTES
# Taken from: GITHUB: zmeri/gist:3c43d3b98a00c02f81c2ab1aaacc3a49
#Reference: Paul H. C. Eilers. "A Perfect Smoother". 
#	Analytical Chemistry, 2003, 75 (14), pp 3631–3636
#   UPDATED: 3-09-2017
# INPUTS
#y = data
#lmda = smoothing paramter
# OUTPUTS
#z = smoothed dataset
#   NOTES
#
#
def whitsm(y, lmda):
  m = len(y)
  E = sp.sparse.identity(m)
  d1 = -1 * np.ones((m),dtype='d')
  d2 = 3 * np.ones((m),dtype='d')
  d3 = -3 * np.ones((m),dtype='d')
  d4 = np.ones((m),dtype='d')
  D = sp.sparse.diags([d1,d2,d3,d4],[0,1,2,3], shape=(m-3, m), format="csr")
  z = sp.sparse.linalg.cg(E + lmda * (D.transpose()).dot(D), y)
  return z[0]
