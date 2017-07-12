# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:41:36 2017

@author: tdong
"""

'Import functions'
import numpy as np
import mne
from matplotlib import pyplot as plt
from mne.preprocessing import ICA, create_ecg_epochs
import spectrum
from pylab import *
from scipy import signal
import statsmodels.api as sm
import statsmodels.tsa.api as tsa




'Import file (this is a .set file)'
#raw_fname = 'C:/Users/tdong/mne_data/eeglab_data.set'
#Montage = mne.channels.read_montage('eeglab_chan32', path = 'C:/Users/tdong/mne_data/')

#raw = mne.io.read_raw_eeglab(raw_fname, montage = Montage, preload=True)
 
'Import file (this is a .bdf file)'
raw_fname = 'C:/Users/tdong/mne_data/AnBe_Rest_wk1.bdf'
Montage = mne.channels.read_montage('biosemi64')

raw = mne.io.read_raw_edf(raw_fname, montage=Montage, eog= ['LEOG', 'REOG', 'LML', 'RML', 'LMH', 'RMH', 'Nose', 'SNOse'], preload=True)

'frequency filtering'
raw.filter(0.1,70)

'Initialize variables'
(data1, times) = raw[::,::]
data1 = data1.T
(timepoints, channels) = data1.shape
frequency = raw.info['sfreq']

'ICA'

ica = ICA(n_components=64)#set up ica
ica.fit(raw) #run the ica

ica.plot_components() #plot the topographs of each component


Montage.plot(show_names=True) #plot the electrode locations

ica.plot_properties(raw, picks=26)#plot the properties of each single component

ica.plot_sources(raw)#plot the timecourse of each component

ica.plot_overlay(raw, exclude=[1,2,3]) #plot the proposed transformation

ica.apply(raw,exclude=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]) #apply the transformation

'choose just 1 second of 1 channel'
for abba in (5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20):
    choiceofsecond = abba
    (newdata, newtimes) = raw[::,::]
    channel1data = newdata[49,::]
    last1second = channel1data[-512*choiceofsecond:-512*(choiceofsecond-1)]
    last1second = last1second - mean(last1second)
    
    
    'pick order based on AIC'
    #order = arange(1, 511)
    #rho = [spectrum.aryule(last1second, i, norm='biased')[1] for i in order]
    #plot(order, spectrum.AIC(len(last1second), rho, order), label='AIC')
    #index_min = np.argmin(spectrum.AIC(len(last1second),rho,order))
    
    'autoregressive spectral analysis'
    [ar, var, reflec] = spectrum.aryule(last1second, 100, norm= 'biased')
    psd = spectrum.arma2psd(ar)
    freqvals = np.linspace(0,frequency/2,len(psd)/2)
    #plt.plot(freqvals,psd[0:round(len(psd)/2)],label='AR')
    #plt.axis([4,9,0,1500])
    
    
    
    'fourier transform'
    #ff = fft(last1second)
    #freqvals = np.fft.fftfreq(len(last1second), 1.0/frequency)
    #powerspectrumvals = abs(ff[0:round(len(ff)/2)])**2
    #figure()
    #plt.plot(freqvals[0:round(len(ff)/2)],powerspectrumvals,label='FFT')
    #plt.axis([0,20,0,0.000001])
    #plt.legend()
    #plt.show()
    
    'pick frequency'
    freqrange = np.where((freqvals >= 4) & (freqvals <=9))
    newpowerspectrumvals = psd[0:round(len(psd)/2)][freqrange]
    
    x = 0
    y = 0
    totalpower = np.trapz(newpowerspectrumvals, dx=1.0/256)
    theta = totalpower
    
    while theta > 0.5 * totalpower:
        if np.trapz(newpowerspectrumvals[1:],dx=1.0/256) > np.trapz(newpowerspectrumvals[:-1], dx=1.0/256):
            x = x + 1
            newpowerspectrumvals = newpowerspectrumvals[1:]
        else:
            y = y +1
            newpowerspectrumvals = newpowerspectrumvals[:-1]
        theta = np.trapz(newpowerspectrumvals, dx = 1.0/256)
        
    if x == 0:
        newfreqrange = freqrange[0][:-y]
    elif y == 0:
        newfreqrange = freqrange[0][x:]
    else:
        newfreqrange = freqrange[0][x:-y]
        
    minfreq = freqvals[newfreqrange][0]
    maxfreq = freqvals[newfreqrange][-1]
    
    'bandpass filter'
    b,a = signal.butter(1, [minfreq/(0.5*512), maxfreq/(0.5*512)], btype='band')
    cleanlast1second = signal.filtfilt(b, a, last1second)
    
    'time series forward prediction'
    
    predictionoverlaplength = 0.1 #in seconds
    frames = round(predictionoverlaplength*len(cleanlast1second))
    ARmodel= tsa.AR(cleanlast1second[frames:(-1*frames)])
    ARmodelfit = ARmodel.fit(ic='aic')
    ARmodelpredict = ARmodel.predict(params=ARmodelfit.params, start = len(cleanlast1second)-2*frames, end = len(cleanlast1second)+0*frames, dynamic = True)
    predicteddata = np.concatenate((cleanlast1second[:-frames],ARmodelpredict))
    
    'plot the prediction'
    #plt.plot(predicteddata,label='predict')
    #plt.plot(cleanlast1second, label='orig')
    #plt.plot(cleanlast1second[:(-1*frames)])
    #plt.legend()
    
    'Hilbert transform to get instantaneous phase and freq'
    Hilberttransform = signal.hilbert(predicteddata)
    inst_amplitude = np.abs(Hilberttransform)
    inst_phase = np.angle(Hilberttransform)
    inst_freq = np.diff(inst_phase)/(2*np.pi)*frequency
    
    phaseguess = degrees(inst_phase[512])
    freqguess = inst_freq[512]
    
    #'plot signal based on calculated instantaneous phase'
    
    #regenerated_carrier = np.cos(inst_phase)
    #plt.plot(regenerated_carrier)
    
    'calculate time delay'
    
    timedelay = (1.0/freqguess) * (phaseguess)/360
    timedelayframes = int(round(timedelay * 512))
    
    'plot when the stimulus would be given'
    plt.figure()
    plt.plot(signal.filtfilt(b, a, channel1data[-512*choiceofsecond:-512*(choiceofsecond-2)]))
    plt.axvline(x=512+timedelayframes)





    

