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

(newdata, newtimes) = raw[::,::]
channel1data = newdata[49,::]
last1second = channel1data[-512*2:-512]
last1second = last1second - mean(last1second)

'pick order based on AIC'
#order = arange(1, 511)
#rho = [spectrum.aryule(last1second, i, norm='biased')[1] for i in order]
#plot(order, spectrum.AIC(len(last1second), rho, order), label='AIC')
#index_min = np.argmin(spectrum.AIC(len(last1second),rho,order))

'autoregressive spectral analysis'
[ar, var, reflec] = spectrum.aryule(last1second, 511, norm= 'biased')
psd = spectrum.arma2psd(ar)
freqvals = np.linspace(0,frequency/2,len(psd)/2)
plt.plot(freqvals,psd[0:round(len(psd)/2)],label='AR')
plt.yscale('log')
plt.axis([0,50,0,1000000])



'fourier transform'
ff = fft(last1second)
freqvals = np.fft.fftfreq(len(last1second), 1.0/frequency)
powerspectrumvals = abs(ff[0:round(len(ff)/2)])**2
plt.plot(freqvals,powerspectrumvals,label='FFT')
plt.yscale('log')
plt.axis([0,50,0,1000000])
plt.legend()
plt.show()


