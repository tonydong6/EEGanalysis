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
raw_fname = 'C:/Users/margyela/Google Drive/projects/TMS/ClosedLoopEEG/EEGdata/AnBe_Rest_wk1.bdf'
Montage = mne.channels.read_montage('biosemi64')

raw = mne.io.read_raw_edf(raw_fname, montage=Montage, eog= ['LEOG', 'REOG', 'LML', 'RML', 'LMH', 'RMH', 'Nose', 'SNOse'], preload=True)

'frequency filtering (this is 1 to 40 Hz)'
raw.filter(1,40)

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

#ica.apply(raw,exclude=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]) #apply the transformation
ica.apply(raw,include=26)

'autoregression'

(newdata, newtimes) = raw[::,::]
channel1data = newdata[49,::]
last1second = channel1data[-512*2:-512]
plot(last1second)

Y    = np.fft.fft(last1second)
freq = np.fft.fftfreq(len(last1second), 1.0/512)

figure()
#plot( freq, np.abs(Y) )
#instead!!
plot(freq[1:50],np.abs(Y)[1:50])
figure()
plot(freq[1:50], np.angle(Y)[1:50] )
show()


#Wavelet transformation

from wavelets import WaveletAnalysis

# given a signal x(t)
x = np.random.randn(1000)
# and a sample spacing
dt = 1.0/512

wa = WaveletAnalysis(x, dt=dt)
wa = WaveletAnalysis(last1second, dt=dt)

# wavelet power spectrum
power = wa.wavelet_power
re=wa.wavelet_transform.imag
im=wa.wavelet_transform.real
phase=arctan(im/re)
# scales 
scales = wa.scales

# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()
#How would you plot this?

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
T, S = np.meshgrid(t, scales)
#ax.contourf(T, S, power, 100)
ax.contourf(T, S, phase, 100)
ax.set_yscale('log')
fig.savefig('test_wavelet_phase_spectrum.png')

fig, ax = plt.subplots()
T, S = np.meshgrid(t, scales)
#ax.contourf(T, S, power, 100)
ax.contourf(T, S, power, 100)
ax.set_yscale('log')
fig.savefig('test_wavelet_power_spectrum.png')



plot(abs(np.fft.rfft(last1second)[0:10]))
xlim(0,10)
show()

[ar, var, reflec] = spectrum.aryule(last1second, 15, norm= 'biased')
psd = spectrum.arma2psd(ar)
plot(psd)

noise = randn(1, 1024)
y = lfilter([1], ar, noise); 

ps = np.abs(np.fft.fft(channel1data))**2
time_step = 1.0 / 512
freqs = np.fft.fftfreq(last1second.size, time_step)
idx = np.argsort(freqs)


plt.plot(freqs[idx], ps[idx])
plt.plot(ps[idx][256:280])

axis([0, 2, 0, 100000])

