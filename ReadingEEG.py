# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:41:36 2017

@author: tdong
"""

'Import functions'
import numpy as np
import mne
from matplotlib import pyplot as plot
from mne.preprocessing import ICA, create_ecg_epochs


'Import file (this is a .set file)'
#raw_fname = 'C:/Users/tdong/mne_data/eeglab_data.set'
#Montage = mne.channels.read_montage('eeglab_chan32', path = 'C:/Users/tdong/mne_data/')

#raw = mne.io.read_raw_eeglab(raw_fname, montage = Montage, preload=True)
 
'Import file (this is a .bdf file)'
raw_fname = 'C:/Users/tdong/mne_data/AnBe_Rest_wk1.bdf'
Montage = mne.channels.read_montage('eeglab_chan32', path = 'C:/Users/tdong/mne_data/')

raw = mne.io.read_raw_edf(raw_fname, montage = Montage, preload=True)

'frequency filtering (this is 1 to 40 Hz)'
raw.filter(1,40)

'Initialize variables'
(data1, times) = raw[::,::]
data1 = data1.T
(timepoints, channels) = data1.shape
frequency = raw.info['sfreq']


'ICA'

ica = ICA(n_components=32)#set up ica
ica.fit(raw) #run the ica

ica.plot_components() #plot the topographs of each component


Montage.plot(show_names=True) #plot the electrode locations

ica.plot_properties(raw, picks=4)#plot the properties of each single component

ica.plot_sources(raw)#plot the timecourse of each component

ica.plot_overlay(raw, exclude=[1,2,3]) #plot the proposed transformation

ica.apply(raw,exclude=[1,2,3]) #apply the transformation
