# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:41:36 2017

@author: tdong
"""

'Import functions'
import numpy as np
import mne
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
import spectrum
from pylab import *
from scipy import signal
import statsmodels.tsa.api as tsa


'Import file (this is a .set file)'
#raw_fname = 'C:/Users/tdong/mne_data/eeglab_data.set'
#Montage = mne.channels.read_montage('eeglab_chan32', path = 'C:/Users/tdong/mne_data/')

#raw = mne.io.read_raw_eeglab(raw_fname, montage = Montage, preload=True)
 
'Import file (this is a .bdf file)'
raw_fname = 'C:/Users/tdong/mne_data/AnKu_Rest_wk5.bdf'#obtain file
Montage = mne.channels.read_montage('2dummy64', path = 'C:/Users/tdong/mne_data/') #obtain montage
raw = mne.io.read_raw_edf(raw_fname, montage=Montage, eog= ['LEOG', 'REOG', 'LML', 'RML', 'LMH', 'RMH', 'Nose', 'SNOse'], preload=True) #get rid of empty channels

'Initial frequency filtering'
raw.filter(0.1,70)#bandpass filter between 0.1 and 70 Hz

'Initialize variables, obtain frequency and # of timepoints, channels'
(data1, times) = raw[::,::]
data1 = data1.T
(timepoints, channels) = data1.shape
frequency = raw.info['sfreq']
errorval = [0]*500 #used for error calculation
errorval2 = [0]*500 #used for error calculation
totaltrials = np.arange(500)+5

'ICA'
ica = ICA(n_components=64)#set up ica
ica.fit(raw) #run the ica
ica.plot_components() #plot the topographs of each component
Montage.plot(show_names=True) #plot the electrode locations
ica.plot_properties(raw, picks=0)#plot the properties of a single component
ica.plot_sources(raw)#plot the timecourse of each component
ica.plot_overlay(raw, exclude=[1,2,5,15,22,23,26,44]) #plot the proposed transformation
ica.apply(raw,exclude=[1,2,5,15,22,23,26,44]) #apply the transformation
   
'Change-able variables'
channeltopick = 49
thetathreshold = 0.5

'Pick 1 channel'
(newdata, newtimes) = raw[::,::]
channel1data = newdata[channeltopick,::]#pick a single channel

'pick order based on AIC, takes a long time'
order = arange(2, 100)
rho = [spectrum.aryule(channel1data, i, norm='biased')[1] for i in order]
plt.plot(order, spectrum.AIC(len(last1second), rho, order), label='AIC')
index_min = np.argmin(spectrum.AIC(len(last1second),rho,order)) #pick minimum rho
plt.title(index_min)

'choose just 1 second of 1 channel'
for abba in totaltrials: #for loop to do multiple second chunks
    choiceofsecond = abba
    last1second = channel1data[-512*choiceofsecond:-512*(choiceofsecond-1)]#pick a second chunk
    last1second = last1second - mean(last1second) #remove mean
    
    'autoregressive spectral analysis'
    [ar, var, reflec] = spectrum.aryule(last1second, index_min, norm= 'biased') 
    psd = spectrum.arma2psd(ar) #power spectrum analysis
    freqvals = np.linspace(0,frequency/2,len(psd)/2) #frequencies
    #plt.plot(freqvals,psd[0:round(len(psd)/2)],label='AR')
    #plt.axis([0,20,0,1500])
    
    'fourier transform'
    #ff = fft(channel1data)
    #freqvals = np.fft.fftfreq(len(channel1data), 1.0/frequency)
    #powerspectrumvals = abs(ff[0:round(len(ff)/2)])**2
    #figure()
    #plt.plot(freqvals[0:round(len(ff)/2)],powerspectrumvals,label='FFT')
    #plt.axis([0,10,0,0.01])
    #plt.legend()
    #plt.show()
    
    'pick frequency'
    freqrange = np.where((freqvals >= 7) & (freqvals <=10)) #frequency range must be between 4Hz and 9Hz
    newpowerspectrumvals = psd[0:round(len(psd)/2)][freqrange] #limit frequency range
    x = 0 #initialize var to track how many points you cut off from the lower range
    y = 0 #initialize var to track how many points you cut off fromt he upper range
    totalpower = np.trapz(newpowerspectrumvals, dx=1.0/256) #calculate area under curve
    theta = totalpower #current area under curve
    
    while theta > thetathreshold * totalpower: #when theta reaches threshold, stop
    #shorten from the end that maintains the highest theta
        if np.trapz(newpowerspectrumvals[1:],dx=1.0/256) > np.trapz(newpowerspectrumvals[:-1], dx=1.0/256):
            x = x + 1
            newpowerspectrumvals = newpowerspectrumvals[1:]
        else:
            y = y +1
            newpowerspectrumvals = newpowerspectrumvals[:-1]
        theta = np.trapz(newpowerspectrumvals, dx = 1.0/256)
        
    #obtain the new narrowed frequency range
    if x == 0:
        newfreqrange = freqrange[0][:-y]
    elif y == 0:
        newfreqrange = freqrange[0][x:]
    else:
        newfreqrange = freqrange[0][x:-y]
    
    #minimum and maximum frequencies    
    minfreq = freqvals[newfreqrange][0]
    maxfreq = freqvals[newfreqrange][-1]
    
    'bandpass filter'
    b,a = signal.butter(1, [minfreq/(0.5*512), maxfreq/(0.5*512)], btype='band') #use min and max freq to bandpass filter
    cleanlast1second = signal.filtfilt(b, a, last1second) #zero phase band pass
    
    'time series forward prediction'
    predictionoverlaplength = 0.1 #how far forward you want to predict, in seconds
    frames = round(predictionoverlaplength*len(cleanlast1second)) #calculate that in frames
    ARmodel= tsa.AR(cleanlast1second[frames:(-1*frames)]) #get autoregressive model
    ARmodelfit = ARmodel.fit(ic='aic') #fit the model
    ARmodelpredict = ARmodel.predict(params=ARmodelfit.params, start = len(cleanlast1second)-2*frames, end = len(cleanlast1second)+0*frames, dynamic = True) #predict
    predicteddata = np.concatenate((cleanlast1second[:-frames],ARmodelpredict)) #get predicted data
    
    'plot the prediction'
    #plt.figure()
    #plt.plot(predicteddata,label='predict')
    #plt.plot(cleanlast1second, label='orig')
    #plt.plot(cleanlast1second[:(-1*frames)])
    #plt.legend()
    
    'Hilbert transform to get instantaneous phase and freq'
    Hilberttransform = signal.hilbert(predicteddata) #perform hilbert transform
    inst_phase = np.angle(Hilberttransform) #get the phases at each timepoint
    inst_freq = np.diff(inst_phase)/(2*np.pi)*frequency #get the frequencies at each timepoint
    phaseguess = degrees(inst_phase[512]) #instant phase at current time point
    freqguess = inst_freq[512] #instant frequency at current time point
    
    #'plot signal based on calculated instantaneous phase'
    #regenerated_carrier = np.cos(inst_phase)
    #plt.plot(regenerated_carrier)
    
    'calculate time delay'
    timedelay = (1.0/freqguess) * (phaseguess)/360 #get timedelay
    timedelay = (1.0/(2*freqguess)) - timedelay #get time to next trough
    timedelayframes = int(round(timedelay * 512)) #get timedelay in frames
    
    'plot when the stimulus would be given'
    ##plt.figure()
    ##plt.plot(signal.filtfilt(b, a, channel1data[-512*choiceofsecond:-512*(choiceofsecond-2)]))
    #plt.plot(predicteddata,label='predict')
    ##plt.axvline(x=512+timedelayframes)
    ##plt.title(theta/np.trapz(psd[0:round(len(psd)/2)],dx=1.0/256))
    #plt.axvline(x=512)
    #plt.plot(regenerated_carrier*.000001)
    
    'calculate error' #error is based on the deviation from Hilbert of actual EEG data
    last2seconds = channel1data[-512*choiceofsecond:-512*(choiceofsecond-2)]
    last2seconds = last2seconds - mean(last2seconds) 
    cleanlast2seconds = signal.filtfilt(b, a, last2seconds) 
    Hilberttransform2 = signal.hilbert(cleanlast2seconds)
    inst_phase2 = np.angle(Hilberttransform2)
    inst_freq2 = np.diff(inst_phase2)/(2*np.pi)*frequency
    phaseguess2 = degrees(inst_phase2[512])
    freqguess2 = inst_freq2[512]
    timedelay2 = (1.0/freqguess2) * (phaseguess2)/360 #get timedelay
    timedelay2 = (1.0/(2*freqguess2)) - timedelay2 #get time to next trough
    timedelayframes2 = int(round(timedelay2 * 512)) #get timedelay in frames
    
    errorval[abba-5] = timedelayframes2 - timedelayframes
    errorval2[abba-5] = 180 - abs(180 - abs(phaseguess2 - phaseguess))

'Plot error'
plt.figure()
plt.hist(errorval,bins=100)
plt.axvline(x=median(errorval))
plt.figure()
plt.hist(errorval2,bins=100)
plt.axvline(x=median(errorval2))
    





    

