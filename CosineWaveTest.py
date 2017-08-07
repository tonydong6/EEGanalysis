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

'Initialize trials'
trialnumbers = np.arange(499)
frameerror = [0]*500
phaseerror = [0]*500

    
'Create cosine wave with noise, bandpass filtered between 0.1 and 70, mean subtracted'
noise = np.random.normal(0,1,512*500)
frequency = 512
f = 6
sample = 512 * 500
x = np.arange(sample)
#y = np.cos(2 * np.pi * f * x[:-512] / Fs) + noise
#y = np.cos(2 * np.pi * f * x[:-512] / frequency)
realsignal = np.cos(2 * np.pi * f * x / frequency)
noisysignal = realsignal + noise

'AIC'
order = arange(2, 100)
rho = [spectrum.aryule(noisysignal, i, norm='biased')[1] for i in order]
plt.plot(order, spectrum.AIC(len(noisysignal), rho, order), label='AIC')
index_min = np.argmin(spectrum.AIC(len(noisysignal),rho,order)) #pick minimum rho
plt.title(index_min)

for trials in trialnumbers:
    y = noisysignal[512*trials:512*(trials+1)]
    
    b,a = signal.butter(1, [0.1/(0.5*512), 70/(0.5*512)], btype='band') 
    last1second = signal.filtfilt(b, a, y)
    last1second = last1second - mean(last1second)
    
    'Run algorithm'    
    [ar, var, reflec] = spectrum.aryule(last1second, 27, norm= 'biased') #replace the number with order
    psd = spectrum.arma2psd(ar) #power spectrum analysis
    freqvals = np.linspace(0,frequency/2,len(psd)/2)
        
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
        
    b,a = signal.butter(1, [minfreq/(0.5*512), maxfreq/(0.5*512)], btype='band') #use min and max freq to bandpass filter
    cleanlast1second = signal.filtfilt(b, a, last1second) #zero phase band pass
    
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
        
    'calculate error'
    realtroughs = np.where(realsignal<=-0.999)[0]
    frameerror[trials] = min(abs(realtroughs-(512*(trials+1)+timedelayframes))) #in number of frames away
    phaseerror[trials] = abs(-1 - realsignal[512*(trials+1)+timedelayframes]) #in y distance from trough
    
'Plot error'
plt.figure()
plt.hist(frameerror,bins=100)
plt.axvline(x=median(frameerror))
plt.figure()
plt.hist(phaseerror,bins=100)
plt.axvline(x=median(phaseerror))

    
    
    
    
    
        
    
    
        
            
