# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 19:59:05 2017

@author: tdong
"""

'Wavelet transformation'
import numpy as np
from wavelets import WaveletAnalysis
from matplotlib import pyplot as plt

# given a signal x(t)
x = np.random.randn(1000)
# and a sample spacing
dt = 1.0/512

wa = WaveletAnalysis(x, dt=dt)
#wa = WaveletAnalysis(last1second, dt=dt)

# wavelet power spectrum
power = wa.wavelet_power
re=wa.wavelet_transform.imag
im=wa.wavelet_transform.real
phase=np.arctan(im/re)

# scales 
scales = wa.scales

# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()
#How would you plot this?

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