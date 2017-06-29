# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:40:09 2017

@author: tdong super

"""

MODE_STD = 0 # standard
MODE_POW = 1 # maybe no use: power
MODE_TWN  =  2  # twin
MODE_DUA = 3 # dual.. use this
# waveform
WF_MONO = 0 # monophasic .. use this
Vf_bif  =  1  # Bifosis
WF_HALF = 2 # half sign
WF_BIBU = 3 # biphasic burst

## settings
t_range = [4, 7] # time range for TMS (sec)
t_init_wait = 3 # time for start TMS (sec)
stim_mode = MODE_DUA # stimulation mode. ref. manual p26. export of data(COM2)
wave_form = WF_MONO # stimulation waveform

## settings: normally unecessary to change after completing setup
p_address = '0xEFF8' # parallel port address
dur_cond_trig = 0.3 * 0.001 # ms -> sec: duration of trigger for conditioning pulse
serial_com_prt = 2 # e.g. COM11 -> 10, COM12 -> 11

## define function
import sys
sys.path.append('./')
import crc8dallas # ref https://gist.github.com/eaydin/768a200c5d68b9bc66e7
def get_com_values(stim_mode, wave_form):
    # ref. manual p26 Export of data(COM2)
    # output bytearray for COM2 using intensity, stim mode, stim waveform, etc
    # params:
    # cond: one of ls_cond
    # Stim_mode: one or stimulation mode (MODE_hoge)
    # wave_form: one of waveform (WF_fuga)
    cond_intensity = 40
    test_intensity = 60
    stim_mode_waveform = int(format(wave_form<<2,'b'),2) + int(format(stim_mode,'b'),2)
    STIM_TYPE = '01' # amplitude ref. manual p26. Export of data(COM2)
    check_string = STIM_TYPE + format(cond_intensity,'02x') + format(test_intensity,'02x') + format(stim_mode_waveform,'02x')
    crc8checksum = crc8dallas.calc(check_string.upper())
    data_bytes = bytearray([0xfe, 0x04, 0x01, cond_intensity, test_intensity, stim_mode_waveform,int(crc8checksum,16),0xff])
    return data_bytes

import numpy 
import serial
from psychopy import parallel

# prepare for serial communication
ser = serial.Serial(serial_com_prt, 38400)

parallel.setPortAddress(address=p_address)

prt_num_test = int('00000001',2)
dur_test_trig = 5 * 0.001 #ms --> sec

#send message to parallel port
parallel.setData(int(prt_num_test))

#send message to serial port
serial_bytearray = get_com_values(stim_mode, wave_form)
ser.write(serial_bytearray)







