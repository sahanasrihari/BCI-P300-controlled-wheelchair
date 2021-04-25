import sys 
from pandas import read_csv
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
from scipy import signal 
import time 

#Function for detrending data
def baseline_subtraction(data):
	detrended_data= signal.detrend(data)   #using detrend function
	return detrended_data
#Function for bandpass filtering
def butter_bandpass(lowcutoff,highcutoff,fs,order):
	nyq=0.5*fs #setting nyquit rate 
	low=lowcutoff/nyq   #low cut off frequency
	high=highcutoff/nyq    #high cut off frequency
	b,a= signal.butter(order,[low,high],btype='band',analog=False,output='ba')     #IIR butterworth bandpass filter
	return b,a
#function for notch filter
def notch(notch_low,notch_high,fs,order):
	nyq=0.5*fs         #setting nyquit rate
	low_bandstop=notch_low/nyq         #low cut off frequency
	high_bandstop=notch_high/nyq       #high cut off frequency
	b1,a1=signal.iirfilter(order,[low_bandstop,high_bandstop],btype='bandstop',analog=False,ftype='butter') ##IIR butterworth notch filter
	return b1,a1 

#function for passing signal thrigh bandpass and notch filter
def bandpass_filter(b1,a1,b,a,data):
	y_notch= signal.filtfilt(b1,a1,data,axis=0)
	y=signal.filtfilt(b,a,y_notch,axis=0)
	return y

'''def freq_domain(y,fs):
	fft=np.fft.fft(y)
	n=len(fft)
	fft=fft[0:n/2]
	freq=np.linspace(0,fs/2,n/2)
	return fft,freq

def reshape(y):
    y=np.reshape(y,(np.shape(y)[0],1))
    return y'''
 
    
#setting parameters
fs=250
notch_low=49
notch_high=51

lowcutoff=0.1
highcutoff=30


#loading data files
data_filt=read_csv('/Applications/MATLAB/filtered files/Rak_1.csv')
data_filt.columns=['Time','Epoch','Fz','Cz','Pz','Oz','P3','P4','PO7','PO8','Col2','Col3','Col4','Col5','Col6','Col7']
#data_filt_Saha=read_csv('/Users/sahanasrihari/Downloads/Data_saha_P300.csv')
#data_filt_Saha.columns=['Time','Epoch','Fz','Cz','Pz','Oz','P3','P4','PO7','PO8','Col2','Col3','Col4','Col5','Col6','Col7']
data_filtOpen=read_csv('/Applications/MATLAB/filtered files/Rak_1_filt.csv')
data_filtOpen.columns=['Time','Epoch','Fz','Cz','Pz','Oz','P3','P4','PO7','PO8','Col2','Col3','Col4','Col5','Col6','Col7']

checkFilt = data_filtOpen.Pz

#Channel filtered data
x1 = data_filt.Fz
x2 = data_filt.Cz
x3 = data_filt.Pz
x4 = data_filt.Oz
x5 = data_filt.P3
x6 = data_filt.P4
x7 = data_filt.PO7
x8 = data_filt.PO8

#dataSaha = data_filt_Saha.Cz



#filter parameters set
b1,a1=notch(notch_low,notch_high,fs,4)
b,a=butter_bandpass(lowcutoff,highcutoff,fs,4)

#detrended data

detrended_data1=baseline_subtraction(x1)
detrended_data2=baseline_subtraction(x2)
detrended_data3=baseline_subtraction(x3)
detrended_data4=baseline_subtraction(x4)
detrended_data5=baseline_subtraction(x5)
detrended_data6=baseline_subtraction(x6)
detrended_data7=baseline_subtraction(x7)
detrended_data8=baseline_subtraction(x8)


#Filtered data
y1=bandpass_filter(b1,a1,b,a,detrended_data1)
y2=bandpass_filter(b1,a1,b,a,detrended_data2)
y3=bandpass_filter(b1,a1,b,a,detrended_data3)
y4=bandpass_filter(b1,a1,b,a,detrended_data4)
y5=bandpass_filter(b1,a1,b,a,detrended_data5)
y6=bandpass_filter(b1,a1,b,a,detrended_data6)
y7=bandpass_filter(b1,a1,b,a,detrended_data7)
y8=bandpass_filter(b1,a1,b,a,detrended_data8)

#plt.plot(checkFilt)
#plt.plot(y3)
#plt.show()

#filtSaha = bandpass_filter(b1,a1,b,a,detrendedSaha)
 
'''y1=reshape(y1)
y2=reshape(y2)
y3=reshape(y3)
y4=reshape(y4)
y5=reshape(y5)
y6=reshape(y6)
y7=reshape(y7)
y8=reshape(y8)
'''

'''val = np.concatenate((y3,y4,y5,y6,y7,y8),axis=1)
print(val.shape)
np.savetxt('RakFilt8s.csv',val,delimiter=',')'''

plt.title("Filtered signal comparison between OpenVibe and Python filters")
plt.subplot(2,1,1)
plt.plot(checkFilt[0:1000],'r',label="OpenVibe")
plt.legend()
plt.plot(y3[0:1000],'b',label="Python")
plt.legend()


'''plt.subplot(2,1,2)
plt.plot(detrendedSaha[1000:2000])
plt.plot(filtSaha[1000:2000])
plt.ylim(-200,200)'''

#plt.show()



