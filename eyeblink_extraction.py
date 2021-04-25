import pywt 
import numpy as np 
from pandas import read_csv
import pandas as pd
import matplotlib 
from scipy import signal 
import matplotlib.pyplot as plt
import sys
import adaptfilt
import csv 

def resample(channel,fs) : 

		m=len(channel)%fs   #resample signal to be multiple of length 256 
		pad=256-m
		n=len(channel)+pad
		chan_resamp=signal.resample(channel,n)
		#print(len(chan_resamp))
		return chan_resamp

def dwt(channel):
		coeffs=pywt.wavedec(channel,'db3',mode='sym',level=7)  #wavelet decomposition, using level 7 db3 mother wavelet to obtain approximate and detail coefficients
		cA1,cD7,cD6,cD5,cD4,cD3,cD2,cD1=coeffs
		return coeffs

def threshold(coeffs):
		cA1,cD7,cD6,cD5,cD4,cD3,cD2,cD1=coeffs
		sd_cD5 = np.std(cD5)  #standard deviation for last three levels of detail and approximate coefficients. cd5,cD6,cD7 and cA1 make up the lower frequency bands in which eyeblinks are present
		sd_cD6 = np.std(cD6)
		sd_cD7 = np.std(cD7)
		sd_cA1 = np.std(cA1)
	
		cD5_new=np.zeros(np.shape(cD5))
		cD6_new=np.zeros(np.shape(cD6))
		cD7_new=np.zeros(np.shape(cD7))
		cA1_new=np.zeros(np.shape(cA1))
		
		for i in range(len(cD5)): #thresholding detail and approximate coefficients based on standard deviations to extract only eyeblink (higher amplitude signals)
				
				if(abs(cD5[i])>(1.5* sd_cD5)):
					cD5_new[i]=cD5[i]
					#cD5_new[i]=0
				else:
					cD5_new[i]=0

		for i in range(len(cD6)):
				
				if(abs(cD6[i])> (1.2*sd_cD6)):
					cD6_new[i]=cD6[i]
					#cD6_new[i]=0
				else:
					cD6_new[i]=0
		
		for i in range(len(cD7)):
				
				if(abs(cD7[i])>1.5*sd_cD7):
					cD7_new[i]=cD7[i]
					#cD7_new[i]=0
				else: 
					cD7_new[i]=0
		for i in range(len(cA1)):
				
				if(abs(cA1[i])>3*sd_cA1):
					cA1_new[i]=cA1[i]
				else: 
					cA1_new[i]=0
	
		return cD5_new,cD6_new,cD7_new,cA1_new #returning thresholded coefficients 



def reconstruct(cD5_t,cD6_t,cD7_t,cA1_t,coeffs):
	cD1=np.zeros_like(coeffs[7])
	cD2 =np.zeros_like(coeffs[6])
	cD3=np.zeros_like(coeffs[5])
	cD4=np.zeros_like(coeffs[4])
	cD5=np.zeros_like(coeffs[3])
	#cA1,cD7,cD6,cD5,cD4,cD3,cD2,cD1=coeffs
	recon_data=pywt.waverec([cA1_t,cD7_t,cD6_t,cD5,cD4,cD3,cD2,cD1], 'db3', mode='sym') #reconstruct thresholded signal with wavelet reconstruction
	#print(recon_data,'reconstructed data')
	return recon_data

def extract_eyeblink(channel,fs): #calling resampling, decomposition, thresholding and reconstruction methods 
	channel_resamp=resample(channel,fs)
	coeffs=dwt(channel_resamp)
	cA1,cD7,cD6,cD5,cD4,cD3,cD2,cD1=coeffs
	cD5_t,cD6_t,cD7_t,cA1_t=threshold(coeffs)
	recon_data=reconstruct(cD5_t,cD6_t,cD7_t,cA1_t,coeffs)
	return channel_resamp,recon_data,cD5_t,cD6_t,cD7_t,cA1_t
	
fs=256

#data=read_csv('blink_data.txt')

data=read_csv('/Users/raksharamesh/Downloads/Data_FpzSaha.csv')
data.columns=['Fpz'] #extract Fpz data - eyeblinks are more prominent in this channel 

channel_resampFpz,eyeblink_Fpz,cD5_tFz, cD6_tFz, cD7_tFz,cA1_tFz=extract_eyeblink(data.Fpz,fs)

u=channel_resampFpz   # u contains input to adaptive filter (raw data with eyeblinks)
d=u-eyeblink_Fpz	# d contains desired signal which is fed to adaptive filter to train 

#np.savetxt('u_Fpz_full.csv',u,delimiter=',')
#np.savetxt('d_Fpz_full.csv',d,delimiter=',')

np.savetxt('d_sahaP300final.csv',d,delimiter=',')
np.savetxt('u_sahaP300final.csv',u,delimiter=',')
plt.figure(1)
plt.subplot(3,1,1)    #plots for raw, eyeblink extracted and eyeblink removed signals 
plt.title('eeg signal with artifacts')
plt.ylabel('Amplitude(uV)')
plt.ylim(-100,100)
plt.plot(channel_resampFpz[1000:6000])
plt.subplot(3,1,2)
plt.title('extracted eyeblink')
plt.ylabel('Amplitude(uV)')
plt.plot(eyeblink_Fpz[1000:6000])
plt.ylim(-100,100)
plt.subplot(3,1,3)
plt.title('artifact removed eeg')
plt.ylabel('Amplitude(uV)')
plt.plot(d[1000:6000])
plt.ylim(-100,100)
plt.xlabel('Time(samples)')
plt.show()

plt.figure(2)
plt.plot(channel_resampFpz[1000:6000],"b",label=" signal with artifacts")
plt.plot(d[1000:6000],"g",label="corrected eeg")
plt.xlabel('Time(samples)')
plt.ylabel('Amplitude(uV)')
plt.legend()
plt.show()

