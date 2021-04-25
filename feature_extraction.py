import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pywt
from numpy.linalg import inv


eeg_data = []
#included_cols = [2,3,4,5,6,7,13]
included_cols = [0,1,2,3,4,5,6,7] #which channels are being selected
#include event codes in the last channel.



#4thoder_ica_filt6thmarch
with open('/Applications/MATLAB/filtered files/Rak_1.csv') as csvfile:
    eeg_reader = csv.reader(csvfile,delimiter = ',')
    for row in eeg_reader:
        eeg_data.append(list(row[i] for i in included_cols))
                
#del(eeg_data[0:1447])
#32769 is the code for expt start. So we have removed all the data that comes before this code. And this
#happens at sample number 1447.


target_epoch = []
non_target_epoch = []

row = {'33025':'r1','33026':'r2','33027':'r3','33028':'r4','33029':'r5','33030':'r6'}
col = {'33031':'c1','33032':'c2','33033':'c3','33034':'c4','33035':'c5','33036':'c6'}
target = '33285' #target event code
non_tar = '33286'
stim_start = '32779'
stim_stop = '32780'

#to make it 256
samp_bef = 0
samp_aft = 255

row_keys = row.keys() #List with the row stim values 
col_keys = col.keys() #same with columns

segments = 24 # per character, how many rounds of 12 row/col flashes
event_code_col = 7

#called in epoching process

def add_epoch(target,epoch_no,target_r,target_c,stim_start_sample):
    epoch_data = []
    epoch_data.append(target_r)
    epoch_data.append(target_c)
    epoch_data.append(eeg_data[stim_start_sample-samp_bef:stim_start_sample+samp_aft + 1])
    
    #temp_arr = []
    #for l in range(samp_aft + samp_bef + 1):
        #temp_arr.append([float(m) for m in eeg_data[stim_start_sample-samp_bef+l]])
    if target == 1:
        target_epoch.append(epoch_data)
        
    else:
        non_target_epoch.append(epoch_data)

#print(eeg_data[0])
i =  0 #goes through each sample in csv
epoch_no = 0  #total epochs
#print(eeg_data[1495:1510])



while i < len(eeg_data):
    #print(str(i)+'-samp-'+eeg_data[i][event_code_col])
    if len(eeg_data[i][event_code_col]) == 0: #If an event code does not exist for a sample, then it is not our epoch centre
        i = i+1
    else:
        #print(str(i)+'-samp-'+eeg_data[i][event_code_col])
        event_code = eeg_data[i][event_code_col].split(':') #split and get a list of event codes for this sample
        #print(len(event_code))
        target_r = []  #stores stim code for that target letter for that trial.
        target_c = []
        
        if(len(event_code) == 2) and event_code[0] in row_keys and event_code[1] in col_keys: # this will get our target letter
            target_r = event_code[0]
            target_c = event_code[1]
            
            
            #print(target_r)
            #print(target_c)
            #print("-------------")
            
            samp_no = i+1 #to iterate till you get the start of a flash, i+1 cause prev it has blue flash
            
            for j in range(segments):
                for k in range(12): # 12 is fixed, because you have 12 stimulations per segment
                    while len(eeg_data[samp_no][event_code_col]) == 0:
                        samp_no = samp_no + 1
                    
                    #now we are "likely" at the smaple that indicates the stimulation start, but let's confirm just in case
                    split_event_code = eeg_data[samp_no][event_code_col].split(':')
                
                    if stim_start in split_event_code: # this is the stim_start sample
                        if split_event_code[0] == target:
                            #print(samp_no)
                            add_epoch(1,epoch_no,target_r,target_c,samp_no)
                        else:
                            #print(samp_no)
                            add_epoch(0,epoch_no,target_r,target_c,samp_no)
                        epoch_no = epoch_no + 1
                            
                    samp_no = samp_no + 1
                    
            i = samp_no + 1
            if(i >= len(eeg_data)):
                break
        else:
            i = i+1
            if(i >len(eeg_data)):
                break
                        


#print("Number of target epochs found:"+str(len(target_epoch)))
#print("Number of non target epochs found:"+str(len(non_target_epoch)))




chan=7
np_tar = np.zeros(shape=(chan,len(target_epoch),samp_bef + samp_aft + 1))
np_nontar = np.zeros(shape=(chan,len(non_target_epoch),samp_bef + samp_aft + 1))
for i in range(6):
    for j in range(len(target_epoch)):
        for k in range(samp_bef + samp_aft +1):
            np_tar[i][j][k] = target_epoch[j][2][k][i]
for i in range(6):
    for j in range(len(non_target_epoch)):
        for k in range(samp_bef + samp_aft +1):
            np_nontar[i][j][k] = non_target_epoch[j][2][k][i]

#print("target epoch", np_tar.shape)
#print("non-target epoch", np_nontar.shape)
    

l1=256
l2=8
m=int(np.floor((l1+l2-1)/2))
n=int(2*m+l2)

#print(np_tar.shape[1])
def pad_matrix(m,q,n,W):
	W_extend=np.zeros((q,n))
	ext_bot=int(np.ceil((q-m)/2))
	ext_top=int(np.floor((q-m)/2))
	
	for i in range(ext_bot):
		W_extend[ext_top+m+i][:]=W[m-i-1][:] # symmetric extend 

	for i in range(ext_top):
		W_extend[ext_top-i-1][:]=W[i][:]
		

	W_extend[ext_top:ext_top+m][:]=W[:][:]
	return W_extend



def transform_mat(l1,l2,m,n,L): 				# computing transformation matrix W function 
    W=np.zeros((m,n))			
    wavelet=pywt.Wavelet('db4')					
	
    h=wavelet.dec_lo							# h contains low pass filter bank coefficients of db4
    c=1
    for i in range(m):
        for j in range(l2):
            k=int(i*2+l2-j)
            W[i][k]=h[j]
    while(c<=L):
        if(L==1):	
            #print(W,'W1')						#returning transformation matrix if level is 1 
            return W
		
        p=int(np.floor((m+l2-1)/2))
        #print(p,'p')
        q=int(2*p+l2)
                                            		#initializing W_extend matrix with qxn dimensions
        W_extend=pad_matrix(m,q,n,W)			#function call
        W=W_extend
        #print(W,"W_extend")
        A=np.zeros((p,q))					#Initializing A matrix with pxq dimensions 
		
        for i in range(p):
            for j in range(l2):
                z=(i*2+l2-j)
                A[i][z]=h[j]					#construct A matrix with low pass filter bank
        #print(A,'A')
        W=np.matmul(A,W)
        #print(W,c,'W post mul') 		
        m=p 								 #m=p for next iteration
        c=c+1
        if(c==L):
            #print(W,'final_W')
        
            return W


def signal_extend(x,n):				# function to symmetrically extend signal to (l1+l2-1)/2))
	sym_extend=int((n-len(x))/2)			# extend signal by this length on either side
	i=0
	samples=np.zeros(int(len(x)+(2*sym_extend))) #initialize 
	while(i<sym_extend):
		samples[i]=x[sym_extend-i-1]		#mirror samples at the start 
		i=i+1
	samples[i:(i+len(x))]=x[:]
	end_index=(sym_extend+len(x)-1)
	j=0
	while(j<sym_extend):
		samples[end_index+j+1]=x[len(x)-j-1] #mirror samples at the end 
		j=j+1
	return samples


trans_matrix=transform_mat(l1,l2,m,n,5) # function call for transformation matrix- pass level parameter 

#function to obtain approximate coefficients
def getdks(signal_epoch,m,n):
    dks = np.zeros((signal_epoch.shape[0],signal_epoch.shape[1],m))
    for i in range(signal_epoch.shape[0]):
        for j in range(signal_epoch.shape[1]):
            temp=signal_extend(signal_epoch[i][j][:],n) #extension of input signal
            dks[i][j][:]=np.matmul(trans_matrix,temp)       #approx coefficients with by obtaining transformation matrix and extended signal
    return dks

#grouping target and non target signal epichs
def dks_group( dks , N, m):
    davg=np.zeros((chan,m))
    for k in range (chan):
        dsum = np.sum(dks[k], axis = 0)
        davg[k] = (1/N) * dsum      #average of target and nontarget signals
    return davg         #returnind Dks avg for both groups


#constructing scatter matrix
   
def scattermatrix(dks,dg,N):
    sgk = np.zeros((chan,dks.shape[2])) #initialization
    for i in range(chan):
        for j in range(N):
            #print("shape of dk",(dks[i][j]).shape)
            #print("dg ", dg.shape)
            sgk[i] += np.matmul((dks[i][j]-dg[i]),(dks[i][j]-dg[i]).T)  #equation for Sgk
    return sgk


           
          
            
dks_tar = getdks(np_tar,trans_matrix.shape[0],trans_matrix.shape[1])  #target approximate coeffs
dks_nontar = getdks(np_nontar,trans_matrix.shape[0],trans_matrix.shape[1])      #nontarget approx coeff 
#print("approx tar",dks_tar)
#print("dks nontar shape", dks_nontar.shape)
#print("dks tar shape",dks_tar.shape)
 
        
dk1=dks_group(dks_tar,dks_tar.shape[1],trans_matrix.shape[0])   #grouped dks for target
dk0=dks_group(dks_nontar,dks_nontar.shape[1],trans_matrix.shape[0])      #grouped dks for non-target
#print("avg target", dk1)
#print("avg non tar", dk0)
#print("dk0 group  ", dk0.shape)
#print("dk1 group", dk1.shape)

sbk=np.zeros((chan,trans_matrix.shape[0]))    #obtaining between class distance    
sbk = dk1 - dk0      #between class distance
#print(sbk,'sbk')

#print("between class", sbk)

#function call for constructing scatter matrices
s1k = scattermatrix(dks_tar,dk1,dks_tar.shape[1]) 
s0k = scattermatrix(dks_nontar,dk0,dks_nontar.shape[1])

swk=np.zeros((chan,dks_tar.shape[2])) #within class distance
swk=s1k + s0k
#print(swk,'swk')
#print("within class", swk)

#print("swk",swk.shape)
#print("sbk ", sbk.shape)

#finding fisher's optimization ratio
inv_val = np.linalg.pinv(swk)
wk =np.matmul(np.linalg.pinv(swk) , sbk)
wk_val = np.dot(np.linalg.pinv(swk), sbk)
#print("Fisher weights", wk.shape)


#finding the eigen values and sorting for most important weights
eigvals, eigvecs = np.linalg.eig(wk)
eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

#print("list shape",len(eiglist))
#print("dec indec", eiglist.shape)


#sorted weights in descending order
cj=sorted(range(len(eigvals)), key=lambda k: eigvals[k],reverse=True)
print("des order", cj)

tar_ext= np.zeros((np_tar.shape[0],np_tar.shape[1],270))
nontar_ext= np.zeros((np_nontar.shape[0],np_nontar.shape[1],270))
for i in range(np_tar.shape[0]):
        for j in range(np_tar.shape[1]):
            tar_ext[i][j][:]=signal_extend(np_tar[i][j][:],n)
            
for i in range(np_nontar.shape[0]):
        for j in range(np_nontar.shape[1]):
            nontar_ext[i][j][:]=signal_extend(np_nontar[i][j][:],n)

#print("shape",tar_ext.shape)
r=14
#print(cj)


#construction of new feature matrix based on the weights
def feature_matrix(r,cj,W):
    mk = np.zeros((r,trans_matrix.shape[1]))
    for i in range(r):
        index = cj[i]  #index according to important weights
        mk[i][:] = W[index][:]
    return mk
mk = feature_matrix(r,cj,trans_matrix) #function call
#print("mk", mk)


#Fisher;s feature vector
def feature_vector(Mk,fks,r):
    vks=np.zeros((chan,fks.shape[1],r))
    #print("vks val",vks.shape)
    for i in range(chan):
        for j in range(fks.shape[1]):
            #print("mk ", mk.shape)
            #print("fks val ", fks[i][j].shape)
            vks[i][j] = np.dot(Mk,fks[i][j][:])             #construction of new features based on new weights
            #print((vks[i][j]).shape)
    return vks #features 


fin_vks_tar = feature_vector(mk,tar_ext,r)  #features for target
fin_vks_nontar = feature_vector(mk,nontar_ext,r)  #features for non-target
#print("tar features ", fin_vks_tar)
#print("non tar feat ",fin_vks_nontar)


coeffs_tar=pywt.wavedec(np_tar,'db4',level=5)		#inbuilt function decomposing signal till level 2 using db4, pad signal symmetrically
coeffs_nontar=pywt.wavedec(np_nontar,'db4',level=5)	

#cA1,cD5,cD4,cD3,cD2,cD1=coeffs
cA1,cD5,cD4,cD3,cD2,cD1=coeffs_tar 									#unpack approximation and detail coeffs. unpack 4 coefficients if level is 3. 
nA1,nD5,nD4,nD3,nD2,nD1=coeffs_nontar
#print("approximation coefficients-inbuilt func",cA1)
#print('app for non tar',nA1)
cD1=np.zeros_like(cD1)
cD2 =np.zeros_like(cD2)
cD3=np.zeros_like(cD3)
cD4=np.zeros_like(cD4)							# setting detail coefficients obtained to zero in order to reconstruct signal only with approximation coefficients
cD5=np.zeros_like(cD5)

nD1=np.zeros_like(nD1)
nD2 =np.zeros_like(nD2)
nD3=np.zeros_like(nD3)
nD4=np.zeros_like(nD4)							# setting detail coefficients obtained to zero in order to reconstruct signal only with approximation coefficients
nD5=np.zeros_like(nD5)

signal_tar=pywt.waverec([cA1,cD5,cD4,cD3,cD2,cD1], 'db4', mode='sym') 			# reconstruct signal using approx coeffs of inbuilt function
signal_nontar=pywt.waverec([nA1,nD5,nD4,nD3,nD2,nD1], 'db4', mode='sym')
signal_trans_tar=pywt.waverec([dks_tar,cD5,cD4,cD3,cD2,cD1],'db4',mode='sym') 		# reconstruct signal using approx coeffs obtained from  computed transformation matrix 
signal_trans_nontar=pywt.waverec([dks_nontar,nD5,nD4,nD3,nD2,nD1],'db4',mode='sym') 

#print("tar", signal_tar.shape)
#print("nt", signal_nontar.shape)
'''sigval = np.zeros((69,14))
avgdks = np.zeros((1,14))
for i in range(1):
    for j in range(69):
        for k in range(14):
            sigval[j][:]= dks_tar[i][j][:]
            avgdks[i][:] = (1/69) * np.sum(sigval[j][:])
            
#            
avgnondks = np.zeros((1,14))            
sig = np.zeros((465,14))
for i in range(1):
    for j in range(465):
        for k in range(14):
            sig[j][:]= dks_nontar[i][j][:]
            avgnondks[i][:] = (1/465) * np.sum(sig[j][:])'''

#print(avgnondks, "non dk")
#print(sigval.shape)
#print(sig.shape
'''fs=256
def freq_domain(data,fs):
    fft=np.fft.fft(data)
    n=len(fft)
    k=int(n/2)
    fft=fft[0:k]
    freq=np.linspace(0,fs/2,k)
    return fft,freq'''

    
'''plt.figure(1)
plt.subplot(2,1,1)
plt.plot(avgdks,'r',label = "dks target ")
plt.plot(avgnondks,'b',label = "dks non target")
plt.legend()								
plt.title('Overlapping reconstruction plot of Target and non targetsignal ')
plt.xlabel("Number of samples")
plt.ylabel("Amplitude")

plt.figure(2)
plt.subplot(2,1,1)
fft_app,freq_app = freq_domain(avgdks,fs)
fft_app2,freq_app2 = freq_domain(avgnondks,fs)
plt.plot(freq_app, abs(fft_app))
plt.plot(freq_app2, abs(fft_app2))
plt.title('Target and non target FFT plot')
plt.xlabel("Frequency")
plt.ylabel("Amplitude")

plt.show()
'''




