import numpy as np
import matplotlib.pyplot as plt
import pywt

#function to symmetrically extend transformation matrtix
def pad_matrix(m,q,n,W):
	W_extend=np.zeros((q,n))   #initial zero-padding
	ext_bot=int(np.ceil((q-m)/2))  #way to extend bottom rows
	ext_top=int(np.floor((q-m)/2))     #way to extend top rows
    
	for i in range(ext_bot):
		W_extend[ext_top+m+i][:]=W[m-i-1][:] # symmetric extend 
	for i in range(ext_top):
		W_extend[ext_top-i-1][:]=W[i][:]
	W_extend[ext_top:ext_top+m][:]=W[:][:]     # symmetric extend 
	return W_extend                            #return extended transformation matrix
 


def transform_mat(l1,l2,m,n,L): 				#initializing W matrix
    W=np.zeros((m,n))			
    wavelet=pywt.Wavelet('db4')				#picking bd$ mother wavelet		
    h=wavelet.dec_lo							# h contains low pass filter bank coefficients of db4
    c=1
    for i in range(m):
        for j in range(l2):
            k=int(i*2+l2-j)
            W[i][k]=h[j]                #equating the elements of the transformation matrix to the filter bank
    while(c<=L):
        if(L==1):	
            return W        #returning transformation matrix if level is 1 
            

		
        p=int(np.floor((m+l2-1)/2))
       
        q=int(2*p+l2)
        
			#initializing W_extend matrix with qxn dimensions
        W_extend=pad_matrix(m,q,n,W)			#function call
		
        W=W_extend
        
        A=np.zeros((p,q))					#Initializing A matrix with pxq dimensions 
		
        for i in range(p):
            for j in range(l2):
                z=(i*2+l2-j)
                A[i][z]=h[j]					#construct A matrix with low pass filter bank
        
        W=np.matmul(A,W)
         		
        m=p 								 #m=p for next iteration
        c=c+1
        if(c==L):
            
        
            return W                #return transfprmation matrix at the required level

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



x=np.loadtxt('/Users/sahanasrihari/Downloads/avg_over_ep_tar.csv',delimiter=',',usecols=0)     #load epoch data
x=x[0:250]


l1=len(x)       #length of input siganl
l2=8            #length of filter bank for Db4
m=int(np.floor((l1+l2-1)/2))

n=int(2*m+l2)


samples=signal_extend(x,n)          #funciton call for signal extension

trans_matrix=transform_mat(l1,l2,m,n,5)   				# function call for transformation matrix- pass level parameter


Dks=np.matmul(trans_matrix,samples)					#computed Approximation coefficients decomposed till specified level. Dks= W X inputsignal 



coeffs=pywt.wavedec(x,'db4',level=5)		#inbuilt function decomposing signal till level 5 using db4, pad signal symmetrically
cA1,cD5,cD4,cD3,cD2,cD1=coeffs
#cA1,cD5,cD4,cD3=coeffs 									#unpack approximation and detail coeffs. 

#print(cA1,'approximation coefficients-inbuilt func')
#cD1=np.zeros_like(cD1)
#cD2 =np.zeros_like(cD2)
#cD3=np.zeros_like(cD3)
#cD4=np.zeros_like(cD4)							# setting detail coefficients obtained to zero in order to reconstruct signal only with approximation coefficients
#cD5=np.zeros_like(cD5)

signal=pywt.waverec([cA1,cD5,cD4,cD3,cD2,cD1], 'db4', mode='sym') 			# reconstruct signal using approx coeffs of inbuilt function

signal_trans=pywt.waverec([Dks,cD5,cD4,cD3,cD2,cD1],'db4',mode='sym') 		# reconstruct signal using approx coeffs obtained from  computed transformation matrix 

print("shape dks", signal.shape)


#plotting reconstructed signal from transformation matrix
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(x)
plt.xlabel("number of samples")
plt.ylabel("amplitude")
plt.subplot(2,1,2)
plt.plot(signal_trans)								
plt.title('reconstructed from transformation function')
plt.xlabel("number of samples")
plt.ylabel("amplitude")
#plot reconstructed from ibuilt function
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(signal_trans,"b",label="from transformation matrix")
plt.plot(signal,"g",label="from inbuilt function")
plt.legend()
plt.title('reconstructed from transformation matrix weights overlapped with reconstruction using inbuilt function')
plt.show()

