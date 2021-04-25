
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pywt
from numpy.linalg import inv


# In[2]:


no_of_files = 4


# In[3]:


eeg_data = []

for i in range(no_of_files):
    eeg_data.append([])


included_cols = [2,3,4,5,6,7,8]


# In[4]:


#4thoder_ica_filt6thmarch

for i in range(no_of_files):
    filename = 'C:\\Users\\sahjoshi\\Downloads\\BCI\\Open_vibe\\Raksha_7_5\\Rak_custom_{}.csv'.format(i)
    with open(filename) as csvfile:
        eeg_reader = csv.reader(csvfile,delimiter = ',')
        for row in eeg_reader:
            eeg_data[i].append(list(row[j] for j in included_cols))
            


# In[5]:


target_epoch = []
non_target_epoch = []

row = {'33025':'r1','33026':'r2','33027':'r3','33028':'r4','33029':'r5','33030':'r6'}
col = {'33031':'c1','33032':'c2','33033':'c3','33034':'c4','33035':'c5','33036':'c6'}
target = '33285'
non_tar = '33286'
stim_start = '32779'
stim_stop = '32780'

samp_bef = 0
samp_aft = 255
row_keys = row.keys()
col_keys = col.keys()

segments = 24 # per character, how many rounds of 12 row/col flashes
event_code_col = 6
channels = 6


# In[6]:


def add_epoch(target,epoch_no,target_r,target_c,stim_start_sample,eeg_data):
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


# In[7]:


for iteration in range(no_of_files):
    i =  0
    epoch_no = 0
    #print(eeg_data[1495:1510])
    while i < len(eeg_data[iteration]):
        #print(str(i)+'-samp-'+eeg_data[i][event_code_col])
        if len(eeg_data[iteration][i][event_code_col]) == 0: #If an event code does not exist for a sample, then it is not our epoch centre
            i = i+1
        else:
            #print(str(i)+'-samp-'+eeg_data[i][event_code_col])
            event_code = eeg_data[iteration][i][event_code_col].split(':') #split and get a list of event codes for this sample
            #print(len(event_code))
            target_r = []
            target_c = []
            if(len(event_code) == 2) and event_code[0] in row_keys and event_code[1] in col_keys: # this will get our target letter
                target_r = event_code[0]
                target_c = event_code[1]
                #print(target_r)
                #print(target_c)
                #print("-------------")

                samp_no = i+1
                for j in range(segments):
                    for k in range(12): # 12 is fixed, because you have 12 stimulations per segment
                        while len(eeg_data[iteration][samp_no][event_code_col]) == 0:
                            samp_no = samp_no + 1

                        #now we are "likely" at the smaple that indicates the stimulation start, but let's confirm just in case
                        split_event_code = eeg_data[iteration][samp_no][event_code_col].split(':')

                        if stim_start in split_event_code: # this is the stim_start sample
                            if split_event_code[0] == target:
                                #print(samp_no)
                                add_epoch(1,epoch_no,target_r,target_c,samp_no,eeg_data[iteration])
                            else:
                                #print(samp_no)
                                add_epoch(0,epoch_no,target_r,target_c,samp_no,eeg_data[iteration])

                            epoch_no = epoch_no + 1

                        samp_no = samp_no + 1

                i = samp_no + 1
                if(i >= len(eeg_data[iteration])):
                    break
            else:
                i = i+1
                if(i >len(eeg_data[iteration])):
                    break
                        
                    


# In[8]:


print("Number of target epochs found:"+str(len(target_epoch)))
print("Number of non target epochs found:"+str(len(non_target_epoch)))


# In[9]:


avg_no = 6
tar_av_no = int(len(target_epoch)/avg_no)
non_tar_av_no = int(len(non_target_epoch)/avg_no)


# In[10]:


np_tar = np.zeros(shape=(channels,len(target_epoch),samp_bef + samp_aft + 1))
np_nontar = np.zeros(shape=(channels,len(non_target_epoch),samp_bef + samp_aft + 1))
for i in range(channels):
    for j in range(len(target_epoch)):
        for k in range(samp_bef + samp_aft +1):
            np_tar[i][j][k] = target_epoch[j][2][k][i]
            
for i in range(channels):
    for j in range(len(non_target_epoch)):
        for k in range(samp_bef + samp_aft +1):
            np_nontar[i][j][k] = non_target_epoch[j][2][k][i]


# In[11]:


dm_np_tar = np.zeros(shape=(channels,tar_av_no,samp_bef + samp_aft + 1))
dm_np_nontar = np.zeros(shape=(channels,non_tar_av_no,samp_bef + samp_aft + 1))

for i in range(channels):
    for j in range(tar_av_no):
        dm_np_tar[i][j] = np.mean(np_tar[i][j*avg_no : (j+1)*avg_no][:], axis = 0)
for i in range(channels):
    for j in range(non_tar_av_no):
        dm_np_nontar[i][j] = np.mean(np_nontar[i][j*avg_no : (j+1)*avg_no][:], axis = 0)


# In[12]:


np_tar = dm_np_tar 
np_nontar = dm_np_nontar

