# coding: utf-8

# In[37]:
import subprocess
# subprocess("sudo","pip","install","--upgrade","-r","requirements.txt")
from scipy.misc import imsave
import os, sys
import pyedflib
import numpy
import numpy, scipy.io
import re
import sklearn
import numpy as np
import scipy
from sklearn import preprocessing
from numpy import *
from scipy import signal
import random
random.seed(1)

# import matplotlib.pyplot as plt
# import logging
# import cloudstorage as gcs

# In[25]:


class epochEEG:
    def __init__(self, intf=20):
        self.signals = np.zeros((320, 320))
        self.start = 0
        self.end = 0
        self.label = False  # False =No seizure, True =seizure
        self.Biggroup = '0,0'


class seizure_log:
    def __init__(self, intf=20):
        self.seizureNumber = 0
        self.szStart = []
        self.szEnd = []
        self.filename = "chb"


# In[26]:


def sparse(fre, signals):
    B1channelfre = fre[0:7]
    B2channelfre = fre[8:14]
    B3channelfre = fre[15:49]
    B1channelsig = signals[0:7]
    B2channelsig = signals[8:14]
    B3channelsig = signals[15:49]
    B1 = numpy.trapz(B1channelsig, x=B1channelfre)
    B2 = numpy.trapz(B2channelsig, x=B2channelfre)
    B3 = numpy.trapz(B3channelsig, x=B3channelfre)
    return B1, B2, B3


# In[27]:


def interpolation(proj, B1, B2, B3):
    x = numpy.linspace(-16, 16, 320)
    y = numpy.linspace(-16, 16, 320)
    [xq, yq] = numpy.meshgrid(x, y)
    # print(proj.shape)
    # print(B1.shape)

    vq1 = scipy.interpolate.griddata(proj, B1.T, (xq, yq), method='cubic')
    vq2 = scipy.interpolate.griddata(proj, B2.T, (xq, yq), method='cubic')
    vq3 = scipy.interpolate.griddata(proj, B3.T, (xq, yq), method='cubic')
    vq1[isnan(vq1)] = 0
    vq2[isnan(vq2)] = 0
    vq3[isnan(vq3)] = 0
    vq10 = np.zeros((320, 320))
    vq10 = vq1[:, :, 0]
    vq20 = np.zeros((320, 320))
    vq20 = vq2[:, :, 0]
    vq30 = np.zeros((320, 320))
    vq30 = vq3[:, :, 0]
    vq1.reshape((320, 320))
    vq2.reshape((320, 320))
    vq3.reshape((320, 320))
    # print(vq10.shape)
    vq1 = np.zeros((320, 320, 1))
    vq1[:, :, 0] = preprocessing.normalize(vq10)
    vq2 = np.zeros((320, 320, 1))
    vq2[:, :, 0] = preprocessing.normalize(vq20)
    vq3 = np.zeros((320, 320, 1))
    vq3[:, :, 0] = preprocessing.normalize(vq30)
    imageepoch = np.concatenate((vq1, vq2, vq3), axis=2)
    return imageepoch


# In[28]:


# x = numpy.linspace(-16,16,321)
# y = numpy.linspace(-16,16,321)
# xy = numpy.meshgrid(x,y)
# print(xy)
k = open("summary.txt")
seizure_data = []
while k:
    name_summary = k.readline()
    name_summary = name_summary.strip('\n')
    summary = name_summary + "-summary.txt"
    print(name_summary)
    # "./chb01-summary.txt"
    if name_summary == "":
        break
    s = open(summary)
    s_f = s.readline()

    while s_f:
        # print(s_f)
        f_str = re.findall(r"\d+\.?\d*", s_f)
        f_num = list(map(float, f_str))
        # print(f_num)

        if 'Number of Seizures' in s_f:
            if not f_num[0] == 0:
                seizure_get = seizure_log(10)
                seizure_get.seizureNumber = f_num[0]
                print(seizure_get.seizureNumber)
                seizure_get.filename = name_summary
                #print(f_num[0])
                for i in range(int(f_num[0])):
                    s_f = s.readline()
                    f_str = re.findall(r"\d+\.?\d*", s_f)
                    #f_num = list(map(float, f_str[-1]))
                    print(f_str[-1])
                    seizure_get.szStart.append(f_str[-1])
                    s_f = s.readline()
                    f_str = re.findall(r"\d+\.?\d*", s_f)
                    #f_num = list(map(float, f_str[-1]))
                    print(f_str[-1])
                    seizure_get.szEnd.append(f_str[-1])
                seizure_data.append(seizure_get)
        s_f = s.readline()

    s.close()

k.close()

# In[29]:
print(len(seizure_data))
print(seizure_data[0].seizureNumber)
print(seizure_data[0].filename)

Fz = 256
# 2 seconds as a sample, so the size of a sample is 512
# original size is 500
key = open("RECORDS")
seizure_key = open("RECORDS-WITH-SEIZURES")
seizure_keys = seizure_key.readline()
seizure_keys = seizure_keys.strip('\n')
seizureNum = -1
numImage = -1
proj = numpy.loadtxt('2d.txt')
la = open("train_labels.txt",'w')
tx = open("test_labels.txt",'w')
labels = []
#stin = 1;
#counter = 0
while key:
    keys = key.readline()
    #keys = 'chb01/chb01_03.edf'
    keys = keys.strip('\n')
    #counter = counter+1
    #if counter == 2500:
    #    break
    if keys == "":
        break
    if seizure_keys == "":
        break
    Is_seizure = False
    #stin = 0;
    
    print(keys)
    print(seizure_keys)
    if keys == seizure_keys:
        Is_seizure = True
        seizureNum = seizureNum + 1
        seizure_keys = seizure_key.readline()
        seizure_keys = seizure_keys.strip('\n')
    ##position = "gs://seizure/seizure/" + keys
    filename = keys[6:len(keys)]
    print(filename)
    print(Is_seizure)

    # In[30]:



    # print(proj)


    # In[36]:


    # print()
    ##subprocess.check_output(["gsutil", "cp", position, "./temp"])
    cc = "seizure/" + filename
    ##print(cc)
    f = pyedflib.EdfReader(cc)
    ##subprocess.check_output(["rm", "./temp/" + filename])
    sigbufs = np.zeros((23, f.getNSamples()[0]))
    for i in range(23):
    #
        sigbufs[i, :] = signal.detrend(f.readSignal(i))
        
        # print(sigbufs.shape)
    f._close()
    del f

    # In[41]:


    width, height = sigbufs.shape
    # y = sigbufs[:,0:512]
    # print(y.shape)
    Data = []
    for j in range(height / 256):
        numImage = numImage + 1
        x = epochEEG(10)
        signals = sigbufs[:, j * 256:(j + 1) * 256]
        label_ofseizure = 0
        if Is_seizure:
            print('seizureNum')
            print(seizureNum)
            print(seizure_data[seizureNum].seizureNumber)
            for numz in range(int(seizure_data[seizureNum].seizureNumber)):
                if j  >= int(seizure_data[seizureNum].szStart[numz]) and j <= int(seizure_data[seizureNum].szEnd[
                    numz]):
                    label_ofseizure = 1
               # if (j + 1)  <= seizure_data[seizureNum].szEnd[numz] and (j + 1)  >= \
               #         seizure_data[seizureNum].szStart[numz]:
               #     label_ofseizure = 1
        if label_ofseizure is 1:
            print(1)
        labels.append(label_ofseizure)
        #la.write(str(label_ofseizure)+" ")
        # print(signals.shape)
        B1 = numpy.zeros((1, 23))
        B2 = numpy.zeros((1, 23))
        B3 = numpy.zeros((1, 23))
        for i in range(sigbufs.shape[0]):
            # print(signals[i,:].shape)
            # print(i)
            signals1 = signal.detrend(signals[i, :])
            signals1 = numpy.fft.fft(signals1)
            signals1 = abs(signals1 / 256)
            signals0 = signals1[0:128]
            signals0[1:127] = signals1[1:127] * 2
            freq = range(129)
            B1c, B2c, B3c = sparse(freq, signals0)
            # x.siganls = x.siganls.fft
            # print(B1)
            # print(B1c)
            B1[:, i] = B1c
            B2[:, i] = B2c
            B3[:, i] = B3c
        epoachImage = interpolation(proj, B1, B2, B3)
        x.signals = epoachImage
        # plt.imshow(x.signals)
        # plt.show()
        x.Biggroup = filename
        Data.append(x)
        # print(x.signals.shape)
        ImageName = str(numImage)
        # sklearn.preprocessing.normalize(epoachImage)
        tx_la = random.random()
        if tx_la<0.7:
            imsave("./train/" + ImageName + ".png", epoachImage * 255)
            la.write("./train/" +ImageName + ".png"+" "+str(label_ofseizure)+"\n")
        if tx_la>=0.7:
            imsave("./validate/" + ImageName + ".png", epoachImage * 255)
            tx.write("./validate/" +ImageName + ".png"+" "+str(label_ofseizure)+" ")
        #subprocess.check_output(["gsutil", "cp", "./temp/" + ImageName + ".png", "gs://seizure/images/"])
        #print("rm" + "./temp/" + ImageName + ".png")
        #subprocess.check_output(["rm", "temp/" + ImageName + ".png"])
        # In[ ]:
la.close()
tx.close()
     



