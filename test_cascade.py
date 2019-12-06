# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:11:42 2019

@author: Xin Hu
"""

import numpy as np
import os
from pro import feature_extracted as et
import cv2
from tkinter import _flatten



def cascade_test(features):
    stage = np.load("cascade50.npy").item()#change file name to load cascade system parameter
    res = 1
    for i in range(40):
        feature = stage[str(i)][0]
        threshold = stage[str(i)][1]
        p = stage[str(i)][2]
        #print(feature, threshold, p)
        if p * features[feature] <= p * threshold:
            continue
        else: 
            res = 0
            return res
    return res


pathf = "dataset/dataset/testset/faces/"
pathn = "dataset/dataset/testset/non-faces/"

err = 0
f_p = 0
f_n = 0
filef = os.listdir(pathf)
filen = os.listdir(pathn)
filt =  [(1,2),(2,1),(3,1),(1,3),(2,2)]



for i, each in enumerate(filef):
    
    features = []
    img = cv2.imread(pathf+each)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    for filt_size in filt:
        feature = et(img,filt_size[0],filt_size[1])
        features.append(feature)
        features = list(_flatten(features))
    
    #test cascae system
    res = cascade_test(features)
    
    if res == 0:
        err += 1
        f_n += 1
        
    print(err, f_n,i)


for i, each in enumerate(filen):
#    if i<1158: continue
#    if i > 471: break
    features = []
    img = cv2.imread(pathn+each)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    for filt_size in filt:
        feature = et(img,filt_size[0],filt_size[1])
        features.append(feature)
        features = list(_flatten(features))
    
    #test cascade system    
    res = cascade_test(features)
    
    if res == 1:
        err += 1
        f_p += 1
        
    print(err, f_p,i)

#print("error is", err/(len(filef)+472)," f_p is",f_p/(len(filef)+472), " f_n is", f_n/(len(filef)+472))
print("error is", err/(len(filef)+len(filen))," f_p is",f_p/(len(filef)+len(filen)), " f_n is", f_n/(len(filef)+len(filen)))

