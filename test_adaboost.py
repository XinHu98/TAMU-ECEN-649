# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:32:07 2019

@author: Xin Hu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pro import feature_extracted as et
import cv2
from tkinter import _flatten


path = ["para.npy","parafp.npy","parafn.npy"]


pathf = "dataset/dataset/testset/faces/"
pathn = "dataset/dataset/testset/non-faces/"

filef = os.listdir(pathf)
filen = os.listdir(pathn)
filt =  [(1,2),(2,1),(3,1),(1,3),(2,2)]
for name in path:
    
    s = np.load(name).item()
    fpd = []
    fnd = []
    errd = []
    
    if name == "para.npy": num = [0,2,4,9]
    else: num = [0,1,2,3,4]
    for e in num:
        p = s[str(e+1)][2]
    
        err = 0
        f_p = 0
        f_n = 0
        
        threshold = s[str(e+1)][1]
        for i, each in enumerate(filef):
            
            features = []
            img = cv2.imread(pathf+each)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            for filt_size in filt:
                feature = et(img,filt_size[0],filt_size[1])
                features.append(feature)
                features = list(_flatten(features))
                

            
            #test normal adaboost
            s_f = features[s[str(e+1)][0]]
            if s_f * p <= threshold * p:
                res = 1
            else:
                res = 0
            
            if res == 0:
                err += 1
                f_n += 1
                
          #  print(err, f_n,i)
        
        
        for i, each in enumerate(filen):

            features = []
            img = cv2.imread(pathn+each)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            for filt_size in filt:
                feature = et(img,filt_size[0],filt_size[1])
                features.append(feature)
                features = list(_flatten(features))
            
            
            #test normal adaboost
            s_f = features[s[str(e+1)][0]]
            if s_f * p <= threshold * p:
                res = 1
            else:
                res = 0
            
            if res == 1:
                err += 1
                f_p += 1
                
        #    print(err, f_p,i)
        
        
        print("error is", err/(len(filef)+len(filen))," f_p is",f_p/(len(filef)+len(filen)), " f_n is", f_n/(len(filef)+len(filen)))
        errd.append(err/(len(filef)+len(filen)))
        fpd.append(f_p/(len(filef)+len(filen)))
        fnd.append(f_n/(len(filef)+len(filen)))
    
    if name == "para.npy": x = [1,3,5,10]
    else: x = [1,2,3,4,5]
    plt.figure(figsize = (5,5))
    
    plt.xlabel("rounds")
    plt.ylabel("value")
    plt.plot(x,errd,label = "$error$",color = "red")
    plt.plot(x,fpd,label = "$false-positive$", color = "blue")
    plt.plot(x,fnd,label = "$false-negative$", color = "green")
    plt.legend()
    plt.show()
