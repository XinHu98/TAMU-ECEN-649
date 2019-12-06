# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:37:20 2019

@author: Xin Hu
"""

import numpy as np
import os


def cascade(pos_num, label_path,setn):
    path = "dataset/dataset/features/"
    labels = np.load(label_path)
    label = []
    for i in pos_num:
        label.append(labels[i])
    total_pos = 0
    total_neg = 0
    for each in label:
        if each == 1:
            total_pos += 1
        else: total_neg += 1
    file_name = os.listdir(path)
    pro = []
    for each in label:
        if each == 1:
            pro.append(1/(2*total_pos))
        else: pro.append(1/(2*total_neg))
    
    false_n = 1000000
    err = 1
    minerr = 1
    for i,each in enumerate(file_name):
        features = np.load(path+each)
        
        feature = []
        
        for j in pos_num:
            feature.append(features[j])
        
        n_pos = 0
        n_neg = 0
        pw = 0
        nw = 0
        tfea = sorted(zip(pro,feature,label),key = lambda x:x[1])
        for w, f, l in tfea:
            minerr = min(nw+1/2-pw,pw+1/2-nw)
            if minerr < err:
                if ((nw+1/2-pw) < (pw+1/2-nw)) and  ((total_pos - n_pos) <= (total_pos  * 0.01)):
                    false_n = total_pos - n_pos
                    p = 1
                    best_feature = int(each[:-4])
                    best_threshold = f
                    err = minerr
                elif ((nw+1/2-pw) >= (pw+1/2-nw)) and (n_pos <= (total_pos  * 0.01)):
                    false_n = n_pos
                    p = -1
                    best_feature = int(each[:-4])
                    best_threshold = f
                    err = minerr
            if l == 1:
                n_pos += 1
                pw += w
            elif l == 0:
                n_neg += 1
                nw += w
    return best_feature, best_threshold, p,err,false_n
        
def select_ins(feature,threshold,p, pnum):
    path = "dataset/dataset/features/"
    features = np.load(path+str(feature)+".npy")
    labels = np.load("labels.npy")
    num = []
    abandon = 0
    for i in pnum:
        if p * features[i] <= p * threshold:
            res = 1
            num.append(i)
        else:res = 0
        if res == 0 and labels[i] == 1:
            num.append(i)
        if res == 0 and labels[i] == 0:
            abandon += 1
    return num, abandon
        
if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    setn = [50]
    e = 0
    for t in setn:
        num = [x for x in range(2499)]
        data = {}
        
        for s in range(40):
            if (s+1)%4 == 0: e=t*((s+1)/4)/10
            feature, threshold, p,err,false_n = cascade(num,"labels.npy",e)
            
            num,abandon = select_ins(feature,threshold,p,num)
            data[str(s)] = [feature,threshold,p,abandon]
            print(s, "err is ", err, "feature is ", feature,"threshold is ", threshold, "p is ",p, "fn is ",false_n, "abandon is ",abandon)
            
            
        
        
        np.save("cascade"+str(t)+".npy",data)
                    
            
            