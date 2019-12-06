# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:06:32 2019

@author: Xin Hu

This page is to find best_threshold , best_p , best_features
"""

import numpy as np
import os

def create_features(path3,r):
    path1 = "dataset/dataset/afaces/"
    path2 = "dataset/dataset/anonfaces/"
    for i in range(28896):
        if r == 1: e = 28895 - i
        else: e = i
        features = []
        files = os.listdir(path1)
        
        for each in files:
            features.append(np.load(path1+each)[e])
        files = os.listdir(path2)
        
        for each in files:
            features.append(np.load(path2+each)[e])
        
        if not os.path.exists(path3): os.makedirs(path3)
        np.save(path3+str(e)+".npy",features)
        
def find_classifier(pro_path, label_path):
    pro = np.loadtxt(pro_path)
    
    labels = np.load(label_path)
    total_pw = 0
    total_nw = 0
    for i in range(499):
        total_pw += pro[i]
    for i in range(499,2499):
        total_nw += pro[i]
    total_pos = 499
    total_neg = 2000
    min_err = 1
    best_feature = None
    best_threshold = None
    best_p = None
    err = 1
    file_name = os.listdir(path3)

    for i,each in enumerate(file_name):
        
        features = np.load(path3+each)
        n_pos = 0
        n_neg = 0
        pw = 0
        nw = 0
        
        tfea = sorted(zip(pro,features,labels),key = lambda x:x[1])
        for w,f,l in tfea:
            err = min(nw+total_pw-pw,pw+total_nw-nw)
            if err < min_err:
                min_err = err
                best_feature = int(each[:-4])
                best_threshold = f
                best_p = 1 if nw+total_pw-pw < pw+total_nw-nw else -1
            if l == 1:
                n_pos += 1
                pw += w
            elif l == 0:
                n_neg += 1
                nw += w
        #print(each,min_err,best_feature,best_threshold,best_p)
            
            
    print(min_err,best_feature,best_threshold, best_p)
    if best_feature < 7440:
        print("min is 1_2")
    elif best_feature < 14880:
        print("min is 2_1")
    elif best_feature < 20088:
        print("min is 3_1")
    elif best_feature < 25296:
        print("min is 1_3")
    else:
        print("min is 2_2")
    return min_err, best_feature, best_threshold, best_p



def initial():
    ##initialize weight
    pro = []
    for i in range(499):
        pro.append(1/(2*499))
    for i in range(2000):
        pro.append(1/(2*2000))
    np.savetxt("pro1.txt",pro)
    
    labels = []
    for i in range(499):
        labels.append(1)
    for i in range(2000):
        labels.append(0)
    np.save("labels.npy",labels)


def change_weights(feature,p,threshold,err,in_weight,labels_path):
    pro = np.loadtxt(in_weight)
    labels = np.load(labels_path)
    
    path = "dataset/dataset/features/" + str(feature) + ".npy"
    features = np.load(path)

    beta = err/(1-err)
    errr = 0
    f_n = 0
    for i,each in enumerate(features):
        if each * p < threshold * p:
            res = 1
        else: res = 0
        if res != labels[i]:
            errr += pro[i]
        if res == 0 and labels[i] == 1:
            f_n += 1
    ## empirical error criteria        
        if res != labels[i]:
            continue
    ## false positive criteria        
    #    if res == 1 and labels[i] == 0:
    #        continue
    ## false negative criteria        
    #    if res == 0 and labels[i] == 1:
    #        continue
        else:
            pro[i] = pro[i] * beta
        
    print(errr,f_n)
    asum = np.sum(pro) 
    for i in range(len(pro)):
        pro[i] = pro[i]/asum
    return pro
    
        
if __name__ == "__main__":
    path3 = "dataset/dataset/features/"
    if not os.path.exists(path3): 
        create_features(path3,0)
        initial()
    pro_path = "pro1.txt"
    labels = "labels.npy"
    para = {}
    for s in range(2,12):
        err, feature, threshold, p = find_classifier(pro_path, labels)
       
        print(err,feature,threshold,p)
        para[str(s-1)] = [feature,threshold,p,err]
        pro = change_weights(feature,p,threshold, err, pro_path, labels)
        pro_path = "pro"+str(s)+".txt"
        np.savetxt(pro_path,pro)
    np.save("para.npy",para)

            

    
        