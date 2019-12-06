# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:03:32 2019

@author: Xin Hu

This page is to extract features and save
"""

import os
import numpy as np
import cv2
import json
from tkinter import _flatten
import matplotlib.pyplot as plt

def feature_extracted(original_img,filt_h,filt_w):
    out_img =np.zeros((np.shape(original_img)))
    #feature = {}
    feature = []
    for i,hpix in enumerate(original_img):
        for j,wpix in enumerate(hpix):
            if(j == 0):
                out_img[i,j,:] = original_img[i,j,:]
            else:
                for t in range(j+1):
                    
                    out_img[i,j,:] += original_img[i,t,:] 
        if(i == 0):continue
        else:
            out_img[i,:,:] = out_img[i-1,:,:] + out_img[i,:,:]
            
    if(8%filt_h == 0):Maxh = 8
    else:Maxh = 8
    if(8%filt_w == 0):Maxw = 8
    else:Maxw = 8
    for sh in range(filt_h, Maxh+filt_h, filt_h):
        for sw in range(filt_w, Maxw+filt_w, filt_w):
            temp = []
#            print(sh,sw)
            for ch in range(0,np.shape(out_img)[0]-sh+1):
                for cw in range(0,np.shape(out_img)[1]-sw+1):
                    topleftx = cw - 1
                    toplefty = ch - 1
                    toprightx = cw + sw - 1
                    toprighty = ch - 1
                    bottomleftx = cw - 1
                    bottomlefty = ch + sh - 1
                    bottomrightx = cw + sw -1
                    bottomrighty = ch + sh -1
                    centralupx = int(cw + sw/2 - 1)
                    centralupy = ch - 1
                    centraldownx = int(cw + sw/2 - 1)
                    
                    centraldowny = ch + sh - 1
                    #print(centraldownx,centraldowny,out_img[centraldownx][centraldowny])
                    centralleftx = cw - 1
                    centrallefty = int(ch + sh/2 - 1)
                    centralrightx = cw + sw -1
                    centralrighty = int(ch + sh/2 - 1)
                    centralx = int(cw + sw/2 - 1)
                    centraly = int(ch + sh/2 - 1)
                    topleft = 0 if(topleftx < 0 or toplefty < 0) else out_img[toplefty][topleftx]
                    topright = 0 if(toprightx < 0 or toprighty < 0) else out_img[toprighty][toprightx]
                    bottomleft = 0 if(bottomleftx < 0 or bottomlefty < 0) else out_img[bottomlefty][bottomleftx]
                    bottomright = 0 if(bottomrightx < 0 or bottomrighty < 0) else out_img[bottomrighty][bottomrightx]
                    centralup = 0 if(centralupx < 0 or centralupy < 0) else out_img[centralupy][centralupx]
                    centraldown = 0 if(centraldownx < 0 or centraldowny < 0) else out_img[centraldowny][centraldownx]
                    centralleft = 0 if(centralleftx < 0 or centrallefty < 0) else out_img[centrallefty][centralleftx]
                    centralright = 0 if(centralrightx < 0 or centralrighty < 0) else out_img[centralrighty][centralrightx]
                    central = out_img[centraly][centralx]
                    radio = filt_h / filt_w
                    #if(cw == 0 and ch == 0):print(topleft, bottomleft, centralup, centraldown,topright, bottomright )
                    if radio == 1/2: dif = bottomright - centraldown - topright + centralup - (centraldown - bottomleft - centralup + topleft)
                    if radio == 2: dif = centralright - centralleft - topright + topleft - (bottomright - bottomleft - centralright + centralleft)
                    if radio == 1: dif = (centralright - central - topright + centralup) + (centraldown + centralleft - bottomleft - central) - (central + topleft - centralleft - centralup) - (bottomright - centraldown - centralright + central)
                    if radio == 1/3 or 3:
                        cleftupx = int(cw + sw/3 - 1) 
                        cleftupy = ch - 1
                        crightupx = int(cw + sw*2/3 - 1)
                        crightupy = ch - 1
                        cleftdownx = int(cw + sw/3 - 1)
                        cleftdowny = ch + sh -1
                        crightdownx = int(cw + sw*2/3 - 1)
                        crightdowny = ch + sh - 1
                        cleftup = 0 if(cleftupx < 0 or cleftupy < 0) else out_img[cleftupy][cleftupx]
                        cleftdown = 0 if(cleftdownx <0 or cleftdowny < 0) else out_img[cleftdowny][cleftdownx]
                        crightup = 0 if(crightupx < 0 or crightupy < 0) else out_img[crightupy][crightupx]
                        crightdown = 0if(crightdownx < 0 or crightdowny < 0)else out_img[crightdowny][crightdownx]
                        if(radio == 1/3):dif = -(cleftdown - bottomleft - cleftup + topleft) - (bottomright - crightdown - topright + crightup) + (crightdown - cleftdown - crightup + cleftup)
                        cleftupx = cw - 1 
                        cleftupy = int(ch + sh/3 - 1)
                        crightupx = cw + sw - 1
                        crightupy = int(ch + sh/3 - 1)
                        cleftdownx = cw - 1
                        cleftdowny = int(ch + sh*2/3 -1)
                        crightdownx = cw + sw - 1
                        crightdowny = int(ch + sh*2/3 - 1)
                        cleftup = 0 if(cleftupx < 0 or cleftupy < 0) else out_img[cleftupy][cleftupx]
                        cleftdown = 0 if(cleftdownx <0 or cleftdowny < 0) else out_img[cleftdowny][cleftdownx]
                        crightup = 0 if(crightupx < 0 or crightupy < 0) else out_img[crightupy][crightupx]
                        crightdown = 0if(crightdownx < 0 or crightdowny < 0)else out_img[crightdowny][crightdownx]
                        if(radio == 3):dif = -(cleftdown - bottomleft - crightdown + bottomright) - (topleft - cleftup - topright + crightup) + (crightdown - cleftdown - crightup + cleftup)
                    
                    #temp.append(dif)
                    feature.append(dif[0])
                    
#                    if(len(feature) == 2308 and radio == 1/2):
#                        print(sh,sw,ch, cw )
            #feature[str(sh)+"_"+str(sw)] = temp
    return feature



if __name__ == "__main__":
    pathw = ["dataset/dataset/trainset/faces/","dataset/dataset/trainset/non-faces/"]
    save_pathw = ["dataset/dataset/afaces/", "dataset/dataset/anonfaces/"]
    filt_h = 1
    filt_w = 3
    filt =  [(1,2),(2,1),(3,1),(1,3),(2,2)]

    for i in range(len(pathw)):

        path = pathw[i]
        save_path = save_pathw[i]
        files = os.listdir(path)
        for i,each in enumerate(files):
    #    if i > 0:break
            features = []
            img = cv2.imread(path+each)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
            for filt_size in filt:
    #            print(filt_size)
                feature = feature_extracted(img,filt_size[0],filt_size[1])
    #            print(str(filt_size[0])+"_"+str(filt_size[1])+" length is",len(feature))
                features.append(feature)
                features = list(_flatten(features))
                
                name = each[:-4]
                if not os.path.exists(save_path): os.makedirs(save_path)
                np.save(save_path+name+".npy",features)
            
            
            
