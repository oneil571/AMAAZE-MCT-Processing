import numpy as np 
import os
import pandas as pd




def retrieve_value(calibration_fname, str2lookfor):
    #str2lookfor = 'Optimum voxel size' #first chunk of line with microns measurement 
                
    with open(calibration_fname, 'r') as file:
        text = file.read()
            
    q = text.split('\n')
        
    for s in q:
        if s[0:len(str2lookfor)]==str2lookfor:
            break
        
    if s[0:len(str2lookfor)]!=str2lookfor:
        print('string to look for not found... uh oh. please change it to continue.')
        raise SystemExit(0)
        
    words = s.split('\ulnone\f1')

    '''
    for i in range(len(words)):
        if words[i] == 'microns':
            i = i-1
            break
        
    dx = float(words[i])*(10**-3) #microns to mm
    '''
    dx = words[-1]
    return dx


lll= []

for i in range(1,11):
    fname = "Technique-AMAAZE_"+str(i)+".rtf"


    dict_list = ['voltage','current','focal spot size','pixel pitch', 'mode','gain', 'framerate','flip','gain map 0', 'gain map 1', 'gain map 2','ug start	tube to detector','tube to part','calculated Ug','zoom factor','# frames averaged', 'delay','duration']
    
    l = []
    for d in dict_list:
        l.append(retrieve_value(calibration_fname, d))
    
    lll.append(l)

    

