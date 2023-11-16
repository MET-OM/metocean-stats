from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os


def convert_latexTab_to_csv(inputFileName, outputFileName):

    input = open(inputFileName, "r")
    output = open(outputFileName, "w")

    for line in input:
        line = line.replace("&",",")
        line = line.replace("\\\\","")
        if not line.lstrip().startswith((",","\\")):
            output.write(line)
        
    input.close()
    output.close()

def Tp_correction(Tp):
    
    new_Tp=1+np.log(Tp/3.244)/0.09525
    index = np.where(Tp>=3.2) # indexes of Tp
    r = np.random.uniform(low=-0.5, high=0.5, size=len(Tp[index])) 
    Tp[index]=np.round(3.244*np.exp(0.09525*(new_Tp[index]-1-r)),1)
    return Tp 
