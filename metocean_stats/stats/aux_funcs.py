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
    """
    This function will correct the Tp from ocean model which are vertical straight lines in Hs-Tp distribution
    """   
    new_Tp=1+np.log(Tp/3.244)/0.09525
    index = np.where(Tp>=3.2) # indexes of Tp
    r = np.random.uniform(low=-0.5, high=0.5, size=len(Tp[index])) 
    Tp[index]=np.round(3.244*np.exp(0.09525*(new_Tp[index]-1-r)),1)
    return Tp 



def readNora10File(file):
    df = pd.read_csv(file, delim_whitespace=True, header=3)
    df.index= pd.to_datetime(df.YEAR*1000000+df.M*10000+df.D*100+df.H,format='%Y%m%d%H')
    df['tp_corr_nora10'] = Tp_correction(df.TP.values)
    return df
