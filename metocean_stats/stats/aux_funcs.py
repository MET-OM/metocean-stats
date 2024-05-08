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

def wind_correction_nora10(df,var='W10'):
    
    U15=15
    p=0.1
    df.loc[df[var] > U15, var]=df.loc[df[var] > U15, var] + p*(df.loc[df[var] > U15, var]-U15)
    
    return df 


def Weibull_method_of_moment(X):
    X=X+0.0001; 
    n=len(X);
    m1 = np.mean(X);
    cm1=np.mean((X-np.mean(X))**1);
    m2 = np.var(X);
    cm2=np.mean((X-np.mean(X))**2);
    m3 = stats.skew(X);
    cm3 = np.mean((X-np.mean(X))**3);
    
    from scipy.special import gamma
    
    def m1fun(a,b,c):
        return a+b*gamma(1+1/c)
        
    def cm2fun(b,c):
        return b**2*(gamma(1+2/c)-gamma(1+1/c)**2)
        
    def cm3fun(b,c):
        return b**3*(gamma(1+3/c)-3*gamma(1+1/c)*gamma(1+2/c)+2*gamma(1+1/c)**3)
    
    def cfun(c):
        return abs(np.sqrt(cm3fun(1,c)**2/cm2fun(1,c)**3)-np.sqrt(cm3**2/cm2**3))
    
    from scipy import optimize
    cHat = optimize.fminbound(cfun, -2, 5) # shape 
    
    
    def bfun(b):
        return abs(cm2fun(b,cHat)-cm2)
    bHat = optimize.fminbound(bfun,-5,30) # scale 
    
    def afun(a):
        return abs(m1fun(a,bHat,cHat)-m1)
    aHat = optimize.fminbound(afun,-5,30) # location
    
    return cHat, aHat, bHat # shape, location, scale 
