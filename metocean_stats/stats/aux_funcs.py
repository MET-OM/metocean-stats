from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from ..stats.spec_funcs import *



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
    df = pd.read_csv(file, sep='\s+', header=3)
    df.index= pd.to_datetime(df.YEAR*1000000+df.M*10000+df.D*100+df.H,format='%Y%m%d%H')
    df.index = pd.to_datetime(dict(year=df.YEAR, month=df.M, day=df.D, hour=df.H))
    # Drop redundant columns
    df.drop(['YEAR', 'M', 'D', 'H'], axis=1, inplace=True)
    df['tp_corr_nora10'] = Tp_correction(df.TP.values)
    df.index.rename('time', inplace=True)
    return df

def wind_correction_nora10(df,var='W10'):
    
    U15=15
    p=0.1
    df.loc[df[var] > U15, var]=df.loc[df[var] > U15, var] + p*(df.loc[df[var] > U15, var]-U15)
    
    return df 

def Weibull_method_of_moment(X):
    import scipy.stats as stats
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

def add_direction_sector(data,var_dir):
    direction_bins = np.arange(15, 360, 30)
    direction_labels = [value for value in np.arange(30, 360, 30)]
    data['direction_sector'] = pd.Series(np.nan, index=data.index)
    for i in range(len(direction_bins)-1):
        condition = (data[var_dir] > direction_bins[i]) & (data[var_dir] <= direction_bins[i + 1])
        data['direction_sector'] = data['direction_sector'].where(~condition, direction_labels[i])

    data['direction_sector'].fillna(0, inplace=True)
    return data 

def consecutive_indices(lst):
    result = []
    start_index = None
    for i in range(len(lst)):
        if ((i == 0) or (lst[i] != (lst[i - 1]+1))):
            if start_index is not None:
                result.append(list(lst[start_index:i]))
            start_index = i
    if start_index is not None:
        result.append(list(lst[start_index:len(lst)]))
    return result


def Hs_Tp_curve(data,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,interval,X=100):
    import scipy.stats as stats
    from scipy.signal import find_peaks

    # RVE of X years 
    shape, loc, scale = Weibull_method_of_moment(data) # shape, loc, scale
    
    if X == 1 : 
        period=1.5873*365.2422*24/interval
    else :
        period=X*365.2422*24/interval
    rve_X = stats.weibull_min.isf(1/period, shape, loc, scale)
    
    # Find index of Hs=value
    epsilon = abs(h - rve_X)
    param = find_peaks(1/epsilon) # to find the index of bottom
    index = param[0][0]     # the  index of Hs=value
    
    # Find peak of pdf at Hs=RVE of X year 
    pdf_Hs_Tp_X = pdf_Hs_Tp[index,:] # Find pdf at RVE of X year 
    param = find_peaks(pdf_Hs_Tp_X) # find the peak
    index = param[0][0]
    f_Hs_Tp_100=pdf_Hs_Tp_X[index]

    
    h1=[]
    t1=[]
    t2=[]
    for i in range(len(h)):
        f3_ = f_Hs_Tp_100/pdf_Hs[i]
        f3 = f_Hs_Tp[i,:]
        epsilon = abs(f3-f3_) # the difference 
        para = find_peaks(1/epsilon) # to find the bottom
        index = para[0]
        if t[index].shape[0] == 2 :
            h1.append(h[i])
            t1.append(t[index][0])
            t2.append(t[index][1])
    
    h1=np.asarray(h1)
    t1=np.asarray(t1)
    t2=np.asarray(t2)
    t3 = np.concatenate((t1, t2[::-1])) # to get correct circle order 
    h3 = np.concatenate((h1, h1[::-1])) # to get correct circle order 
    t3 = np.concatenate((t3, t1[0:1])) # connect the last to the first point  
    h3 = np.concatenate((h3, h1[0:1])) # connect the last to the first point  

    df = pd.DataFrame()
    df['hs']=h1
    df['t1']=t1
    df['t2']=t2
    
    return t3,h3,X,df
    


def Gauss3(x, a1, a2):
    y = a1 + a2*x**0.36
    return y

def Gauss4(x, b2, b3):
    y = 0.005 + b2*np.exp(-x*b3)
    return y


def DVN_steepness(df,h,t,periods,interval):
    import scipy.stats as stats
    ## steepness 
    max_y=max(periods)
    X = max_y # get max 500 year 
    period=X*365.2422*24/interval
    shape, loc, scale = Weibull_method_of_moment(df.hs.values) # shape, loc, scale
    rve_X = stats.weibull_min.isf(1/period, shape, loc, scale)
    
    h1=[]
    t1=[]
    h2=[]
    t2=[]
    h3=[]
    t3=[]
    g = 9.80665
    j15 = 10000
    for j in range(len(t)):
        if t[j]<=8 :
            Sp=1/15
            temp = Sp * g * t[j]**2 /(2*np.pi)
            if temp <= rve_X:
                h1.append(temp)
                t1.append(t[j])
        
            j8=j # t=8
            h1_t8=temp
            t8=t[j]
        elif t[j]>=15 :
            Sp=1/25 
            temp = Sp * g * t[j]**2 /(2*np.pi)
            if temp <= rve_X:
                h3.append(temp)
                t3.append(t[j])
            if j < j15 :
                j15=j # t=15
                h3_t15=temp
                t15=t[j]

    xp = [t8, t15]
    fp = [h1_t8, h3_t15]
    t2_=t[j8+1:j15]
    h2_=np.interp(t2_, xp, fp)
    for i in range(len(h2_)):
        if h2_[i] <= rve_X:
            h2.append(h2_[i])
            t2.append(t2_[i])

    h_steepness=np.asarray(h1+h2+h3)
    t_steepness=np.asarray(t1+t2+t3)
    
    return t_steepness, h_steepness

def find_percentile(data,pdf_Hs_Tp,h,t,p,periods,interval):
    import scipy.stats as stats
    from scipy.signal import find_peaks

    ## find pecentile
    # RVE of X years 
    max_y=max(periods)
    X = max_y # get max 500 year 
    period=X*365.2422*24/interval
    shape, loc, scale = Weibull_method_of_moment(data) # shape, loc, scale
    rve_X = stats.weibull_min.isf(1/period, shape, loc, scale)
    epsilon = abs(h - rve_X)
    param = find_peaks(1/epsilon) # to find the index of bottom
    index_X = param[0][0]     # the  index of Hs=value
    
    
    h1=[]
    t1=[]
    # Find peak of pdf at Hs=RVE of X year 
    for i in range(index_X):
        pdf_Hs_Tp_X = pdf_Hs_Tp[i,:] # Find pdf at RVE of X year 
        sum_pdf = sum(pdf_Hs_Tp_X)
        for j in range(len(pdf_Hs_Tp_X)):
            if (sum(pdf_Hs_Tp_X[:j])/sum_pdf <= p/100) and (sum(pdf_Hs_Tp_X[:j+1])/sum_pdf >= p/100) : 
                #print (i, h[i],j,t[j])
                t1.append(t[j])
                h1.append(h[i])
                break 
    h1=np.asarray(h1)
    t1=np.asarray(t1)

    return t1,h1


def estimate_Tz(T_p):
    return T_p / 1.28

def calculate_Us_Tu(H_s, T_p, depth, ref_depth,spectrum='JONSWAP'):
    df = 0.01
    f=np.arange(0,1,df)
    S_u = np.zeros((len(H_s),len(f)))
    for i in range(len(H_s)):
        if spectrum=='JONSWAP':
            E = jonswap(f=f,hs=H_s[i],tp=T_p[i])
        elif spectrum=='TORSEHAUGEN':
            E = torsethaugen(f=f,hs=H_s[i],tp=T_p[i]) 

        S_u[i,:] = velocity_spectrum(f, E, depth=depth, ref_depth=ref_depth)
    
    M0 = np.trapz(S_u*df,axis=1)
    M2 = np.trapz((f**2)*S_u*df,axis=1)
    Us = 2*np.sqrt(M0)
    Tu = np.sqrt(M0/M2)
    return Us, Tu

import numpy as np
from scipy.optimize import curve_fit

# Define the function H(U)
def Hs_as_function_of_U(U, a, b, c, d):
    return a + b * U**(c + d * U)

# Define the function Uc(U) # Uc for current speed 
def Uc_as_function_of_U(U, a, b, c, d):
    return a + b * U**(c + d * U)

# Define the function Uc(Hs) # Uc for current speed 
def Uc_as_function_of_Hs(Hs, a, b, c):
    return a + b * Hs**c

# Define the function S(Hs) # S for storm surge 
def S_as_function_of_Hs(Hs, a, b, c):
    return a + b * Hs + c * np.log(Hs)

# Function to fit the parameters a, b, c, and d
def fit_hs_wind_model(U, H_values, initial_guesses=None, maxfev=10000):
    if initial_guesses is None:
        # If no initial guesses are provided, use some default values
        initial_guesses = [1, 0.032, 1.6, 0.003]
    
    # Use curve_fit to fit the function to the data
    params, covariance = curve_fit(Hs_as_function_of_U, U, H_values, p0=initial_guesses, maxfev=maxfev)
    
    # Extract the parameters
    a, b, c, d = params
    #print(a,b,c,d)
    return a, b, c, d


# Function to fit the parameters a, b, c, and d
def fit_Uc_wind_model(U, Uc, initial_guesses=None, maxfev=10000):
    if initial_guesses is None:
        # If no initial guesses are provided, use some default values
        initial_guesses = [13.5, 0.005, 2.0, 0.003]
    
    # Use curve_fit to fit the function to the data
    params, covariance = curve_fit(Uc_as_function_of_U, U, Uc, p0=initial_guesses, maxfev=maxfev)
    
    # Extract the parameters
    a, b, c, d = params
    #print(a,b,c,d)
    return a, b, c, d

# Function to fit the parameters a, b, c, and d
def fit_Uc_Hs_model(H_values, Uc, initial_guesses=None, maxfev=10000):
    if initial_guesses is None:
        # If no initial guesses are provided, use some default values
        initial_guesses = [14.5, 0.60, 1.8]
    
    # Use curve_fit to fit the function to the data
    params, covariance = curve_fit(Uc_as_function_of_Hs, H_values, Uc, p0=initial_guesses, maxfev=maxfev)
    
    # Extract the parameters
    a, b, c = params
    #print(a,b,c)
    return a, b, c

def air_temperature_correction_nora10(df,var='T2m'):
    
    T15=15
    p=0.3
    
    df.loc[df[var] < 0, var]=df.loc[df[var] < 0, var]*1.07 
    df.loc[df[var] > T15, var]=df.loc[df[var] > T15, var] + p*(df.loc[df[var] > T15, var]-T15) 

    return df 


def wind_gust(df,var='W10',var0='W10',z=10):
    # this assume the 3-hour interval = 1-h mean wind speed 
    # the calculation folow Norce Report for LUNA page 122/130 
    Uo=df[var0]
    Uref=10 # m/s
    zr = 10
    Iu = 0.06*(1+0.43*Uo/Uref)*(z/zr)**(-0.22)
    Uzt = df[var]*(1-0.41*Iu*np.log(10/60))
    Uzt90 = df[var]*(1-0.59*Iu*np.log(10/60))
    df['Wind_gust'] = Uzt
    df['Wind_gust_P90'] = Uzt90
    return df 


def estimate_Tz(Tp,gamma = 2.5):
    Tz = (0.6673 + 0.05037 * gamma - 0.006230 * gamma ** 2 + 0.0003341 * gamma ** 3) * Tp
    return Tz

def estimate_Tm01(Tp,gamma = 2.5):
    Tm01 = (0.7303 + 0.04936 * gamma - 0.006556 * gamma ** 2 + 0.0003610 * gamma ** 3) * Tp
    return Tm01

def fit_profile_polynomial(z, speeds, degree=4):
    """
    Fit a polynomial of a given degree to the data.
    
    Parameters:
    z (array-like): Depths or heights
    speeds (array-like): Current speeds at the given depths
    degree (int): Degree of the polynomial to fit
    
    Returns:
    numpy.ndarray: Coefficients of the fitted polynomial
    """
    # Convert inputs to numpy arrays
    z = np.array(z)
    speeds = np.array(speeds)
    
    # Fit a polynomial of the given degree to the data
    coefficients = np.polyfit(z, speeds, degree)
    
    return coefficients



def fit_profile_spline(z, speeds, s=None):
    from scipy.interpolate import UnivariateSpline
    """
    Fit a spline to the data.
    
    Parameters:
    z (array-like): Depths or heights
    speeds (array-like): Current speeds at the given depths
    s (float or None): Smoothing factor. If None, the spline will interpolate through all points.
    
    Returns:
    UnivariateSpline: A spline representation of the data
    """
    # Convert inputs to numpy arrays
    z = np.array(z)
    speeds = np.array(speeds)
    
    # Fit a spline to the data
    spline = UnivariateSpline(z, speeds, s=s)
    
    return spline

def extrapolate_speeds(fit_model, z, target_speed, target_z, method='polynomial'):
    """
    Extrapolate speeds at given depths using the fitted model and adjust 
    so that the speed at the target depth matches the target speed.
    
    Parameters:
    fit_model (array-like or UnivariateSpline): Coefficients of the polynomial or the spline model
    z (array-like): Depths at which to calculate the speeds
    target_speed (float): The known speed at the target depth
    target_z (float): The depth at which the speed is known
    method (str): Method used for fitting, either 'polynomial' or 'spline'
    
    Returns:
    numpy.ndarray: Adjusted extrapolated speeds at the given depths
    """
    # Convert depths to numpy array
    z = np.array(z)
    
    if method == 'polynomial':
        # Evaluate the polynomial at the given depths
        extrapolated_speeds = np.polyval(fit_model, z)
        
        # Calculate the current speed at the target depth
        current_speed_at_target_z = np.polyval(fit_model, target_z)
    elif method == 'spline':
        from scipy.interpolate import UnivariateSpline
        # Evaluate the spline at the given depths
        extrapolated_speeds = fit_model(z)
        
        # Calculate the current speed at the target depth
        current_speed_at_target_z = fit_model(target_z)
    else:
        raise ValueError("Method must be either 'polynomial' or 'spline'")
    
    # Calculate the adjustment factor
    adjustment_factor = target_speed / current_speed_at_target_z
    
    # Adjust the speeds
    adjusted_speeds = extrapolated_speeds * adjustment_factor
    
    return adjusted_speeds


# Function to fit the parameters a, b, c, and d
def fit_S_Hs_model(H_values, S, initial_guesses=None, maxfev=10000):
    # S is storm surge
    # H_values is significant wave height 

    if initial_guesses is None:
        # If no initial guesses are provided, use some default values
        initial_guesses = [-2.5, 1.25, 4.4]
    
    # Use curve_fit to fit the function to the data
    params, covariance = curve_fit(S_as_function_of_Hs, H_values, S, p0=initial_guesses, maxfev=maxfev)
    
    # Extract the parameters
    a, b, c = params
    #print(a,b,c)
    return a, b, c

def detrend_ts(df, column_name='tide'):
    """
    Detrend the specified column in the given DataFrame using scipy.signal.detrend.

    Parameters:
    ds_ocean (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to detrend. Default is 'tide'.

    Returns:
    pd.DataFrame: The DataFrame with the detrended column.
    """
    import scipy.signal as signal
    df[column_name] = signal.detrend(df[column_name])
    return df


def degminsec_to_decimals(degrees,minutes,seconds):
    if degrees<=0:
        loc_decimals=degrees-(minutes/60)-(seconds/3600)
    else:
        loc_decimals=degrees+(minutes/60)+(seconds/3600)
    return loc_decimals
