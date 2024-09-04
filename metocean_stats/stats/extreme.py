from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from .aux_funcs import *


def return_levels_pot(data, var, dist='Weibull_2P', 
                      periods=[50, 100, 1000], 
                      threshold=None, r="48h"):
    """
    Calulates return value estimates for different periods, fitting a 
    given distribution to threshold excess values of the data.  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    dist (str): POT distribution to choose from 'GP' for Generalized Pareto,
                'Weibull_2P' for Weibull 2-paremeters or 'EXP' for exponential
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    threshold (float): Threshold used to define the extremes
    r (str): Minimum period of time between two peaks. Default 48h.
    
    return (pandas DataFrame): return levels estimates and corresponding
                               probability of non-exceedance indexed by 
                               corresponding periods. Also contains attrs for 
                               the method, the distribution, the threshold, 
                               and the period considered between independent 
                               extremes (r).  
    
    Function written by dung-manh-nguyen and KonstantinChri.
    """
    import scipy.stats as stats
    from pyextremes import get_extremes

    if threshold == None:
        threshold = get_threshold_os(data=data, var=var)
    else:
        pass
    
    extremes = get_extremes(data[var], method="POT", threshold=threshold, r=r)
    
    years = data.index.year
    yr_num = years[-1]-years[0]+1
    length_data = extremes.shape[0]
    # in hours 
    time_step = yr_num*365.2422*24/length_data
    # years is converted to K-th
    return_periods = np.array(periods)*24*365.2422/time_step
    
    if dist == 'Weibull_2P':
        shape, loc, scale = stats.weibull_min.fit(extremes-threshold, floc=0)
        return_levels = stats.weibull_min.isf(1/return_periods, 
                                              shape, loc, scale)\
                                              + threshold
    elif dist == 'EXP':
        loc, scale = stats.expon.fit(extremes-threshold)
        return_levels = stats.expon.isf(1/return_periods, loc, scale)\
                                        + threshold
    elif dist == 'GP':
        shape, loc, scale = stats.genpareto.fit(extremes-threshold)
        return_levels = stats.genpareto.isf(1/return_periods, 
                                            shape, loc, scale)\
                                            + threshold
    else:
        print ('please check method/distribution, must be one of: EXP, \
                GP or Weibull_2P')
                           
    df = pd.DataFrame({'return_levels':return_levels,
                       'periods': periods})
    df = df.set_index('periods')
    df.loc[df.index >= 1, 'prob_non_exceedance'] = \
                                               1 - 1 / df.index[df.index >= 1]
    df.attrs['method'] = 'pot'
    df.attrs['dist'] = dist
    df.attrs['r'] = '48h'
    df.attrs['threshold'] = threshold
    df.attrs['var'] = var

    return df


def return_levels_annual_max(data, var='hs', dist='GEV', 
                             periods=[50, 100, 1000]): 
    """
    Calulates return value estimates for different periods, fitting a 
    Generalized Extreme Value ('GEV') or a Gumbel ('GUM') distribution to given 
    data.  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable 
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    method (str): Distribution to fit to the data. Either 'GEV' for Generalized
    Extreme Value or 'GUM' for Gumbel.
    
    return (pandas DataFrame): return levels estimates and corresponding
                               probability of non-exceedance indexed by 
                               corresponding periods. 
                               Also contains attrs for the method, 
                               and the distribution. 
    
    Function written by dung-manh-nguyen and KonstantinChri.   
    """
    it_selected_max = data.groupby(data.index.year)[var].idxmax().values
    # get annual maximum with actual dates that maximum occured
    data_am = data[var].loc[it_selected_max] 
    periods = np.array(periods, dtype=float)
    for i in range(len(periods)) :
        if periods[i] == 1 : 
            periods[i] = 1.6

    if dist == 'GEV':
        from scipy.stats import genextreme
        # fit data
        shape, loc, scale = genextreme.fit(data_am) 
        # Compute the return levels for several return periods       
        return_levels = genextreme.isf(1/periods, shape, loc, scale)

    elif dist == 'GUM': 
        from scipy.stats import gumbel_r 
        loc, scale = gumbel_r.fit(data_am) # fit data
        # Compute the return levels for several return periods.
        return_levels = gumbel_r.isf(1/periods, loc, scale)

    else:
        print ('please check method/distribution, must be either GEV or GUM')

    df = pd.DataFrame({'return_levels':return_levels,
                       'periods': periods})
    df = df.set_index('periods')
    df.loc[df.index >= 1, 'prob_non_exceedance'] = \
                                               1 - 1 / df.index[df.index >= 1]
    df.attrs['method'] = 'AM'
    df.attrs['dist'] = dist
    df.attrs['var'] = var

    return df


def return_levels_idm(data, var, dist='Weibull_3P', 
                      periods=[50, 100, 1000]):
    """
    Calulates return value estimates for different periods, fitting a 
    given distribution to all available data (initial distribution method).  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    dist (str): 'Weibull_3P' for 3-parameters Weibull distribution
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    
    return (pandas DataFrame): return levels estimates and corresponding
                               probability of non-exceedance indexed by 
                               corresponding periods. 
                               Also contains attrs for the method, 
                               and the distribution.
    
    """
    import scipy.stats as stats

    # time step between each data, in hours
    time_step = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
    # years is converted to K-th
    return_periods = np.array(periods)*24*365.2422/time_step
    
    if dist == 'Weibull_3P':
        shape, loc, scale = param = Weibull_method_of_moment(data[var])
        return_levels = stats.weibull_min.isf(1/return_periods, 
                                              shape, loc, scale)
        
    else:
        print ('please check method/distribution, must be one of: EXP, \
                GP or Weibull_2P')

    df = pd.DataFrame({'return_levels':return_levels,
                       'periods': periods})
    df = df.set_index('periods')
    df.loc[df.index >= 1, 'prob_non_exceedance'] = \
                                               1 - 1 / df.index[df.index >= 1]
    df.attrs['method'] = 'idm'
    df.attrs['dist'] = dist
    df.attrs['var'] = var

    return df
 

def get_threshold_os(data, var):
    """
    Follows the method by Outten and Sobolowski (2021) to define the threshold 
    to be used in subroutine return_levels_pot
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    
    return: Threshold as defined in Outten and Sobolowski (2021)    
    
    Function written by clio-met (July 2023)
    """
    ts = data[var]
    yearly_max = ts.resample('YS').max()
    # Take the minimum of the yearly maxima found
    min_ym=yearly_max.min()
    
    return min_ym



def probplot(data, sparams):    
    import scipy.stats as stats
    stats.probplot(data, sparams=sparams, 
                   dist=stats.genpareto,fit=True, plot=plt)
    plt.grid()
    plt.axline((data[0], data[0]), slope=1,label='y=x')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


def get_empirical_return_levels(data, var, method="POT",
                                block_size="365.2425D",
                                threshold=None):
    """
    Returns an estimation of observed return periods and return levels.
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    method (str): Method of definition for the extremes, 
                  'BM' for Block Maxima or 'POT' for Peak over threshold
    block_size (str): Size of the block for block maxima
    threshold (float): Threshold to be used for peak-over-threshold, default
                       None.
    
    return (pandas DataFrame): df, dataframe containing empirical return levels
                               and return periods. df.attrs contains meta-data
                               about the method used, threshold or block size 
                               used, and variable of interest.   
    """
    from pyextremes import get_extremes, get_return_periods
    
    if method == 'BM':
        extremes = get_extremes(ts=data[var],
                                method=method,
                                block_size=block_size)
        rp = get_return_periods(ts=data[var],
                                extremes=extremes,
                                extremes_method=method,
                                extremes_type="high",
                                return_period_size=block_size)
                                
        df = pd.DataFrame(rp).rename(columns = {
                                 'return period':'return_periods',
                                 var: 'return_levels'
                                 })\
                             .loc[:,['return_levels', 'return_periods']]
        df.attrs['method'] = 'BM'
        df.attrs['block_size'] = block_size
        df.attrs['var'] = var
        
    elif method == 'POT':
    
        if threshold is None:
             threshold = get_threshold_os(data=data, var=var)   
             
        extremes = get_extremes(ts=data[var],
                                method=method,
                                threshold=threshold)
        rp = get_return_periods(ts=data[var],
                                extremes=extremes,
                                extremes_method=method,
                                extremes_type="high")
                                
        df = pd.DataFrame(rp).rename(columns = {
                                 'return period':'return_periods',
                                 var: 'return_levels'
                                 })\
                             .loc[:,['return_levels', 'return_periods']]
                             
        df.loc[df['return_periods'] >= 1, 'prob_non_exceedance'] = \
                        1 - 1 / df[df['return_periods'] >= 1]['return_periods']
        df.attrs['method'] = 'POT'
        df.attrs['threshold'] = threshold
        df.attrs['var'] = var

    return df


def threshold_sensitivity(data, var, thresholds, 
                           dist_list=['GP','EXP','Weibull_2P'], 
                           period=100):
    """
    Returns theoretical return level for given return period and distribution,
    as a function of the threshold. Plots the return levels in function of the 
    thresholds, for each method and saves the result into the given output_file
    if output_file is not None.

    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    thresholds (float): Range of thresholds used for POT methods
    dist_list (list of str): list of the names of the models to fit the
                                data and display
    period (int or float): Return period
    output_file (str): path of the output file to save the plot, else None.
    
    return: 
        dict_rl (dict of list of floats): Contains the return levels 
                                          corresponding to the given period for
                                          each distribution (key) 
                                          and each threshold
        thresholds (float): Range of thresholds used for POT methods
    """
    # Initialize the dictionnary with one key for each distribution
    dict_rl = {dist:[] for dist in dist_list}

    # For each threshold, and each method get the return value corresponding to 
    # the input period and add it to the corresponding list in the dictionnary
    for thresh in thresholds:
        for dist in dist_list:
            if dist in ['GP', 'Weibull_2P','EXP']:
                dict_rl[dist].append(return_levels_pot(data, 
                                                   var,
                                                   dist=dist,
                                                   threshold=thresh,
                                                   periods=[period])\
                                                   .iloc[0,0])

            elif dist in ['GEV','GUM']:
                dict_rl[dist].append(return_levels_annual_max(data, 
                                                          var,
                                                          dist=dist,
                                                          periods=[period])\
                                                          .iloc[0,0])

    df = pd.DataFrame(dict_rl)
    df['Thresholds'] = thresholds
    df = df.set_index('Thresholds')

    df.attrs['period'] = period
    df.attrs['var'] = var

    return df



    
def return_levels_weibull_2p(data, var,
                             periods=[50, 100, 1000], 
                             threshold=None, r="48h"):
    """
    Calulates return value estimates for different periods, fitting a 
    2-parameters Weibull distribution to given data.  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable 
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    threshold (float): Threshold used to define the extrems
    r (str): Minimum period of time between two peaks. Default 48h.
    
    return (list of float): return value estimates corresponding to input
                            periods 
    
    Function written by dung-manh-nguyen and KonstantinChri.
    """                         
    import scipy.stats as stats
    from pyextremes import get_extremes
    # how to use, to test this 
    #return_period = np.array([1,10,100,10000],dtype=float)
    #return_values = RVE_Weibull_2P(df.hs,return_period,threshold=6.2)
    #print (return_values)
    if threshold == None:
        threshold = get_threshold_os(data=data, var=var)
    else:
        pass
    extremes = get_extremes(data[var], method="POT", threshold=threshold, r=r)

    # Fit a 2-parameter Weibull distribution to the data
    shape, loc, scale = stats.weibull_min.fit(extremes, floc=0)

    years = data.index.year
    yr_num = years[-1]-years[0]+1
    length_data = extremes.shape[0]
    # in hours 
    time_step = yr_num*365.2422*24/length_data
    # years is converted to K-th
    return_period = np.array(periods)*24*365.2422/time_step 

    # Calculate the 2-parameter Weibull return value
    return_value = stats.weibull_min.isf(1/return_period, shape, loc, scale)

    #for i in range(len(return_value)) : 
    #    return_value[i] = round(return_value[i],1)
    #    rl[i] = round(return_value[i],1)

    return return_value


def return_levels_exp(data, var='hs', periods=[50, 100, 1000], 
                      threshold=None, r="48h"):
    """
    Calulates return value estimates for different periods, fitting an 
    exponential distribution to given data.  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable 
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    threshold (float): Threshold used to define the extrems
    r (str): Minimum period of time between two peaks. Default 48h.
    
    return (list of float): return value estimates corresponding to input
                            periods 
    
    Function written by dung-manh-nguyen and KonstantinChri.
    """        
    from scipy.stats import expon
    from pyextremes import get_extremes
    # how to use this function 
    #return_periods = np.array([1.6, 10, 100, 10000]) # in years
    #return_values = RVE_EXP(df.hs,return_periods,4)
    #print (return_values)
    if threshold == None:
        threshold = get_threshold_os(data=data, var=var)
    else:
        pass
    extremes = get_extremes(data[var], method="POT", threshold=threshold, r=r)
    loc, scale = expon.fit(extremes)
    #print (loc,scale)

    years = data.index.year
    yr_num = years[-1]-years[0]+1
    length_data = extremes.shape[0]
    interval = yr_num*365.2422*24/length_data # in hours 
    # years is converted to K-th
    return_periods = np.array(periods)*24*365.2422/interval 

    return_levels = expon.isf(1/return_periods, loc, scale)

    return return_levels
        

def old_return_levels_GP(data, var, threshold=None, 
                          periods=[50,100], output_file=False):
    """
    Subroutine written by clio-met (July 2023)
    Purpose: calculates the return levels (rl) from a time series (ts) and given threshold for different return periods (periods)
    - Inputs:
    1) data is a dataframe with time enabled containing the hourly time series of the variable of interest (ex: wind speed)
    2) threshold is a scalar
    3) periods is a numpy array containing the return periods (in years) of interest (ex: np.array([20,30,50,100]))
    - Follows the POT method
    """
    if threshold == None:
        threshold = get_threshold_os(data=data, var=var)
    else:
        pass
    periods = np.array(periods, dtype=float)
    ts = data[var]
    # select the values above the threshold in the timeseries ts
    it=np.where(ts.values>=threshold)[0]
    # Group consecutive indices with a +/-2 days difference
    # to select only one event and not consecutive timesteps with values above threshold
    number_days_diff = 2
    larrs=np.split(it,np.where(np.diff(it)>number_days_diff*24)[0]+1)
    it_selected_max=[]
    # Pick the maximum value of the variable within groups
    for l in range(len(larrs)):
        mxi=np.argmax(ts.iloc[larrs[l]].values) # If several values are equal to the max, argmax takes the first occurrence 
        it_selected_max.append(larrs[l][mxi]) # index of points used in the analysis
        del mxi
    # Get the selected values and time for the extreme value analysis
    sel_val=ts.iloc[it_selected_max]

    #sel_time=sel_val.index
    # Total number of selected values
    ns=len(sel_val)
    # Number of selected values in each year
    ns_yr=sel_val.resample("YS").count() # Should give an array of 30 values if 30 years are considered
    # Expected rate of occurrence (mean number of occurrences per year)
    lambd=np.mean(ns_yr.values,axis=0)
    # Fit the Generalized Pareto distribution to the selected values
    from scipy.stats import genpareto
    shape,loc,scale = genpareto.fit(sel_val.values-threshold)
    rl = genpareto.isf(1/(lambd*periods), shape, loc, scale) + threshold

    # If the return level is unrealistic (usually very high), set rl to a special number
    rl=np.where(rl>=3*threshold,np.nan,rl)
    if output_file == False:
        pass
    else:
      plot_return_levels(data,var,rl,periods,output_file,
                         it_selected_max=ts.iloc[it_selected_max].index)
        #probplot(data=sel_val.values, sparams=(shape, loc, scale))

    return rl

def RVE_ALL(dataframe,var='hs',periods=[1,10,100,1000],distribution='Weibull3P',method='default',threshold='default'):
    
    # data : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # distribution: 'EXP', 'GEV', 'GUM', 'LoNo', 'Weibull2P' or 'Weibull3P'
    # method: 'default' (all data), 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 

    import scipy.stats as stats
    from pyextremes import get_extremes
    shape, loc, scale = [], [], []
    periods = np.array(periods)
    it_selected_max = dataframe.groupby(dataframe.index.year)[var].idxmax().values
    df = dataframe[var]
    
    period = periods
    # get data for fitting 
    if method == 'default' : # all data 
        data = df.values
    elif method == 'AM' : # annual maxima
        annual_maxima = df.resample('Y').max() # get annual maximum 
        data = annual_maxima
    elif method == 'POT' : # Peak over threshold 
        if threshold == 'default' :
            annual_maxima = df.resample('Y').max() 
            threshold=annual_maxima.min()
        data = get_extremes(df, method="POT", threshold=threshold, r="48h")
    else:
        print ('Please check the method of filtering data')
    
    # Return periods in K-th element 
    try:
        for i in range(len(period)) :
            if period[i] == 1 : 
                period[i] = 1.5873
    except:
        if period == 1 : 
            period = 1.5873
            
    duration = (df.index[-1]-df.index[0]).days + 1 
    length_data = data.shape[0]
    #interval = duration*24/length_data # in hours 
    interval = ((df.index[-1]-df.index[0]).days + 1)*24/df.shape[0] # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a distribution to the data
    if distribution == 'EXP' : 
        loc, scale = stats.expon.fit(data)
        value = stats.expon.isf(1/period, loc, scale)
        #value = stats.expon.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'GEV' :
        shape, loc, scale = stats.genextreme.fit(data) # fit data   
        value = stats.genextreme.isf(1/period, shape, loc, scale)
        #value = stats.genextreme.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'GUM' :
        loc, scale = stats.gumbel_r.fit(data) # fit data
        value = stats.gumbel_r.isf(1/period, loc, scale)
        #value = stats.gumbel_r.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'GUM_L' : # Gumbel Left-skewed (for minimum order statistic) Distribution
        loc, scale = stats.gumbel_l.fit(data) # fit data
        value = stats.gumbel_l.ppf(1/period, loc, scale)
        #value = stats.gumbel_l.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'LoNo' :
        shape, loc, scale = stats.lognorm.fit(data)
        value = stats.lognorm.isf(1/period, shape, loc, scale)
        #value = stats.lognorm.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull2P' :
        shape, loc, scale = stats.weibull_min.fit(data, floc=0) # (ML)
        value = stats.weibull_min.isf(1/period, shape, loc, scale)
        #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull3P' : 
        shape, loc, scale = stats.weibull_min.fit(data) # (ML)
        value = stats.weibull_min.isf(1/period, shape, loc, scale)
        #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull3P_MOM' : 
        shape, loc, scale = Weibull_method_of_moment(data)
        value = stats.weibull_min.isf(1/period, shape, loc, scale)
        #value = stats.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    else:
        print ('Please check the distribution')    
        
    #if method == 'default' :  
    # 	output_file= distribution + '.png'
    #else:
    #    output_file= distribution + '(' + method + ')' + '.png'   
    #plot_return_levels(dataframe,var,value,periods,output_file,it_selected_max)
       
    return shape, loc, scale, value


def joint_distribution_Hs_Tp(data,var_hs='hs',var_tp='tp',periods=[1,10,100,10000], adjustment=None):  
    
    """
    This fuction will plot Hs-Tp joint distribution using LogNoWe model (the Lognormal + Weibull distribution) 
    df : dataframe, 
    var1 : Hs: significant wave height,
    var2 : Tp: Peak period 
    file_out: Hs-Tp joint distribution, optional
    """
    if adjustment == 'NORSOK':
        periods_adj = np.array([x * 6 for x in periods])
    else:
        periods_adj = periods
    df  = data
    max_y = max(periods)
    period = np.array(periods)
    pd.options.mode.chained_assignment = None  # default='warn'
    df.loc[:,'hs'] = df[var_hs].values
    df.loc[:,'tp'] = Tp_correction(df[var_tp].values)
    
    import scipy.stats as stats
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
   
    # calculate lognormal and weibull parameters and plot the PDFs 
    mu = np.mean(np.log(df.hs.values)) # mean of ln(Hs)
    std = np.std(np.log(df.hs.values)) # standard deviation of ln(Hs)
    alpha = mu
    sigma = std
    
    h = np.linspace(start=0.01, stop=30, num=1500)
    pdf_Hs1 = h*0
    pdf_Hs2 = h*0
    
    if 0 < mu < 5 : 
    	pdf_Hs1 = 1/(np.sqrt(2*np.pi)*alpha*h)*np.exp(-(np.log(h)-sigma)**2/(2*alpha**2))
    else:
    	param = stats.lognorm.fit(df.hs.values,) # shape, loc, scale
    	pdf_lognorm = stats.lognorm.pdf(h, param[0], loc=param[1], scale=param[2])
    	pdf_Hs1 = pdf_lognorm
    
    param = Weibull_method_of_moment(df.hs.values) #stats.weibull_min.fit(df.hs.values) # shape, loc, scale
    pdf_Hs2 = stats.weibull_min.pdf(h, param[0], loc=param[1], scale=param[2])
    
    
    # Find the index where two PDF cut, between P60 and P99 
    for i in range(len(h)):
        if abs(h[i]-np.percentile(df.hs.values,60)) < 0.1:
            i1=i
            
        if abs(h[i]-np.percentile(df.hs.values,99)) < 0.1:
            i2=i
            
    epsilon=abs(pdf_Hs1[i1:i2]-pdf_Hs2[i1:i2])
    param = find_peaks(1/epsilon)
    try:
    	index = param[0][1]
    except:
    	try:
    	    index = param[0][0]
    	except:
            index = np.where(epsilon == epsilon.min())[0]
    index = index + i1
        
    # Merge two functions and do smoothing around the cut 
    eta = h[index]
    pdf_Hs = h*0
    for i in range(len(h)):
        if h[i] < eta : 
            pdf_Hs[i] = pdf_Hs1[i]
        else:
            pdf_Hs[i] = pdf_Hs2[i]
            
    for i in range(len(h)):
        if eta-0.5 < h[i] < eta+0.5 : 
            pdf_Hs[i] = np.mean(pdf_Hs[i-10:i+10])
    
            
    #####################################################
    # calcualte a1, a2, a3, b1, b2, b3 
    # firstly calcualte mean_hs, mean_lnTp, variance_lnTp 
    Tp = df.tp.values
    Hs = df.hs.values
    maxHs = max(Hs)
    if maxHs<2 : 
        intx=0.05
    elif maxHs>=2 and maxHs<3 :
        intx=0.1
    elif maxHs>=3 and maxHs<4 :
        intx=0.2
    elif maxHs>=4 and maxHs<10 :
        intx=0.5
    else : 
        intx=1.0;
    
    mean_hs = []
    variance_lnTp = []
    mean_lnTp = []
    
    hs_bin = np.arange(0,maxHs+intx,intx)
    for i in range(len(hs_bin)-1):
        idxs = np.where((hs_bin[i]<=Hs) & (Hs<hs_bin[i+1]))
        if Hs[idxs].shape[0] > 15 : 
            mean_hs.append(np.mean(Hs[idxs]))
            mean_lnTp.append(np.mean(np.log(Tp[idxs])))
            variance_lnTp.append(np.var(np.log(Tp[idxs])))
    
    mean_hs = np.asarray(mean_hs)
    mean_lnTp = np.asarray(mean_lnTp)
    variance_lnTp = np.asarray(variance_lnTp)

    # calcualte a1, a2, a3 
    parameters, covariance = curve_fit(Gauss3, mean_hs, mean_lnTp)
    a1 = parameters[0]
    a2 = parameters[1]
    a3 = 0.36
    
    # calcualte b1, b2, b3 
    start = 1
    x = mean_hs[start:]
    y = variance_lnTp[start:]
    parameters, covariance = curve_fit(Gauss4, x, y)
    b1 = 0.005
    b2 = parameters[0]
    b3 = parameters[1]
    
    
    # calculate pdf Hs, Tp 
    t = np.linspace(start=0.01, stop=40, num=2000)
    
    f_Hs_Tp = np.zeros((len(h), len(t)))
    pdf_Hs_Tp=f_Hs_Tp*0
    
    for i in range(len(h)):
        mu = a1 + a2*h[i]**a3
        std2 = b1 + b2*np.exp(-b3*h[i])
        std = np.sqrt(std2)
        
        f_Hs_Tp[i,:] = 1/(np.sqrt(2*np.pi)*std*t)*np.exp(-(np.log(t)-mu)**2/(2*std2))
        pdf_Hs_Tp[i,:] = pdf_Hs[i]*f_Hs_Tp[i,:]
    
    interval = ((df.index[-1]-df.index[0]).days + 1)*24/df.shape[0] # in hours 

    t3 = []
    h3 = []
    X = []
    hs_tpl_tph = pd.DataFrame()

    # Assuming Hs_Tp_curve() returns four values, otherwise adjust accordingly
    for i in range(len(periods)):
        t3_val, h3_val, X_val, hs_tpl_tph_val = Hs_Tp_curve(df.hs.values, pdf_Hs, pdf_Hs_Tp, f_Hs_Tp, h, t, interval, X=periods_adj[i])
        t3.append(t3_val)
        h3.append(h3_val)
        X.append(X_val)
        hs_tpl_tph_val.columns = [f'{col}_{periods[i]}' for col in hs_tpl_tph_val.columns]
        hs_tpl_tph = pd.concat([hs_tpl_tph, hs_tpl_tph_val], axis=1)
    
    #if save_rve:
    #    hs_tpl_tph[3].to_csv(str(param[2])+'_year.csv', index=False)  

    return a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph 

def monthly_extremes(data, var='hs', periods=[1, 10, 100, 10000], distribution='Weibull3P_MOM', method='default', threshold='default'):
    from scipy.stats import weibull_min
    # Calculate parameters for each month based on different method
    params = []
    return_values = [] #np.zeros((13, len(periods)))
    num_events_per_year = []
    time_step = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
    # years is converted to K-th
    periods1 = np.array(periods)*24*365.2422/time_step
    threshold_values = []
    for month in range(1, 13):
        month_data = data[data.index.month == month]
        
        if isinstance(threshold, str) and threshold.startswith('P'):
            threshold_value = month_data[var].quantile(int(threshold.split('P')[1])/100)
            threshold_values.append(threshold_value)
            # Calculate the number of events exceeding a threshold:
            num_years = (month_data.index[-1]  - month_data.index[0] ).days / 365.25  # Using 365.25 to account for leap years
            num_events_per_year.append((month_data[var] >= threshold_value).sum()/num_years)
        else:
            threshold_value = threshold

        if method == 'minimum': # used for negative temperature
            shape, loc, scale, value = RVE_ALL(month_data.resample('ME').min().dropna(),var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
        elif method == 'maximum': # used for positive temperature
            shape, loc, scale, value = RVE_ALL(month_data.resample('ME').max().dropna(),var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
        elif method == 'default':
            shape, loc, scale, value = RVE_ALL(month_data,var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
        elif method == 'POT':
            shape, loc, scale, value = RVE_ALL(month_data,var=var,periods=periods,distribution=distribution,method=method,threshold=threshold_value)
        
        params.append((shape, loc, scale))
        return_values.append(value)

    # add annual
    if isinstance(threshold, str) and threshold.startswith('P'):
        threshold_value = data[var].quantile(int(threshold.split('P')[1])/100)
        threshold_values.append(threshold_value)
        # Calculate the number of events exceeding a threshold:
        num_years = (data.index[-1]  - data.index[0] ).days / 365.25  # Using 365.25 to account for leap years
        num_events_per_year.append((data[var] >= threshold_value).sum()/num_years)
    else:
        threshold_value = threshold

    if method == 'minimum':
        shape, loc, scale, value = RVE_ALL(data.resample('YE').min(),var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
    elif method == 'maximum':
        shape, loc, scale, value = RVE_ALL(data.resample('YE').max(),var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
    elif method == 'default':
        shape, loc, scale, value = RVE_ALL(data,var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
    elif method == 'POT':
        shape, loc, scale, value = RVE_ALL(data,var=var,periods=periods,distribution=distribution,method=method,threshold=threshold_value)
            

    params.append((shape, loc, scale))       
    return_values.append(value)
    return_values = np.array(return_values)

    # Define the threshold values (annual values) for each column
    thresholds = return_values[-1]

    # Replace values in each column that exceed the thresholds
    if method != 'minimum':
        for col in range(return_values.shape[1]):
            return_values[:, col] = np.minimum(return_values[:, col], thresholds[col])

    return params, return_values, threshold_values, num_events_per_year

def directional_extremes(data: pd.DataFrame, var: str, var_dir: str, periods=[1, 10, 100, 10000], distribution='Weibull3_MOM', adjustment='NORSOK', method='default', threshold='default'):
    
    from scipy.stats import weibull_min, gumbel_r
    # Your implementation of monthly_extremes_weibull function
    # Calculate Weibull parameters for each month
    sector_prob = []
    params = []
    threshold_values = []
    num_events_per_year = []
    return_values = []
    add_direction_sector(data=data,var_dir=var_dir)
    # time step between each data, in hours
    time_step = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
    
    for dir in range(0,360,30):
        sector_data = data[data['direction_sector']==dir]

        if isinstance(threshold, str) and threshold.startswith('P'):
            threshold_value = sector_data[var].quantile(int(threshold.split('P')[1])/100)
            threshold_values.append(threshold_value)
            # Calculate the number of events exceeding a threshold:
            num_years = (sector_data.index[-1]  - sector_data.index[0] ).days / 365.25  # Using 365.25 to account for leap years
            num_events_per_year.append((sector_data[var] >= threshold_value).sum()/num_years)
        else:
            threshold_value = threshold

        periods_adj = [x * 6 for x in periods]#*24*365.2422/time_step
        periods_noadj = periods#*24*365.2422/time_step
        
        if adjustment == 'NORSOK':
            pass
        else:
            periods_adj = periods_noadj

        if method == 'minimum': # used for negative temperature
            shape, loc, scale, value = RVE_ALL(sector_data.min().dropna(),var=var,periods=periods_adj,distribution=distribution,method='default',threshold=threshold_value)
        elif method == 'maximum': # used for positive temperature
            shape, loc, scale, value = RVE_ALL(sector_data.max().dropna(),var=var,periods=periods_adj,distribution=distribution,method='default',threshold=threshold_value)
        elif method == 'default':
            shape, loc, scale, value = RVE_ALL(sector_data,var=var,periods=periods_adj,distribution=distribution,method=method,threshold=threshold_value)
        elif method == 'POT':
            shape, loc, scale, value = RVE_ALL(sector_data,var=var,periods=periods_adj,distribution=distribution,method=method,threshold=threshold_value)
    
        sp = 100*len(sector_data)/len(data[var])
        sector_prob.append(sp)
        params.append((shape, loc, scale))
        return_values.append(value)

    # add annual
    if isinstance(threshold, str) and threshold.startswith('P'):
        threshold_value = data[var].quantile(int(threshold.split('P')[1])/100)
        threshold_values.append(threshold_value)
        # Calculate the number of events exceeding a threshold:
        num_years = (data.index[-1]  - data.index[0] ).days / 365.25  # Using 365.25 to account for leap years
        num_events_per_year.append((data[var] >= threshold_value).sum()/num_years)
    else:
        threshold_value = threshold

    if method == 'minimum':
        shape, loc, scale, value = RVE_ALL(data.resample('YE').min(),var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
    elif method == 'maximum':
        shape, loc, scale, value = RVE_ALL(data.resample('YE').max(),var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
    elif method == 'default':
        shape, loc, scale, value = RVE_ALL(data,var=var,periods=periods,distribution=distribution,method='default',threshold=threshold_value)
    elif method == 'POT':
        shape, loc, scale, value = RVE_ALL(data,var=var,periods=periods,distribution=distribution,method=method,threshold=threshold_value)
            
    params.append((shape, loc, scale))
    return_values.append(value)
    return_values = np.array(return_values)
    # Define the threshold values (annual values) for each column
    thresholds = return_values[-1]
    
    # Replace values in each column that exceed the thresholds
    for col in range(return_values.shape[1]):
        return_values[:, col] = np.minimum(return_values[:, col], thresholds[col])

    return params, return_values, sector_prob,  threshold_values, num_events_per_year

def monthly_joint_distribution_Hs_Tp_weibull(data, var='hs', periods=[1, 10, 100, 10000]):
    from scipy.stats import weibull_min
    # Your implementation of monthly_extremes_weibull function
    # Calculate Weibull parameters for each month
    weibull_params = []
    for month in range(1, 13):
        month_data = data[data.index.month == month][var]
        shape, loc, scale = Weibull_method_of_moment(month_data)
        weibull_params.append((shape, loc, scale))
    # add annual
    shape, loc, scale = Weibull_method_of_moment(data[var])
    weibull_params.append((shape, loc, scale))       
    # time step between each data, in hours
    time_step = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
    # years is converted to K-th
    periods1 = np.array(periods)*24*365.2422/time_step
    #breakpoint()
    # Calculate return periods for each month and period
    return_values = np.zeros((13, len(periods)))
    for i, (shape, loc, scale) in enumerate(weibull_params):
        for j, period in enumerate(periods1):
            return_value = weibull_min.isf(1/period, shape, loc, scale)
            return_values[i, j] = round(return_value, 1)

    return weibull_params, return_values
    
def prob_non_exceedance_fitted_3p_weibull(data, var='hs'):
    from scipy.stats import percentileofscore, weibull_min
    #step = 0.01
    #hs = np.arange(0.5,data[var].max()+step,step)
    y_obs = np.arange(1,int(data[var].max())+0.5,0.5)
    prob_non_exceedance_obs = percentileofscore(data[var], y_obs)
    print(prob_non_exceedance_obs)
    shape, location, scale =  Weibull_method_of_moment(data[var])
    weibull_dist = weibull_min(c=shape, scale=scale, loc=location)
    # Generate random samples
    num_samples = 1000000  # Number of samples to generate
    y_fitted = weibull_dist.rvs(size=num_samples)
    cdf = weibull_dist.cdf(y_fitted)
    #pdf=weib_3(np.arange(0.001,15.001,0.001),shape,location,scale)
    return y_obs, y_fitted, cdf, prob_non_exceedance_obs, shape, location, scale
    

def weib_3(arr,sh,lo,sc):
    a=sh/sc
    b=((arr-lo)/sc)**(sh-1)
    c=0.0-((arr-lo)/sc)**(sh)
    d=np.exp(c)
    return d


def model_tp_given_hs(hs: float, a1, a2, a3, b1, b2, b3):
    mi = a1 +a2 * hs**a3
    variance = b1 + b2*np.exp(-b3*hs)
    tp_mean = np.exp(mi + 0.5*variance)
    std_tp = tp_mean * np.sqrt(np.exp(variance)-1) 
    P5_model = tp_mean - 1.645*std_tp
    P95_model = tp_mean + 1.645*std_tp
    Mean_model = tp_mean

    return P5_model,Mean_model,P95_model

def estimate_forristal_maxCrest(Hs, Tp, depth=50, twindow=3, sea_state = 'short-crested'):
    from scipy.optimize import minimize
    # Example usage
    #Hs = 3.0  # Significant wave height
    #Tp = 10.0  # Peak period
    #depth = 50.0  # Depth
    #twindow = 3.0  # Duration in hours
    
    # Constants
    LAT = 0  # Lowest astronomical Tide
    SWL = 0  # Still Water Level
    g = 9.81  # Gravity

    Tz = estimate_Tz(Tp,gamma = 2.5)
    Tm01 = estimate_Tm01(Tp,gamma = 2.5)
    S1 = (2 * np.pi) / g * (Hs / Tm01 ** 2)
    k1 = (2 * np.pi) ** 2 / (g * Tm01 ** 2)  # Deep water

    # Function to find k for finite depth
    def delta(delta):
        return abs((g / (2 * np.pi)) * Tm01 ** 2 * np.tanh(2 * np.pi * (depth / delta)) - delta)

    res = minimize(delta, 100)
    dlt = res.x[0]
    k1 = (2 * np.pi) / dlt

    Urs = Hs / (k1 ** 2 * depth ** 3)

    if sea_state == 'long-crested' :
        alphaFC = 0.3536 + 0.2892 * S1 + 0.1060 * Urs 
        betaFC = 2 - 2.1597 * S1 + 0.0969 * Urs ** 2
    elif sea_state == 'short-crested' :
        alphaFC = 0.3536 + 0.2568 * S1 + 0.0800 * Urs 
        betaFC = 2 - 1.7912 * S1 - 0.5302 * Urs + 0.284 * Urs ** 2 
    else:
        print ('please check sea state')

    # Number of waves
    N = (twindow * 3600) / Tz

    #Cmax = (SWL - LAT) + alphaFC * Hs * (np.log(N)) ** (1 / betaFC)
    p = 0.85
    Cmax = (SWL - LAT) + alphaFC * Hs * (-np.log(1 - p ** (1 / N))) ** (1 / betaFC)
    
    return Cmax

def estimate_Hmax(Hs, Tp, twindow=3, k=0.9):
    Tz = estimate_Tz(Tp,gamma = 2.5)
    N = (twindow*3600)/ Tz
    Hmax =  Hs * ( np.sqrt(np.log(N)/2) + 0.2886/np.sqrt(2*np.log(N))  ) # Expected largest Hmax based on Max Wave Distribution
    return Hmax

