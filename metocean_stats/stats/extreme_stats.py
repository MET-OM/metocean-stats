import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def return_levels_pot(data,var,threshold=None,periods=[50,100], output_file='return_levels_pot.pdf'):
    """
    Subroutine written by clio-met (July 2023)
    Purpose: calculates the return levels (rl) from a time series (ts) and given threshold for different return periods (periods)
    - Inputs:
    1) ts is a dataframe with time enabled containing the hourly time series of the variable of interest (ex: wind speed)
    2) threshold is a scalar 
    2) periods is a numpy array containing the return periods (in years) of interest (ex: np.array([20,30,50,100]))
    - Follows the POT method 
    """
    if threshold == None:
        threshold = get_threshold_os(data=data, var=var)
    else:
        pass
    # Required modules
    periods = np.array(periods, dtype=float)
    # Take yearly maximum
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
    shape,loc,scale = genpareto.fit(sel_val.values,loc=threshold)
    rl = genpareto.isf(1/(lambd*periods), shape, loc, scale)
    # If the return level is unrealistic (usually very high), set rl to a special number
    rl=np.where(rl>=3*threshold,np.nan,rl)
    if output_file == False:
        pass
    else:
        plot_return_levels(data,var,rl,periods,output_file,it_selected_max=ts.iloc[it_selected_max].index)

    return rl


def return_levels_annual_max(data,var='hs',periods=[50,100,1000],method='GEV',output_file='return_levels_annual_max.pdf'): 
    """
    The function is written by dung-manh-nguyen and KonstantinChri.
    It calulates return value estimates for different periods using different methods 
    data = pandas dataframe
    var  = variable of the dataframe 
    periods = [50, 100,1000]
    methods = 'GEV', 'GUM' 
    out: return_values
    """
 
    #data_am = data[var].resample('Y').max() # get annual maximum with index YYYY-12
    it_selected_max = data.groupby(data.index.year)[var].idxmax().values
    data_am = data[var].loc[it_selected_max] # get annual maximum with actual dates that maximum occured
    periods = np.array(periods, dtype=float)
    for i in range(len(periods)) :
        if periods[i] == 1 : 
            periods[i] = 1.6
                
    if method == 'GEV' : 
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(data_am) # fit data
        # Compute the return levels for several return periods       
        rl = genextreme.isf(1/periods, shape, loc, scale)
    
    elif method == 'GUM' : 
        from scipy.stats import gumbel_r 
        loc, scale = gumbel_r.fit(data_am) # fit data
        # Compute the return levels for several return periods.
        rl = gumbel_r.isf(1/periods, loc, scale)

    else : 
        print ('please check method/distribution, must be either GEV or GUM')

    if output_file == False:
        pass
    else:
        plot_return_levels(data,var,rl,periods,output_file,it_selected_max)


    return rl

def get_threshold_os(data,var):
    """
    Subroutine written by clio-met (July 2023)
    Follows the method by Outten and Sobolowski (2021) to define the threshold to be used 
    in subroutine return_levels_pot
    """
    ts = data[var]
    yearly_max = ts.resample('YS').max()
    # Take the minimum of the yearly maxima found
    min_ym=yearly_max.min()
    return min_ym

def plot_return_levels(data,var,rl,periods, output_file,it_selected_max=[]):
    color = plt.cm.rainbow(np.linspace(0, 1, len(rl)))
    fig, ax = plt.subplots(figsize=(12,6))
    data[var].plot(color='lightgrey')
    for i in range(len(rl)):
        ax.hlines(y=rl[i], xmin=data[var].index[0], xmax=data[var].index[-1],color=color[i] ,linestyle='dashed', linewidth=2,label=str(rl[i].round(2))+' ('+str(int(periods[i]))+'y)' )

    plt.scatter(data[var][it_selected_max].index,data[var].loc[it_selected_max],s=10,marker='o',color='black',zorder=2)
    #plt.scatter(data[var].index[it_selected_max],data[var].iloc[it_selected_max],s=10,marker='o',color='black',zorder=2)
    
    plt.grid(linestyle='dotted')
    plt.ylabel(var,fontsize=18)
    plt.xlabel('time',fontsize=18)
    plt.legend(loc='center left')
    plt.title(output_file.split('.')[0],fontsize=18)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return