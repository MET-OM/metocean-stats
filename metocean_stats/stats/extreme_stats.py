# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt

def return_levels_pot(data,var,periods):
    """
    Subroutine created by Clio (July 2023)
    Goal: calculates the return levels (rl) from a time series (ts) for different return periods (periods)
    - Inputs:
    1) ts is an xarray with time enabled containing the time series of the daily or monthly maximum of the variable of interest (ex: wind speed)
    2) periods is a numpy array containing the return periods (in years) of interest (ex: np.array([20,30,50,100]))
    - Follows the method by Outten and Sobolowski (2021) (POT method)
    """
    # Required modules
    import pandas as pd
    import numpy as np
    from scipy.stats import genpareto as genpareto
    # Take yearly maximum
    ts = data[var]
    #yearly_max=ts.groupby("time.year").max("time")
    yearly_max = ts.resample('YS').max()
    # Take the minimum of the yearly maxima found
    min_ym=yearly_max.min()
    # min_ym is the threshold to select the extreme values in the timeseries ts
    it=np.where(ts.values>=min_ym)[0]
    # Group consecutive indices with a +/-2 days difference
    # to select only one event and not consecutive timesteps with values above threshold
    larrs=np.split(it,np.where(np.diff(it)>2)[0]+1)
    it_selected_max=[]
    # Pick the maximum value of the variable within groups
    for l in range(len(larrs)):
        mxi=np.argmax(ts.iloc[larrs[l]].values) # If several values are equal to the max, argmax takes the first occurrence 
        it_selected_max.append(larrs[l][mxi])
        del mxi
    # Get the selected values and time for the extreme value analysis
    sel_val=ts.iloc[it_selected_max]
    sel_time=sel_val.index
    # Total number of selected values
    ns=len(sel_val)
    # Number of selected values in each year
    ns_yr=sel_val.resample("YS").count() # Should give an array of 30 values if 30 years are considered
    # Expected rate of occurrence (mean number of occurrences per year)
    lambd=np.mean(ns_yr.values,axis=0)
    # Fit the Generalized Pareto distribution to the selected values
    shape,loc,scale = genpareto.fit(sel_val.values,loc=min_ym)
    # Calculate the return levels
    rl = genpareto.isf(1/(lambd*periods), shape, loc, scale)
    # If the return level is unrealistic (usually very high), set rl to a special number
    if (rl>=3*min_ym):
        rl=np.nan
    return rl
