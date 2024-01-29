from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def return_levels_pot(data,var,threshold=None,periods=[50,100], output_file=False):
    """
    Subroutine written by clio-met (July 2023)
    Purpose: calculates the return levels (rl) from a time series (ts) and given threshold for different return periods (periods)
    - Inputs:
    1) ts is a dataframe with time enabled containing the hourly time series of the variable of interest (ex: wind speed)
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
        plot_return_levels(data,var,rl,periods,output_file,it_selected_max=ts.iloc[it_selected_max].index)
        #probplot(data=sel_val.values, sparams=(shape, loc, scale))
    return rl


def return_levels_annual_max(data,var='hs',periods=[50,100,1000],method='GEV',output_file=False): 
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
        #probplot(data=data[var].loc[it_selected_max], sparams=(shape, loc, scale))

    
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


def return_levels_weibull_2p(data,var,periods=[50,100,1000],threshold=None,r="48H"):
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
    time_step = yr_num*365.2422*24/length_data # in hours 
    return_period = np.array(periods)*24*365.2422/time_step # years is converted to K-th
    
    # Calculate the 2-parameter Weibull return value
    return_value = stats.weibull_min.isf(1/return_period, shape, loc, scale)
    
    for i in range(len(return_value)) : 
        return_value[i] = round(return_value[i],1)
    
    return return_value

def return_levels_exp(data,var='hs',periods=[50,100,1000],threshold=None, r="48H"):
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
    
    return_periods = np.array(periods)*24*365.2422/interval # years is converted to K-th
        
    return_levels = expon.isf(1/return_periods, loc, scale)
    
    return return_levels


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


def probplot(data, sparams):    
    import scipy.stats as stats
    stats.probplot(data,sparams=sparams, dist=stats.genpareto,fit=True, plot=plt)
    plt.grid()
    plt.axline((data[0], data[0]), slope=1,label='y=x')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def return_value_plot(obs_return_periods, obs_return_levels,model_return_periods, model_return_levels, model_method=''):    
    fig, ax = plt.subplots()   
    ax.scatter(obs_return_periods, obs_return_levels, marker="o",s=20,lw=1,facecolor="k",edgecolor="w",zorder=20)
    ax.plot(model_return_periods, model_return_levels,color='red',label=model_method)
    ax.semilogx()
    ax.grid(True, which="both")
    ax.set_xlabel("Return period")
    ax.set_ylabel("Return levels")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #alpha = 0.95
    #cil, ciu = np.quantile(a=rl_sample, q=[(1 - alpha) / 2, (1 + alpha) / 2])
    return




def get_empirical_return_levels(data,var,method="POT",block_size="365.2425D"):
    """
    Return an estimation of empirical/observed [return levels, return periods] 
    """
    from pyextremes import get_extremes, get_return_periods
    if method == 'BM':
        extremes = get_extremes(ts=data[var],method=method,block_size=block_size)
        rp = get_return_periods(ts=data[var],extremes=extremes,extremes_method=method,extremes_type="high",return_period_size=block_size)
    elif method == 'POT':
        extremes = get_extremes(ts=data[var],method=method,threshold=get_threshold_os(data=data, var=var))
        rp = get_return_periods(ts=data[var],extremes=extremes,extremes_method=method,extremes_type="high")
    return rp['return period'], rp[var]


def diagnostic_return_level_plot(data,var,model_method,periods=np.arange(0.1,1000,0.1),threshold=None):
    if model_method == 'POT':
        return_value_plot(obs_return_periods=get_empirical_return_levels(data,var)[0], obs_return_levels=get_empirical_return_levels(data,var)[1],
                          model_return_periods=periods, model_return_levels=return_levels_pot(data,var,threshold=threshold,periods=periods), model_method=model_method)
    elif model_method == 'GEV':
        return_value_plot(obs_return_periods=get_empirical_return_levels(data,var)[0], obs_return_levels=get_empirical_return_levels(data,var)[1],
                          model_return_periods=periods, model_return_levels=return_levels_annual_max(data,var,method='GEV',periods=periods), model_method=model_method)
    elif model_method == 'GUM':
        return_value_plot(obs_return_periods=get_empirical_return_levels(data,var)[0], obs_return_levels=get_empirical_return_levels(data,var)[1],
                          model_return_periods=periods, model_return_levels=return_levels_annual_max(data,var,method='GUM',periods=periods), model_method=model_method)
    elif model_method == 'Weibull_2P':
        return_value_plot(obs_return_periods=get_empirical_return_levels(data,var)[0], obs_return_levels=get_empirical_return_levels(data,var)[1],
                          model_return_periods=periods, model_return_levels=return_levels_weibull_2p(data,var,threshold=threshold,periods=periods), model_method=model_method)
    elif model_method == 'EXP':
        return_value_plot(obs_return_periods=get_empirical_return_levels(data,var)[0], obs_return_levels=get_empirical_return_levels(data,var)[1],
                          model_return_periods=periods, model_return_levels=return_levels_exp(data,var,threshold=threshold,periods=periods), model_method=model_method)
        


def get_joint_2D_contour(data=pd.DataFrame,var1='hs', var2='tp', periods=[50,100]) -> Sequence[Dict]:
    """Compute a joint contour for the given return periods. 
    Input:
        data: pd.DataFrame
        var1: e.g., 'hs'
        var2: e.g., 'tp'
        return_periods: A list of return periods in years, default [50,100]
    Output:
         contours : list of joint contours, 
         i.e.,  number of contours based on given return_periods 
    """
    from virocon import get_OMAE2020_Hs_Tz,get_DNVGL_Hs_U,get_OMAE2020_Hs_Tz,get_OMAE2020_V_Hs, GlobalHierarchicalModel,IFORMContour, ISORMContour     
    # Define 2D joint distribution model
    dist_descriptions, fit_descriptions, _ = get_OMAE2020_Hs_Tz()
    model = GlobalHierarchicalModel(dist_descriptions)
    dt = (data.index[1]-data.index[0]).total_seconds()/3600  # duration in hours
    data_2D =  np.transpose(np.array([data[var1],data[var2]]))
    model.fit(data_2D, fit_descriptions=fit_descriptions)
    contours = []
    for rp in periods:
        alpha = 1 / (rp * 365.25 * 24 / dt)
        contour = IFORMContour(model, alpha)
        coords = contour.coordinates
        x = coords[:, 1].tolist()
        y = coords[:, 0].tolist()
        contours.append({
            'return_period': rp,
            'x': x,
            'y': y
        })
    return contours, data_2D

def plot_joint_2D_contour(data=pd.DataFrame,var1='hs', var2='tp', periods=[50,100], output_file='2D_contours.png') -> Sequence[Dict]:
    contours, data_2D = get_joint_2D_contour(data=data,var1=var1,var2=var2, periods=periods)
    """Plot joint contours for the given return periods. 
    Input:
        data: pd.DataFrame
        var1: e.g., 'hs'
        var2: e.g., 'tp'
        return_periods: A list of return periods in years, default [50,100]
        output_file: Path to save the plot 
    Output:
         figure with 2Djoint contours
    """
    # Plot the contours and save the plot

    fig, ax = plt.subplots()
    if len(data_2D)>0:
        import seaborn as sns
        sns.set_theme(style="ticks")
        sns.scatterplot(x=data_2D[:,1], y=data_2D[:,0], ax=ax, color='black')
    else:
        pass

    color = plt.cm.rainbow(np.linspace(0, 1, len(periods)))
    # Compute an IFORM contour with a return period of N years
    #loop over contours, get index

    for i,contour in enumerate(contours):
        # Plot the contour
        x = []
        x.extend(contour['x'])
        x.append(x[0])

        y = []
        y.extend(contour['y'])
        y.append(y[0])

        rp = contour['return_period']
        ax.plot(x, y, color=color[i],label=str(rp)+'y')

        ax.set_xlabel(var2,fontsize='16')
        ax.set_ylabel(var1,fontsize='16')

    plt.grid()
    plt.title(output_file.split('.')[0],fontsize=18)
    plt.legend()
    plt.savefig(output_file,dpi=300)
    return fig, ax
