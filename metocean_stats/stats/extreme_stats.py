from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def return_levels_pot(data, var, dist='Weibull_2P', 
                      periods=[50, 100, 1000], 
                      threshold=None, r="48h",
                      output_file=False):
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
    
    return (list of float): return value estimates corresponding to input
                            periods 
    
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

    if output_file == False:
        pass
    else:
        plot_return_levels(data, var, return_levels, 
                           periods, output_file,
                           it_selected_max=extremes.index.values)

    return return_levels


def return_levels_annual_max(data, var='hs', dist='GEV', 
                             periods=[50, 100, 1000], 
                             output_file=False): 
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
    output_file (str): If not False, save the plot of the return value 
                       estimates at the given path.
    
    return (list of float): return value estimates corresponding to input
                            periods 
    
    Function written by dung-manh-nguyen and KonstantinChri.   
    """
    it_selected_max = data.groupby(data.index.year)[var].idxmax().values
    # get annual maximum with actual dates that maximum occured
    data_am = data[var].loc[it_selected_max] 
    periods = np.array(periods, dtype=float)
    for i in range(len(periods)) :
        if periods[i] == 1 : 
            periods[i] = 1.6

    if dist == 'GEV' :
        from scipy.stats import genextreme
        # fit data
        shape, loc, scale = genextreme.fit(data_am) 
        # Compute the return levels for several return periods       
        return_levels = genextreme.isf(1/periods, shape, loc, scale)

    elif dist == 'GUM' : 
        from scipy.stats import gumbel_r 
        loc, scale = gumbel_r.fit(data_am) # fit data
        # Compute the return levels for several return periods.
        return_levels = gumbel_r.isf(1/periods, loc, scale)

    else : 
        print ('please check method/distribution, must be either GEV or GUM')

    if output_file == False:
        pass
    else:
        plot_return_levels(data, var, return_levels, periods, 
                           output_file, it_selected_max)

    return return_levels


def return_levels_idm(data, var, dist='Weibull_3P', 
                      periods=[50, 100, 1000],
                      time_step=3,
                      output_file=False):
    """
    Calulates return value estimates for different periods, fitting a 
    given distribution to all available data (initial distribution method).  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    dist (str): 'Weibull_3P' for 3-parameters Weibull distribution
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    time_step (float): Average duration time of a state for the observed
                       variable, in hours. Default 3, for sea state.
    return (list of float): return value estimates corresponding to input
                            periods 
    
    """
    import scipy.stats as stats

    # in hours
    time_step = time_step 
    # years is converted to K-th
    return_periods = np.array(periods)*24*365.2422/time_step
    
    if dist == 'Weibull_3P':
        shape, loc, scale = stats.weibull_min.fit(data['hs'])
        return_levels = stats.weibull_min.isf(1/return_periods, 
                                              shape, loc, scale)
    else:
        print ('please check method/distribution, must be one of: EXP, \
                GP or Weibull_2P')

    if output_file == False:
        pass
    else:
        plot_return_levels(data, var, return_levels, 
                           periods, output_file,
                           it_selected_max=data['hs'].index.values)

    return return_levels
 

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


def plot_return_levels(data, var, rl, periods, 
                       output_file, it_selected_max=[]):
    """
    Plots data, extremes from these data and associated return levels.  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable 
    rl (1D-array or list): Return level estimates
    periods (1D-array or list): List of periods to for which to return return
                                value estimates
    output_file (str):  Path to save the plot to.
    it_selected_max (1D-array or list): Index corresponding to the extremes
    
    Function written by dung-manh-nguyen and KonstantinChri.
    """  
    color = plt.cm.rainbow(np.linspace(0, 1, len(rl)))
    fig, ax = plt.subplots(figsize=(12,6))
    data[var].plot(color='lightgrey')
    for i in range(len(rl)):
        ax.hlines(y=rl[i], 
                  xmin=data[var].index[0], 
                  xmax=data[var].index[-1],
                  color=color[i], linestyle='dashed', linewidth=2,
                  label=str(rl[i].round(2))+' ('+str(int(periods[i]))+'y)')

    plt.scatter(data[var][it_selected_max].index, 
                data[var].loc[it_selected_max], 
                s=10, marker='o', color='black', zorder=2)
    #plt.scatter(data[var].index[it_selected_max],
    #            data[var].iloc[it_selected_max],
    #            s=10, marker='o', color='black', zorder=2)

    plt.grid(linestyle='dotted')
    plt.ylabel(var, fontsize=18)
    plt.xlabel('time', fontsize=18)
    plt.legend(loc='center left')
    plt.title(output_file.split('.')[0], fontsize=18)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


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


def return_value_plot(obs_return_periods, obs_return_levels, 
                      model_return_periods, model_return_levels, 
                      dist =''):    
    """
    Plots empirical return level plot along one fitted distribution.

    obs_return_periods (1D-array): return periods obtained 
                                   from the observations
    obs_return_levels (1D-array): return levels obtained 
                                  from the observations
    model_return_periods (1D-array): return periods obtained 
                                     from the fitted model
    model_return_levels (1D-array): return levels obtained 
                                     from the fitted model
    dist (str): name of the fitted distribution (used for label here)
    
    return: plots the return value plot
    """
    fig, ax = plt.subplots()   
    ax.scatter(obs_return_periods, 
               obs_return_levels, 
               marker="o", s=20, lw=1, facecolor="k", 
               edgecolor="w", zorder=20)
    ax.plot(model_return_periods, 
            model_return_levels,
            color='red',
            label=dist)
    ax.semilogx()
    ax.grid(True, which="both")
    ax.set_xlabel("Return period")
    ax.set_ylabel("Return levels")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #alpha = 0.95
    #cil, ciu = np.quantile(a=rl_sample, q=[(1 - alpha) / 2, (1 + alpha) / 2])


def get_empirical_return_levels(data, var, method="POT",
                                block_size="365.2425D"):
    """
    Returns an estimation of empirical/observed [return periods, return levels]
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    method (str): Method of definition for the extremes, 
                  'BM' for Block Maxima or 'POT' for Peak over threshold
    block_size (str): Size of the block for block maxima
    
    return: return periods, return levels  
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
    elif method == 'POT':
        extremes = get_extremes(ts=data[var],
                                method=method,
                                threshold=get_threshold_os(data=data, var=var))
        rp = get_return_periods(ts=data[var],
                                extremes=extremes,
                                extremes_method=method,
                                extremes_type="high")
            
    return rp['return period'], rp[var]


def diagnostic_return_level_plot(data, var, dist, 
                                 periods=np.arange(0.1, 1000, 0.1),
                                 threshold=None):
    """
    Plots empirical return level plot along one fitted distribution.

    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    dist (str): name of the distribution to fit with the data
                         ('GP', 'GEV', 'GUM', 'Weibull_2P', 'EXP')
    periods (1D-array or list): Range of periods to compute
    threshold (float): Threshold used for POT methods
    
    return: plots the return value plot
    """
    # Get empirical return levels and periods
    empirical_rl_res = get_empirical_return_levels(data, var)
    obs_return_periods = empirical_rl_res[0]
    obs_return_levels = empirical_rl_res[1]

    if dist in ['GP', 'EXP', 'Weibull_2P']:
        return_value_plot(obs_return_periods=obs_return_periods, 
                          obs_return_levels=obs_return_levels,
                          model_return_periods=periods, 
                          model_return_levels=return_levels_pot(data, var, 
                                                          dist=dist,
                                                          threshold=threshold, 
                                                          periods=periods), 
                          dist=dist)

    elif dist in ['GEV', 'GUM']:
        return_value_plot(obs_return_periods=obs_return_periods, 
                          obs_return_levels=obs_return_levels,
                          model_return_periods=periods, 
                          model_return_levels=return_levels_annual_max(data, 
                                                              var,
                                                              dist=dist,
                                                              periods=periods),
                          dist=dist)

def diagnostic_return_level_plot_multi(data, var, 
                                       dist_list=['GP','EXP',
                                                  'Weibull_2P'],
                                       periods=np.arange(0.1, 1000, 0.1),
                                       threshold=None,
                                       yaxis='prob',
                                       output_file=False,
                                       display=False):
    """
    Plots empirical value plot along fitted distributions.

    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    dist_list (list of str): list of the names of the models to fit the
                                data and display
    periods (1D-array or list): Range of periods to compute
    threshold (float): Threshold used for POT methods
    yaxis (str): Either 'prob' (default) or 'rp', displays probability of
    non-exceedance or return period on yaxis respectively.
    output_file (str): path of the output file to save the plot, else None.
    display (bool): if True, displays the figure
    
    return: plots the return value plot
    """
    # Get empirical return levels and periods
    empirical_rl_res = get_empirical_return_levels(data, var)
    obs_return_periods = empirical_rl_res[0]
    obs_return_levels = empirical_rl_res[1]

    # Give default threshold if not given as argument
    if threshold == None:
        threshold = get_threshold_os(data=data, var=var)
    else:
        pass

    # Initialize plot and fill in empirical return levels
    fig, ax = plt.subplots()

    # Plot points (return level, return period) corresponding to
    # empirical values
    if yaxis == 'prob':
        # Troncate all series only to keep values corresponding to return 
        # period greater than 1 year
        periods_cut = [rp for rp in periods if rp >= 1]
        obs_return_periods_cut = [rp for rp in obs_return_periods if rp >= 1]
        obs_return_levels_cut = [obs_return_levels.iloc[i]
                                 for i, rp in enumerate(obs_return_periods)
                                 if rp >= 1]
        ax.scatter(obs_return_levels_cut,
                   obs_return_periods_cut,
                   marker="o", s=20, lw=1,
                   facecolor="k", edgecolor="w", zorder=20)

    elif yaxis == 'rp':
         ax.scatter(obs_return_levels,
                    obs_return_periods,
                    marker="o", s=20, lw=1,
                    facecolor="k", edgecolor="w", zorder=20)

    # Get return levels for the distributions given as argument,
    dist_rl_res={}

    for dist in dist_list:

        if dist in ['GP','Weibull_2P','EXP']:
            dist_rl_res[dist] = return_levels_pot(data, var,
                                                   dist=dist,
                                                   threshold=threshold,
                                                   periods=periods)

        elif dist in ['GEV','GUM']:
            dist_rl_res[dist] = return_levels_annual_max(data, var,
                                                          dist=dist,
                                                          periods=periods)

        # Troncate all series only to keep values corresponding to return
        # period greater than 1 year
        dist_rl_cut = [dist_rl_res[dist][i]
                      for i, rp in enumerate(periods)
                      if rp >= 1]

        # Plot (return levels, return periods) lines corresponding
        if yaxis == 'prob':
            ax.plot(dist_rl_cut,
                    periods_cut,
                    label=dist)

        elif yaxis == 'rp':
            ax.plot(dist_rl_res[dist],
                    periods,
                    label=dist)

    plt.yscale('log')
    ax.grid(True, which="both")

    # Change label and ticklabels of y-axis, to display either probabilities
    # of non-exceedance or return period
    if yaxis == 'prob':
        ax.set_ylabel("Probability of non-exceedance")
        list_yticks = [1/0.9, 2, 10]\
                    + [10**i for i in range(2, 5) if max(periods) > 10**i]
        if max(periods) > max(list_yticks):
            list_yticks = list_yticks + [max(periods)]

        ax.set_yticks(list_yticks)
        ax.set_yticklabels([round(1-1/rp,5) for rp in list_yticks])

    elif yaxis == 'rp':
        ax.set_ylabel("Return period")

    ax.set_xlabel("Return levels")
    plt.legend()
    plt.tight_layout()

    # Save the plot if a path is given
    if output_file == False:
        pass
    else:
        plt.savefig(output_file)

    if display:
        plt.show()


def return_level_threshold(data, var, thresholds, 
                           dist_list=['GP','EXP','Weibull_2P'], 
                           period=100,
                           output_file=False):
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
                dict_rl[dist].append(return_levels_pot(data, var,
                                                   dist=dist,
                                                   threshold=thresh,
                                                   periods=[period])[0])

            elif dist in ['GEV','GUM']:
                dict_rl[dist].append(return_levels_annual_max(data, var,
                                                          dist=dist,
                                                          periods=[period])[0])

    if output_file == False:
        pass
    else:
        plot_return_level_threshold(data, var, dict_rl, period,
                                    thresholds, output_file)

    return dict_rl, thresholds


def plot_return_level_threshold(data, var, dict_rl, period,
                                thresholds, output_file):
    """
    Plots theoretical return level for given return period and distribution,
    as a function of the threshold.

    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    period (int or float): Return period
    dict_rl (dict of list of floats): Contains the return levels corresponding
                                      to the given period for each method (key)
                                      and each threshold
    thresholds (float): Range of thresholds used for POT methods
    output_file (str): path of the output file to save the plot

    return: plots return levels in function of the thresholds, for each method 
    and saves the result into the given output_file 
    """
    # Plotting the result
    fig, ax = plt.subplots(1,1)

    threshold_os = get_threshold_os(data, var)

    for dist in dict_rl.keys():

        ax.plot(thresholds, dict_rl[dist], label=dist)

    ax.axvline(threshold_os, color = 'black', 
               label = 'Threshold Outten and Sobolowski (2021)',
               alpha=0.5, linestyle = 'dashed')

    ax.grid(True, which="both")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(var)
    ax.legend()
    plt.title('{} year return value estimate'.format(period))

    # Save the plot if a path is given
    plt.savefig(output_file)

    plt.show()


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
    from virocon import get_OMAE2020_Hs_Tz, get_DNVGL_Hs_U,get_OMAE2020_Hs_Tz,get_OMAE2020_V_Hs, GlobalHierarchicalModel,IFORMContour, ISORMContour     
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
    # loop over contours, get index

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

def RVE_ALL(dataframe,var='hs',periods=np.array([1,10,100,1000]),distribution='Weibull3P',method='default',threshold='default'):
    
    # data : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # distribution: 'EXP', 'GEV', 'GUM', 'LoNo', 'Weibull2P' or 'Weibull3P'
    # method: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 

    import scipy.stats as stats
    from pyextremes import get_extremes
    
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
        data = get_extremes(df, method="POT", threshold=threshold, r="48H")
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
    interval = duration*24/length_data # in hours 
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
    else:
        print ('Please check the distribution')    
        
    if method == 'default' :  
    	output_file= distribution + '.png'
    else:
        output_file= distribution + '(' + method + ')' + '.png'
        
    plot_return_levels(dataframe,var,value,periods,output_file,it_selected_max)
       
    return 
