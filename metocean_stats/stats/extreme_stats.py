from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from .aux_funcs import Tp_correction


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


def plot_return_levels(data, var, rl, output_file):
    """
    Plots data, extremes from these data and associated return levels.  
    
    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable 
    rl (pandas DataFrame): Return level estimates, output of one of the
                           return_levels function
    output_file (str): Path to save the plot to.
    
    Function written by dung-manh-nguyen and KonstantinChri.
    """
    from pyextremes import get_extremes
    
    method = rl.attrs['method']

    color = plt.cm.rainbow(np.linspace(0, 1, len(rl)))
    fig, ax = plt.subplots(figsize=(12,6))
    data[var].plot(color='lightgrey')
    for i in range(len(rl.index)):
        ax.hlines(y=rl.return_levels.iloc[i], 
                  xmin=data[var].index[0], 
                  xmax=data[var].index[-1],
                  color=color[i], linestyle='dashed', linewidth=2,
                  label=str(rl.return_levels.iloc[i].round(2))+\
                            ' ('+str(int(rl.index[i]))+'y)')

    if method == 'pot':
        r = rl.attrs['r']
        threshold = rl.attrs['threshold']
        extremes = get_extremes(data[var], method="POT", 
                                threshold=threshold, r=r)
        it_selected_max = extremes.index.values

    elif method == 'AM':
        it_selected_max = data.groupby(data.index.year)[var].idxmax().values

    elif method == 'idm':
        it_selected_max = data[var].index.values

    plt.scatter(data[var][it_selected_max].index, 
                data[var].loc[it_selected_max], 
                s=10, marker='o', color='black', zorder=2)

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


def plot_diagnostic_return_levels(data, var, dist, 
                                  periods=np.arange(0.1, 1000, 0.1),
                                  threshold=None,
                                  output_file=None):
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
    
    if dist in ['GP', 'EXP', 'Weibull_2P']:
        
        if threshold is None:
             threshold = get_threshold_os(data=data, var=var)  
        
        # Get empirical return levels and periods 
        df_emp_rl = get_empirical_return_levels(data=data, var=var, 
                                                method='POT', 
                                                threshold=threshold)
        # Get model return levels and periods                                         
        df_model_rl = return_levels_pot(data, var, 
                                        dist=dist,
                                        threshold=threshold, 
                                        periods=periods)
        
    elif dist in ['GEV', 'GUM']:
    
        # Get empirical return levels and periods 
        df_emp_rl = get_empirical_return_levels(data=data, var=var, 
                                                method='BM')
        # Get model return levels and periods                                        
        df_model_rl = return_levels_annual_max(data, 
                                               var,
                                               dist=dist,
                                               periods=periods)
     
    # Plot obtained data   
    fig, ax = plt.subplots()   
    ax.scatter(df_emp_rl['return_periods'], 
               df_emp_rl['return_levels'], 
               marker="o", s=20, lw=1, facecolor="k", 
               edgecolor="w", zorder=20)
    ax.plot(df_model_rl.index, 
            df_model_rl['return_levels'],
            color='red',
            label=dist)
    ax.semilogx()
    ax.grid(True, which="both")
    ax.set_xlabel("Return period")
    ax.set_ylabel("Return levels")
    plt.legend()
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()
    

def plot_multi_diagnostic_return_levels(data, var, 
                                        dist_list=['GP','EXP',
                                                   'Weibull_2P'],
                                        periods=np.arange(0.1, 1000, 0.1),
                                        threshold=None,
                                        yaxis='prob',
                                        output_file=None):
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

    return: plots the return value plot
    """

    if ('GP' or 'EXP' or 'Weibull_2P') in dist_list:
        
        # Get empirical return levels and periods
        if threshold is None:
             threshold = get_threshold_os(data=data, var=var)  
        
        # Get empirical return levels and periods 
        df_emp_rl = get_empirical_return_levels(data=data, 
                                                var=var, 
                                                method='POT', 
                                                threshold=threshold)

    elif ('GEV' or 'GUM') in dist_list:
    
        # Get empirical return levels and periods 
        df_emp_rl = get_empirical_return_levels(data=data, 
                                                var=var, 
                                                method='BM')
    elif ('Weibull_3P') in dist_list:
        
        interval = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
        interval_day = 1/(24/interval)
        
        df_emp_rl = get_empirical_return_levels(data=data, 
                                                var=var, 
                                                method='BM',
                                                block_size= str(interval_day) + "D")
        df_emp_rl.loc[:,'return_periods'] = df_emp_rl.loc[:,'return_periods']/(365.24/interval_day)
        df_emp_rl = df_emp_rl.loc[df_emp_rl.loc[:,'return_periods'] >= 1.0,:]
                                                
    # Initialize plot and fill in empirical return levels
    fig, ax = plt.subplots()

    # Plot points (return level, return period) corresponding to
    # empirical values
    if yaxis == 'prob':
        # Troncate all series only to keep values corresponding to return 
        # period greater than 1 year
        df_emp_rl_cut = df_emp_rl[df_emp_rl['return_periods'] >= 1]

        ax.scatter(df_emp_rl_cut['return_levels'],
                   df_emp_rl_cut['return_periods'],
                   marker="o", s=20, lw=1,
                   facecolor="k", edgecolor="w", zorder=20)

    elif yaxis == 'rp':
         ax.scatter(df_emp_rl['return_levels'],
                    df_emp_rl['return_periods'],
                    marker="o", s=20, lw=1,
                    facecolor="k", edgecolor="w", zorder=20)

    for dist in dist_list:

        if dist in ['GP','Weibull_2P','EXP']:
            df_model_rl_tmp = return_levels_pot(data, var,
                                                dist=dist,
                                                threshold=threshold,
                                                periods=periods)

        elif dist in ['GEV','GUM']:
            df_model_rl_tmp = return_levels_annual_max(data, var,
                                                       dist=dist,
                                                       periods=periods)
                                                       
        elif dist in ['Weibull_3P']:
            df_model_rl_tmp = return_levels_idm(data, var, 
                                               dist=dist, 
                                               periods=periods)

        # Troncate all series only to keep values corresponding to return
        # period greater than 1 year
        df_model_rl_tmp_cut = df_model_rl_tmp[df_model_rl_tmp.index >= 1]

        # Plot (return levels, return periods) lines corresponding
        if yaxis == 'prob':
            ax.plot(df_model_rl_tmp_cut['return_levels'],
                    df_model_rl_tmp_cut.index,
                    label=dist)

        elif yaxis == 'rp':
            ax.plot(df_model_rl_tmp['return_levels'],
                    df_model_rl_tmp.index,
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
    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


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


def plot_threshold_sensitivity(df, output_file):
    """
    Plots theoretical return level for given return period and distribution,
    as a function of the threshold.

    df (pd.DataFrame): dataframe containing the return levels in function of 
                       the thresholds. The year of the corresponding return
                       levels as well as the variable are 
    output_file (str): path of the output file to save the plot

    return: plots return levels in function of the thresholds, for each method 
    and saves the result into the given output_file 
    """  
    fig, ax = plt.subplots(1,1)

    for dist in df.columns:

        ax.plot(df.index, df[dist], label=dist)

    ax.grid(True, which="both")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(df.attrs['var'])
    ax.legend()
    plt.title('{} year return value estimate'.format(df.attrs['period']))

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


def joint_distribution_Hs_Tp(df,var1='hs',var2='tp',periods=np.array([1,10,100,10000]),save_rve=False, title='Hs-Tp joint distribution',file_out='Hs.Tp.joint.distribution.png',density_plot=False):  
    
    """
    This fuction will plot Hs-Tp joint distribution using LogNoWe model (the Lognormal + Weibull distribution) 
    df : dataframe, 
    var1 : Hs: significant wave height,
    var2 : Tp: Peak period 
    file_out: Hs-Tp joint distribution, optional
    """
    max_y = max(periods)
    
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
         
    df['hs'] = df[var1].values
    df['tp'] = Tp_correction(df[var2].values)
    
    interval = ((df.index[-1]-df.index[0]).days + 1)*24/df.shape[0] # in hours 
    
    import scipy.stats as stats
    from scipy.signal import find_peaks
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
   
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
    def Gauss3(x, a1, a2):
        y = a1 + a2*x**0.36
        return y
    parameters, covariance = curve_fit(Gauss3, mean_hs, mean_lnTp)
    a1 = parameters[0]
    a2 = parameters[1]
    a3 = 0.36
    
    # calcualte b1, b2, b3 
    def Gauss4(x, b2, b3):
        y = 0.005 + b2*np.exp(-x*b3)
        return y
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
        
        
    def Hs_Tp_curve(data,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,X=100):
        
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


        
    def DVN_steepness(df,h,t,max_y=max(periods)):
        ## steepness 
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
        
        
        
    def find_percentile(data,pdf_Hs_Tp,h,t,p,max_y=max(periods)):
        ## find pecentile
        # RVE of X years 
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
        
           
    t_steepness, h_steepness = DVN_steepness(df,h,t)
    percentile05 = find_percentile(df.hs.values,pdf_Hs_Tp,h,t,5)
    percentile50 = find_percentile(df.hs.values,pdf_Hs_Tp,h,t,50)
    percentile95 = find_percentile(df.hs.values,pdf_Hs_Tp,h,t,95)
    

    plt.figure(figsize=(8,6))
    df = df[df['hs'] >= 0.1]
    fig, ax = plt.subplots()
    if density_plot is False: 
        plt.scatter(df.tp.values,df.hs.values,c='red',label='data',s=3)
    else:
        import seaborn as sns
        sns.set_theme(style="ticks")
        g = sns.jointplot(x=df.tp.values, y=df.hs.values, kind="hex", color="#5d5d60",joint_kws={'gridsize':40, 'bins':'log'})


    for i in range(len(periods)):
        param = Hs_Tp_curve(df.hs.values,pdf_Hs,pdf_Hs_Tp,f_Hs_Tp,h,t,X=periods[i])
        plt.plot(param[0],param[1],label=str(param[2])+'-year')
        if save_rve :
            param[3].to_csv(str(param[2])+'_year.csv', index=False)    
        
   
    plt.plot(t_steepness,h_steepness,'k--',label='steepness')
    
    plt.plot(percentile50[0],percentile50[1],'g',label='Tp-mean',linewidth=5)
    plt.plot(percentile05[0],percentile05[1],'g:',label='Tp-5%',linewidth=2)
    plt.plot(percentile95[0],percentile95[1],'g--',label='Tp-95%',linewidth=2)

    plt.xlabel('Tp - Peak Period [s]')
    plt.suptitle(title)
    plt.ylabel('Hs - Significant Wave Height [m]')
    plt.grid()
    plt.legend() 
    plt.savefig(file_out,dpi=100,facecolor='white',bbox_inches='tight')
    
    return 

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


def plot_bounds(file='NORA10_6036N_0336E.1958-01-01.2022-12-31.txt'):
    from metocean_stats.stats import general_stats, extreme_stats, dir_stats, aux_funcs
    import pandas as pd 
    import numpy as np 
    from scipy.stats import weibull_min
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    
    def readNora10File(file):
        df = pd.read_csv(file, delim_whitespace=True, header=3) # sep=' ', header=None,0,1,2,3
        df.index= pd.to_datetime(df.YEAR*1000000+df.M*10000+df.D*100+df.H,format='%Y%m%d%H')
        df['tp_corr_nora10'] = aux_funcs.Tp_correction(df.TP.values)
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
    
    
    
    df = readNora10File(file)
    
    # Fit Weibull distribution to your data and estimate parameters
    data = df.HS.values  # Your data
    shape, loc, scale = Weibull_method_of_moment(data)
    
    # Define return periods
    periods = np.arange(1.5873, 10000, 100) 
    return_periods = periods
    return_periods = return_periods*365.2422*24/3
    # Calculate return values
    return_values = weibull_min.ppf(1 - 1 / return_periods, shape, loc, scale)
    
    # Bootstrap to estimate confidence bounds
    num_bootstrap_samples = 1000
    bootstrap_return_values = []
    for _ in range(num_bootstrap_samples):
        # Resample data with replacement
        bootstrap_sample = np.random.choice(data, size=1000, replace=True)
        
        # Fit Weibull distribution to resampled data
        shape_b, loc_b, scale_b = Weibull_method_of_moment(bootstrap_sample)
        # Calculate return values for resampled distribution
        bootstrap_return_values.append(weibull_min.ppf(1 - 1 / return_periods, shape_b, loc_b, scale_b))
    
    # Calculate confidence bounds
    lower_bounds = np.percentile(bootstrap_return_values, 2.5, axis=0)
    upper_bounds = np.percentile(bootstrap_return_values, 97.5, axis=0)
    
    
    
    modelled_data = pd.DataFrame()
    modelled_data['return value'] = return_values
    modelled_data['lower ci'] =     lower_bounds
    modelled_data['upper ci'] =     upper_bounds
    modelled_data['return period'] = periods
    
    
    
    # Generate some random data
    x = modelled_data['return period'].values #np.linspace(0, 10, 100)
    y = modelled_data['return value'].values ##np.sin(x)
    
    # Generate upper and lower bounds
    upper_bound = modelled_data['upper ci'].values
    lower_bound = modelled_data['lower ci'].values
    
    # Plot the data
    plt.plot(x, y, label='Weibull')
    plt.xscale('log')
    plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.3, label='Bounds')
    plt.xlabel('Years')
    plt.ylabel('Waves')
    plt.title('Plot with Bounds')
    plt.legend()
    
    return 
