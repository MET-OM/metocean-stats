import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import scipy.stats as st

from cycler import cycler
from pyextremes import get_extremes

from .. import stats
from .. import tables
from ..stats import aux_funcs
from ..CMA import JointProbabilityModel

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
    
    method = rl.attrs['method']

    color = plt.cm.rainbow(np.linspace(0, 1, len(rl)))
    fig, ax = plt.subplots(figsize=(12,6))
    data[var].plot(color='lightgrey')
    for i in range(len(rl.index)):
        ax.hlines(y=rl.return_levels.iloc[i], 
                  xmin=data[var].index[0], 
                  xmax=data[var].index[-1],
                  color=color[i], linestyle='dashed', linewidth=2,
                  label=str(np.round(rl.return_levels.iloc[i],2))+\
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
    if output_file != "": 
        plt.savefig(output_file)
    plt.close()


def probplot(data, sparams):    
    st.probplot(data, sparams=sparams, 
                   dist=st.genpareto,fit=True, plot=plt)
    plt.grid()
    plt.axline((data[0], data[0]), slope=1,label='y=x')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


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
             threshold = stats.get_threshold_os(data=data, var=var)  
        
        # Get empirical return levels and periods 
        df_emp_rl = stats.get_empirical_return_levels(data=data, var=var, 
                                                method='POT', 
                                                threshold=threshold)
        # Get model return levels and periods                                         
        df_model_rl = stats.return_levels_pot(data, var, 
                                        dist=dist,
                                        threshold=threshold, 
                                        periods=periods)
        
    elif dist in ['GEV', 'GUM']:
    
        # Get empirical return levels and periods 
        df_emp_rl = stats.get_empirical_return_levels(data=data, var=var, 
                                                method='BM')
        # Get model return levels and periods                                        
        df_model_rl = stats.return_levels_annual_max(data, 
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
             threshold = stats.get_threshold_os(data=data, var=var)  
        
        # Get empirical return levels and periods 
        df_emp_rl = stats.get_empirical_return_levels(data=data, 
                                                var=var, 
                                                method='POT', 
                                                threshold=threshold)

    elif ('GEV' or 'GUM') in dist_list:
    
        # Get empirical return levels and periods 
        df_emp_rl = stats.get_empirical_return_levels(data=data, 
                                                var=var, 
                                                method='BM')
    elif ('Weibull_3P') in dist_list:
        
        interval = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
        interval_day = 1/(24/interval)
        
        df_emp_rl = stats.get_empirical_return_levels(data=data, 
                                                var=var, 
                                                method='BM',
                                                block_size= str(interval_day) + "D")
        df_emp_rl.loc[:,'return_periods'] = df_emp_rl.loc[:,'return_periods']/(365.24/interval_day)
        df_emp_rl = df_emp_rl.loc[df_emp_rl.loc[:,'return_periods'] >= 1.0,:]

    # Initialize plot and fill in empirical return levels
    fig, ax = plt.subplots()
    ax.margins(x=0,y=0)

    # Plot points (return level, return period) corresponding to
    # empirical values
    if yaxis == 'prob':
        # Troncate all series only to keep values corresponding to return 
        # period greater than 1 year
        df_emp_rl_cut = df_emp_rl[df_emp_rl['return_periods'] >= 1]

        #ax.scatter(df_emp_rl_cut['return_levels'],
        #           df_emp_rl_cut['return_periods'],
        #           marker="o", s=20, lw=1,
        #           facecolor="k", edgecolor="w", zorder=20)
        ax.scatter(df_emp_rl_cut['return_levels'],
                   1-df_emp_rl_cut['prob_non_exceedance'],
                   marker="o", s=20, lw=1,
                   facecolor="k", edgecolor="w", zorder=20)

    elif yaxis == 'rp':
         ax.scatter(df_emp_rl['return_levels'],
                    df_emp_rl['return_periods'],
                    marker="o", s=20, lw=1,
                    facecolor="k", edgecolor="w", zorder=20)

    for dist in dist_list:

        if dist in ['GP','Weibull_2P','EXP']:
            df_model_rl_tmp = stats.return_levels_pot(data, var,
                                                dist=dist,
                                                threshold=threshold,
                                                periods=periods)

        elif dist in ['GEV','GUM']:
            df_model_rl_tmp = stats.return_levels_annual_max(data, var,
                                                       dist=dist,
                                                       periods=periods)
                                                       
        elif dist in ['Weibull_3P']:
            df_model_rl_tmp = stats.return_levels_idm(data, var, 
                                               dist=dist, 
                                               periods=periods)

        # Troncate all series only to keep values corresponding to return
        # period greater than 1 year
        df_model_rl_tmp_cut = df_model_rl_tmp[df_model_rl_tmp.index >= 1]

        # Plot (return levels, return periods) lines corresponding
        if yaxis == 'prob':
            #ax.plot(df_model_rl_tmp_cut['return_levels'],
            #        df_model_rl_tmp_cut.index,
            #        label=dist)
            ax.plot(df_model_rl_tmp_cut['return_levels'],
                    1-df_model_rl_tmp_cut['prob_non_exceedance'],
                    label=dist)

        elif yaxis == 'rp':
            ax.plot(df_model_rl_tmp['return_levels'],
                    df_model_rl_tmp.index,
                    label=dist)

    ax.set_yscale('log')
    ax.grid(True, which="both")

    # Change label and ticklabels of y-axis, to display either probabilities
    # of non-exceedance or return period
    if yaxis == 'prob':
        ax.invert_yaxis() # only invert if probability plot.
        ticks = np.array([0,0.5,0.9,0.99,0.999,0.9999,0.99999,0.999999,0.9999999])
        ax.set_ylabel("Probability of non-exceedance")
        ylim = ax.get_ylim()
        ax.set_yticks(1-ticks,ticks)
        # This line should expand ylim to the nearest tick above.
        ax.set_ylim(ylim[0],1-ticks[np.searchsorted(ticks,1-ylim[1])])

    elif yaxis == 'rp':
        ax.set_ylabel("Return period [yr]")

    ax.set_xlabel("Return levels "+var)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save the plot if a path is given
    if output_file is not None:
        plt.savefig(output_file)

    return fig

def plot_multi_diagnostic_return_levels_uncertainty(data, var, 
                                        dist_list=['GP','EXP',
                                                   'Weibull_2P'],
                                        periods=np.arange(0.1, 10000.1, 0.1).tolist(),
                                        threshold=None,
                                        yaxis='prob',
                                        uncertainty=None,
                                        output_file=None):
    """
    Plots empirical value plot along fitted distributions.

    data (pd.DataFrame): dataframe containing the time series
    var (str): name of the variable
    dist_list (list of str): list of the names of the models to fit the
                                data and display
    periods (list): Range of periods to compute
    threshold (float): Threshold used for POT methods
    yaxis (str): Either 'prob' (default) or 'rp', displays probability of
    non-exceedance or return period on yaxis respectively.
    uncertainty (float): confidence interval between 0 and 1 (recommended 0.95)
    output_file (str): path of the output file to save the plot, else None.

    return: plots the return value plot
    !!!!! Do not plot GEV or GUM with GP or EXP or Weibull_2P (do not mix POT and annual max)
    Modified by clio-met
    """
    if (('GP' in dist_list) or ('EXP' in dist_list) or ('Weibull_2P' in dist_list)):
        # Get empirical return levels and periods
        if threshold is None:
             threshold = stats.get_threshold_os(data=data, var=var)

        # Get empirical return levels and periods 
        df_emp_rl = stats.get_empirical_return_levels_new(data=data, 
                                                var=var, 
                                                method='POT', 
                                                threshold=threshold)
    elif (('GEV' in dist_list) or ('GUM' in dist_list)):
    
        # Get empirical return levels and periods 
        df_emp_rl = stats.get_empirical_return_levels_new(data=data, 
                                                var=var, 
                                                method='BM')
    elif ('Weibull_3P' in dist_list):

        interval = ((data.index[-1]-data.index[0]).days + 1)*24/data.shape[0]
        interval_day = 1/(24/interval)

        df_emp_rl = stats.get_empirical_return_levels_new(data=data, 
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
        ax.scatter(df_emp_rl['return_levels'],
                   df_emp_rl['prob_non_exceedance'],
                   marker="o", s=20, lw=1,
                   facecolor="k", edgecolor="w", zorder=20)
    elif yaxis == 'rp':
        ax.scatter(df_emp_rl['return_periods'],
                    df_emp_rl['return_levels'],
                    marker="o", s=10, lw=1,
                    facecolor="k", edgecolor=None, zorder=20)

    for dist in dist_list:

        if dist in ['GP','Weibull_2P','EXP']:
            df_model_rl_tmp = stats.return_levels_pot_uncertainty(data, var,
                                                dist=dist,
                                                threshold=threshold,
                                                periods=periods,
                                                uncertainty=uncertainty)
        elif dist in ['GEV','GUM']:
            df_model_rl_tmp = stats.return_levels_annual_max_uncertainty(data, var,
                                                       dist=dist,
                                                       periods=periods,
                                                       uncertainty=uncertainty)
        elif dist in ['Weibull_3P']:
            df_model_rl_tmp = stats.return_levels_idm(data, var, 
                                               dist=dist, 
                                               periods=periods)
        df_model_rl_tmp=df_model_rl_tmp.dropna()
        # Plot (return levels, return periods) lines corresponding
        if yaxis == 'prob':
            pl=ax.plot(df_model_rl_tmp['return_levels'],
                       df_model_rl_tmp['prob_non_exceedance'],
                       label=dist,zorder=10)
            if uncertainty is not None:
                ax.fill_betweenx(df_model_rl_tmp['prob_non_exceedance'],
                                df_model_rl_tmp['ci_lower_rl'],
                                df_model_rl_tmp['ci_upper_rl'],
                                color=pl[0].get_color(),
                                alpha=0.3)
                ax.plot(df_model_rl_tmp['ci_lower_rl'],
                    df_model_rl_tmp['prob_non_exceedance'],
                    color=pl[0].get_color(),
                    linestyle='dashed',
                    linewidth=1)
                ax.plot(df_model_rl_tmp['ci_upper_rl'],
                    df_model_rl_tmp['prob_non_exceedance'],
                    color=pl[0].get_color(),
                    linestyle='dashed',
                    linewidth=1)
        elif yaxis == 'rp':
            pl=ax.plot(df_model_rl_tmp.index,
                       df_model_rl_tmp['return_levels'],
                       label=dist,zorder=10)
            if uncertainty is not None:
                ax.fill_between(df_model_rl_tmp.index,
                                df_model_rl_tmp['ci_lower_rl'],
                                df_model_rl_tmp['ci_upper_rl'],
                                color=pl[0].get_color(),
                                alpha=0.2)
                ax.plot(df_model_rl_tmp.index,
                    df_model_rl_tmp['ci_lower_rl'],
                    color=pl[0].get_color(),
                    linestyle='dashed',
                    linewidth=0.5)
                ax.plot(df_model_rl_tmp.index,
                    df_model_rl_tmp['ci_upper_rl'],
                    color=pl[0].get_color(),
                    linestyle='dashed',
                    linewidth=0.5)

    ax.grid(True, which="both")

    # Change label and ticklabels of y-axis, to display either probabilities
    # of non-exceedance or return period
    if yaxis == 'prob':
        ax.set_ylabel("Probability of non-exceedance")
        ax.set_xlabel("Return levels "+var)
        #min_y=df_model_rl_tmp_cut['prob_non_exceedance'].min()
        #ax.set_ylim(min_y,1.01)
        #list_yticks = [1/0.9, 2, 10]\
        #            + [10**i for i in range(2, 5) if max(periods) > 10**i]
        #if max(periods) > max(list_yticks):
        #    list_yticks = list_yticks + [max(periods)]
        #ax.set_yticks(list_yticks)
        #ax.set_yticklabels([round(1-1/rp,5) for rp in list_yticks])
        ax.set_ylim(0.1,1)
        plt.legend(loc='lower right')
    elif yaxis == 'rp':
        ax.set_xscale('log')
        ax.set_xlabel("Return period [yr]", fontsize=22)
        ax.set_ylabel("Return levels "+var, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(loc='upper left')

    plt.tight_layout()

    # Save the plot if a path is given
    if output_file is not None:
        plt.savefig(output_file,dpi=250)

    return fig


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
    if output_file != "": plt.savefig(output_file)

    return fig

def plot_RVE_ALL(dataframe,var='hs',periods=np.array([1,10,100,1000]),distribution='Weibull3P',method='default',threshold='default'):
    
    # data : dataframe, should be daily or hourly
    # period: a value, or an array return periods =np.array([1,10,100,10000],dtype=float)
    # distribution: 'EXP', 'GEV', 'GUM', 'LoNo', 'Weibull2P' or 'Weibull3P'
    # method: 'default', 'AM' or 'POT'
    # threshold='default'(min anual maxima), or a value 

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
            
    # duration = (df.index[-1]-df.index[0]).days + 1 
    # length_data = data.shape[0]
    #interval = duration*24/length_data # in hours 
    interval = ((df.index[-1]-df.index[0]).days + 1)*24/df.shape[0] # in hours 
    period = period*365.2422*24/interval # years is converted to K-th
    
    # Fit a distribution to the data
    if distribution == 'EXP' : 
        loc, scale = st.expon.fit(data)
        value = st.expon.isf(1/period, loc, scale)
        #value = st.expon.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'GEV' :
        shape, loc, scale = st.genextreme.fit(data) # fit data   
        value = st.genextreme.isf(1/period, shape, loc, scale)
        #value = st.genextreme.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'GUM' :
        loc, scale = st.gumbel_r.fit(data) # fit data
        value = st.gumbel_r.isf(1/period, loc, scale)
        #value = st.gumbel_r.ppf(1 - 1 / period, loc, scale)
    elif distribution == 'LoNo' :
        shape, loc, scale = st.lognorm.fit(data)
        value = st.lognorm.isf(1/period, shape, loc, scale)
        #value = st.lognorm.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull2P' :
        shape, loc, scale = st.weibull_min.fit(data, floc=0) # (ML)
        value = st.weibull_min.isf(1/period, shape, loc, scale)
        #value = st.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    elif distribution == 'Weibull3P' : 
        shape, loc, scale = st.weibull_min.fit(data) # (ML)
        value = st.weibull_min.isf(1/period, shape, loc, scale)
        #value = st.weibull_min.ppf(1 - 1 / period, shape, loc, scale)
    else:
        print ('Please check the distribution')    
        
    if method == 'default' :  
        output_file= distribution + '.png'
    else:
        output_file= distribution + '(' + method + ')' + '.png'
        
    plot_return_levels(dataframe,var,value,periods,output_file,it_selected_max)
       
    return 


def plot_joint_2D_contour(
        data,
        var1='hs', 
        var2='tp',
        model='hs_tp',
        return_periods=[10,100,1000], 
        state_duration = 1,
        output_file='contours.png'):
    """
    Plot joint contours for the given return periods. 
    
    Parameters
    -----------
    data: pd.DataFrame
        Dataframe containing data columns.
    var1 : str
        A column of the dataframe corresponding to the primary variable of the model.
    var2 : str
        A column of the dataframe corresponding to the second variable of the model.
    model : str, default hs_tp
        The model. One of 

             - Hs_Tp: DNV-RP-C205 model of significant wave height and peak wave period
             - Hs_U: DNV-RP-C205 model of wind speed and significant wave height
    return_periods : list[float], default [50,100]
        A list of return periods for which to create contours.
    state_duration : int
        Duration of each entry in the data, in hours. E.g., 1 or 3 hours per entry.
    output_file : str
        Filename for saved figure.
    
    Notes
    ------
    Written by efvik.

    This is a simplified functional interface to the CMA module.
    There are many more configuration options available by using 
    the JointProbabilityModel class from that module directly.

    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2)
    ax = model.plot_contours(periods=return_periods,state_duration=state_duration)
    model.plot_data_density(ax)
    model.plot_dependent_percentiles(ax)
    model.plot_DNVGL_steepness_criterion(ax,label="Steepness\nlimit")
    model.plot_legend(ax)
    if output_file!="":ax.get_figure().savefig(output_file,bbox_inches="tight")
    return ax

def plot_joint_3D_contour(
        data,var1,var2,var3,
        model="u_hs_tp",
        return_period=100,
        state_duration=1,
        output_file="3D_contour.png"
        ):
    """
    Plot 3-dimensional IFORM contour.

    Parameters
    -----------
    data: pd.DataFrame
        Dataframe containing data columns.
    var1 : str
        A column of the dataframe corresponding to the primary variable of the model.
    var2 : str
        A column of the dataframe corresponding to the second variable of the model.
    var3 : str
        A column of the dataframe corresponding to the third variable of the model.
    model : str, default hs_tp
        The model. One of 

            - U_Hs_Tp: Model of wind speed, significant wave height, and peak wave period, based on Li et. al. (2015).
            - cT_Hs_Tp: Model of current, significant wave height, and peak wave period.

    return_period : float, default 100
        The return period represented by the contour.
    state_duration : int
        Duration of each entry in the data, in hours. E.g., 1 or 3 hours per entry.
    
    output_file : str
        Filename for saved figure.

    Notes
    ------
    Written by efvik.

    This is a simplified functional interface to the CMA module.
    There are many more configuration options available by using 
    the JointProbabilityModel class from that module directly.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2,var3)
    ax = model.plot_3D_contour(
        return_period = return_period,
        state_duration = state_duration)
    model.plot_legend(ax)
    ax.set_box_aspect(None,zoom=0.85)
    if output_file!="":ax.get_figure().savefig(output_file,bbox_inches="tight")
    return ax

def plot_joint_3D_contour_slices(
        data,var1,var2,var3,
        model="u_hs_tp",
        slice_values=[5,10,15,20,25],
        return_period=100,
        state_duration=1,
        output_file="3D_contour_slices.png"
        ):
    """
    Plot 3-dimensional IFORM contour.

    Parameters
    -----------
    data: pd.DataFrame
        Dataframe containing data columns.
    var1 : str
        A column of the dataframe corresponding to the primary variable of the model.
    var2 : str
        A column of the dataframe corresponding to the second variable of the model.
    var3 : str
        A column of the dataframe corresponding to the third variable of the model.
    model : str, default hs_tp
        The model. One of 

            - U_Hs_Tp: Model of wind speed, significant wave height, and peak wave period, based on Li et. al. (2015).
            - cT_Hs_Tp: Model of current, significant wave height, and peak wave period.
    slice_values : list[float]
        The values of the primary variable (e.g., wind speed) to "cut" the 3D contour at.
    return_period : float, default 100
        The contour return period.
    state_duration : int
        Duration of each entry in the data, in hours. E.g., 1 or 3 hours per entry.
    
    output_file : str
        Filename for saved figure.

    Notes
    ------
    Written by efvik.

    This is a simplified functional interface to the CMA module.
    There are many more configuration options available by using 
    the JointProbabilityModel class from that module directly.
    """
    model = JointProbabilityModel(model)
    model.fit(data,var1,var2,var3)
    ax = model.plot_3D_contour_slices(
        subplots=False,
        return_period=return_period,
        state_duration=state_duration,
        slice_values=slice_values
        )
    model.plot_legend(ax)
    if output_file!="":ax.get_figure().savefig(output_file)
    return ax

def plot_multi_joint_distribution_Hs_Tp_var3(data,var_hs='hs',var_tp='tp',var3='W10',var3_units='m/s',periods=[100],var3_bin=5,threshold_min=100,output_file='Hs.Tp.joint.distribution.multi.binned.var3.png'):  
    """
    Plot joint distribution of Hs-Tp for a given return period for eached binned var3 data
    Input:
        var3: e.g., 'W10' 
        var3_units: e.g., 'm/s'
        var3_bin: sets the bin size, e.g., 5 
        threshold_min: provide is the minimum number of datapoints in the dataset, e.g., 100
    """
    var3_name = var3 
    df = data.dropna()
    max_var3 = max(df[var3])
    window = np.arange(0, max_var3 + var3_bin, var3_bin)
    t3_ = []; h3_= []

    fig, ax = plt.subplots(figsize=(8,6))
    #custom_cycler = cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    #ax.set_prop_cycle(custom_cycler)
    cmap = plt.cm.Blues
    ii = np.linspace(0.2,1,len(window)-1)

    for i in range(len(window)-1):
        var_3 = df[var3].where((df[var3] > window[i]) & (df[var3] <= window[i+1])).dropna() #w10 = (wind1,wind2]
        if len(var_3)< threshold_min:
            continue
        hs = df[var_hs].where(var_3.notnull()).dropna()
        tp = df[var_tp].where(var_3.notnull()).dropna()
        dff = pd.DataFrame({var_hs: hs, var_tp: tp, var3_name: var_3})
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = stats.joint_distribution_Hs_Tp(data=dff,var_hs=var_hs,var_tp=var_tp,periods=periods)
        t3_.append(t3)
        h3_.append(h3)
        linestyle = '-' if i % 2 == 0 else '--'
        labels = str(np.round(window[i],2))+'-'+str(np.round(window[i+1],2))
        #plt.plot(t3[0],h3[0], linestyle=linestyle, label=var3_name+'$\\in$' + labels+' ['+var3_units+']')  
        plt.plot(t3[0],h3[0], color=cmap(ii[i]), linestyle=linestyle, label=var3_name+'$\\in$' + labels+' ['+var3_units+']')  #color=cmap(0.5)

    #include the whole dataset also:
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = stats.joint_distribution_Hs_Tp(data=df,var_hs=var_hs,var_tp=var_tp,periods=periods)
    labels = str(np.round(window[0],2))+'-'+str(np.round(window[-1],2))
    plt.plot(t3[0],h3[0],label=var3_name+'$_{all}$'+'$\\in$ ' + labels+ ' ['+var3_units+']') 
    plt.xlabel('Tp - Peak Period [s]')
    plt.suptitle('Return period = '+ str(periods[0])+'-year')
    plt.ylabel('Hs - Significant Wave Height [m]')
    plt.grid()
    plt.legend()
    #plt.xlim([0,np.ceil(max(max(t3_[0])))]) #When the dataset is split 
    #plt.ylim([0,np.ceil(max(max(h3_[-1])))]) #When the dataset is split
    plt.xlim([0,np.ceil(max(max(t3)))]) #When the whole dataset is included 
    plt.ylim([0,np.ceil(max(max(h3)))]) #When the whole dataset is included    
    #plt.show()
    plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')

    return fig

##################

def plot_joint_distribution_Hs_Tp(
        data,var_hs='hs',var_tp='tp',
        periods=[1,10,100,10000], 
        title='Hs-Tp joint distribution',
        output_file='Hs.Tp.joint.distribution.png',
        density_plot=False,
        model = "lonowe",
        state_duration = 1,
        ):

    if model != "lonowe":
        model = JointProbabilityModel(model)
        model.fit(data,var_hs,var_tp)
        ax = model.plot_contours(
            periods=periods,
            state_duration=state_duration
            )
        if density_plot:
            model.plot_data_density(ax,bins=50)
        else:
            model.plot_data_scatter(ax)
        model.plot_dependent_percentiles(ax)
        model.plot_DNVGL_steepness_criterion(ax)
        model.plot_legend(ax)
        if output_file != "": ax.get_figure().savefig(output_file)
        return ax

    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = stats.joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
    df = data
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
    t_steepness, h_steepness = aux_funcs.DNV_steepness(df,h,t,periods,interval)
    percentile05 = aux_funcs.find_percentile(df.hs.values,pdf_Hs_Tp,h,t,5,periods,interval)
    percentile50 = aux_funcs.find_percentile(df.hs.values,pdf_Hs_Tp,h,t,50,periods,interval)
    percentile95 = aux_funcs.find_percentile(df.hs.values,pdf_Hs_Tp,h,t,95,periods,interval)
    
    fig, ax = plt.subplots(figsize=(8,6))
    df = df[df['hs'] >= 0.1]
    if density_plot is False: 
        plt.scatter(df.tp.values,df.hs.values,c='red',label='data',s=3)
    else:
        #plt.scatter(df.tp.values,df.hs.values,c='red',label='data',s=3)
        plt.hist2d(df['tp'].values, df['hs'].values,bins=50, cmap='hot',cmin=1)
        plt.colorbar()


    for i in range(len(periods)):
        plt.plot(t3[i],h3[i],label=str(X[i])+'-year')  


    plt.plot(t_steepness,h_steepness,'k--',label='steepness')
    
    plt.plot(percentile50[0],percentile50[1],'g',label='Tp-mean',linewidth=5)
    plt.plot(percentile05[0],percentile05[1],'g:',label='Tp-5%',linewidth=2)
    plt.plot(percentile95[0],percentile95[1],'g--',label='Tp-95%',linewidth=2)

    plt.xlabel('Tp - Peak Period [s]')
    plt.suptitle(title)
    plt.ylabel('Hs - Significant Wave Height [m]')
    plt.grid()
    plt.legend()
    plt.xlim([0,np.round(hs_tpl_tph['t2_'+str(np.max(periods))].max()+1)])
    plt.ylim([0,np.round(hs_tpl_tph['hs_'+str(np.max(periods))].max()+1)])
    plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')
    
    return fig

def plot_bounds(file='NORA10_6036N_0336E.1958-01-01.2022-12-31.txt'):
    
    def readNora10File(file):
        df = pd.read_csv(file, delim_whitespace=True, header=3) # sep=' ', header=None,0,1,2,3
        df.index= pd.to_datetime(df.YEAR*1000000+df.M*10000+df.D*100+df.H,format='%Y%m%d%H')
        df['tp_corr_nora10'] = stats.aux_funcs.Tp_correction(df.TP.values)
        return df
    
    df = readNora10File(file)
    
    # Fit Weibull distribution to your data and estimate parameters
    data = df.HS.values  # Your data
    shape, loc, scale = aux_funcs.Weibull_method_of_moment(data)
    
    # Define return periods
    periods = np.arange(1.5873, 10000, 100) 
    return_periods = periods
    return_periods = return_periods*365.2422*24/3
    # Calculate return values
    return_values = st.weibull_min.ppf(1 - 1 / return_periods, shape, loc, scale)
    
    # Bootstrap to estimate confidence bounds
    num_bootstrap_samples = 1000
    bootstrap_return_values = []
    for _ in range(num_bootstrap_samples):
        # Resample data with replacement
        bootstrap_sample = np.random.choice(data, size=1000, replace=True)
        
        # Fit Weibull distribution to resampled data
        shape_b, loc_b, scale_b = aux_funcs.Weibull_method_of_moment(bootstrap_sample)
        # Calculate return values for resampled distribution
        bootstrap_return_values.append(st.weibull_min.ppf(1 - 1 / return_periods, shape_b, loc_b, scale_b))
    
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
    plt.plot(x, y, label='Weibull3P_MOM')
    plt.xscale('log')
    plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.3, label='Bounds')
    plt.xlabel('Years')
    plt.ylabel('Waves')
    plt.title('Plot with Bounds')
    plt.legend()
    
    return 

def plot_monthly_return_periods(data, var='hs', periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM',method='default',threshold='default', units='m',output_file='monthly_extremes_weibull.png'):
    df = tables.table_monthly_return_periods(data=data,var=var, periods=periods,distribution=distribution,method=method,threshold=threshold, units=units, output_file=None)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0,1,len(periods)))
    for i in range(len(periods)):
        plt.plot(df['Month'][1:-1], df.iloc[1:-1,-i-1],marker = 'o',label=df.keys()[-i-1].split(':')[1],color=colors[i])

    plt.title('Return values for '+str(var)+' ['+units+']',fontsize=16)
    plt.xlabel('Month',fontsize=15)
    plt.legend()
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig


def plot_directional_return_periods(data, var='hs',var_dir='Pdir', periods=[1, 10, 100, 10000],distribution='Weibull', units='m',adjustment='NORSOK',method='default',threshold='default', output_file='monthly_extremes_weibull.png'):
    df = tables.table_directional_return_periods(data=data,var=var,var_dir=var_dir, periods=periods, distribution=distribution, units=units,adjustment=adjustment,method=method,threshold=threshold, output_file=None)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0,1,len(periods)))
    a=0
    for i in periods:
        plt.plot(df['Direction sector'][1:-1], df[f'Return period: {i} [years]'][1:-1],marker = 'o',label=f'{i} years',color=colors[a])
        a=a+1
    
    plt.title('Return values for '+str(var)+' ['+units+']',fontsize=16)
    plt.xlabel('Direction',fontsize=15)
    plt.legend()
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig


def plot_polar_directional_return_periods(data, var='hs', var_dir='Pdir', periods=[1, 10, 100, 10000], distribution='Weibull', units='m', adjustment='NORSOK',method='default',threshold='default', output_file='monthly_extremes_weibull.png'):
    df = tables.table_directional_return_periods(data=data, var=var, var_dir=var_dir, periods=periods, distribution=distribution, units=units, adjustment=adjustment, method=threshold,threshold=threshold,output_file=None)

    # Remove degree symbols and convert to numeric
    directions_str = df['Direction sector'][1:-1].str.rstrip('°')
    directions = pd.to_numeric(directions_str, errors='coerce')
    directions_rad = np.deg2rad(directions.dropna())  # Convert valid directions to radians
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})    
    
    for i in range(len(periods)):
        values = df.iloc[1:-1, i+5].astype(float)
        marker_sizes = values ** 1.8  # Scale factor, adjust as needed
        ax.scatter(directions_rad, values, marker='o', s=marker_sizes, label=df.keys()[i+5].split(':')[1])
    
    ax.set_theta_direction(-1)  # Set the direction of the theta axis
    ax.set_theta_offset(np.pi / 2.0)  # Set the location of 0 degrees to the top

    # Add the cardinal directions
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])

    plt.title('Return values for ' + str(var) + ' [' + units + ']', fontsize=16)
    plt.legend(loc='upper left',  bbox_to_anchor=(1.1, 0.6))
    plt.grid(True)
    plt.tight_layout()
    
    if output_file != "": plt.savefig(output_file)
    return fig




def plot_prob_non_exceedance_fitted_3p_weibull(data, var='hs', output_file='plot_prob_non_exceedance_fitted_3p_weibull.png'):
    hs, y, cdf, prob_non_exceedance_obs, shape, location, scale = stats.prob_non_exceedance_fitted_3p_weibull(data=data, var=var)
    fig, ax = plt.subplots()

    a = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
    b1 = np.array([0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, 7, 7.5, 8, 8.5])
    a1 = prob_non_exceedance_obs / 100
    yinterp = np.interp(a1, a, b1)
    # b = np.arange(0, len(yinterp), 1)
    a2 = cdf
    yinterp2 = np.interp(a2, a, b1)
    # b2 = np.arange(0, len(yinterp2), 1)
    ax.plot(np.log(np.sort(y)), np.log(np.sort(yinterp2)), label='fitted')
    ax.plot(np.log(hs), np.log(yinterp), 'r*', label='data')
    
    ax.set_yticks(np.log(b1))
    ax.set_yticklabels([str(i) for i in a.tolist()])
    ax.set_ylim(-0.5, 2.2)
    
    # Set grid lines for minor ticks
    itv=int(data[var].max())-int(data[var].min())
    if itv<=10:
        x_grid_ticks = np.arange(0.5, int(data[var].max()) + 0.5, 0.5)
    else:
        x_grid_ticks = np.arange(1, int(data[var].max()) + 1, 1)
    ax.set_xticks(np.log(x_grid_ticks), minor=True)
    
    # Set major ticks and labels
    if itv<=10:
        x_major_ticks = np.arange(1, int(data[var].max()) + 1, 1)
    else:
        x_major_ticks = np.arange(2, int(data[var].max()) + 2, 2)
    x_major_ticks = np.arange(5, int(data[var].max()) + 5, 5)
    ax.set_xticks(np.log(x_major_ticks))
    ax.set_xticklabels([f'{x}' for x in x_major_ticks])
    
    # Center the value 5 in the middle of the x-axis
    x_min, x_max = np.log(data[var].min()), np.log(data[var].max())
    mid_point = (x_min + x_max) / 2
    offset = np.log(5) - mid_point
    ax.set_xlim(x_min + offset, x_max)

    ax.set_xlabel(var, fontsize=12)
    ax.set_ylabel('Probability of non-exceedance', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Add minor grid lines
    # Add solid black grid lines for major ticks and dashed lines for minor ticks
    for tick in x_major_ticks:
        ax.axvline(np.log(tick), color='black', linestyle='-')    
    
    # Move y-axis ticks to the right, but keep the label on the left
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', labelleft=False, labelright=True)
    plt.tight_layout()
    
    if output_file != "": plt.savefig(output_file)
    plt.close()

    return fig


def plot_tp_for_given_hs(data: pd.DataFrame, var_hs: str, var_tp: str, output_file='tp_for_given_hs.png'):

    df = tables.table_tp_for_given_hs(data=data, var_hs=var_hs, var_tp=var_tp, output_file=False)
    # Plot the 2D histogram
    fig, ax = plt.subplots()
    plt.hist2d(data[var_hs], data[var_tp], bins=50, cmap='hot', cmin=1)
    plt.scatter(df['Hs[m]'], df['Tp(P5-obs) [s]'], marker='x', color='grey', label='P5')
    plt.scatter(df['Hs[m]'], df['Tp(Mean-obs) [s]'], marker='x', color='blue', label='Mean')
    plt.scatter(df['Hs[m]'], df['Tp(P95-obs) [s]'], marker='x', color='magenta', label='P95')
    plt.plot(df['Hs[m]'], df['Tp(P5-model) [s]'], color='grey', label='P5 fitted')
    plt.plot(df['Hs[m]'], df['Tp(Mean-model) [s]'], color='blue', label='Mean fitted')
    plt.plot(df['Hs[m]'], df['Tp(P95-model) [s]'], color='magenta', label='P95 fitted')
    plt.xlabel('$H_s$ [m]',fontsize=16)
    plt.ylabel('$T_p$ [s]',fontsize=16)
    plt.xlim(0,df['Hs[m]'].max())
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig

def plot_hs_for_given_wind(data: pd.DataFrame, var_hs: str, var_wind: str,output_file='hs_for_given_wind.png'):
    df = tables.table_hs_for_given_wind(data=data, var_hs=var_hs, var_wind=var_wind, bin_width=2,max_wind=np.ceil(data[var_wind].max()), output_file=False)
    # Plot the 2D histogram
    fig, ax = plt.subplots()
    plt.hist2d(data[var_wind],data[var_hs], bins=50, cmap='hot', cmin=1)
    plt.scatter(df['U[m/s]'], df['Hs(P5-obs) [m]'], marker='x', color='grey', label='P5')
    plt.scatter(df['U[m/s]'], df['Hs(Mean-obs) [m]'], marker='x', color='blue', label='Mean')
    plt.scatter(df['U[m/s]'], df['Hs(P95-obs) [m]'], marker='x', color='magenta', label='P95')
    plt.plot(df['U[m/s]'],df['Hs(P5-model) [m]'],  color='grey', label='P5 fitted')
    plt.plot(df['U[m/s]'],df['Hs(Mean-model) [m]'],  color='blue', label='Mean fitted')
    plt.plot(df['U[m/s]'],df['Hs(P95-model) [m]'],  color='magenta', label='P95 fitted')

    plt.ylabel('$H_s$ [m]',fontsize=16)
    plt.xlabel('$U$ [m/s]',fontsize=16)
    plt.xlim(0,df['U[m/s]'].max())
    plt.ylim(0,df['Hs(P95-model) [m]'].max())
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig

def plot_profile_return_values(data,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150], periods=[1, 10, 100, 10000],reverse_yaxis=False,title='Return Periods over z',units = 'm/s',distribution='Weibull3P',method='default',threshold='default', output_file='RVE_wind_profile.png'):
    df = tables.table_profile_return_values(data,var=var, z=z, periods=periods,units = units ,distribution=distribution,method=method,threshold=threshold, output_file=None)
    fig, ax = plt.subplots()
    df.columns = [col.replace('Return period ', '') for col in df.columns] # for legends
    plt.yticks(z)  # Set yticks to be the values in z
    ax.yaxis.set_major_locator(mticker.MultipleLocator(int(max(z)/4)))  # Set major y-ticks at intervals of 10

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0,1,len(df.columns[1:])))
    a=0
    for column in df.columns[1:]:
        plt.plot(df[column][1:],z,marker='.', label=column, color=colors[a])
        a=a+1
    plt.ylabel('z [m]')
    plt.xlabel('[m/s]')
    plt.title(title)
    if reverse_yaxis is True:
        plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig


def plot_current_for_given_wind(data: pd.DataFrame, var_curr: str, var_wind: str,output_file='curr_for_given_wind.png'):
    df = tables.table_current_for_given_wind(data=data, var_curr=var_curr, var_wind=var_wind, bin_width=2,max_wind=np.ceil(data[var_wind].max()), output_file=False)
    # Plot the 2D histogram
    fig, ax = plt.subplots()
    plt.hist2d(data[var_wind],data[var_curr], bins=50, cmap='hot', cmin=1)
    plt.scatter(df['U[m/s]'], df['Uc(P5-obs) [m/s]'], marker='x', color='grey', label='P5')
    plt.scatter(df['U[m/s]'], df['Uc(Mean-obs) [m/s]'], marker='x', color='blue', label='Mean')
    plt.scatter(df['U[m/s]'], df['Uc(P95-obs) [m/s]'], marker='x', color='magenta', label='P95')
    plt.plot(df['U[m/s]'],df['Uc(P5-model) [m/s]'],  color='grey', label='P5 fitted')
    plt.plot(df['U[m/s]'],df['Uc(Mean-model) [m/s]'],  color='blue', label='Mean fitted')
    plt.plot(df['U[m/s]'],df['Uc(P95-model) [m/s]'],  color='magenta', label='P95 fitted')

    plt.ylabel('Current speed, $U_c$ [m/s]',fontsize=16)
    plt.xlabel('Wind speed, $U$ [m/s]',fontsize=16)
    plt.xlim(0,df['U[m/s]'].max())
    plt.ylim(0,1.5*df['Uc(Mean-model) [m/s]'].max())
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig


def plot_current_for_given_hs(data: pd.DataFrame, var_curr: str, var_hs: str,max_hs=20,output_file='curr_for_given_hs.png'):
    df = tables.table_current_for_given_hs(data=data, var_curr=var_curr, var_hs=var_hs, bin_width=2,max_hs=max_hs, output_file=False)
    # Plot the 2D histogram
    fig, ax = plt.subplots()
    plt.hist2d(data[var_hs],data[var_curr], bins=50, cmap='hot', cmin=1)
    plt.scatter(df['Hs[m]'], df['Uc(P5-obs) [m/s]'], marker='x', color='grey', label='P5')
    plt.scatter(df['Hs[m]'], df['Uc(Mean-obs) [m/s]'], marker='x', color='blue', label='Mean')
    plt.scatter(df['Hs[m]'], df['Uc(P95-obs) [m/s]'], marker='x', color='magenta', label='P95')
    plt.plot(df['Hs[m]'],df['Uc(P5-model) [m/s]'],  color='grey', label='P5 fitted')
    plt.plot(df['Hs[m]'],df['Uc(Mean-model) [m/s]'],  color='blue', label='Mean fitted')
    plt.plot(df['Hs[m]'],df['Uc(P95-model) [m/s]'],  color='magenta', label='P95 fitted')

    plt.ylabel('Current speed, $U_c$ [m/s]',fontsize=16)
    plt.xlabel('$H_s$ [m]',fontsize=16)
    plt.xlim(0,df['Hs[m]'].max())
    plt.ylim(0,df['Uc(P95-model) [m/s]'].max())
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig


def plot_storm_surge_for_given_hs(data: pd.DataFrame, var_surge: str, var_hs: str,max_hs=20,output_file='surge_for_given_hs.png'):
    df, df_coeff = tables.table_storm_surge_for_given_hs(data=data, var_surge=var_surge, var_hs=var_hs, bin_width=2,max_hs=max_hs, output_file=False)
    # Plot the 2D histogram
    fig, ax = plt.subplots()
    plt.hist2d(data[var_hs],data[var_surge], bins=50, cmap='hot', cmin=1)
    plt.scatter(df['Hs[m]'], df['S(P5-obs) [m]'], marker='x', color='grey', label='P5')
    plt.scatter(df['Hs[m]'], df['S(Mean-obs) [m]'], marker='x', color='blue', label='Mean')
    plt.scatter(df['Hs[m]'], df['S(P95-obs) [m]'], marker='x', color='magenta', label='P95')
    plt.plot(df['Hs[m]'],df['S(P5-model) [m]'],  color='grey', label='P5 fitted')
    plt.plot(df['Hs[m]'],df['S(Mean-model) [m]'],  color='blue', label='Mean fitted')
    plt.plot(df['Hs[m]'],df['S(P95-model) [m]'],  color='magenta', label='P95 fitted')

    plt.ylabel('Storm Surge, S [m]',fontsize=16)
    plt.xlabel('$H_s$ [m]',fontsize=16)
    plt.xlim(0,df['Hs[m]'].max())
    plt.ylim(df['S(P5-model) [m]'].min(),df['S(P95-model) [m]'].max())
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file)

    return fig


def plot_cca_profiles(data,var='current_speed_',month=None,percentile=None,return_period=None,distribution='GUM',method='default',threshold='default',unit_var='m/s',unit_lev='m',output_file='plot_cca_profiles.png'):
    """
    This function plots the CCA profiles for a specific percentile or return period

    Parameters
    ----------
    data: pd.DataFrame
        Contains the time series
    var: string
        Prefix of the variable name of interest e.g. 'current_speed_' for names such as 'current_speed_{depth}m'
    month: string
        Month of interest (e.g. 'January' or 'Jan'), default (None) takes all months
    percentile: float
        Percentile associated with the worst case scenario
    return_period: float
        Return-period e.g., 10 for a 10-yr return period
    distribution, method, and threshold: string, string, string
        To specify only if return_period is given 
    unit_var: string
        Unit of the variable var, default is 'm/s'
    unit_lev: string
        Units of the vertical levels, default is 'm'
    output_file: string
        Name of the figure file

    Returns
    -------
    fig: matplotlib figure in a new file

    Authors
    -------
    Function written by clio-met
    """
    if ((percentile is None) and (return_period is None)):
        raise ValueError('Please specify either a percentile or a return period in years')
    if not(percentile is None):
        lev,woca,cca=stats.cca_profiles(data,var=var,month=month,percentile=percentile)
    if not(return_period is None):
        lev,woca,cca=stats.cca_profiles(data,var=var,month=month,return_period=return_period,distribution=distribution,method=method,threshold=threshold)
    n_lines = len(lev)
    cmap = plt.get_cmap('jet_r')
    colors = cmap(np.linspace(0, 1, n_lines))
    fig, ax = plt.subplots(figsize=(9,10))
    for d in range(len(lev)):
        ax.plot(lev,cca[:,d],color=colors[d],label=str(int(lev[d]))+' m',linewidth=2)
    ax.plot(lev,woca,label='Worst case',color='k',linewidth=3.5)
    ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    ax.set_ylabel(var+' ['+unit_var+']',fontsize=16)
    ax.set_xlabel('Level ['+unit_lev+']',fontsize=16)
    if month is None:
        month_str=''
    else:
        month_str=' - Month: '+month
    if percentile is not None:
        pp=f"{percentile:.4f}"
        ax.set_title('CCA profile - P'+pp+month_str,fontsize=16)
    if return_period is not None:
        rp=f"{return_period:.0f}"
        ax.set_title('CCA profile - RP '+rp+' years'+month_str,fontsize=16)
    ax.tick_params(axis='both', labelsize= 16)
    ax.set_xlim(lev[0],lev[-1])
    plt.grid(color='lightgray',linestyle=':')
    plt.tight_layout()
    if output_file != "": plt.savefig(output_file,dpi=250,facecolor='white',bbox_inches='tight')
    return fig

