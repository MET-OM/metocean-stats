from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..stats.extreme import *
from ..tables.extreme import *


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

        #ax.scatter(df_emp_rl_cut['return_levels'],
        #           df_emp_rl_cut['return_periods'],
        #           marker="o", s=20, lw=1,
        #           facecolor="k", edgecolor="w", zorder=20)
        ax.scatter(df_emp_rl_cut['return_levels'],
                   df_emp_rl_cut['prob_non_exceedance'],
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
            #ax.plot(df_model_rl_tmp_cut['return_levels'],
            #        df_model_rl_tmp_cut.index,
            #        label=dist)
            ax.plot(df_model_rl_tmp_cut['return_levels'],
                    df_model_rl_tmp_cut['prob_non_exceedance'],
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
        #list_yticks = [1/0.9, 2, 10]\
        #            + [10**i for i in range(2, 5) if max(periods) > 10**i]
        #if max(periods) > max(list_yticks):
        #    list_yticks = list_yticks + [max(periods)]
        #ax.set_yticks(list_yticks)
        #ax.set_yticklabels([round(1-1/rp,5) for rp in list_yticks])
        list_yticks = [10**-1,10**(np.log10(0.5)),10**0]
        list_ytickslabels = ['0.1','0.5','1']
        ax.set_yticks(list_yticks)
        ax.set_yticklabels(list_ytickslabels)
        ax.set_ylim(0.05,1.02)

    elif yaxis == 'rp':
        ax.set_ylabel("Return period [yr]")

    ax.set_xlabel("Return levels "+var)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save the plot if a path is given
    if output_file is not None:
        plt.savefig(output_file)

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
    plt.savefig(output_file)

    return fig

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

def plot_RVE_ALL(dataframe,var='hs',periods=np.array([1,10,100,1000]),distribution='Weibull3P',method='default',threshold='default'):
    
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

def plot_joint_distribution_Hs_Tp(data,var_hs='hs',var_tp='tp',periods=[1,10,100,10000], title='Hs-Tp joint distribution',output_file='Hs.Tp.joint.distribution.png',density_plot=False):
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
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
    t_steepness, h_steepness = DVN_steepness(df,h,t,periods,interval)
    percentile05 = find_percentile(df.hs.values,pdf_Hs_Tp,h,t,5,periods,interval)
    percentile50 = find_percentile(df.hs.values,pdf_Hs_Tp,h,t,50,periods,interval)
    percentile95 = find_percentile(df.hs.values,pdf_Hs_Tp,h,t,95,periods,interval)
    
    fig, ax = plt.subplots(figsize=(8,6))
    df = df[df['hs'] >= 0.1]
    if density_plot is False: 
        plt.scatter(df.tp.values,df.hs.values,c='red',label='data',s=3)
    else:
        from matplotlib.colors import LogNorm
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
    plt.plot(x, y, label='Weibull3P_MOM')
    plt.xscale('log')
    plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.3, label='Bounds')
    plt.xlabel('Years')
    plt.ylabel('Waves')
    plt.title('Plot with Bounds')
    plt.legend()
    
    return 

def plot_monthly_return_periods(data, var='hs', periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM',method='default',threshold='default', units='m',output_file='monthly_extremes_weibull.png'):
    df = table_monthly_return_periods(data=data,var=var, periods=periods,distribution=distribution,method=method,threshold=threshold, units=units, output_file=None)
    fig, ax = plt.subplots()
    for i in range(len(periods)):
        plt.plot(df['Month'][1:-1], df.iloc[1:-1,-i-1],marker = 'o',label=df.keys()[-i-1].split(':')[1])

    plt.title('Return values for '+str(var)+' ['+units+']',fontsize=16)
    plt.xlabel('Month',fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    return fig


def plot_directional_return_periods(data, var='hs',var_dir='Pdir', periods=[1, 10, 100, 10000],distribution='Weibull', units='m',adjustment='NORSOK',method='default',threshold='default', output_file='monthly_extremes_weibull.png'):
    df = table_directional_return_periods(data=data,var=var,var_dir=var_dir, periods=periods, distribution=distribution, units=units,adjustment=adjustment,method=method,threshold=threshold, output_file=None)
    fig, ax = plt.subplots()
    for i in periods:
        plt.plot(df['Direction sector'][1:-1], df[f'Return period: {i} [years]'][1:-1],marker = 'o',label=f'{i} years')
        
    plt.title('Return values for '+str(var)+' ['+units+']',fontsize=16)
    plt.xlabel('Direction',fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    return fig


def plot_polar_directional_return_periods(data, var='hs', var_dir='Pdir', periods=[1, 10, 100, 10000], distribution='Weibull', units='m', adjustment='NORSOK',method='default',threshold='default', output_file='monthly_extremes_weibull.png'):
    df = table_directional_return_periods(data=data, var=var, var_dir=var_dir, periods=periods, distribution=distribution, units=units, adjustment=adjustment, method=threshold,threshold=threshold,output_file=None)

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
    
    plt.savefig(output_file)
    return fig




def plot_prob_non_exceedance_fitted_3p_weibull(data, var='hs', output_file='plot_prob_non_exceedance_fitted_3p_weibull.png'):
    hs, y, cdf, prob_non_exceedance_obs, shape, location, scale = prob_non_exceedance_fitted_3p_weibull(data=data, var=var)
    fig, ax = plt.subplots()

    a = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
    b1 = np.array([0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, 7, 7.5, 8, 8.5])
    a1 = prob_non_exceedance_obs / 100
    yinterp = np.interp(a1, a, b1)
    b = np.arange(0, len(yinterp), 1)
    a2 = cdf
    yinterp2 = np.interp(a2, a, b1)
    b2 = np.arange(0, len(yinterp2), 1)
    ax.plot(np.log(np.sort(y)), np.log(np.sort(yinterp2)), label='fitted')
    ax.plot(np.log(hs), np.log(yinterp), 'r*', label='data')
    
    ax.set_yticks(np.log(b1))
    ax.set_yticklabels([str(i) for i in a.tolist()])
    ax.set_ylim(-0.5, 2.2)
    
    # Set grid lines for each integer
    x_grid_ticks = np.arange(1, int(data[var].max()) + 2)
    ax.set_xticks(np.log(x_grid_ticks), minor=True)
    
    # Set major ticks and labels at intervals of 5
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
    
    plt.savefig(output_file)
    plt.close()

    return fig


def plot_tp_for_given_hs(data: pd.DataFrame, var_hs: str, var_tp: str,output_file='tp_for_given_hs.png'):
    df = table_tp_for_given_hs(data=data, var_hs=var_hs, var_tp=var_tp, bin_width=1,output_file=False)
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
    plt.savefig(output_file)

    return fig

def plot_hs_for_given_wind(data: pd.DataFrame, var_hs: str, var_wind: str,output_file='hs_for_given_wind.png'):
    df = table_hs_for_given_wind(data=data, var_hs=var_hs, var_wind=var_wind, bin_width=2,max_wind=40, output_file=False)
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
    plt.savefig(output_file)

    return fig

def plot_profile_return_values(data,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150], periods=[1, 10, 100, 10000],reverse_yaxis=False,title='Return Periods over z',units = 'm/s',distribution='Weibull3P',method='default',threshold='default', output_file='RVE_wind_profile.png'):
    import matplotlib.ticker as ticker
    df = table_profile_return_values(data,var=var, z=z, periods=periods,units = units ,distribution=distribution,method=method,threshold=threshold, output_file=None)
    fig, ax = plt.subplots()
    df.columns = [col.replace('Return period ', '') for col in df.columns] # for legends
    plt.yticks(z)  # Set yticks to be the values in z
    ax.yaxis.set_major_locator(ticker.MultipleLocator(int(max(z)/4)))  # Set major y-ticks at intervals of 10

    for column in df.columns[1:]:
        plt.plot(df[column][1:],z,marker='.', label=column)
    plt.ylabel('z[m]')
    plt.xlabel('[m/s]')
    plt.title(title)
    if reverse_yaxis == True:
        plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file)

    return fig


def plot_current_for_given_wind(data: pd.DataFrame, var_curr: str, var_wind: str,max_wind=40,output_file='curr_for_given_wind.png'):
    df = table_current_for_given_wind(data=data, var_curr=var_curr, var_wind=var_wind, bin_width=2,max_wind=max_wind, output_file=False)
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
    plt.savefig(output_file)

    return fig


def plot_current_for_given_hs(data: pd.DataFrame, var_curr: str, var_hs: str,max_hs=20,output_file='curr_for_given_hs.png'):
    df = table_current_for_given_hs(data=data, var_curr=var_curr, var_hs=var_hs, bin_width=2,max_hs=max_hs, output_file=False)
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
    plt.savefig(output_file)

    return fig


def plot_storm_surge_for_given_hs(data: pd.DataFrame, var_surge: str, var_hs: str,max_hs=20,output_file='surge_for_given_hs.png'):
    df, df_coeff = table_storm_surge_for_given_hs(data=data, var_surge=var_surge, var_hs=var_hs, bin_width=2,max_hs=max_hs, output_file=False)
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
    plt.savefig(output_file)

    return fig
