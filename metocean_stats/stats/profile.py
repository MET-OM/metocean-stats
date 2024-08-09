import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def mean_profile(data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], perc = [25,75], output_file='mean_wind_profile.png'):  
    """
    The function is written by clio-met and KonstantinChri.
    It creates a figure with a mean profile in black and lower/higher percentile in gray.
    data = dataframe  
    vars = list of variables in dataframe e.g., ['wind_speed_10m','wind_speed_20m','wind_speed_50m']
    height_levels = [10,20,50] 
    perc = lower, higher percentile e.g., [25,75] 
    output_file = 'wind_profile.png' or False for no 
    return mean_values as numpy array.
    """
    df = data[vars]
    nparr=df.to_numpy()
    mean_values = np.mean(nparr,axis=0) 
    #breakpoint()

    if output_file==False:
        pass
    else:
        fig, ax = plt.subplots(figsize=(4,8))
        plt.fill_betweenx(np.array(height_levels),np.percentile(nparr,perc[0],axis=0),np.percentile(nparr,perc[1],axis=0),color='lightgray',label=f'{perc[0]}-{perc[1]}%')
        plt.plot(mean_values,np.array(height_levels),'k-',linewidth=2.5,label='Mean wind speed')
        plt.ylim(np.min(height_levels),np.max(height_levels))
        plt.xlabel('wind speed [m/s]')
        plt.ylabel('height [m]')
        plt.grid()
        plt.title(output_file.split('.')[0],fontsize=18)
        plt.tight_layout()
        plt.legend()
        plt.savefig(output_file)
        plt.close()
    return mean_values

def profile_shear(data, vars = ['wind_speed_10m','wind_speed_20m','wind_speed_50m','wind_speed_100m','wind_speed_250m','wind_speed_500m','wind_speed_750m'],height_levels=[10,20,50,100,250,500,750], z=[20,250], perc = [25,75], output_file='shear_distribution.png'):  
    """ This function is written by birgitterf.
    It calculates and plot the shear between two height levels as a histogram 
    with the median value in black and and lower/higher percentile in gray.
    data = dataframe
    vars = list of variables in dataframe e.g., ['wind_speed_10m','wind_speed_20m','wind_speed_50m']
    height_levels = [10,20,50] 
    z = interval to calculate the shear over e.g. [10,50]
    perc = lower, higher percentile e.g., [25,75] 
    output_file = 'shear_histogram.png' or False for no 
    return shear values alfa as numpy array.
    """
    df = data[vars]
    nparr = df.to_numpy()
    H = np.array(height_levels)
        
    i_upper = np.arange(len(H))[H==z[1]]
    i_lower = np.arange(len(H))[H==z[0]]
    if len(i_lower)==1:
        i_lower = i_lower[0]
        i_upper = i_upper[0]
    else:
        raise Exception
    
    a = np.log(nparr[:,i_upper]/nparr[:,i_lower])
    b = np.log(z[1]/z[0])
    alfa = a/b
    
    if output_file==False:
        pass
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        N, bins, range = plt.hist(alfa,50)
        locs, _ = plt.yticks() 
        print(locs)
        plt.yticks(locs,np.round(100*(locs/len(alfa)),1))
        plt.fill_betweenx([0,np.max(N)],np.percentile(alfa,perc[0]),np.percentile(alfa,perc[1]),alpha=0.5,color='lightgray',label=f'{perc[0]}-{perc[1]}%')
        plt.plot(np.ones(2)*np.median(alfa),[0,np.max(N)],color='black',label='median')
        #ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        plt.xlabel(f'Shear between {z[0]} m and {z[1]} m')
        plt.ylabel('Density [%]')
        plt.legend()
        plt.savefig(output_file)
        plt.close()
    return alfa
