import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    if output_file==False:
        pass
    else:
        fig, ax = plt.subplots(figsize=(4,8))
        plt.fill_betweenx(np.array(height_levels),np.percentile(nparr,perc[0],axis=0),np.percentile(nparr,perc[1],axis=0),color='lightgray')
        plt.plot(mean_values,np.array(height_levels),'k-',linewidth=2.5)
        plt.ylim(np.min(height_levels),np.max(height_levels))
        plt.xlabel('wind speed [m/s]')
        plt.ylabel('height [m]')
        plt.grid()
        plt.title(output_file.split('.')[0],fontsize=18)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    return mean_values

