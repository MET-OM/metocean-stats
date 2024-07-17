import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import windrose
import matplotlib.cm as cm
import os
from .aux_funcs import convert_latexTab_to_csv, Tp_correction


def rose(wd,ws,max_ws,step_ws,min_percent, max_percent, step_percent):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="windrose")
    ax.bar(wd, ws, bins=np.arange(0, max_ws, step_ws), cmap=cm.rainbow, normed=True, opening=0.9, edgecolor='white')
    ax.set_yticks(np.arange(min_percent, max_percent, step_percent))
    ax.set_yticklabels(np.arange(min_percent, max_percent,step_percent))
    ax.legend(bbox_to_anchor=(0.90,-0.05),framealpha=0.5)
    return fig


def var_rose(data, direction,intensity,max_perc,decimal_places=1, units='m/s', method='overall', output_file='rose.png'):

    direction2 = data[direction]
    intensity2 = data[intensity]
    size = 5
    bins_range = np.array([0, np.percentile(intensity2,40),
                   np.percentile(intensity2,60),
                   np.percentile(intensity2,80),
                   np.percentile(intensity2,99)])
    
    if method == 'overall':
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection="windrose")
        ax.bar(direction2, intensity2, normed=True, bins=bins_range, opening=0.99, nsector=12)
        ax.set_yticks(np.arange(5, max_perc, step=10))
        ax.set_yticklabels(np.arange(5, max_perc, step=10))
        ax.set_legend(decimal_places=decimal_places, units=units)
        ax.set_title('Overall')
        ax.figure.set_size_inches(size, size)
        plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')

    elif method == 'monthly':
        monthly_var_rose(data,direction,intensity,max_perc=decimal_places,bins_range=bins_range,decimal_places=decimal_places, units=units, output_file=output_file)
    
    plt.close()
    return fig


def monthly_var_rose(data, direction,intensity,bin_range,max_perc=40,decimal_places=1, units='m/s',output_file='rose.png') : 

    # this function make monthly wind/wave rose
    # direction, intensity: panda series 
    # get month from panda series 
    direction = data[direction]
    intensity = data[intensity]
    M = intensity.index.month.values
    
    # get month names 
    import calendar
    months = calendar.month_name[1:] # eliminate the first insane one 
    for i in range(len(months)) : 
        months[i] = months[i][:3] # get the three first letters 
    
    # sort them outh by months 
    dic_intensity = {} # dic_intensity
    dic_direction = {} # dic_direction
    for i in range(len(months)) : 
        dic_intensity[months[i]] = [] 
        dic_direction[months[i]] = [] 
        
    for i in range(len(intensity)) : 
        m_idx = int(M[i]-1)
        dic_intensity[months[m_idx]].append(intensity.iloc[i])
        dic_direction[months[m_idx]].append(direction.iloc[i])
        
    for j in range(12):
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection="windrose")
        ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True,bins=bin_range, opening=0.99, nsector=12)
        ax.set_legend()
        ax.set_title(months[j])
        size = 5
        ax.figure.set_size_inches(size, size)
        plt.savefig(months[j]+'_'+output_file,dpi=100,facecolor='white',bbox_inches='tight')
        plt.close()
    return fig
