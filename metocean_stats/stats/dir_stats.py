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


def var_rose(data, direction,intensity, output_file, method='overall'):

    direction = data[direction]
    intensity = data[intensity]
    size = 5
    if method == 'overall':
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection="windrose")
        ax.bar(direction, intensity, normed=True, opening=0.8, nsector=12)
        ax.set_legend()
        ax.figure.set_size_inches(size, size)
        plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')

    elif method == 'monthly':
        monthly_var_rose(direction,intensity,output_file)
    
    plt.close()
    return 

def directional_min_mean_max(data, direction, intensity, output_file) : 
    direction = data[direction]
    intensity = data[intensity]
    temp_file  = output_file.split('.')[0]

    time = intensity.index  
    
    # sorted by sectors/directions, keep time for the next part 
    bins_dir = np.arange(0,360,30) # 0,30,...,300,330
    dic_Hs = {}
    dic_time = {}
    for i in range(len(bins_dir)) : 
        dic_Hs[str(int(bins_dir[i]))] = [] 
        dic_time[str(int(bins_dir[i]))] = [] 
    
    for i in range(len(intensity)): 
        if 345 <= direction.iloc[i] :
            dic_time[str(int(bins_dir[0]))].append(time[i])
            dic_Hs[str(int(bins_dir[0]))].append(intensity.iloc[i]) 
        else: 
            for j in range(len(bins_dir)): 
                if bins_dir[j]-15 <= direction.iloc[i] < bins_dir[j] + 15 : # -15 --> +345 
                    dic_time[str(int(bins_dir[j]))].append(time[i])
                    dic_Hs[str(int(bins_dir[j]))].append(intensity.iloc[i]) 
                    
    # write to file 
    with open(temp_file, 'w') as f :
        f.write('\\begin{tabular}{l | c c c }' + '\n')
        f.write('Direction & Minimum & Mean & Maximum \\\\' + '\n')
        f.write('\hline' + '\n')
        
        # sorted by years, get max in each year, and statistical values 
        for j in range(len(bins_dir)):
            df_dir = pd.DataFrame()
            df_dir.index = dic_time[str(int(bins_dir[j]))]
            df_dir['Hs'] = dic_Hs[str(int(bins_dir[j]))]
            annual_max_dir = df_dir.resample('Y').max()
            mind = round(annual_max_dir.min()['Hs'],1)
            meand = round(annual_max_dir.mean()['Hs'],1)
            maxd = round(annual_max_dir.max()['Hs'],1)
            start = bins_dir[j] - 15
            if start < 0 : 
                start = 345 
            f.write(str(start) + '-' + str(bins_dir[j]+15) + ' & ' + str(mind) + ' & ' + str(round(meand,1)) + ' & ' + str(maxd) + ' \\\\' + '\n')
            
        ## annual row 
        annual_max = intensity.resample('Y').max()
        mind = round(annual_max.min(),1)
        meand = round(annual_max.mean(),1)
        maxd = round(annual_max.max(),1)
        f.write('Annual & ' + str(mind) + ' & ' + str(meand) + ' & ' + str(maxd) + ' \\\\' + '\n')
        f.write('\hline' + '\n')
        f.write('\end{tabular}' + '\n')

    if output_file.split('.')[1] == 'csv':
        convert_latexTab_to_csv(temp_file, output_file)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)

    return

def monthly_var_rose(data, direction,intensity,output_file) : 

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
        ax.bar(dic_direction[months[j]], dic_intensity[months[j]], normed=True, opening=0.8, nsector=12)
        ax.set_legend()
        ax.set_title(months[j])
        size = 5
        ax.figure.set_size_inches(size, size)
        plt.savefig(months[j]+'_'+output_file,dpi=100,facecolor='white',bbox_inches='tight')
        plt.close()
    return
