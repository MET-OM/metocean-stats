from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..stats.aux_funcs import Tp_correction, add_direction_sector, Hs_Tp_curve, Gauss3, Gauss4, Weibull_method_of_moment, DVN_steepness, find_percentile
from ..stats.extreme import *


def table_monthly_joint_distribution_Hs_Tp_param(data,var1='hs',var2='tp',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_param.csv'):
    # Calculate LoNoWe parameters for each month
    params = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']

    for month in range(1,len(months)):
        month_data = data[data.index.month == month]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=month_data,var1=var1,var2=var2,periods=periods)
        params.append((a1, a2, a3, b1, b2, b3))
    # add annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph =  joint_distribution_Hs_Tp(data=data,var1=var1,var2=var2,periods=periods)
    params.append((a1, a2, a3, b1, b2, b3))   
    headers = ['Month', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
        # Create DataFrame
    df = pd.DataFrame(params, columns=headers[1:], index=months)
    df.index.name='Month'
    df = df.round(3)

    if output_file:
        df.to_csv(output_file)    

    return df

def table_directional_joint_distribution_Hs_Tp_param(data,var1='hs',var2='tp',var_dir='pdir',periods=[1,10,100,10000],output_file='directional_Hs_Tp_joint_param.csv'):
    # Calculate LoNoWe parameters for each month
    params = []
    dir_label = [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']

    add_direction_sector(data=data,var_dir=var_dir)
    for dir in range(0,360,30):
        sector_data = data[data['direction_sector']==dir]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=sector_data,var1=var1,var2=var2,periods=periods)
        params.append((a1, a2, a3, b1, b2, b3))
    # add annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var1=var1,var2=var2,periods=periods)
    params.append((a1, a2, a3, b1, b2, b3))       
    headers = ['Direction', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    # Create DataFrame

    df = pd.DataFrame(params, columns=headers[1:], index=dir_label)
    df.index.name='Direction'
    df = df.round(3)

    if output_file:
        df.to_csv(output_file)    

    return df  


def table_monthly_weibull_return_periods(data, var='hs', periods=[1, 10, 100, 10000], units='m',output_file='monthly_extremes_weibull.csv'):
    weibull_params, return_periods = monthly_extremes_weibull(data=data, var=var, periods=periods)
    months = ['-','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']
    
    # Initialize lists to store table data
    annual_prob = ['%'] + [np.round(100/12,2)] * 12 + [100.00]
    weibull_shape =   ['-'] + [round(shape, 3) for shape, _, _ in weibull_params]
    weibull_scale =   [units] + [round(scale, 3) for _, _, scale in weibull_params]
    weibull_location =   [units] + [round(loc, 2) for _, loc, _ in weibull_params]

    # Create the table data dictionary
    table_data = {
        'Month': months,
        'Annual prob.': annual_prob,
        'Shape': weibull_shape,
        'Scale': weibull_scale,
        'Location': weibull_location,
    }
    return_periods = return_periods.T.tolist()
    # Fill in return periods for each period
    for i, period in enumerate(periods):
        table_data[f'Return period: {period} [years]'] = [units] + return_periods[i]
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file)
    
    return df

def table_directional_weibull_return_periods(data: pd.DataFrame, var='hs', var_dir='dir', periods=[1, 10, 100, 10000], units='m',output_file='directional_extremes_weibull.csv'):
    weibull_params, return_periods, sector_prob = directional_extremes_weibull(data=data, var=var, var_dir=var_dir, periods=periods)
    dir = ['-'] + [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']
    # Initialize lists to store table data
    sector_prob = ['%'] + [round(value, 2) for value in sector_prob] + [100.00]
    weibull_shape = ['-'] + [round(shape, 3) for shape, _, _ in weibull_params]
    weibull_scale = [units] + [round(scale, 3) for _, _, scale in weibull_params]
    weibull_location = [units] + [round(loc, 2) for _, loc, _ in weibull_params]
    # Create the table data dictionary
    table_data = {
        'Direction sector': dir,
        'Sector prob.': sector_prob,
        'Shape': weibull_shape,
        'Scale': weibull_scale,
        'Location': weibull_location,
    }
    return_periods = return_periods.T.tolist()
    # Fill in return values for each period
    for i, period in enumerate(periods):
        table_data[f'Return period: {period} [years]'] = [units] + return_periods[i]
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file)
    
    return df



def table_monthly_joint_distribution_Hs_Tp_return_values(data,var1='hs',var2='tp',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_reurn_values.csv'):
    # Calculate LoNoWe parameters for each month
    rv_hs = np.zeros((13,len(periods)))
    rv_tp = np.zeros((13,len(periods)))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']

    for month in range(1,len(months)):
        print(month)
        month_data = data[data.index.month == month]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=month_data,var1=var1,var2=var2,periods=periods)
        for i in range(len(periods)):
            rv_hs[month-1,i] = hs_tpl_tph['hs_'+str(periods[i])].max().round(2)
            rv_tp[month-1,i] = hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max().round(2)

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var1=var1,var2=var2,periods=periods)
    for i in range(len(periods)):
        rv_hs[12,i] = hs_tpl_tph['hs_'+str(periods[i])].max().round(2)
        rv_tp[12,i] = hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max().round(2)
    
    # Initialize lists to store table data
    annual_prob =  [np.round(100/12,2)] * 12 + [100.00]

    # Create the table data dictionary
    table_data = {
        'Month': months,
        'Annual prob.[%]': annual_prob,
    }
    # Fill in return values for each period
    for i, period in enumerate(periods):
        table_data[f'H_s [{period} years]'] = rv_hs[:,i].tolist()
        table_data[f'T_p [{period} years]'] = rv_tp[:,i].tolist()
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file,index=False)

    return df
    
def table_directional_joint_distribution_Hs_Tp_return_values(data,var1='hs',var2='tp',var_dir='pdir',periods=[1,10,100,10000],output_file='directional_Hs_Tp_joint_reurn_values.csv'):
    weibull_params, return_periods, sector_prob = directional_extremes_weibull(data=data, var=var1, var_dir=var_dir, periods=periods)
    dir = ['-'] + [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']    
    dir = ['-'] + [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']
    # Initialize lists to store table data
    sector_prob =  [round(value, 2) for value in sector_prob] + [100.00]
    rv_hs = np.zeros((13,len(periods)))
    rv_tp = np.zeros((13,len(periods)))
    dir_label = [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']

    add_direction_sector(data=data,var_dir=var_dir)
    k=0
    for dir in range(0,360,30):
        k=k+1
        sector_data = data[data['direction_sector']==dir]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=sector_data,var1=var1,var2=var2,periods=periods)
        for i in range(len(periods)):
            rv_hs[k-1,i] = hs_tpl_tph['hs_'+str(periods[i])].max().round(2)
            rv_tp[k-1,i] = hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max().round(2)

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var1=var1,var2=var2,periods=periods)
    for i in range(len(periods)):
        rv_hs[12,i] = hs_tpl_tph['hs_'+str(periods[i])].max().round(2)
        rv_tp[12,i] = hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max().round(2)
    

    # Create the table data dictionary
    table_data = {
        'Direction': dir_label,
        'Sector prob.[%]': sector_prob,
    }
    # Fill in return values for each period
    for i, period in enumerate(periods):
        table_data[f'H_s [{period} years]'] = rv_hs[:,i].tolist()
        table_data[f'T_p [{period} years]'] = rv_tp[:,i].tolist()
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file,index=False)
    return df


def table_Hs_Tpl_Tph_return_values(data,var1='hs',var2='tp',periods=[1,10,100,10000],output_file='Hs_Tpl_Tph_joint_reurn_values.csv'):
    # Calculate LoNoWe parameters for each month
    table = np.zeros((10,3*len(periods)))

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var1=var1,var2=var2,periods=periods)
    k=0
    for i in range(len(periods)):
        max_hs = hs_tpl_tph['hs_'+str(periods[i])].max().round(1)
        for j in range(10):
            closest_index = (hs_tpl_tph['hs_'+str(periods[i])] - (max_hs-j)).abs().idxmin()
            #print(j,k,k+3)
            table[j,k:k+3] = hs_tpl_tph.loc[closest_index][k:k+3].values.round(1)
        k = k+4
    breakpoint()


    # Create DataFrame
    df = pd.DataFrame(table)
    if output_file:
        df.to_csv(output_file,index=False)

    return df
