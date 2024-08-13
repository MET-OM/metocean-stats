from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..stats.aux_funcs import *
from ..stats.extreme import *
from ..stats.general import *


def table_monthly_joint_distribution_Hs_Tp_param(data,var_hs='hs',var_tp='tp',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_param.csv'):
    # Calculate LoNoWe parameters for each month
    params = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']

    for month in range(1,len(months)):
        month_data = data[data.index.month == month]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=month_data,var_hs=var_hs,var_tp=var_tp,periods=periods)
        params.append((a1, a2, a3, b1, b2, b3))
    # add annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph =  joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
    params.append((a1, a2, a3, b1, b2, b3))   
    headers = ['Month', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    # Create DataFrame
    df = pd.DataFrame(params, columns=headers[1:], index=months)
    df.index.name='Month'
    df = df.round(3)

    if output_file:
        df.to_csv(output_file)    

    return df

def table_directional_joint_distribution_Hs_Tp_param(data,var_hs='hs',var_tp='tp',var_dir='pdir',periods=[1,10,100,10000],output_file='directional_Hs_Tp_joint_param.csv'):
    # Calculate LoNoWe parameters for each month
    params = []
    dir_label = [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']

    add_direction_sector(data=data,var_dir=var_dir)
    for dir in range(0,360,30):
        sector_data = data[data['direction_sector']==dir]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=sector_data,var_hs=var_hs,var_tp=var_tp,periods=periods)
        params.append((a1, a2, a3, b1, b2, b3))
    # add annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
    params.append((a1, a2, a3, b1, b2, b3))       
    headers = ['Direction', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    # Create DataFrame

    df = pd.DataFrame(params, columns=headers[1:], index=dir_label)
    df.index.name='Direction'
    df = df.round(3)

    if output_file:
        df.to_csv(output_file)    

    return df  

def table_monthly_return_periods(data, var='hs', periods=[1, 10, 100, 10000],distribution='Weibull3P_MOM',method='default',threshold='default', units='m',output_file='monthly_extremes_weibull.csv'):
    months = ['-','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']
    params, return_periods, threshold_values, num_events_per_year = monthly_extremes(data=data, var=var, periods=periods, distribution=distribution, method=method, threshold=threshold)   

    # Initialize lists to store table data
    annual_prob = ['%'] + [np.round(100/12,2)] * 12 + [100.00]
    shape = ['-'] + [round(shape, 3) if isinstance(shape, (int, float)) else shape for shape, _, _ in params]    
    scale = [units] + [round(scale, 3) if isinstance(scale, (int, float)) else scale for _, _, scale in params]
    location = [units] + [round(loc, 2) if isinstance(loc, (int, float)) else loc for _, loc, _ in params]
    shape = ['-' if element == [] else element for element in shape]   
    
    table_data = {
        'Month': months,
        'Annual prob.': annual_prob,
        'Shape': shape,
        'Scale': scale,
        'Location': location,
    }
    if threshold_values:
        table_data['Threshold'] = [units] + [round(x, 2) for x in threshold_values]
    if num_events_per_year:
        table_data['Events'] = ['1/year'] + [round(x, 1) for x in num_events_per_year]
    return_periods = return_periods.T.tolist()
    # Fill in return periods for each period
    for i, period in enumerate(periods):
        table_data[f'Return period: {period} [years]'] = [units] + [round(x, 2) for x in return_periods[i]]
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file)
    
    return df

def table_directional_return_periods(data: pd.DataFrame, var='hs', var_dir='dir', periods=[1, 10, 100, 10000], distribution='Weibull3P_MOM', units='m',adjustment='NORSOK',method='default', threshold='default',output_file='directional_extremes_weibull.csv'):
    params, return_periods, sector_prob,  threshold_values, num_events_per_year = directional_extremes(data=data, var=var, var_dir=var_dir, periods=periods,distribution=distribution, adjustment=adjustment, method=method, threshold=threshold)    

    dir = ['-'] + [str(angle) + '°' for angle in np.arange(0,360,30)] + ['Omni']
    # Initialize lists to store table data
    sector_prob = ['%'] + [round(value, 2) for value in sector_prob] + [100.00]
    shape = ['-'] + [round(shape, 3) if isinstance(shape, (int, float)) else shape for shape, _, _ in params]    
    scale = [units] + [round(scale, 3) if isinstance(scale, (int, float)) else scale for _, _, scale in params]
    location = [units] + [round(loc, 2) if isinstance(loc, (int, float)) else loc for _, loc, _ in params]
    shape = ['-' if element == [] else element for element in shape]  

    # Create the table data dictionary
    table_data = {
        'Direction sector': dir,
        'Sector prob.': sector_prob,
        'Shape': shape,
        'Scale': scale,
        'Location': location,
    }

    if threshold_values:
        table_data['Threshold'] = [units] + [round(x, 2) for x in threshold_values]
    if num_events_per_year:
        table_data['Events'] = ['1/year'] + [round(x, 1) for x in num_events_per_year]
    
    return_periods = return_periods.T.tolist()
    # Fill in return values for each period
    for i, period in enumerate(periods):
        table_data[f'Return period: {period} [years]'] = [units] +  [round(x, 2) for x in return_periods[i]]
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file)

    return df

def table_monthly_joint_distribution_Hs_Tp_return_values(data,var_hs='hs',var_tp='tp',periods=[1,10,100,10000],output_file='monthly_Hs_Tp_joint_reurn_values.csv'):
    # Calculate LoNoWe parameters for each month
    rv_hs = np.zeros((13,len(periods)))
    rv_tp = np.zeros((13,len(periods)))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']

    for month in range(1,len(months)):
        print(month)
        month_data = data[data.index.month == month]
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=month_data,var_hs=var_hs,var_tp=var_tp,periods=periods)
        for i in range(len(periods)):
            rv_hs[month-1,i] = np.round(hs_tpl_tph['hs_'+str(periods[i])].max(),2)
            rv_tp[month-1,i] = np.round(hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max(),2)

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
    for i in range(len(periods)):
        rv_hs[12,i] = np.round(hs_tpl_tph['hs_'+str(periods[i])].max(),2)
        rv_tp[12,i] = np.round(hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max(),2)
    
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
    
def table_directional_joint_distribution_Hs_Tp_return_values(data,var_hs='hs',var_tp='tp',var_dir='pdir',periods=[1,10,100,10000],adjustment='NORSOK', output_file='directional_Hs_Tp_joint_reurn_values.csv'):
    weibull_params, return_periods, sector_prob, threshold_values, num_events_per_year = directional_extremes(data=data, var=var_hs, var_dir=var_dir, periods=periods,distribution='Weibull3P_MOM', adjustment=adjustment)

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
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=sector_data,var_hs=var_hs,var_tp=var_tp,periods=periods,adjustment=adjustment)
        for i in range(len(periods)):
            rv_hs[k-1,i] = round(hs_tpl_tph['hs_'+str(periods[i])].max(),2)
            rv_tp[k-1,i] = round(hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max(),2)

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods,adjustment=None)
    for i in range(len(periods)):
        rv_hs[12,i] = round(hs_tpl_tph['hs_'+str(periods[i])].max(),2)
        rv_tp[12,i] = round(hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max(),2)

    # Define the threshold values (annual values) for each column
    thresholds_hs = rv_hs[12,:]
    thresholds_tp = rv_tp[12,:]

    # Replace values in each column that exceed the thresholds
    for col in range(rv_hs.shape[1]):
        rv_hs[:, col] = np.minimum(rv_hs[:, col], thresholds_hs[col])
        rv_tp[:, col] = np.minimum(rv_tp[:, col], thresholds_tp[col])

    # Create the table data dictionary
    table_data = {
        'Direction': dir_label,
        'Sector prob.[%]': sector_prob,
    }
    # Fill in return values for each period
    for i, period in enumerate(periods):
        table_data[f'Hs[m] [{period} years]'] = rv_hs[:,i].tolist()
        table_data[f'Tp[s] [{period} years]'] = rv_tp[:,i].tolist()
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file,index=False)
    return df


def table_Hs_Tpl_Tph_return_values(data,var_hs='hs',var_tp='tp',periods=[1,10,100,10000],output_file='Hs_Tpl_Tph_joint_reurn_values.csv'):
    # Calculate LoNoWe parameters for each month
    table = np.zeros((10,3*len(periods)))

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
    k=0
    for i in range(len(periods)):
        max_hs = np.round(hs_tpl_tph['hs_'+str(periods[i])].max(),1)
        for j in range(10):
            closest_index = (hs_tpl_tph['hs_'+str(periods[i])] - (max_hs-j)).abs().idxmin()
            #print(j,k,k+3)
            table[j,k:k+3] = hs_tpl_tph.loc[closest_index][k:k+3].values.round(1)
        k = k+3

    # Create a DataFrame for better readability
    columns = [item for period in periods for item in (f'HS [{period} years]', f'TpL [{period} years]', f'TpH [{period} years]')]

    # Create DataFrame
    df = pd.DataFrame(table,  columns=columns)
    if output_file:
        df.to_csv(output_file,index=False)

    return df


def table_tp_for_given_hs(data: pd.DataFrame, var_hs: str,var_tp: str, bin_width=1, max_hs=20, output_file='table_perc_tp_for_hs.csv'):
    df=data
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=[1000])
    # Create bins
    min_hs = 0.5
    max_hs = max_hs
    bins = np.arange(min_hs, max_hs + bin_width, bin_width)
    bin_labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

    # Bin the HS values
    df[var_hs+'_bin'] = pd.cut(df[var_hs], bins=bins, labels=bin_labels, right=False)
    
    # Calculate P5, Mean, and P95 for each bin
    result = []
    for i, bin_label in enumerate(bin_labels):
        bin_df = df[df[var_hs+'_bin'] == bin_label]
        if len(bin_df.index)>10:
            P5 = bin_df[var_tp].quantile(0.05)
            Mean = bin_df[var_tp].mean()
            P95 = bin_df[var_tp].quantile(0.95)
        else:
            P5 = np.nan
            Mean = np.nan
            P95 = np.nan

        P5_model,Mean_model,P95_model = model_tp_given_hs(hs=bin_centers[i], a1=a1, a2=a2, a3=a3, b1=b1, b2=b2, b3=b3)
        result.append([bin_label, bin_centers[i], P5, Mean, P95,P5_model,Mean_model,P95_model])


    # Create a new dataframe with the results
    result_df = pd.DataFrame(result, columns=[var_hs+'_bin', 'Hs[m]','Tp(P5-obs) [s]','Tp(Mean-obs) [s]','Tp(P95-obs) [s]', 'Tp(P5-model) [s]','Tp(Mean-model) [s]','Tp(P95-model) [s]'])
    if output_file:
        result_df[['Hs[m]', 'Tp(P5-model) [s]','Tp(Mean-model) [s]','Tp(P95-model) [s]']].round(2).to_csv(output_file,index=False)

    return result_df


def table_tp_for_rv_hs(data: pd.DataFrame, var_hs: str,var_tp: str, bin_width=1, periods=[1,10,1000,10000], output_file='table_perc_tp_for_hs.csv'):
    df=data
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var_hs=var_hs,var_tp=var_tp,periods=periods)
    # Create bins
    rv_hs = np.zeros(len(periods))
    result = []
    for i in range(len(periods)):
        rv_hs = np.round(hs_tpl_tph['hs_'+str(periods[i])].max(),2)
        P5_model,Mean_model,P95_model = model_tp_given_hs(hs=rv_hs, a1=a1, a2=a2, a3=a3, b1=b1, b2=b2, b3=b3)
        result.append([periods[i],rv_hs,P5_model,Mean_model,P95_model])


    # Create a new dataframe with the results
    result_df = pd.DataFrame(result, columns=['Return period [years]', 'Hs[m]','Tp(P5-model) [s]','Tp(Mean-model) [s]','Tp(P95-model) [s]'])
    #result_df = result_df.dropna()
    if output_file:
        result_df.round(2).to_csv(output_file,index=False)

    return result_df


def table_hs_for_given_wind(data: pd.DataFrame, var_hs: str,var_wind: str, bin_width=2, max_wind=40, output_file='table_perc_tp_for_wind.csv'):
    df=data
    # Create bins
    min_wind = 0
    bins = np.arange(min_wind, max_wind + bin_width, bin_width)
    bin_labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    # Bin the wind values
    df[var_wind+'_bin'] = pd.cut(df[var_wind], bins=bins, labels=bin_labels, right=False)
    
    # Calculate P5, Mean, and P95 for each bin
    result = []
    for i, bin_label in enumerate(bin_labels):
        bin_df = df[df[var_wind+'_bin'] == bin_label]
        if len(bin_df.index)>10:
            P5 = bin_df[var_hs].quantile(0.05)
            Mean = bin_df[var_hs].mean()
            sigma = bin_df[var_hs].std()
            P95 = bin_df[var_hs].quantile(0.95)
        else:
            P5 = np.nan
            Mean = np.nan
            sigma  = np.nan
            P95 = np.nan

        result.append([bin_label, bin_centers[i], P5, Mean, sigma, P95])

    result_df = pd.DataFrame(result, columns=[var_hs+'_bin', 'U[m/s]','Hs(P5-obs) [m]','Hs(Mean-obs) [m]','Hs(std-obs) [m]', 'Hs(P95-obs) [m]'])
    a_mean, b_mean, c_mean, d_mean = fit_hs_wind_model(result_df.dropna()['U[m/s]'].values,result_df.dropna()['Hs(Mean-obs) [m]'].values) 
    a_sigma, b_sigma, c_sigma, d_sigma = fit_hs_wind_model(result_df.dropna()['U[m/s]'].values,result_df.dropna()['Hs(std-obs) [m]'].values) 
    result_df['Hs(Mean-model) [m]'] = Hs_as_function_of_U(result_df['U[m/s]'], a_mean, b_mean, c_mean, d_mean)
    result_df['Hs(std-model) [m]'] = Hs_as_function_of_U(result_df['U[m/s]'], a_sigma, b_sigma, c_sigma, d_sigma)
    result_df['Hs(P5-model) [m]'] =  result_df['Hs(Mean-model) [m]'] - 1.65*result_df['Hs(std-model) [m]']
    result_df['Hs(P95-model) [m]'] =  result_df['Hs(Mean-model) [m]'] + 1.65*result_df['Hs(std-model) [m]']

    # Create the table data dictionary
    table_data = {
        'Hs(U)': ['Mean','Std. Dev.'],
        'a': [a_mean,a_sigma],
        'b': [b_mean,b_sigma],
        'c': [c_mean,c_sigma],
        'd': [c_mean,d_sigma],
    }
    df_coeff = pd.DataFrame(table_data)
    if output_file:
        df_coeff.round(3).to_csv(output_file.split('.')[0]+'_coeff.csv',index=False)

    # Create a new dataframe with the results
    if output_file:
        result_df[['U[m/s]', 'Hs(P5-model) [m]','Hs(Mean-model) [m]','Hs(P95-model) [m]']].round(2).to_csv(output_file,index=False)
    return result_df

def table_wave_induced_current(ds, var_hs,var_tp,max_hs= 20, depth=200,ref_depth=50,spectrum='JONSWAP',output_file='wave_induced_current_depth200.csv'):
    df = table_tp_for_given_hs(ds, var_hs,var_tp, bin_width=1, max_hs=max_hs, output_file=None)
    df['Us(P5) [m/s]'], df['Tu(P5-model) [s]'] = calculate_Us_Tu(df['Hs[m]'],df['Tp(P5-model) [s]'], depth=depth, ref_depth=ref_depth,spectrum=spectrum)
    df['Us(Mean) [m/s]'], df['Tu(Mean-model) [s]'] = calculate_Us_Tu(df['Hs[m]'],df['Tp(Mean-model) [s]'], depth=depth, ref_depth=ref_depth,spectrum=spectrum)
    df['Us(P95) [m/s]'], df['Tu(P95-model) [s]'] = calculate_Us_Tu(df['Hs[m]'],df['Tp(P95-model) [s]'], depth=depth, ref_depth=ref_depth,spectrum=spectrum)
    if output_file:
        df[['Hs[m]','Tp(P5-model) [s]', 'Us(P5) [m/s]','Tu(P5-model) [s]','Tp(Mean-model) [s]', 'Us(Mean) [m/s]','Tu(Mean-model) [s]','Tp(P95-model) [s]', 'Us(P95) [m/s]','Tu(P95-model) [s]']].round(2).to_csv(output_file,index=False)
    
    return df


def table_profile_return_values(data,var=['W10','W50','W80','W100','W150'], z=[10, 50, 80, 100, 150], periods=[1, 10, 100, 10000],units = 'm/s' ,distribution='Weibull3P',method='default',threshold='default', output_file='RVE_wind_profile.csv'):
    df = pd.DataFrame()
    df['z']= ['m'] + [str(num) for num in z] 
    for p in periods: 
        df[f'Return period {p} [years]'] = [units] + [RVE_ALL(data,var=var1,periods=p,distribution=distribution,method=method,threshold=threshold)[3].round(2) for var1 in var]
    
    if output_file:
        df.to_csv(output_file, index=False)  
    
    return df

def table_Hmax_crest_return_periods(ds,var_hs='HS', var_tp = 'TP',depth=200, periods=[1, 10, 100,10000], sea_state = 'short-crested', output_file='table_Hmax_crest_return_values.csv'):
    df = table_tp_for_rv_hs(ds, var_hs, var_tp,periods=periods,output_file=None)
    time_step = ((ds.index[-1]-ds.index[0]).days + 1)*24/ds.shape[0]
    df['T_Hmax(P5-model) [s]'] =  0.9 * df['Tp(P5-model) [s]'] # according to Goda (1988)
    df['T_Hmax(Mean-model) [s]'] =  0.9 * df['Tp(Mean-model) [s]'] # according to Goda (1988)
    df['T_Hmax(P95-model) [s]'] =  0.9 * df['Tp(P95-model) [s]'] # according to Goda (1988)
    df['Crest heigh[m]'] = np.nan
    df['H_max[m]'] = np.nan
    df['H_max/Hs'] = np.nan
    for i in range(df.shape[0]):
        df.loc[i,'Crest heigh[m]'] = estimate_forristal_maxCrest(df.loc[i,'Hs[m]'],df.loc[i,'T_Hmax(Mean-model) [s]'],depth=depth, twindow=time_step, sea_state=sea_state)
        #df.loc[i,'Crest heigh[m]'] = estimate_forristal_maxCrest(df.loc[i,'Hs[m]'],df.loc[i,'Tp(Mean-model) [s]'],depth=depth, twindow=time_step, sea_state=sea_state)
        df.loc[i,'H_max[m]'] = estimate_Hmax(df.loc[i,'Hs[m]'], df.loc[i,'T_Hmax(Mean-model) [s]'], twindow=3, k=1.0)
        #df.loc[i,'H_max[m]'] = estimate_Hmax(df.loc[i,'Hs[m]'], df.loc[i,'Tp(Mean-model) [s]'], twindow=3, k=1.0)
        df.loc[i,'H_max/Hs'] = df.loc[i,'H_max[m]']/ df.loc[i,'Hs[m]']

    if output_file:
        df[['Return period [years]','Hs[m]', 'H_max/Hs','H_max[m]','Crest heigh[m]','T_Hmax(P5-model) [s]','T_Hmax(Mean-model) [s]','T_Hmax(P95-model) [s]']].round(2).to_csv(output_file,index=False)
    
    return df

def table_directional_Hmax_return_periods(ds, var_hs='HS', var_tp='TP', var_dir='DIRM', periods=[1, 10, 100, 10000], adjustment='NORSOK', output_file='table_dir_Hmax_return_values.csv'):
    df = table_directional_joint_distribution_Hs_Tp_return_values(ds, var_hs=var_hs, var_tp=var_tp, var_dir=var_dir, periods=periods, adjustment=adjustment, output_file=None)
    hs_columns = [col for col in df.columns if col.startswith('Hs')]
    tp_columns = [col for col in df.columns if col.startswith('Tp')]
    hmax_columns = [col.replace('Hs[m]', 'Hmax[m]') for col in df.columns if col.startswith('Hs[m]')]

    if not hs_columns or not tp_columns:
        raise ValueError("Hs or Tp columns are missing or empty in the dataframe.")

    df[hmax_columns] = np.nan
    
    # Joint distribution calculation
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3, h3, X, hs_tpl_tph = joint_distribution_Hs_Tp(ds, var_hs=var_hs, var_tp=var_tp, periods=[1000])
    
    # Model prediction
    P5_model, Mean_model, P95_model = model_tp_given_hs(hs=df[hs_columns], a1=a1, a2=a2, a3=a3, b1=b1, b2=b2, b3=b3)
    
    # Assign T_Hmax values
    for i in range(len(hs_columns)):
        df[f'T_Hmax(P5-model) [s] [{periods[i]} years]'] = 0.9 * P5_model[hs_columns[i]]
        df[f'T_Hmax(Mean-model) [s] [{periods[i]} years]'] = 0.9 * Mean_model[hs_columns[i]]
        df[f'T_Hmax(P95-model) [s] [{periods[i]} years]'] = 0.9 * P95_model[hs_columns[i]]
    
    tHmax_columns = [col for col in df.columns if col.startswith('T_Hmax')]
    
    # Estimate Hmax
    for i in range(len(hmax_columns)):
        for j in range(df.shape[0]):
            df.loc[j, hmax_columns[i]] = estimate_Hmax(df.loc[j, hs_columns[i]], df.loc[j, tp_columns[i]], twindow=3, k=1.0)
    
    # Output to file if specified
    if output_file:
        selected_columns = ['Direction'] + hmax_columns + tHmax_columns
        df[selected_columns].round(2).to_csv(output_file, index=False)
    
    return df

def table_hs_for_rv_wind(data, var_wind='W10', var_hs='HS',periods=[1,10,100,10000],output_file='hs_for_rv_wind.csv'):
    df = table_hs_for_given_wind(data, var_hs='HS',var_wind='W10', bin_width=2, max_wind=40, output_file=None)
    shape, loc, scale, value = RVE_ALL(data,var='W10',periods=periods,distribution='Weibull3P_MOM',method='default',threshold='default')
    result_df = pd.DataFrame(value, columns=['U[m/s]'])
    result_df['Return period [years]'] = periods
    a_mean, b_mean, c_mean, d_mean = fit_hs_wind_model(df.dropna()['U[m/s]'].values,df.dropna()['Hs(Mean-obs) [m]'].values) 
    a_sigma, b_sigma, c_sigma, d_sigma = fit_hs_wind_model(df.dropna()['U[m/s]'].values,df.dropna()['Hs(std-obs) [m]'].values) 
    result_df['Hs(Mean-model) [m]'] = Hs_as_function_of_U(result_df['U[m/s]'], a_mean, b_mean, c_mean, d_mean)
    result_df['Hs(std-model) [m]'] = Hs_as_function_of_U(result_df['U[m/s]'], a_sigma, b_sigma, c_sigma, d_sigma)
    result_df['Hs(P5-model) [m]'] =  result_df['Hs(Mean-model) [m]'] - 1.65*result_df['Hs(std-model) [m]']
    result_df['Hs(P95-model) [m]'] =  result_df['Hs(Mean-model) [m]'] + 1.65*result_df['Hs(std-model) [m]']
    if output_file:
        result_df[['Return period [years]','U[m/s]', 'Hs(P5-model) [m]', 'Hs(Mean-model) [m]','Hs(P95-model) [m]' ]].round(2).to_csv(output_file,index=False)
    
    return result_df

def table_current_for_given_wind(data: pd.DataFrame, var_curr: str,var_wind: str, bin_width=2, max_wind=40, output_file='table_perc_curr_for_wind.csv'):
    df=data
    # Create bins
    min_wind = 0
    bins = np.arange(min_wind, max_wind + bin_width, bin_width)
    bin_labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    # Bin the wind values
    df[var_wind+'_bin'] = pd.cut(df[var_wind], bins=bins, labels=bin_labels, right=False)
    
    # Calculate P5, Mean, and P95 for each bin
    result = []
    for i, bin_label in enumerate(bin_labels):
        bin_df = df[df[var_wind+'_bin'] == bin_label]
        if len(bin_df.index)>10:
            P5 = bin_df[var_curr].quantile(0.05)
            Mean = bin_df[var_curr].mean()
            sigma = bin_df[var_curr].std()
            P95 = bin_df[var_curr].quantile(0.95)
        else:
            P5 = np.nan
            Mean = np.nan
            sigma  = np.nan
            P95 = np.nan

        result.append([bin_label, bin_centers[i], P5, Mean, sigma, P95])

    result_df = pd.DataFrame(result, columns=[var_curr+'_bin', 'U[m/s]','Uc(P5-obs) [m/s]','Uc(Mean-obs) [m/s]','Uc(std-obs) [m/s]', 'Uc(P95-obs) [m/s]'])
    a_mean, b_mean, c_mean, d_mean = fit_Uc_wind_model(result_df.dropna()['U[m/s]'].values,result_df.dropna()['Uc(Mean-obs) [m/s]'].values) 
    a_sigma, b_sigma, c_sigma, d_sigma = fit_Uc_wind_model(result_df.dropna()['U[m/s]'].values,result_df.dropna()['Uc(std-obs) [m/s]'].values) 
    result_df['Uc(Mean-model) [m/s]'] = Uc_as_function_of_U(result_df['U[m/s]'], a_mean, b_mean, c_mean, d_mean)
    result_df['Uc(std-model) [m/s]'] = Uc_as_function_of_U(result_df['U[m/s]'], a_sigma, b_sigma, c_sigma, d_sigma)
    result_df['Uc(P5-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] - 1.65*result_df['Uc(std-model) [m/s]']
    result_df['Uc(P95-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] + 1.65*result_df['Uc(std-model) [m/s]']


    # Create the table data dictionary
    table_data = {
        'Uc(U)': ['Mean','Std. Dev.'],
        'a': [a_mean,a_sigma],
        'b': [b_mean,b_sigma],
        'c': [c_mean,c_sigma],
        'd': [c_mean,d_sigma],
    }
    df_coeff = pd.DataFrame(table_data)
    if output_file:
        df_coeff.round(3).to_csv(output_file.split('.')[0]+'_coeff.csv',index=False)

    # Create a new dataframe with the results
    if output_file:
        result_df[['U[m/s]', 'Uc(P5-model) [m/s]','Uc(Mean-model) [m/s]','Uc(P95-model) [m/s]']].round(2).to_csv(output_file,index=False)
    return result_df


def table_current_for_given_hs(data: pd.DataFrame, var_curr: str,var_hs: str, bin_width=2, max_hs=20, output_file='table_perc_curr_for_hs.csv'):
    df=data
    # Create bins
    min_hs = 0
    bins = np.arange(min_hs, max_hs + bin_width, bin_width)
    bin_labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    # Bin the wind values
    df[var_hs+'_bin'] = pd.cut(df[var_hs], bins=bins, labels=bin_labels, right=False)
    
    # Calculate P5, Mean, and P95 for each bin
    result = []
    for i, bin_label in enumerate(bin_labels):
        bin_df = df[df[var_hs+'_bin'] == bin_label]
        if len(bin_df.index)>10:
            P5 = bin_df[var_curr].quantile(0.05)
            Mean = bin_df[var_curr].mean()
            sigma = bin_df[var_curr].std()
            P95 = bin_df[var_curr].quantile(0.95)
        else:
            P5 = np.nan
            Mean = np.nan
            sigma  = np.nan
            P95 = np.nan

        result.append([bin_label, bin_centers[i], P5, Mean, sigma, P95])

    result_df = pd.DataFrame(result, columns=[var_curr+'_bin', 'Hs[m]','Uc(P5-obs) [m/s]','Uc(Mean-obs) [m/s]','Uc(std-obs) [m/s]', 'Uc(P95-obs) [m/s]'])
    a_mean, b_mean, c_mean = fit_Uc_Hs_model(result_df.dropna()['Hs[m]'].values,result_df.dropna()['Uc(Mean-obs) [m/s]'].values) 
    a_sigma, b_sigma, c_sigma = fit_Uc_Hs_model(result_df.dropna()['Hs[m]'].values,result_df.dropna()['Uc(std-obs) [m/s]'].values) 
    result_df['Uc(Mean-model) [m/s]'] = Uc_as_function_of_Hs(result_df['Hs[m]'], a_mean, b_mean, c_mean)
    result_df['Uc(std-model) [m/s]'] = Uc_as_function_of_Hs(result_df['Hs[m]'], a_sigma, b_sigma, c_sigma)
    result_df['Uc(P5-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] - 1.65*result_df['Uc(std-model) [m/s]']
    result_df['Uc(P95-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] + 1.65*result_df['Uc(std-model) [m/s]']

    # Create the table data dictionary
    table_data = {
        'Uc(Hs)': ['Mean','Std. Dev.'],
        'a': [a_mean,a_sigma],
        'b': [b_mean,b_sigma],
        'c': [c_mean,c_sigma],
    }
    df_coeff = pd.DataFrame(table_data)
    if output_file:
        df_coeff.round(3).to_csv(output_file.split('.')[0]+'_coeff.csv',index=False)

    # Create a new dataframe with the results
    if output_file:
        result_df[['Hs[m]', 'Uc(P5-model) [m/s]','Uc(Mean-model) [m/s]','Uc(P95-model) [m/s]']].round(2).to_csv(output_file,index=False)
    return result_df

def table_extreme_current_profile_rv(data: pd.DataFrame, var: str, z=[10, 20, 30], periods=[1,10,100], percentile=95, fitting_method='polynomial', fmt=".2f", output_file='table_extreme_current_profile_rv.csv'):
    for period in periods:
        df = table_profile_return_values(data=data, var=var, z=z, periods=periods, output_file=None)
        df[[f'{i}' for i in z]] = np.nan
        df.loc[0, [f'{i}' for i in z]] = df[f'Return period {period} [years]'][0] # add units
        
        # Create a list of columns to drop, excluding the current period
        columns_to_drop = [f'Return period {p} [years]' for p in periods if p != period]
        # Drop the columns from the DataFrame
        df = df.drop(columns=columns_to_drop)
        
        for i in range(len(z)):
            index_events = data[data[var[i]] > data[var[i]].quantile(percentile / 100)].index
            mean_profile = data.loc[index_events][var].mean(axis=0)
            if fitting_method == 'polynomial':
                coeffients = fit_profile_polynomial(z, mean_profile.values, degree=5)
                df.loc[1:, f'{z[i]}'] = np.round(extrapolate_speeds(coeffients, z=z, target_speed=df[f'Return period {period} [years]'][i+1], target_z=z[i], method='polynomial'), 2)
            elif fitting_method == 'spline':
                coeffients = fit_profile_spline(z, mean_profile.values, s=None)
                df.loc[1:, f'{z[i]}'] = np.round(extrapolate_speeds(coeffients, z=z, target_speed=df[f'Return period {period} [years]'][i+1], target_z=z[i], method='spline'), 2)
        
        # Create a new dataframe with the results
        if output_file:
            file_extension = output_file.split('.')[1] 
            if file_extension == 'csv':
                df.to_csv(output_file.split('.')[0] + f'period_{period}.csv', index=False)
            elif file_extension == 'png' or 'pdf':
                import seaborn as sns
                plt.figure(figsize=(10, 8))
                df.rename(columns={ f'Return period {period} [years]': f'RP {period} [yrs]'}, inplace=True)
                ax = sns.heatmap(df.iloc[1:, 1:].astype(float), annot=True, cmap="pink_r", fmt=fmt, cbar=False, yticklabels=z)
                #plt.title(f'Return Period {period} Years')
                plt.xlabel('Associated values [m/s] per z')
                plt.ylabel('z[m]')
                ax.xaxis.tick_top()  # Move x-axis to the top
                ax.xaxis.set_label_position('top')  # Move x-axis label to the top
                plt.xticks(rotation=45, ha='left')  # Rotate x-axis labels by 45 degrees
                plt.yticks(rotation=45, ha='right')  # Rotate y-axis labels by 45 degrees
                #plt.tight_layout()
                plt.savefig(output_file.split('.')[0] + f'period_{period}.'+file_extension,dpi=100)
            else:
                print('File format is not supported')
    return df

def table_current_for_rv_wind(data, var_curr='current_speed_0m', var_wind='W10',periods=[1,10],output_file='Uc_for_rv_wind.csv'):
    df = table_current_for_given_wind(data=data, var_curr=var_curr,var_wind=var_wind, bin_width=2, max_wind=40, output_file=None)
    shape, loc, scale, value = RVE_ALL(data,var=var_wind,periods=periods,distribution='Weibull3P_MOM',method='default',threshold='default')
    result_df = pd.DataFrame(value, columns=['U[m/s]'])
    result_df['Return period [years]'] = periods
    a_mean, b_mean, c_mean, d_mean = fit_Uc_wind_model(df.dropna()['U[m/s]'].values,df.dropna()['Uc(Mean-obs) [m/s]'].values) 
    a_sigma, b_sigma, c_sigma, d_sigma = fit_Uc_wind_model(df.dropna()['U[m/s]'].values,df.dropna()['Uc(std-obs) [m/s]'].values) 
    result_df['Uc(Mean-model) [m/s]'] = Uc_as_function_of_U(result_df['U[m/s]'], a_mean, b_mean, c_mean, d_mean)
    result_df['Uc(std-model) [m/s]'] = Uc_as_function_of_U(result_df['U[m/s]'], a_sigma, b_sigma, c_sigma, d_sigma)
    result_df['Uc(P5-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] - 1.65*result_df['Uc(std-model) [m/s]']
    result_df['Uc(P95-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] + 1.65*result_df['Uc(std-model) [m/s]']
    # set to 0 values that are negative, especially for P5
    result_df[result_df < 0] = 0
    
    if output_file:
        result_df[['Return period [years]','U[m/s]', 'Uc(P5-model) [m/s]', 'Uc(Mean-model) [m/s]','Uc(P95-model) [m/s]' ]].round(2).to_csv(output_file,index=False)
    
    return result_df

def table_current_for_rv_hs(data, var_curr='current_speed_0m', var_hs='HS',periods=[1,10],output_file='Uc_for_rv_hs.csv'):
    df = table_current_for_given_hs(data=data, var_curr=var_curr,var_hs=var_hs, bin_width=2, max_hs=20, output_file=None)
    shape, loc, scale, value = RVE_ALL(data,var=var_hs,periods=periods,distribution='Weibull3P_MOM',method='default',threshold='default')
    result_df = pd.DataFrame(value, columns=['Hs[m]'])
    result_df['Return period [years]'] = periods
    a_mean, b_mean, c_mean = fit_Uc_Hs_model(df.dropna()['Hs[m]'].values,df.dropna()['Uc(Mean-obs) [m/s]'].values) 
    a_sigma, b_sigma, c_sigma = fit_Uc_Hs_model(df.dropna()['Hs[m]'].values,df.dropna()['Uc(std-obs) [m/s]'].values) 
    result_df['Uc(Mean-model) [m/s]'] = Uc_as_function_of_Hs(result_df['Hs[m]'], a_mean, b_mean, c_mean)
    result_df['Uc(std-model) [m/s]'] = Uc_as_function_of_Hs(result_df['Hs[m]'], a_sigma, b_sigma, c_sigma)
    result_df['Uc(P5-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] - 1.65*result_df['Uc(std-model) [m/s]']
    result_df['Uc(P95-model) [m/s]'] =  result_df['Uc(Mean-model) [m/s]'] + 1.65*result_df['Uc(std-model) [m/s]']
    # set to 0 values that are negative, especially for P5
    result_df[result_df < 0] = 0
    
    if output_file:
        result_df[['Return period [years]','Hs[m]', 'Uc(P5-model) [m/s]', 'Uc(Mean-model) [m/s]','Uc(P95-model) [m/s]' ]].round(2).to_csv(output_file,index=False)
    
    return result_df


def table_storm_surge_for_given_hs(data: pd.DataFrame, var_surge: str,var_hs: str, bin_width=1, max_hs=20, output_file='table_perc_storm_surge_for_hs.csv'):
    df=data
    # Create bins
    min_hs = 0
    bins = np.arange(min_hs, max_hs + bin_width, bin_width)
    bin_labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    # Bin the wind values
    df[var_hs+'_bin'] = pd.cut(df[var_hs], bins=bins, labels=bin_labels, right=False)
    
    # Calculate P5, Mean, and P95 for each bin
    result = []
    for i, bin_label in enumerate(bin_labels):
        bin_df = df[df[var_hs+'_bin'] == bin_label]
        if len(bin_df.index)>10:
            P5 = bin_df[var_surge].quantile(0.05)
            Mean = bin_df[var_surge].mean()
            sigma = bin_df[var_surge].std()
            P95 = bin_df[var_surge].quantile(0.95)
        else:
            P5 = np.nan
            Mean = np.nan
            sigma  = np.nan
            P95 = np.nan

        result.append([bin_label, bin_centers[i], P5, Mean, sigma, P95])

    result_df = pd.DataFrame(result, columns=[var_surge+'_bin', 'Hs[m]','S(P5-obs) [m]','S(Mean-obs) [m]','S(std-obs) [m]', 'S(P95-obs) [m]'])
    a_mean, b_mean, c_mean = fit_S_Hs_model(result_df.dropna()['Hs[m]'].values,result_df.dropna()['S(Mean-obs) [m]'].values) 
    a_sigma, b_sigma, c_sigma = fit_S_Hs_model(result_df.dropna()['Hs[m]'].values,result_df.dropna()['S(std-obs) [m]'].values) 
    a_p5, b_p5, c_p5 = fit_S_Hs_model(result_df.dropna()['Hs[m]'].values,result_df.dropna()['S(P5-obs) [m]'].values) 
    a_p95, b_p95, c_p95 = fit_S_Hs_model(result_df.dropna()['Hs[m]'].values,result_df.dropna()['S(P95-obs) [m]'].values) 


    result_df['S(Mean-model) [m]'] = S_as_function_of_Hs(result_df['Hs[m]'], a_mean, b_mean, c_mean)
    result_df['S(std-model) [m]'] = S_as_function_of_Hs(result_df['Hs[m]'], a_sigma, b_sigma, c_sigma)
    result_df['S(P5-model) [m]'] =  result_df['S(Mean-model) [m]'] - 1.65*result_df['S(std-model) [m]']
    result_df['S(P95-model) [m]'] =  result_df['S(Mean-model) [m]'] + 1.65*result_df['S(std-model) [m]']

    # Create the table data dictionary
    table_data = {
        'S(Hs)': ['Mean','Std. Dev.', 'P5', 'P95'],
        'a': [a_mean,a_sigma,a_p5,a_p95],
        'b': [b_mean,b_sigma,b_p5,b_p95],
        'c': [c_mean,c_sigma,c_p5,c_p95],
    }
    df_coeff = pd.DataFrame(table_data)
    if output_file:
        df_coeff.round(3).to_csv(output_file.split('.')[0]+'_coeff.csv',index=False)

    # Create a new dataframe with the results
    if output_file:
        result_df[['Hs[m]', 'S(P5-model) [m]','S(Mean-model) [m]','S(P95-model) [m]']].round(2).to_csv(output_file,index=False)
    return result_df, df_coeff


def table_extreme_total_water_level(data: pd.DataFrame, var_hs='HS',var_tp = 'TP', var_surge='zeta', var_tide='tide',depth=200,periods=[100,10000], sea_state = 'short-crested', output_file='table_extreme_total_water_level.csv'):
    df = table_Hmax_crest_return_periods(data,var_hs=var_hs, var_tp =var_tp,depth=depth, periods=periods, sea_state = 'short-crested', output_file=None)
    df['Tidal level(HAT)[m]'] = data[var_tide].max()
    shape, loc, scale, df['Storm surge[m]'] = RVE_ALL(data,var=var_surge,periods=periods,distribution='GEV',method='default',threshold='default')
    df['Total water level[m]'] = df['Crest heigh[m]'] + df['Storm surge[m]'] + df['Tidal level(HAT)[m]']

    # Create a new dataframe with the results
    if output_file:
        df[['Return period [years]', 'Storm surge[m]','Tidal level(HAT)[m]','Crest heigh[m]','Total water level[m]']].round(2).to_csv(output_file,index=False)
    return df



def table_storm_surge_for_rv_hs(data: pd.DataFrame, var_hs='HS',var_tp='TP',var_surge='zeta_0m', var_tide='tide',depth=200, periods=[1,10,100,10000], output_file='table_storm_surge_for_rv_hs.csv'):
    df = table_Hmax_crest_return_periods(data,var_hs=var_hs, var_tp =var_tp,depth=depth, periods=periods, sea_state = 'short-crested', output_file=None)
    df['Tidal level(HAT)[m]'] = data[var_tide].max()
    df_S, df_coeff = table_storm_surge_for_given_hs(data, var_surge=var_surge,var_hs=var_hs, bin_width=1, max_hs=20, output_file=None)
    a_mean = df_coeff.loc[df_coeff['S(Hs)'] == 'Mean', 'a'].values[0]
    b_mean = df_coeff.loc[df_coeff['S(Hs)'] == 'Mean', 'b'].values[0]
    c_mean = df_coeff.loc[df_coeff['S(Hs)'] == 'Mean', 'c'].values[0]

    a_sigma = df_coeff.loc[df_coeff['S(Hs)'] == 'Std. Dev.', 'a'].values[0]
    b_sigma = df_coeff.loc[df_coeff['S(Hs)'] == 'Std. Dev.', 'b'].values[0]
    c_sigma = df_coeff.loc[df_coeff['S(Hs)'] == 'Std. Dev.', 'c'].values[0]

    df['S(Mean-model) [m]'] = S_as_function_of_Hs(df['Hs[m]'], a_mean, b_mean, c_mean)
    df['S(std-model) [m]'] = S_as_function_of_Hs(df['Hs[m]'], a_sigma, b_sigma, c_sigma)

    df['S(P5-model) [m]'] =  df['S(Mean-model) [m]'] - 1.65*df['S(std-model) [m]']
    df['S(P95-model) [m]'] = df['S(Mean-model) [m]'] + 1.65*df['S(std-model) [m]']

    df['Total water level[m]'] = df['Crest heigh[m]'] + df['S(P95-model) [m]'] + df['Tidal level(HAT)[m]']

    if output_file:
        df[['Return period [years]', 'Hs[m]','Crest heigh[m]','Tidal level(HAT)[m]','S(P5-model) [m]','S(Mean-model) [m]','S(P95-model) [m]','Total water level[m]']].round(2).to_csv(output_file,index=False)
    return df
