from typing import Dict, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..stats.aux_funcs import *
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

def table_monthly_return_periods(data, var='hs', periods=[1, 10, 100, 10000],distribution='Weibull',method='default', units='m',output_file='monthly_extremes_weibull.csv'):
    months = ['-','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Year']
    params, return_periods = monthly_extremes(data=data, var=var, periods=periods, distribution=distribution, method=method)    

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
    return_periods = return_periods.T.tolist()
    # Fill in return periods for each period
    for i, period in enumerate(periods):
        table_data[f'Return period: {period} [years]'] = [units] + [round(x, 2) for x in return_periods[i]]
    # Create DataFrame
    df = pd.DataFrame(table_data)
    if output_file:
        df.to_csv(output_file)
    
    return df

def table_directional_return_periods(data: pd.DataFrame, var='hs', var_dir='dir', periods=[1, 10, 100, 10000], distribution='Weibull', units='m',adjustment='NORSOK',output_file='directional_extremes_weibull.csv'):
    params, return_periods, sector_prob = directional_extremes(data=data, var=var, var_dir=var_dir, periods=periods,distribution=distribution, adjustment=adjustment)    

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
    
def table_directional_joint_distribution_Hs_Tp_return_values(data,var1='hs',var2='tp',var_dir='pdir',periods=[1,10,100,10000],adjustment='NORSOK', output_file='directional_Hs_Tp_joint_reurn_values.csv'):
    weibull_params, return_periods, sector_prob = directional_extremes(data=data, var=var1, var_dir=var_dir, periods=periods,distribution='Weibull', adjustment=adjustment)

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
        a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=sector_data,var1=var1,var2=var2,periods=periods,adjustment=adjustment)
        for i in range(len(periods)):
            rv_hs[k-1,i] = round(hs_tpl_tph['hs_'+str(periods[i])].max(),2)
            rv_tp[k-1,i] = round(hs_tpl_tph['t2_'+str(periods[i])].where(hs_tpl_tph['hs_'+str(periods[i])]==hs_tpl_tph['hs_'+str(periods[i])].max()).max(),2)

    #append annual
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph  =  joint_distribution_Hs_Tp(data=data,var1=var1,var2=var2,periods=periods,adjustment=None)
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
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var1=var_hs,var2=var_tp,periods=[1000])
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
    a1, a2, a3, b1, b2, b3, pdf_Hs, h, t3,h3,X,hs_tpl_tph = joint_distribution_Hs_Tp(data=data,var1=var_hs,var2=var_tp,periods=periods)
    # Create bins
    rv_hs = np.zeros(len(periods))
    result = []
    for i in range(len(periods)):
        rv_hs = hs_tpl_tph['hs_'+str(periods[i])].max().round(2)
        P5_model,Mean_model,P95_model = model_tp_given_hs(hs=rv_hs, a1=a1, a2=a2, a3=a3, b1=b1, b2=b2, b3=b3)
        result.append([periods[i],rv_hs,P5_model,Mean_model,P95_model])


    # Create a new dataframe with the results
    result_df = pd.DataFrame(result, columns=['Return period [years]', 'Hs[m]','Tp(P5-model) [s]','Tp(Mean-model) [s]','Tp(P95-model) [s]'])
    #result_df = result_df.dropna()
    if output_file:
        result_df.round(2).to_csv(output_file,index=False)

    return result_df


def table_tp_for_given_wind(data: pd.DataFrame, var_hs: str,var_wind: str, bin_width=2, max_wind=40, output_file='table_perc_tp_for_wind.csv'):
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


def table_profile_return_values(data,var=['W10','W50','W80','W100','W150'], heights=[10, 50, 80, 100, 150], period=[1, 10, 100, 10000], file_out='RVE_wind_profile.csv'):
    df_wind_profile = pd.DataFrame()
    df_wind_profile['Height above MSL']=heights
    for p in period:
        df_wind_profile[str(p)+'-year Return'] = [RVE_Weibull(data,var1,period=p) for var1 in var]
        
    df_wind_profile.to_csv(file_out, index=False)  
    
    return df_wind_profile



def table_wave_crest_return_periods(ds,var_hs='HS', var_tp = 'TP',depth=200, periods=[1, 10, 100,10000], sea_state = 'short-crested', output_file='table_Hmax_crest_return_values.csv'):
    df = table_tp_for_rv_hs(ds, var_hs, var_tp,periods=periods,output_file=None)
    time_step = ((ds.index[-1]-ds.index[0]).days + 1)*24/ds.shape[0]
    df['T_Hmax(P5-model) [s]'] =  0.9 * df['Tp(P5-model) [s]'] # according to Goda (1988)
    df['T_Hmax(Mean-model) [s]'] =  0.9 * df['Tp(Mean-model) [s]'] # according to Goda (1988)
    df['T_Hmax(P95-model) [s]'] =  0.9 * df['Tp(P95-model) [s]'] # according to Goda (1988)
    df['Crest height[m]'] = np.nan
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
