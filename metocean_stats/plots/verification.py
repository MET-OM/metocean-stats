import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import scipy.stats as st
from matplotlib.patches import Patch


#from ..stats.general import *
from .. import stats
from ..tables.verification import *



def plot_scatter(df,var1,var2,var1_units='m', var2_units='m',title=' ',regression_line='effective-variance',qqplot=True,density=True,output_file='scatter_plot.png'):
    """
    Plots a scatter plot with optional density, regression line, and QQ plot. 
    Calculates and displays statistical metrics on the plot.
    
    Parameters:
        df (DataFrame): Pandas DataFrame containing the data.
        var1 (str): Column name for the x-axis variable.
        var2 (str): Column name for the y-axis variable.
        var1_units (str): Units for the x-axis variable. Default is 'm'.
        var2_units (str): Units for the y-axis variable. Default is 'm'.
        title (str): Title of the plot. Default is an empty string.
        regression_line (str or bool): Type of regression line ('least-squares', 'mean-slope', 'effective-variance',None). Default is effective-variance.
        qqplot (bool): Whether to include QQ plot. Default is True.
        density (bool): Whether to include density plot. Default is True.
        output_file (str): Filename for saving the plot. Default is 'scatter_plot.png'.
    
    Returns:
        fig: The matplotlib figure object.
    """

    x=df[var1].values
    y=df[var2].values
    fig, ax = plt.subplots()
    if density is False:
        ax.scatter(x,y,marker='.',s=10,c='g')
    else:
        plt.hist2d(x, y,bins=50, cmap='hot',cmin=1)
        plt.colorbar()

    dmin, dmax = np.min([x,y])*0.9, np.max([x,y])*1.05
    diag = np.linspace(dmin, dmax, 1000)
    plt.plot(diag, diag, color='r', linestyle='--')
    plt.gca().set_aspect('equal')
    plt.xlim([0,dmax])
    plt.ylim([0,dmax])
    
    if qqplot :    
        percs = np.linspace(0,100,101)
        qn_x = np.nanpercentile(x, percs)
        qn_y = np.nanpercentile(y, percs)    
        ax.scatter(qn_x,qn_y,marker='.',s=80,c='b')

    if regression_line == 'least-squares':
        slope=st.linregress(x,y).slope
        intercept=st.linregress(x,y).intercept
    elif regression_line == 'mean-slope':
        slope = np.mean(y)/np.mean(x)
        intercept = 0 
    elif regression_line == 'effective-variance':
        slope = stats.linfitef(x,y)[0] 
        intercept = stats.linfitef(x,y)[1] 
    else:
        slope = None
        intercept = None

    if slope is not None and intercept is not None:
        if intercept > 0:
            cm0 = f"$y = {slope:.2f}x + {intercept:.2f}$"
        elif intercept < 0:
            cm0 = f"$y = {slope:.2f}x - {abs(intercept):.2f}$"
        else:
            cm0 = f"$y = {slope:.2f}x$"
    
        plt.plot(x, slope * x + intercept, 'k--', label=cm0)
        plt.legend(loc='best')


    rmse = np.sqrt(((y - x) ** 2).mean())
    bias = np.mean(y-x)
    mae = np.mean(np.abs(y-x))
    corr = np.corrcoef(y,x)[0][1]
    si = np.std(x-y)/np.mean(x)

    plt.annotate('rmse = '+str(np.round(rmse,3))
                 +'\nbias = '+str(np.round(bias,3))
                 +'\nmae = '+str(np.round(mae,3))
                 +'\ncorr = '+str(np.round(corr,3))
                 +'\nsi = '+str(np.round(si,3)), xy=(dmin+1,0.6*(dmin+dmax)))
    plt.xlabel(var1+'['+var1_units+']', fontsize=15)
    plt.ylabel(var2+'['+var2_units+']', fontsize=15)

    plt.title("$"+(title +', N=%1.0f'%(np.count_nonzero(~np.isnan(x))))+"$",fontsize=15)
    plt.grid()
    if output_file != "": plt.savefig(output_file)
    return fig



def taylor_diagram(df,var_ref,var_comp,norm_std=True,output_file='Taylor_diagram.png'):
    """
    Plot a Taylor diagram
    df: dataframe with all timeseries
    var_ref: list of string with the name of the timeseries of reference
    var_comp: list of strings with the names of the timeseries to be compared with the reference
    norm_std: option to define normalized or non-normalized standard deviation

    Option 1: #[[A,3],[B,3],[C,3]]
    var_ref   = ['hs_sulaA','hs_sulaB','hs_sulaC'] 
    var_comp = ['hs_nora3']
    norm_std = True #can only run with this option

    Option 2 : #Originalen [[A,3],[A,4],[A,5]]
    var_ref   = ['hs_sulaA']
    var_comp = ['hs_nora3','hs_nora4','hs_nora5']
    norm_std = True/False #can run with both options

    Option 3 : #[[A,3],[B,4],[C,5]]
    var_ref   = ['hs_sulaA','hs_sulaB','hs_sulaC']
    var_comp = ['hs_nora3','hs_nora4','hs_nora5']
    norm_std = True #can only run with this option

    """

    def run_taylor(var_ref,var_comp,maxx,index,show,fig, ax):
        def correlation(var_ref,var_comp,max_std,radius):
            # Calculate the coordinates of the points x and y
            # Correlation coefficient between the reference and the other(s)
            ccf=np.zeros((len(var_comp)+1))
            ccf[0]=np.corrcoef(df[var_ref[0]].to_numpy(),df[var_ref[0]].to_numpy())[0,1] # Should be 1
            for i in range(len(var_comp)):
                ccf[i+1]=np.corrcoef(df[var_ref[0]].to_numpy(),df[var_comp[i]].to_numpy())[0,1]     

            # Coordinates of the lines for the correlation
            xbc1=np.arange(0.0,max_std+0.015,0.001)
            corr=np.array([0.2,0.4,0.6,0.8,0.9,0.95,0.99])
            ycr=np.zeros((len(corr),len(xbc1)))
            for r in range(len(corr)):
                for a in range(len(xbc1)):
                    ycr[r,a]=np.tan(np.acos(corr[r]))*xbc1[a]
                    d=np.sqrt(ycr[r,a]**2+xbc1[a]**2)
                    if d>np.max(radius):
                        ycr[r,a]=np.nan
                    del d
            return ccf,xbc1,ycr,corr

        def set_axes_and_std(var_ref,var_comp,maxx):
            std=np.zeros((len(var_comp)+1))
            std[0]=np.std(df[var_ref].to_numpy())
            for i in range(len(var_comp)):
                std[i+1]=np.std(df[var_comp[i]].to_numpy())
            if norm_std is True:
                std=std/std[0]

            # Coordinates of the big circles
            min_std=0
            max_std=maxx + 0.5 #to set the max of x-y

            if max_std<=5:
                step=0.5
            elif ((max_std>5) & (max_std<=10)):
                step=1
            elif ((max_std>10) & (max_std<=20)):
                step=3
            else:
                step=5

            radius=np.arange(min_std+step,max_std,step)
            radius=np.concatenate([radius,np.array([max_std])])
            radius1=radius
            xbc=np.arange(0.0,max_std+0.01,0.0001)
            ybc=np.zeros((len(radius),len(xbc)))
            ysc=np.zeros((len(radius),len(xbc)))
            for r in range(len(radius)):
                for a in range(len(xbc)):
                    ybc[r,a]=np.sqrt(radius[r]**2-xbc[a]**2)
                    ysc[r,a]=np.sqrt(radius1[r]**2-(xbc[a]-std[0])**2)
                    d=np.sqrt(ysc[r,a]**2+xbc[a]**2)
                    if d>np.max(radius):
                        ysc[r,a]=np.nan
                    del d

            return std,max_std,radius,xbc,ybc,step

        def plotting(var_ref,var_comp,maxx,index):
            #Plot the data
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            #Get the data
            std,max_std,radius,xbc,ybc,step=set_axes_and_std(var_ref,var_comp,maxx)
            ccf,xbc1,ycr,corr = correlation(var_ref,var_comp,max_std,radius)
            # Coordinates of the correlation labels
            corr1=np.array([0.0,0.2,0.4,0.6,0.8,0.9,0.95,0.99,1.0])
            text=[]
            xt=np.zeros((len(corr1)))
            yt=np.zeros((len(corr1)))
            #angle=np.zeros((len(corr1)))
            angle=np.array([0,-10,-24,-37,-50,-62,-71,-82,-90])
            for i in range(len(corr1)):
                text.append(str(corr1[i]))
                xt[i]=corr1[i]*(max_std+0.1)
                yt[i]=np.sin(np.acos(corr1[i]))*(max_std+0.1)
                #angle[i]=math.acos(corr1[i])*180.0/math.pi
                angle[i]=angle[i]
            #angle=0.0-angle[::-1]
            
            if show:
                for r in range(len(radius)):
                    ax.plot(xbc[:],ybc[r,:],'k',linewidth=1.0)
                for r in range(len(corr)):
                    ax.plot(xbc1[:],ycr[r,:],'k',linewidth=1.0,linestyle='dotted')

            rmsdif=np.sqrt(std**2+std[0]**2-2*std*std[0]*ccf)
            
            # Coordinates
            x=ccf*std
            y=np.sqrt((rmsdif**2)-((std[0]-x)**2))
            xa=np.linspace(0,max_std+1,1000)
            xx,yy=np.meshgrid(xa,xa)
            del xa
            zz=np.sqrt((xx-std[0])**2+yy**2)
            zz1=np.sqrt(xx**2+yy**2)
            zz=np.where(zz1>max_std,np.nan,zz)
            del zz1

            levels_crms=np.arange(step,np.nanmax(zz)-step,step) if step<1 else np.arange(step,np.nanmax(zz),step)
            CS = ax.contour(xx,yy,zz,colors='gray',linewidths=1.0,linestyles='--',levels=levels_crms)
            fmt = mticker.ScalarFormatter()
            fmt.create_dummy_axis()
            list_tup=[(std[0],yv) for yv in levels_crms]
            ax.clabel(CS, CS.levels, fmt=fmt, fontsize=14,inline=True,inline_spacing=-2,manual=list_tup)

            # Plot reference point in black
            ax.plot(std[0],0.0,'o',clip_on=False,color='k', markersize=14)
            ax.text(std[0]+0.03,0.01,'ref',fontsize=18)#, weight='bold',verticalalignment='center')
            # Plot reference circle
            ybc_r=np.sqrt(std[0]**2-xbc**2)
            ax.plot(xbc,ybc_r,'k',linewidth=2.0)

            # Plot the other points
            # List of potential markers
            list_mrk = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
            # List of potential colors
            list_col = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            combinations = [[x, y] for x in list_mrk for y in list_col] # First index is the marker, second the color
            combinations=combinations[0:index] if index>0 else combinations[0:len(var_comp)]

            if (index==0) or (index==1):
                legends.append(mlines.Line2D([], [], color='k', marker='o', linestyle='None',markersize=14, label=var_ref[0]))
            else:
                legends.append(mlines.Line2D([], [], color='k', marker='o', linestyle='None',markersize=14, label=var_ref[0]))
        
            if index==0:
                index = index+1
            for ri,rj in zip(range(1,len(var_comp)+1),range(index,len(var_comp)+index)): # Actual plotting of the data
                ax.plot(x[ri],y[ri],clip_on=False,marker=combinations[rj-1][0],color=combinations[rj-1][1],markersize=14,markerfacecolor=combinations[rj-1][1])
                legends.append(mlines.Line2D([],[],marker=combinations[rj-1][0],color=combinations[rj-1][1], linestyle='None',markersize=14, label=var_comp[ri-1]))

            if show:
                num_lines = len(legends)
                ncol=1 if num_lines < 7 else 2 if num_lines < 13 else 3
                Size='x-large' if ncol<3 else 'large' 
                plt.legend(prop=dict(size=Size),handles=legends,ncol=ncol,loc="upper right", fontsize="20",bbox_to_anchor=(1.1, 1.))

            ax.text(max_std*0.73, max_std*0.73, 'Correlation',ha="center", va="center", size=18, rotation=-45.)
            
            for i in range(len(corr1)):
                ax.plot(xt[i],yt[i],clip_on=False,marker='o',color='None',markersize=6,markerfacecolor='None')
                ax.text(xt[i],yt[i],text[i],ha="center", va="center", size=16, rotation=angle[i])
            
            if norm_std:
                ax.set_xlabel('Normalized standard deviation',fontsize=18)
            else:
                ax.set_xlabel('Standard deviation',fontsize=18)
            
            if show:
                plt.xlim(0,max_std+0.02)
                plt.ylim(0,max_std+0.02)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                #plt.show()
            return 

        plotting(var_ref,var_comp,maxx,index)
        return
    
    legends = []
    if (len(var_ref)<len(var_comp)) and (len(var_ref)==1): #for option 2
        fig, ax = plt.subplots(figsize=(10, 10))
        model_ref = df[var_ref[0]] 
        std_mod = df[model_ref.name].std()
        var=[*var_ref,*var_comp] 
        #to find the max of the variables to set the len of axis 
        maxx=int(np.max(df[var].std())/std_mod if norm_std else np.max(df[var].std())) #max value on the x-y axis
        show=True #Always true in this case
        index=0        
        var_ref = np.array(var_ref)
        run_taylor(var_ref,var_comp,maxx,index,show,fig, ax)

    elif (len(var_ref)>len(var_comp)) and (len(var_comp)==1): #for option 1
        if norm_std is not True:
            print('This option can only be run with normalized standard deviation as True.')
            return
        fig, ax = plt.subplots(figsize=(10, 10))
        var=[*var_ref,*var_comp]
        i_end = len(var_ref)
        minn = np.nanmin(df[var_ref].std())
        stdd_c = df[var_comp].std()
        maxx = int(np.max(stdd_c/minn))
        index = 0
        #loop over len of var_ref
        for i in range(len(var_ref)):
            var_ref1 = var_ref[i]
            var_comp1 = var_comp[0]
            model_ref = df[var_ref1]
            std_mod = df[model_ref.name].std()
            if i==i_end-1:
                show=True
                index = index+1
                run_taylor([var_ref1],[var_comp1],maxx,index,show,fig,ax)
            else:
                show=False
                index = index + 1
                run_taylor([var_ref1],[var_comp1],maxx,index,show,fig,ax)

    elif len(var_ref)==len(var_comp): #for option 3
        if norm_std is not True:
            print('This option can only be run with normalized standard deviation as True.')
            return
        fig, ax = plt.subplots(figsize=(10, 10))
        var=[*var_ref,*var_comp]
        i_end = len(var_ref)
        #to find the max value to set on the x-y axis
        std_max = []
        for i in range(len(var_ref)):
            std_m = (df[var_comp[i]].std()/df[var_ref[i]].std())
            std_max.append(std_m)
        maxx = np.nanmax(std_max)
        index = 0
        #loop over len of var_ref
        for i in range(len(var_ref)):
            var_ref1 = var_ref[i]
            var_comp1 = var_comp[i]
            model_ref = df[var_ref1]
            std_mod = df[model_ref.name].std()

            if i==i_end-1:
                index = index + 1
                show=True
                run_taylor([var_ref1],[var_comp1],maxx,index,show,fig,ax)
            else:
                index = index + 1 
                show=False
                run_taylor([var_ref1],[var_comp1],maxx,index,show,fig,ax)
            
    else:   
        print('The option you have sent in is invalid.')
    if output_file != "": plt.savefig(output_file,dpi=200,facecolor='white',bbox_inches='tight')
    
    plt.close()
    return fig


def plot_binned_error_metric(data,var_bin,var_bin_size,var_bin_unit,var_ref,var_comp,var_comp_unit,threshold_min=100,error_metric='bias',output_file='plot_binned_error_metric.png'):
    """
    Plots the error_metric between two or more datasets binned along one variable.

    Parameters
    ----------
    data: pd.DataFrame

    var_bin: string
        Name of the variable the data should be binned into. This is the variable that will be on the x-axis.
    var_bin_unit: string
        Uunit of var_bin
    var_ref: string
        Name of the reference variable
    var_comp: string or list of strings
        Name(s) of the variable(s) to be compared to the reference
    var_comp_unit: string
        Unit of var_comp (should be the same as fro var_ref)
    var_bin_size: float
        Bin size for var_bin
    threshold_min: integer
        If the number of data within the given bin is less than the threshold, this data is highlighted in the plot
    error_metric: string
        Error/statisctic to be calculated and plotted. The user decides which of ['rmse','bias','mae','corr','scatter_index'] should be calculated.
    output_file: string
        Name of the figure
    
    Returns
    -------
    Matplotlib figure
    If there is only one variable to compare to the reference, a bar plot will be output.
    If multiple variables are compared to the reference, a line plot will be output.

    Authors
    -------
    Written by teaje and clio-met
    """
    if error_metric in ['scatter_index','corr']:
        var_comp_unit=''

    fig, ax = plt.subplots(figsize=(8,6))

    if len(var_comp)==1:
        ds = table_binned_error_metric(data=data,var_bin=var_bin,var_bin_size=var_bin_size,var_ref=var_ref,var_comp=var_comp[0],threshold_min=0,error_metric=[error_metric],output_file='')
        bins = ds.index.to_numpy()
        hatch_symb = [None if nn>=threshold_min else '//' for nn in ds['nb_val']]
        ax.bar(bins,ds[error_metric],width=0.9*var_bin_size,bottom=[0]*len(ds[error_metric]),color='lightgray',hatch=hatch_symb)
        ax.hlines(y=0,xmin=0,xmax=100,colors='k',linestyles='-',linewidth=1)
        ax.set_ylabel(error_metric+' ['+var_comp_unit+']',fontsize=16)  # Depends on the metric plotted
        ax.set_xlabel(var_bin+' ['+var_bin_unit+']',fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(var_comp[0]+' vs '+var_ref,fontsize=16)
        ax.set_xlim(bins[0]-var_bin_size/2,bins[-1]+var_bin_size/2)
        legend_elements = [Patch(color='k',fill=False,hatch='//',label='#val<'+str(threshold_min))]
        ax.legend(handles=legend_elements,fontsize=14,frameon=False)
    elif len(var_comp)>1:
        linestyles=['-','--','.','-.']
        legend_elements=[]
        for i in range(len(var_comp)):
            ds = table_binned_error_metric(data=data,var_bin=var_bin,var_bin_size=var_bin_size,var_ref=var_ref,var_comp=var_comp[i],threshold_min=0,error_metric=[error_metric],output_file='')
            if i==0:
                bins = ds.index.to_numpy()
            ax.plot(bins,ds[error_metric],linestyle=linestyles[i],color='k',label=var_comp[i])
            marker_symb = ['o' if nn>=threshold_min else 'x' for nn in ds['nb_val']]
            for j in range(len(bins)):
                ax.scatter(bins[j],ds[error_metric].iloc[j],color='k',marker=marker_symb[j])
            del ds,marker_symb
            legend_elements.append(mlines.Line2D([0],[0],lw=1.5,color='k',linestyle=linestyles[i],label=var_comp[i]))
        ax.hlines(y=0,xmin=0,xmax=100,colors='k',linestyles='-',linewidth=1)
        legend_elements.append(mlines.Line2D([0],[0],marker='x',lw=0,color='k',label='#val<'+str(threshold_min)))
        ax.legend(handles=legend_elements,fontsize=14)
        ax.set_ylabel(error_metric+' ['+var_comp_unit+']',fontsize=16)  # Depends on the metric plotted
        ax.set_xlabel(var_bin+' ['+var_bin_unit+']',fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(str(var_comp)+' vs '+var_ref,fontsize=16)
        ax.set_xlim(bins[0]-var_bin_size/2,bins[-1]+var_bin_size/2)
        ax.grid()

    plt.tight_layout()

    if output_file!='':
        plt.savefig(output_file,dpi=100,facecolor='white',bbox_inches='tight')

    plt.close()
    return fig
