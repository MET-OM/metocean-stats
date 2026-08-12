import typing
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .jointmodel import JointProbabilityModel
from . import predefined

from ..utils import groupby_month, groupby_sector

def _get_n_axes(n_intervals, max_cols=4):
    if n_intervals <= 0:
        raise ValueError(f"n_intervals should be a positive integer, but got {n_intervals}.")
    elif n_intervals < max_cols**2:
        table = np.array(
            [
                (rows, cols)
                for cols in range(1, max_cols + 1)
                for rows in range(1, cols + 1)
            ]
        )
        table = table[np.prod(table, axis=1) >= n_intervals]
        (nrows, ncols) = table[np.argmin(np.prod(table, axis=1) - n_intervals)]
    else:
        nrows = np.ceil(n_intervals / max_cols).astype(int)
        ncols = max_cols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False)
    return fig, axes.ravel()


class BivariateEVA:
    model_omni:JointProbabilityModel
    models_monthly:dict[str,JointProbabilityModel]
    models_sectors:dict[str,JointProbabilityModel]

    def __init__(
        self,
        data:pd.DataFrame,
        var1:str,
        var2:str,
        var_dir:str,
        sectors=12,
        model=predefined.get_DNVGL_Hs_Tz
        ):
        """
        Initialize multivariate extreme value analysis module.
        """
        
        data = data[[var1,var2,var_dir]].copy()
        # data.index = pd.to_datetime(data.index)
        # data = data.sort_index()
        # bins = np.linspace(0, 360, sectors+1,dtype=int)
        # dir_offset = (bins[1]-bins[0])/2
        # labels = [f"{(bins[i]-dir_offset)%360:.0f}-{(bins[i+1]-dir_offset)%360:.0f}°" for i in range(sectors)]
        # data["sector"] = pd.cut((data[var_dir]+dir_offset)%360, bins=bins, labels=labels, right=False)

        # Group the data
        self.data_omni = data
        self.data_monthly = groupby_month(data)
        self.data_sectors = groupby_sector(
            data,
            var_dir=var_dir,
            sectors=sectors,
            )

        # Vars and predefined
        self.var1 = var1
        self.var2 = var2
        self.var_dir = var_dir
        self.model_description = model

    def fit(
            self,
            sector_errors:typing.Literal["ignore","raise"] = "ignore",
            month_errors:typing.Literal["ignore","raise"] = "ignore"):

        pbar = tqdm(total=25)

        pbar.set_description("Fitting Omni")
        model = JointProbabilityModel(self.model_description)
        model.fit(self.data_omni,self.var1,self.var2)
        self.model_omni = model
        pbar.update()

        # Extract semantics
        self.var1_symbol,self.var2_symbol = self.model_omni.semantics["symbols"]
        self.var1_name,self.var2_name = self.model_omni.semantics["names"]
        self.var1_unit,self.var2_unit = self.model_omni.semantics["units"]

        self.models_monthly = {}
        for k,g in self.data_monthly.items():
            pbar.set_description(f"Fitting {k}")
            model = JointProbabilityModel(self.model_description)
            try:
                model.fit(g,self.var1,self.var2)
            except Exception as e:
                if month_errors == "ignore":
                    warnings.warn(f"Could not fit month {k} - future results will omit this month.")
                else:
                    raise e
                
            self.models_monthly[k] = model
            pbar.update()

        self.models_sectors = {}
        for k,g in self.data_sectors.items():
            pbar.set_description(f"Fitting {k}")
            model = JointProbabilityModel(self.model_description)
            try: 
                model.fit(g,self.var1,self.var2)
            except Exception as e:
                if sector_errors == "ignore":
                    warnings.warn(f"Could not fit sector {k} - future results will skip this sector.")
                else:
                    raise e

            self.models_sectors[k] = model
            pbar.update()

    def _subplot_isodensity_contours(
            self,
            rve:pd.Series,
            ax:plt.Axes,
            model:JointProbabilityModel,
            **kwargs,
        ):
        
        if ax is None: fig,ax = plt.subplots()
        model.reset_labels()
        model.plot_isodensity_contours(ax,points=rve.values,labels=rve.index,**kwargs)
        model.plot_dependent_percentiles(ax)
        model.plot_data_density(ax)
        model.plot_DNVGL_steepness_criterion(ax)
        model.plot_legend(ax)
        model.reset_labels()
        return ax

    def plot_isodensity_contours(
            self,
            grouping:typing.Literal["omni","sectors","monthly"],
            RVE:pd.DataFrame,
            ax:plt.Axes|list[plt.Axes]=None,
            contour_kwargs:dict={},
            subplot_figures:int=2,
            subplot_columns:int=None,
            title_N_points=True
            ):

        contour_kwargs = {"cmap":"viridis"} | contour_kwargs

        if grouping == "omni":
            if ax is None:
                fig,ax = plt.subplots()
            elif not isinstance(ax,plt.axes):
                raise TypeError(f"Expected a single matplotlib.Axes object as ax, got {type(ax)}")

            if "Omni" in RVE.index: rve = RVE.loc["Omni"]
            elif "Yearly" in RVE.index: rve = RVE.loc["Yearly"]
            elif "Annual" in RVE.index: rve = RVE.loc["Annual"]
            else: raise ValueError("Could not find Omni, Annual or Yearly in the RVE index, please check.")

            self._subplot_isodensity_contours(rve=rve,ax=ax,model=self.model_omni,**contour_kwargs)
            ax.set_title("Omni")
            return ax

        elif grouping == "monthly":
            models = self.models_monthly
        elif grouping == "sectors":
            models = self.models_sectors
        else:
            raise ValueError(f"grouping should be one of [omni, sectors, monthly], got {grouping}")

        N_plots = len(models)
        if ax is None:
            if subplot_figures == 1:
                if subplot_columns is None: subplot_columns = 3
                fig,ax = _get_n_axes(N_plots,max_cols = subplot_columns)
                figures = np.array([fig])
            else:
                if subplot_columns is None: subplot_columns = 2
                base = N_plots // subplot_figures
                rem = N_plots % subplot_figures
                subplots = [base+1]*rem + [base]*(subplot_figures-rem)
                subplots = [_get_n_axes(n,max_cols=subplot_columns) for n in subplots]
                figures = np.array([s[0] for s in subplots])
                ax = np.concat([s[1] for s in subplots])
        else:
            ax = np.array(ax).ravel()
            if len(ax) < N_plots:
                raise ValueError(f"Expected ax to contain {N_plots} axes, got {len(ax)}.")

        for i,(key,model) in enumerate(models.items()):
            self._subplot_isodensity_contours(RVE.loc[key],ax=ax[i],model=model,**contour_kwargs)
            
            title = "Month:" if grouping == "monthly" else "Sector:"
            title += f" {key}"
            if title_N_points: title += f", N={len(model.data)}"
            ax[i].set_title(title)

        for fig in figures:
            fig.subplots_adjust(hspace = 0.2)

        return ax

    def table_isodensity_contours(
        self,
        grouping:typing.Literal["monthly","sectors"],
        RVE:pd.DataFrame,
        range_interval:float=0.5,
        replace_cols=True,
        **kwargs,
        ):
        """
        Obtain tables of iso-density contour values.

        Parameters
        -----------
        grouping : str
            Select "monthly" or "sectors" for which set of tables to produce.
        RVE : pd.DataFrame
            Dataframe of return values, with the index being sectors or months,
            and columns corresponding to different return periods.
        range_interval : float
            The interval between the table rows.
        replace_cols : bool
            Replace column names based on the return period columns of RVE.
        **kwargs : keyword arguments 
            These are passed to JointProbabilityModel.table_isodensity.contour()
        """

        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
        else:
            raise ValueError(f"models should be monthly or sectors, got {models}")
        
        tables = {}
        for subset,model in models.items():
            rve = RVE.loc[subset]
            table = model.table_isodensity_contour(
                points=rve,range_interval=range_interval,**kwargs)
            if replace_cols:
                table.index.name = f"${self.var1_symbol}$"
                if 2*RVE.shape[1] == table.shape[1]-1:
                    # Assume first col is index
                    table = table.set_index(table.columns[0])
                if 2*RVE.shape[1] != table.shape[1]:
                    raise ValueError(f"Expected {RVE.shape[1]} columns, "
                                     f"2 per return value, but got {table.shape[1]}")
                table.columns = [f"{rp.replace("year","yr")} ${self.var2_symbol}$$_{lh}$" for rp in RVE.columns for lh in ["L","H"]]
            tables[subset] = table

        return tables

    def table_conditional_return_values(
            self,
            grouping:typing.Literal["monthly","sectors"],
            RVE:pd.DataFrame,
    ):
        """
        Given a DataFrame of return values for the primary variable,
        return a DataFrame with corresponding conditional return values.
        """
        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
        else:
            raise ValueError(f"models should be monthly or sectors, got {models}")

        table = {}
        return_periods = RVE.columns
        for subset,model in models.items():
            main = RVE.loc[subset]
            cond = model.get_dependent_given_marginal(main)
            row = {}
            for rp,var1,var2 in zip(return_periods,main,cond):
                row[f"{rp.replace("year","yr")} ${self.var1_symbol}$"] = var1
                row[f"{rp.replace("year","yr")} ${self.var2_symbol}$"] = var2
            table[subset] = row
        return pd.DataFrame.from_dict(table,orient="index")


    def table_return_values(
            self,
            grouping:typing.Literal["monthly","sectors"],
            return_periods:list[float],
            state_duration:float = None,
            contour_method:str = "IFORM",
    ):
        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
        else:
            raise ValueError(f"models should be monthly or sectors, got {models}")

        data_interval = self.data_omni.sort_index().index.diff().mean().total_seconds()/3600

        if state_duration is None:
            state_duration = data_interval
        elif abs(state_duration-data_interval) > (1/6):
            warnings.warn("Average data time-interval is "
                          f"{data_interval:.2f}h, not {state_duration:.2f}h.")

        table = {}
        for subset,model in models.items():
            row = {}
            for rp in return_periods:
                hstp = model.get_contour_maximum(
                    rp,state_duration=state_duration,contour_method=contour_method)
                row[f"{rp}-year {self.var1_symbol}"] = hstp["hs"]
                row[f"{rp}-year {self.var2_symbol}"] = hstp["tp"]
            table[subset] = row

        return pd.DataFrame.from_dict(table,orient="index")

    def table_model_parameters(
            self,
            grouping:typing.Literal["monthly","sectors"],
    ):
        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
        else:
            raise ValueError(f"models should be monthly or sectors, got {models}")
        
        table = {k:m.parameters(False) for k,m in models.items()}
        return pd.DataFrame.from_dict(table,orient="index")

    def plot_marginal_quantiles(
            self,
            grouping:typing.Literal["omni","monthly","sectors"],
            axes:list[plt.Axes] = None,
            max_cols:int = 4,
            title_N_points=False,
    ):
        if grouping == "omni":
            if axes is None:
                fig,axes = plt.subplots(1,2)
            elif len(axes)<2:
                raise TypeError(f"Need at least two axes to plot on.")
            return self.model_omni.plot_marginal_quantiles(axes=axes)
        if grouping == "monthly":
            models = self.models_monthly
        elif grouping == "sectors":
            models = self.models_sectors
        else:
            raise ValueError(f"models should be monthly or sectors, got {models}")
        
        if axes is None:
            fig,axes = _get_n_axes(2*len(models),max_cols=max_cols)
        elif len(axes) < 2*len(models):
            raise ValueError(f"Need at least {2*len(models)} axes, got {len(axes)}.")
        
        for i,(k,m) in enumerate(models.items()):
            ax = axes[2*i:2*i+2]
            m.plot_marginal_quantiles(axes=ax)
            title = f"{k}"
            if title_N_points: 
                title += f", N={len(m.data)}"
            ax[0].set_title(title)
            ax[1].set_title(title)

        fig.subplots_adjust(hspace=0.25)
        return axes
    

    def plot_dependence_functions(
            self,
            grouping:typing.Literal["omni","monthly","sectors"],
            axes:list[plt.Axes] = None,
            max_cols:int = 4,
            title_N_points=False,
    ):
        if grouping == "omni":
            if axes is None:
                fig,axes = plt.subplots(1,2)
            elif len(axes)<2:
                raise ValueError(f"Need at least two axes to plot on.")
            return self.model_omni.plot_dependence_functions(axes=axes)
        if grouping == "monthly":
            models = self.models_monthly
        elif grouping == "sectors":
            models = self.models_sectors
        else:
            raise ValueError(f"models should be monthly or sectors, got {models}")
        
        if axes is None:
            fig,axes = _get_n_axes(2*len(models),max_cols=max_cols)
        elif len(axes) < 2*len(models):
            raise ValueError(f"Need at least {2*len(models)} axes, got {len(axes)}.")
        
        for i,(k,m) in enumerate(models.items()):
            ax = axes[2*i:2*i+2]
            m.plot_dependence_functions(axes=ax)
            title = f"{k}"
            if title_N_points: 
                title += f", N={len(m.data)}"
            ax[0].set_title(title)
            ax[1].set_title(title)

        fig.subplots_adjust(hspace=0.25)
        return axes
    
