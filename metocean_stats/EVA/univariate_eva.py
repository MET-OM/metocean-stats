import copy
import warnings
import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pyextremes as pyex

from tqdm import tqdm

from ..utils import groupby_month, groupby_sector

def _dist_name_map(dist,abbreviate=True):
    """
    Mapping from scipy distribution to abbreviations, e.g. genpareto -> gp
    """
    mapping = {
        "expon": "EXP",         # Exponential
        "genextreme": "GEV",    # Generalised Extreme Value
        "genpareto": "GP",      # Generalised Pareto
        "gumbel_r": "GUM",      # Gumbel
        "weibull_min": "WEI",   # Weibull
        "exponweib": "EW",      # Exponentiated Weibull
        "lognorm": "LOGN"       # Log-Normal
    }

    if abbreviate: return mapping.get(dist,dist)
    reverse_mapping = {v:k for k,v in mapping.items()}
    return reverse_mapping.get(dist,dist)

def _dist_param_map(dist,params):
    """Mapping from pyextremes distribution parameter names to proper names."""
    default = {"c":"Shape","loc":"Location","scale":"Scale"}
    return [default.get(p,p) for p in params]

def _dist_fit_method(dist,extreme_method):
    """
    Return appropriate fitting method (MLE or MM) 
    for any combination of distributions and extremes.
    """
    if dist == "exponweib": return "MLE"
    if extreme_method == "IDM": return "MOM"
    if extreme_method == "POT" and dist =="weibull_min": return "Lmoments"
    if extreme_method == "POT" and dist == "expon": return "Lmoments"
    return "MLE"

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

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False, layout="constrained")
    return fig, axes.ravel()

class UnivariateEVA:
    data_omni:pd.Series
    data_monthly:dict[str,pd.Series]
    data_sectors:dict[str,pd.Series]
    models_omni:dict[str,dict[tuple[str,str],pyex.EVA]]
    models_monthly:dict[str,dict[tuple[str,str],pyex.EVA]]
    models_sectors:dict[str,dict[tuple[str,str],pyex.EVA]]
    
    def __init__(self,
                 data:pd.DataFrame,
                 var:str,
                 var_dir:str,
                 var_name:str,
                 var_symbol:str,
                 var_unit:str,
                 sectors:int=12,
                 dist_name_map:callable=_dist_name_map,
                 ):
        """
        Initialize extreme value analysis module.
        This module applies pyextremes.EVA to monthly and sectors.
        """

        # Sort into sectors
        if var_dir is not None:
            data = data[[var,var_dir]].copy()
        else:
            data = data[[var]].copy()
        # data.index = pd.to_datetime(data.index)
        # bins = np.linspace(0, 360, sectors+1,dtype=int)
        # dir_offset = (bins[1]-bins[0])/2
        # labels = [f"{(bins[i]-dir_offset)%360:.0f}-{(bins[i+1]-dir_offset)%360:.0f}°" for i in range(sectors)]
        # data["sector"] = pd.cut((data[var_dir]+dir_offset)%360, bins=bins, labels=labels, right=False)

        # Group the data
        self.data_omni = data[var]
        self.data_monthly = groupby_month(data,var=var)
        if var_dir is not None:
            self.data_sectors = groupby_sector(
                data,
                var_dir=var_dir,
                sectors=sectors,
                var=var
                )
        else:
            self.data_sectors = None

        # Set keys
        self.keys_sectors = list(self.data_sectors.keys()) if self.data_sectors is not None else None
        self.keys_monthly = list(self.data_monthly.keys())

        # State flags
        self._got_extremes = False
        self._fitted_models = False

        # Semantics
        self.var = var
        self.var_dir = var_dir
        self.var_name = var_name
        self.var_symbol = var_symbol
        self.var_unit = var_unit
        self.dist_name_map = dist_name_map

    def _poisson_correction(self,T,method = None):
        T = np.array(T)
        if method not in ["AM","BM"]: return T
        return 1/(1-np.exp(-(1/T)))

    def _check_sectors_available(self):
        if self.var_dir is None:
            raise ValueError(
                "Sector results are unavailable because var_dir was not provided on init."
            )

    def plot_threshold_diagnostics(
            self,
            dist=["genpareto"],
            alpha=0.95,
        ):
        """
        Plot diagnostics to evaluate threshold stability.
        """
        raise NotImplementedError()

    def get_extremes(
            self,
            th_omni:float=None,
            th_monthly:list[float]=None,
            th_sectors:list[float]=None,
            block_size="365.2425D",
            min_last_block = 0.9,
            r = "48h",
            extremes_type="high",
            errors="raise",
            th_percentile=0.98,
        ):
        """
        Get POT and AM extremes of all models.
        """

        if th_omni is None:
            th_omni = self.data_omni.quantile(th_percentile)
        if th_monthly is None:
            th_monthly = {k:v.quantile(th_percentile)
                          for k,v in self.data_monthly.items()}
        if self.data_sectors is not None and th_sectors is None:
            th_sectors = {k:v.quantile(th_percentile)
                          for k,v in self.data_sectors.items()}

        # if not len(th_monthly) < len(self.data_monthly):
        #     raise ValueError(f"Expected {len(self.data_monthly)} monthly thresholds, got {len(th_monthly)}.")
        if not isinstance(th_monthly,dict):
            th_monthly = {k:t for k,t in zip(self.keys_monthly,th_monthly)}

        if self.data_sectors is not None:
            if not len(th_sectors) == len(self.data_sectors):
                raise ValueError(f"Expected {len(self.data_sectors)} sector thresholds, got {len(th_sectors)}.")
            if not isinstance(th_sectors,dict):
                th_sectors = {k:t for k,t in zip(self.keys_sectors,th_sectors)}

        # Omni
        self.am_omni = pyex.get_extremes(
            self.data_omni,"BM",extremes_type,
            errors=errors,
            block_size=block_size,
            min_last_block=min_last_block,
            )
        self.pot_omni = pyex.get_extremes(
            self.data_omni,"POT",extremes_type,
            threshold=th_omni,r=r)

        # Sectors
        if self.data_sectors is not None:
            self.am_sectors = {k:pyex.get_extremes(
                self.data_sectors[k],"BM",extremes_type,
                errors=errors,
                block_size=block_size,
                min_last_block=min_last_block*
                (len(self.data_sectors[k])/len(self.data_omni)))
                for k in self.keys_sectors}
            self.pot_sectors = {k:pyex.get_extremes(
                self.data_sectors[k],"POT",extremes_type,
                threshold=th_sectors[k],r=r)
                for k in self.keys_sectors}
        else:
            self.am_sectors = None
            self.pot_sectors = None

        # Monthly
        self.am_monthly = {k:pyex.get_extremes(
            self.data_monthly[k],"BM",extremes_type,
            errors=errors,
            block_size=block_size,
            min_last_block=min_last_block/12)
            for k in self.keys_monthly}
        self.pot_monthly = {k:pyex.get_extremes(
            self.data_monthly[k],"POT",extremes_type,
            threshold=th_monthly[k],r=r)
            for k in self.keys_monthly}

        self.th_omni = th_omni
        self.th_monthly = th_monthly
        self.th_sectors = th_sectors
        self.extremes_type = extremes_type
        self.min_last_block = min_last_block
        self.block_size = block_size
        self.r = r
        self.N_years = len(self.am_omni)
        self._got_extremes = True

    def fit(self,
            AM_dist = [],
            POT_dist = [],
            IDM_dist = [],
            dist_fit_method:callable = _dist_fit_method,
            errors = "raise",
            ):
        """
        Fit all combinations of distributions, methods and data subsets.
        Distributions may be given as shorthand (2-3 letters) or scipy names.
        """

        if not self._got_extremes:
            raise ValueError("No extremes to fit on. You must first use .get_extremes().")

        AM_dist = [self.dist_name_map(d,abbreviate=False) for d in AM_dist]
        POT_dist = [self.dist_name_map(d,abbreviate=False) for d in POT_dist]
        IDM_dist = [self.dist_name_map(d,abbreviate=False) for d in IDM_dist]

        # # This part is unnecessary if pyextremes can handle IDM
        # data_interval = self.data_omni.index.diff().mean().total_seconds()/3600
        # if hours_per_entry is None:
        #     hours_per_entry = data_interval
        # elif isinstance(hours_per_entry,str):
        #     hours_per_entry = pd.to_timedelta(hours_per_entry).total_seconds()/3600
        # elif abs(hours_per_entry-data_interval) > (1/6):
        #     warnings.warn(f"Average data time-interval is {data_interval:.2f}h, not {hours_per_entry:.2f}h.")
        # else:
        #     hours_per_entry = pd.to_timedelta(hours_per_entry,"h")

        all_dist = [("AM",d) for d in AM_dist]+[("POT",d) for d in POT_dist]+[("IDM",d) for d in IDM_dist]
        if not len(all_dist): raise ValueError("No distributions to fit.")

        self.models_omni = {}
        self.models_sectors = {k:{} for k in self.keys_sectors} if self.data_sectors is not None else None
        self.models_monthly = {k:{} for k in self.keys_monthly}

        pbar = tqdm(all_dist)
        for (method,dist) in pbar:
            pbar.set_description(f"Fitting {dist} to {method}")
            # Omni
            model = pyex.EVA(self.data_omni)
            if method == "AM":
                model.set_extremes(self.am_omni,"BM",self.extremes_type,
                                   block_size=self.block_size,
                                   min_last_block=self.min_last_block)
                model.fit_model(dist_fit_method(dist,method),dist)
            if method == "POT":
                model.set_extremes(self.pot_omni,"POT",self.extremes_type,
                                   threshold=self.th_omni)
                model.fit_model(dist_fit_method(dist,method),dist)
            if method == "IDM":
                model.set_extremes(self.data_omni,"POT",self.extremes_type,
                                #    block_size=hours_per_entry
                                   )
                model.fit_model(dist_fit_method(dist,method),dist)
            self.models_omni[(method,dist)] = model

            # Sectors
            if self.data_sectors is not None:
                for sector,data in self.data_sectors.items():
                    try:
                        model = pyex.EVA(data)
                        if method == "AM":
                            model.set_extremes(self.am_sectors[sector],
                                            "BM",self.extremes_type,
                                                block_size=self.block_size,
                                                min_last_block=self.min_last_block)
                            model.fit_model(dist_fit_method(dist,method),dist)
                        if method == "POT":
                            model.set_extremes(self.pot_sectors[sector],
                                            "POT",self.extremes_type,
                                                threshold=self.th_sectors[sector])
                            model.fit_model(dist_fit_method(dist,method),dist)
                        if method == "IDM":
                            model.set_extremes(self.data_sectors[sector],
                                            "POT",self.extremes_type,
                                                # block_size=hours_per_entry
                                                )
                            model.fit_model(dist_fit_method(dist,method),dist)
                        self.models_sectors[sector][(method,dist)] = model

                    except Exception as e:
                        if errors == "raise":
                            raise
                        elif errors == "warn":
                            warnings.warn(f"Failed to fit {dist} ({method}) for sector '{sector}': {e}")

            # Monthly
            for month,data in self.data_monthly.items():
                try:
                    model = pyex.EVA(data)
                    if method == "AM":
                        model.set_extremes(self.am_monthly[month],
                                        "BM",self.extremes_type,
                                            block_size=self.block_size,
                                            min_last_block=self.min_last_block)
                        model.fit_model(dist_fit_method(dist,method),dist)
                    if method == "POT":
                        model.set_extremes(self.pot_monthly[month],
                                        "POT",self.extremes_type,
                                            threshold=self.th_monthly[month])
                        model.fit_model(dist_fit_method(dist,method),dist)
                    if method == "IDM":
                        model.set_extremes(self.data_monthly[month],
                                        "POT",self.extremes_type,
                                            # block_size=hours_per_entry
                                            )
                        model.fit_model(dist_fit_method(dist,method),dist)
                    self.models_monthly[month][(method,dist)] = model
                except Exception as e:
                    if errors == "raise":
                        raise
                    elif errors == "warn":
                        warnings.warn(f"Failed to fit {dist} ({method}) for sector '{sector}': {e}")

        self._fitted_models = True

    def _subplot_return_value_comparison(
            self,
            models:dict[tuple[str,str],pyex.EVA],
            return_periods:list[float],
            return_period_size:str="365.2425D",
            ax = None,
            scatter_kwargs_AM  = {},
            scatter_kwargs_POT = {},
            scatter_kwargs_IDM = {},
            plot_kwargs_AM  = {},
            plot_kwargs_POT = {},
            plot_kwargs_IDM = {},
            table_kwargs = {},
            legend_kwargs = {},
            ):
        """
        This plots a set of EVA distributions on a axes object.
        """

        # Input checks
        return_periods = np.array(return_periods)

        # Customization
        if ax is None: fig,ax = plt.subplots()
        scatter_kwargs_AM  = {"marker":"o","color":"black","facecolor":"none","s":30,"rasterized":True} | scatter_kwargs_AM
        scatter_kwargs_POT = {"marker":"x","color":"black","s":15,"rasterized":True} | scatter_kwargs_POT
        scatter_kwargs_IDM = {"marker":".","color":"black","s":15,"rasterized":True} | scatter_kwargs_IDM
        plot_kwargs_AM = {"cmap":"winter","linestyle":"-"}    | plot_kwargs_AM
        plot_kwargs_POT = {"cmap":"autumn","linestyle":"--"}  | plot_kwargs_POT
        plot_kwargs_IDM = {"cmap":"copper","linestyle":":"}   | plot_kwargs_IDM
        table_kwargs = {"include":True,"loc":"top","rp_index":True,"bbox":[0,1,1,0.07*len(return_periods)]} | table_kwargs
        legend_kwargs = {"standing":False} | legend_kwargs

        # Legend
        legend = []
        methods_list = [m[0] for m in models.keys()]
        N_dist_per_method = [methods_list.count(k) for k in ["AM","POT","IDM"]]
        if legend_kwargs.pop("standing",False):
            legend_fill = [0,0,0]
            legend_kwargs = {"loc":"upper left","ncols":1} | legend_kwargs
        else:
            legend_fill = [np.max(N_dist_per_method)-n if n else 0 for n in N_dist_per_method]
            legend_kwargs = {"loc":"lower right","ncols":np.count_nonzero(N_dist_per_method)} | legend_kwargs

        # Plotting
        smallest_extreme = np.inf
        plot_rp = np.power(10.,np.linspace(-1,np.log10(np.max(return_periods))*1.1,100))
        colors_am = plt.get_cmap(plot_kwargs_AM["cmap"])(np.linspace(0,0.9,N_dist_per_method[0]))
        colors_pot = plt.get_cmap(plot_kwargs_POT["cmap"])(np.linspace(0,0.9,N_dist_per_method[1]))
        colors_idm = plt.get_cmap(plot_kwargs_IDM["cmap"])(np.linspace(0,0.9,N_dist_per_method[2]))

        # RVEs
        models_AM = {k:v for k,v in models.items() if k[0]=="AM"}
        models_POT = {k:v for k,v in models.items() if k[0]=="POT"}
        models_IDM = {k:v for k,v in models.items() if k[0]=="IDM"}
        corrected_return_periods = self._poisson_correction(return_periods,method="AM")
        RVE = {}

        # Iterate AM
        for i,((method,dist),model) in enumerate(models_AM.items()):
            if i == 0:
                N_extremes_AM = len(model.extremes)
                empirical_extremes = pyex.get_return_periods(
                    model.data,model.extremes,"BM",
                    self.extremes_type,self.block_size)
                ax.scatter(empirical_extremes["return period"],
                        empirical_extremes[self.var],
                        **scatter_kwargs_AM)
                legend.append(f"AM, N={N_extremes_AM}")
                smallest_extreme = np.minimum(smallest_extreme,model.extremes.min())
                
            plot_rv = model.get_return_value(plot_rp,return_period_size)[0]
            ax.plot(plot_rp,plot_rv,c=colors_am[i],
                    linestyle=plot_kwargs_AM["linestyle"])
            RVE[(method,dist)] = model.get_return_value(corrected_return_periods,return_period_size)[0]
            legend.append(f"{self.dist_name_map(dist)}")

        for i in range(legend_fill[0]):
            ax.plot(np.nan,np.nan,"-",color="none",label="")
            legend.append("")
        
        # Iterate POT
        for i,((method,dist),model) in enumerate(models_POT.items()):
            if i == 0:
                N_extremes_POT = len(model.extremes)
                empirical_extremes = pyex.get_return_periods(
                    model.data,model.extremes,"POT",self.extremes_type)
                ax.scatter(empirical_extremes["return period"],
                        empirical_extremes[self.var],
                        **scatter_kwargs_POT)
                legend.append(f"POT, N={N_extremes_POT}")
                smallest_extreme = np.minimum(smallest_extreme,model.extremes.min())
                
            plot_rv = model.get_return_value(plot_rp,return_period_size)[0]
            ax.plot(plot_rp,plot_rv,c=colors_pot[i],
                    linestyle=plot_kwargs_POT["linestyle"])
            RVE[(method,dist)] = model.get_return_value(return_periods,return_period_size)[0]
            legend.append(f"{self.dist_name_map(dist)}")

        for i in range(legend_fill[1]):
            ax.plot(np.nan,np.nan,"-",color="none",label="")
            legend.append("")
        
        # Iterate IDM
        for i,((method,dist),model) in enumerate(models_IDM.items()):
            if i == 0:
                N_extremes_IDM = len(model.extremes)
                empirical_extremes = pyex.get_return_periods(
                    model.data,model.data,"POT",self.extremes_type)
                #empirical_extremes = empirical_extremes[empirical_extremes["return period"]>0.1]
                ax.scatter(empirical_extremes["return period"],
                        empirical_extremes[self.var],
                        **scatter_kwargs_IDM)
                legend.append(f"IDM, N={N_extremes_IDM}")
                if smallest_extreme == np.inf: # only use idm if no pot or am available
                    smallest_extreme = model.extremes.min()
                
            plot_rv = model.get_return_value(plot_rp,return_period_size)[0]
            ax.plot(plot_rp,plot_rv,c=colors_idm[i],
                    linestyle=plot_kwargs_IDM["linestyle"])
            RVE[(method,dist)] = model.get_return_value(return_periods,return_period_size)[0]
            legend.append(f"{self.dist_name_map(dist)}")

        for i in range(legend_fill[2]):
            ax.plot(np.nan,np.nan,"-",color="none",label="")
            legend.append("")

        # Ticks and grid
        ax.semilogx()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True,which="both")
        ax.grid(True,which="minor",linestyle="--")

        # Axis limits
        ax.set_xlim([0.9*np.min(return_periods),1.1*np.max(return_periods)])
        ax.set_ylim([smallest_extreme,ax.get_ylim()[1]])

        # Labels and legend
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel(f"{self.var_name}, {self.var_symbol} ({self.var_unit})")
        ax.legend(legend,**legend_kwargs)

        # Return values table
        if table_kwargs.pop("include",False):
            RVE = pd.DataFrame(RVE,index=[f"{rp}-year" for rp in return_periods])
            RVE.columns = [f"{self.dist_name_map(c[1],True)} ({c[0]})" for c in RVE.columns]
            def strformat(cell): return f"{cell:.1f}"
            RVE = RVE.map(strformat).replace("-inf","-")
            rp_index = table_kwargs.pop("rp_index")
            RVE = RVE if rp_index else RVE.T
            ax.table(RVE,**table_kwargs)

        return ax

    def plot_return_value_comparison(
            self,
            grouping:typing.Literal["omni","sectors","monthly"],
            return_periods:list[float],
            ax: plt.Axes | list[plt.Axes] = None,
            subplot_columns = None,
            subplot_figures = 1,
            **kwargs,
            ):
        """
        Parameters
        ----------
        grouping : str or dict of EVA models
            Str ("omni", "monthly", or "sectors").
        return_periods : list[float]
            List of return periods, in years.
        ax : matplotlib axes or list of axes
            Axes to plot on. Must correspond to the number of models.
        subplot_columns : int
            Only used if axes are not provided. Sets ncols in subplots.
        subplot_figures : int
            Only used if axes are not provided. Option to divide plots over several figures.
        kwargs : keyword arguments passed to the return value plots:
         - include_table (bool): Include a table of return values above the plot.
         - scatter_kwargs_AM (dict): Keyword arguments passed to AM scatter.
         - scatter_kwargs_POT (dict): Keyword arguments passed to POT scatter.
         - scatter_kwargs_IDM (dict): Keyword arguments passed to IDM scatter.
         - plot_kwargs_AM (dict): Keyword arguments passed to AM-based distributions.
         - plot_kwargs_POT (dict): Keyword arguments passed to POT-based distributions.
         - plot_kwargs_IDM (dict): Keyword arguments passed to IDM-based distributions.
         - table_kwargs (dict): Keyword arguments passed to matplotlib Table (if used).
         - legend_kwargs (dict): Keyword arguments passed to matplotlib legend.
        """
        if not self._fitted_models:
            raise ValueError("Models not fitted. Run .fit() first.")

        if grouping == "sectors":
            self._check_sectors_available()

        # Check if table is included, and its size, to correcly place title
        titlepos = 1 + 0.07*len(return_periods)
        if "table_kwargs" in kwargs:
            if not kwargs["table_kwargs"].get("include",True):
                titlepos = 1
            elif "bbox" in kwargs["table_kwargs"]:
                titlepos = 1 + kwargs["table_kwargs"]["bbox"][3]

        if grouping == "omni":
            if ax is None:
                fig,ax = plt.subplots()
            elif not isinstance(ax,plt.Axes):
                raise TypeError(f"Expected a single matplotlib.Axes object as ax, got {type(ax)}")
            self._subplot_return_value_comparison(self.models_omni,return_periods,ax=ax,**kwargs)
            ax.set_title("Omni",y=titlepos)
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
        for i,(subset,subset_models) in enumerate(models.items()):
            self._subplot_return_value_comparison(subset_models,return_periods,**kwargs,ax=ax[i])
            ax[i].set_title(f"{subset}",y=titlepos)

        # for fig in figures:
        #     fig.subplots_adjust(hspace = 0.6, wspace=0.3)

        return ax

    def _subplot_return_value_confidence(
            self,
            method:typing.Literal["AM","POT","IDM"],
            model:pyex.EVA,
            return_periods:list[float],
            return_period_size:str="365.2425D",
            alphas:float|list[float]=[0.99,0.9],
            samples=300,
            cmap="viridis",
            table_scale = 0.07,
            table_flip = False,
            ax=None):
        """
        Plot return value confidence intervals based on bootstrapping.
        """
        if not hasattr(alphas,"__len__"): alphas = [alphas]
        alphas = np.sort(alphas)
        corrected_rp = self._poisson_correction(return_periods,method)

        # Semantics
        if ax is None: fig,ax = plt.subplots()
        plot_rp = np.power(10.,np.linspace(-1,np.log10(np.max(corrected_rp))*1.1,100))
        if not isinstance(cmap,mcolors.Colormap): cmap = plt.get_cmap(cmap)
        colors = cmap(np.linspace(0,1,len(alphas)+1))

        # Plotting
        table = {}
        legend = [f"{method}","Fitted"]
        for i,alpha in enumerate(alphas):
            rv,cl,cu = model.get_return_value(plot_rp,alpha=alpha,n_samples=samples,
                                              return_period_size=return_period_size)
            if i == 0:
                extremes_method = "BM" if method=="AM" else method
                extremes = pyex.get_return_periods(
                    model.data,model.extremes,extremes_method,self.extremes_type)
                ax.scatter(extremes["return period"], extremes[self.var],s=10,c="black")
                ax.plot(plot_rp,rv,c=colors[0])

            ax.plot(plot_rp,cu,c=colors[i+1])
            ax.plot(plot_rp,cl,c=colors[i+1],label="_nolegend_")
            
            rv,cl,cu = model.get_return_value(corrected_rp,alpha=alpha,n_samples=samples,
                                              return_period_size=return_period_size)
            exceedance = 1-alpha
            table[(100*(exceedance)/2)] = cl
            table[(100*(alpha+exceedance/2))] = cu
            if i == 0: table[50] = rv
            legend.append(f"{100*alpha:.0f}% CI")

        # Ticks and grid
        ax.semilogx()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True,which="both")
        ax.grid(True,which="minor",linestyle="--")

        ax.set_xlim([0.9*np.min(return_periods),1.1*np.max(return_periods)])
        ax.set_ylim([model.extremes.min(),ax.get_ylim()[1]])

        ax.set_xlabel("Return period (years)")
        ax.set_ylabel(f"{self.var_name}, {self.var_symbol} ({self.var_unit})")
        ax.legend(legend)

        table = pd.DataFrame(table,index=[f"{rp}-year" for rp in return_periods])
        table = table.sort_index(axis=1)
        table.columns = [f"{p:.1f}%" if abs(p-50)>0.001 else "Fitted" for p in table.columns]
        if table_flip: table = table.T
        def float_format(float): return f"{float:.1f}"
        table = table.map(float_format)
        if table_scale: ax.table(table,loc="top",bbox=[0,1,1,table_scale*table.shape[0]])

        return ax

    def plot_return_value_confidence(
            self,
            grouping:typing.Literal["omni","sectors","monthly"],
            method:str,
            dist:str,
            return_periods:list[float],
            ax:plt.Axes | list[plt.Axes] = None,
            subplot_columns = None,
            subplot_figures = 1,
            alphas = [0.99,0.9],
            cmap = "viridis",
            table_scale = 0.07,
            table_flip = False,
            samples = 200,
            ):

        if not self._fitted_models:
            raise ValueError("Models not fitted. Run .fit() first.")

        if grouping == "sectors":
            self._check_sectors_available()

        dist = _dist_name_map(dist,False)
        if table_flip: titlepos = 1 + table_scale*(1+2*len(alphas))
        else: titlepos = 1 + table_scale*len(return_periods)

        if grouping == "omni":
            if ax is None:
                fig,ax = plt.subplots()
            elif not isinstance(ax,plt.Axes):
                raise TypeError(f"Expected a single matplotlib.Axes object as ax, got {type(ax)}")
            model = self.models_omni[(method,dist)]
            self._subplot_return_value_confidence(
                method,model,return_periods,alphas,samples,cmap,table_scale,table_flip,ax=ax)
            ax.set_title("Omni",y=titlepos)
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

        for i,(subset,subset_models) in enumerate(models.items()):
            model = subset_models[(method,dist)]
            self._subplot_return_value_confidence(
                method,model,return_periods,alphas,samples,cmap,table_scale,table_flip,ax=ax[i])
            ax[i].set_title(f"{subset}",y=titlepos)

        for fig in figures:
            fig.subplots_adjust(hspace = 0.6, wspace=0.3)

        return ax


    def table_model_parameters(
            self,
            grouping:typing.Literal["monthly","sectors"],
            method:str,
            dist:str,
            ):
        """
        Return table of monthly/sector fitted distribution parameters.

        Parameters
        ------------
        subset : str
            One of [omni, monthly, sectors].

        Notes
        -------
        Columns will use the default names from pyextremes, such as (a, c, loc, scale).
        """
        if not self._fitted_models:
            raise ValueError("Models not fitted. Run .fit() first.")

        if grouping == "sectors":
            self._check_sectors_available()

        if grouping == "monthly": 
            models = self.models_monthly | {"Yearly":self.models_omni}
            thresholds = self.th_monthly | {"Yearly":self.th_omni} 
        elif grouping == "sectors": 
            models = self.models_sectors | {"Omni":self.models_omni}
            thresholds = self.th_sectors | {"Omni":self.th_omni}
        else: raise ValueError(f"grouping should be one of [monthly,sectors], got {grouping}")
        
        table = {}
        for subset,subset_models in models.items():
            model = subset_models[(method,self.dist_name_map(dist,False))]
            if method == "POT": entries = {"Peaks per year":len(model.extremes) / self.N_years}
            if method == "IDM": entries = {"Entries per year":len(model.extremes) / self.N_years}
            if method == "AM":  entries = {}
            param = model.distribution.mle_parameters | model.distribution.fixed_parameters
            threshold = {"Threshold":thresholds[subset]} if method == "POT" else {}
            table[subset] = entries | param | threshold

        table = pd.DataFrame.from_dict(table,orient="index")
        table.columns = _dist_param_map(dist,table.columns)
        return table

    def table_return_values_final(
            self,
            grouping:typing.Literal["monthly","sectors"],
            method:str,
            dist:str,
            return_periods:list[float],
            return_period_size:str="365.2425D"
            ):
        """
        Table of return values for a given distribution.
        """
        
        if not self._fitted_models:
            raise ValueError("Models not fitted. Run .fit() first.")

        if grouping == "sectors":
            self._check_sectors_available()

        if grouping == "omni":
            models = {"Omni":self.models_omni}
        elif grouping == "monthly": 
            models = self.models_monthly | {"Yearly":self.models_omni}
        elif grouping == "sectors": 
            models = self.models_sectors | {"Omni":self.models_omni}
        else: raise ValueError(f"grouping should be one of [omni,monthly,sectors], got {grouping}")

        RVE = {}
        for subset,subset_models in models.items():
            model = subset_models[(method,self.dist_name_map(dist,False))]
            RVE[subset] = {f"{rp}-year":model.get_return_value(
                self._poisson_correction(rp,method),
                return_period_size=return_period_size)[0]
                for rp in return_periods}
        
        RVE = pd.DataFrame.from_dict(RVE,orient="index")

        if grouping == "monthly":
            self.rve_monthly = RVE
            self.rve_monthly_model = (method,dist)
        elif grouping == "sectors":
            self.rve_sectors = RVE
            self.rve_sectors_model = (method,dist)

        return RVE
    
    def plot_return_values_final(
            self,
            grouping:typing.Literal["monthly","sectors"],
            method:str,
            dist:str,
            return_periods:list[float],
            plot_kwargs = {},
            table_kwargs = {},
            ):
        """
        Plot of return values for a given distribution.
        """
        if grouping == "sectors":
            self._check_sectors_available()

        table = self.table_return_values_final(grouping,method,dist,return_periods)
        if grouping == "monthly": N = 12
        else: N = len(self.models_sectors)
        table = table[:N]

        plot_kwargs = {"marker":"o","colormap":"viridis","xticks":np.arange(N)} | plot_kwargs
        table_kwargs = {"include":bool(table_kwargs),"loc":"top","float_format":".1f"} | table_kwargs

        ax = table.plot.line(**plot_kwargs)
        if table_kwargs.pop("include"):
            fmt = table_kwargs.pop("float_format")
            def float_format(f): return f"{f:{fmt}}"
            ax.table(table.round(1).T.map(float_format),**table_kwargs)

        ax.set_xlabel("Month" if grouping=="monthly" else "Sector")
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.grid(True)
        ax.set_ylabel(f"{self.var_name}, {self.var_symbol} ({self.var_unit})")