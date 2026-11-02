"""
Bivariate extreme value analysis with joint probability models.

`BivariateEVA` fits a joint model to two variables for all data, for each
calendar month and, if a direction variable is given, for each directional
sector. It then tabulates return values and plots isodensity contours.
"""

import typing
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .jointmodel import JointProbabilityModel
from . import predefined

from ..utils import groupby_with_metadata, get_n_axes, resolve_axes_grid


class BivariateEVA:
    """
    Joint extreme value analysis of two variables.

    Fits a joint probability model to all data, to each calendar month and,
    if a direction variable is given, to each directional sector. Return
    values are then tabulated and plotted as isodensity contours.

    Typical workflow: `fit`, then `table_return_values`, then the table and
    plot methods.
    """

    #: Full time series of the two variables, plus the direction if given.
    data_omni:pd.DataFrame
    #: Monthly subsets, keyed by month label.
    data_monthly:dict[str,pd.DataFrame]
    #: Directional sector subsets, keyed by sector label. Empty if no
    #: direction variable was given.
    data_sectors:dict[str,pd.DataFrame]
    #: Joint model fitted to all data.
    model_omni:JointProbabilityModel
    #: Joint models keyed by month label. Months that failed to fit are
    #: absent.
    models_monthly:dict[str,JointProbabilityModel]
    #: Joint models keyed by sector label. Sectors that failed to fit are
    #: absent.
    models_sectors:dict[str,JointProbabilityModel]

    def __init__(
        self,
        data:pd.DataFrame,
        var1:str,
        var2:str,
        var_dir:str=None,
        sectors=12,
        model=predefined.get_DNVGL_Hs_Tz
        ):
        """
        Group the data into omni, monthly and sector subsets.

        Parameters
        ----------
        data : pandas.DataFrame
            Data indexed by a datetime-compatible index. Must contain `var1`,
            `var2`, and `var_dir` if given.
        var1, var2 : str
            Column names of the two variables in the joint model.
        var_dir : str, optional
            Column name of a directional variable in degrees (0-360). If
            None, sector-based analysis is unavailable; omni and monthly
            analysis are unaffected.
        sectors : int, default 12
            Number of directional sectors. Ignored if `var_dir` is None.
        model : callable, default predefined.get_DNVGL_Hs_Tz
            Model description passed to `JointProbabilityModel`.
        """

        # Direction is optional - sector-based analysis is only
        # available when a direction column is supplied.
        self._has_direction = var_dir is not None

        cols = [var1,var2] + ([var_dir] if self._has_direction else [])
        data = data[cols].copy()
        # data.index = pd.to_datetime(data.index)
        # data = data.sort_index()
        # bins = np.linspace(0, 360, sectors+1,dtype=int)
        # dir_offset = (bins[1]-bins[0])/2
        # labels = [f"{(bins[i]-dir_offset)%360:.0f}-{(bins[i+1]-dir_offset)%360:.0f}°" for i in range(sectors)]
        # data["sector"] = pd.cut((data[var_dir]+dir_offset)%360, bins=bins, labels=labels, right=False)

        # Group the data. observed=True: a month/sector with zero
        # observations can't fit a joint model, so it's dropped rather
        # than kept as an empty group.
        self.data_omni = data
        month_meta = groupby_with_metadata(data, by="month", observed=True)
        self.data_monthly = dict(zip(month_meta["label"], month_meta["data"]))
        if self._has_direction:
            sector_meta = groupby_with_metadata(
                data, by="direction", var_dir=var_dir, sectors=sectors, observed=True,
                )
            self.data_sectors = dict(zip(sector_meta["label"], sector_meta["data"]))
        else:
            self.data_sectors = {}

        # Vars and predefined
        self.var1 = var1
        self.var2 = var2
        self.var_dir = var_dir
        self.model_description = model

    def _require_direction(self,method_name:str) -> None:
        """Raise a clear error if direction was not provided at initialisation."""
        if not self._has_direction:
            raise AttributeError(
                f"{method_name} requires a direction variable (var_dir), "
                "but none was provided when this BivariateEVA was created. "
                "Sector-based analysis is unavailable; 'omni' and 'monthly' "
                "groupings remain fully available."
            )

    def _rve_var1_columns(self,RVE:pd.DataFrame) -> tuple[list,list]:
        """
        Pick out the marginal (var1) return-value columns of an RVE table,
        and the return-period label (e.g. "10-year") for each one.

        `table_return_values` lays out columns as one var1 column and one
        var2 column per return period. Methods that condition on var1's
        return value (isodensity contours, dependent percentiles) must use
        only the var1 half; mixing in the var2 columns would treat a Tp
        return value as if it were an Hs point.
        """
        suffix = f" {self.var1_symbol}"
        var1_cols = [c for c in RVE.columns if c.endswith(suffix)]
        if not var1_cols:
            raise ValueError(
                f"Could not find any '{self.var1_symbol}' columns in RVE. "
                f"Available columns: {list(RVE.columns)}."
            )
        rp_labels = [c[:-len(suffix)] for c in var1_cols]
        return var1_cols, rp_labels

    def fit(
            self,
            sector_errors:typing.Literal["ignore","raise"] = "ignore",
            month_errors:typing.Literal["ignore","raise"] = "ignore"):
        """
        Fit the joint model to the omni, monthly and sector subsets.

        Parameters
        ----------
        sector_errors : {"ignore", "raise"}, default "ignore"
            With "ignore", a failed sector fit is reported as a warning and the
            sector is skipped. With "raise", the error is re-raised.
        month_errors : {"ignore", "raise"}, default "ignore"
            As `sector_errors`, applied to monthly fits.

        Notes
        -----
        The omni fit is not wrapped and always raises on failure. Sector
        models are skipped entirely, leaving `models_sectors` as an empty dict,
        when no direction variable was given.
        """

        n_sectors = len(self.data_sectors)
        pbar = tqdm(total=1+len(self.data_monthly)+n_sectors)

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
                self.models_monthly[k] = model
            except Exception as e:
                if month_errors == "ignore":
                    warnings.warn(f"Could not fit month {k} - future results will skip the model for this month.")
                else:
                    raise e
            pbar.update()

        self.models_sectors = {}
        if self._has_direction:
            for k,g in self.data_sectors.items():
                pbar.set_description(f"Fitting {k}")
                model = JointProbabilityModel(self.model_description)
                try:
                    model.fit(g,self.var1,self.var2)
                    self.models_sectors[k] = model
                except Exception as e:
                    if sector_errors == "ignore":
                        warnings.warn(f"Could not fit sector {k} - future results will skip the model for this sector.")
                    else:
                        raise e
                pbar.update()

    def _subplot_isodensity_contours(
            self,
            rve:pd.Series,
            ax:plt.Axes,
            model:JointProbabilityModel,
            data:pd.DataFrame = None,
            **kwargs,
        ):
        """Draw isodensity contours for one subset, or its data if the model is missing."""
        if ax is None: fig,ax = plt.subplots()

        if model is None:
            if data is not None:
                ax.scatter(data[self.var2], data[self.var1], s=4, alpha=0.4, color="grey")
                ax.set_xlabel(self.var2)
                ax.set_ylabel(self.var1)
            return ax

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
            nfigs:int=2,
            ncols:int=2,
            title_N_points=True
            ):
        """
        Plot isodensity contours of the return values per subset.

        Parameters
        ----------
        grouping : {"omni", "monthly", "sectors"}
            Subsets to plot. "sectors" requires a direction variable.
        RVE : pandas.DataFrame
            Return values from `table_return_values`, one row per subset.
        ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes, optional
            Axes to draw on. "omni" needs a single axes. The other groupings
            need one axes per subset. Created if None.
        contour_kwargs : dict, optional
            Passed to the model's contour plot. Defaults to
            ``{"cmap": "viridis"}``.
        nfigs : int, default 2
            Number of figures to split the subset panels over, used only if
            `ax` is None.
        ncols : int, default 2
            Number of subplot columns, used only if `ax` is None.
        title_N_points : bool, default True
            Add the number of observations to the subplot titles.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
            A single axes for "omni", otherwise the axes array.

        Notes
        -----
        The omni row of `RVE` is looked up as "Omni", "Yearly" or "Annual",
        in that order. Subsets whose model failed to fit are plotted with
        their data only and marked "(failed to fit)".
        """
        if grouping == "sectors":
            self._require_direction("plot_isodensity_contours(grouping='sectors')")

        contour_kwargs = {"cmap":"viridis"} | contour_kwargs

        if grouping == "omni":
            if ax is None:
                fig,ax = plt.subplots()
            elif not isinstance(ax,plt.Axes):
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
            all_groups = self.data_monthly
        elif grouping == "sectors":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"grouping should be one of [omni, sectors, monthly], got {grouping}")

        N_plots = len(all_groups)
        figures, ax = resolve_axes_grid(N_plots, ax=ax, ncols=ncols, nfigs=nfigs)

        for i,(key,data) in enumerate(all_groups.items()):
            model = models.get(key)  # None if this fit failed/was skipped
            rve = RVE.loc[key] if (model is not None and key in RVE.index) else None
            self._subplot_isodensity_contours(rve, ax=ax[i], model=model, data=data, **contour_kwargs)

            title = "Month:" if grouping == "monthly" else "Sector:"
            title += f" {key}"
            if title_N_points: title += f", N={len(data)}"
            if model is None:
                title += " (failed to fit)"

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
        Tables of isodensity contour points per subset.

        Parameters
        ----------
        grouping : {"monthly", "sectors"}
            Subsets to tabulate. "sectors" requires a direction variable. The
            omni model is added as "Yearly" for monthly and "Omni" for sectors.
        RVE : pandas.DataFrame
            Return values from `table_return_values`, one row per subset.
        range_interval : float, default 0.5
            Passed to the model's contour table method.
        replace_cols : bool, default True
            Rename the index and columns for display.
        **kwargs
            Passed to the model's contour table method.

        Returns
        -------
        dict of pandas.DataFrame
            One table per subset. A subset whose model failed to fit gets a
            single NaN row labelled "(failed to fit)".
        """
        if grouping == "sectors":
            self._require_direction("table_isodensity_contours(grouping='sectors')")

        if grouping == "monthly":
            models = self.models_monthly
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif grouping == "sectors":
            models = self.models_sectors
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be monthly or sectors, got {grouping}")

        models = models | {("Yearly" if grouping == "monthly" else "Omni"): self.model_omni}

        var1_cols, rp_labels = self._rve_var1_columns(RVE)

        tables = {}
        template_cols = None
        for subset in all_keys:
            model = models.get(subset)
            if model is None:
                tables[subset] = None  # filled in with a blank row below
                continue
            rve = RVE.loc[subset,var1_cols]
            table = model.table_isodensity_contour(
                points=rve.values,range_interval=range_interval,**kwargs)
            if replace_cols:
                table.index.name = f"${self.var1_symbol}$"
                if 2*len(var1_cols) == table.shape[1]-1:
                    table = table.set_index(table.columns[0])
                if 2*len(var1_cols) != table.shape[1]:
                    raise ValueError(f"Expected {len(var1_cols)} columns, "
                        f"2 per return value, but got {table.shape[1]}")
                table.columns = [f"{rp.replace('year','yr')} ${self.var2_symbol}$$_{lh}$"
                                for rp in rp_labels for lh in ["L","H"]]
            tables[subset] = table
            if template_cols is None:
                template_cols = table.columns

        # Fill blanks for failed fits using the shape/columns seen from a
        # successful fit, so every subset has a table of matching shape -
        # empty rather than simply missing, to make fit failures visible.
        for subset, table in tables.items():
            if table is None:
                cols = template_cols if template_cols is not None else []
                tables[subset] = pd.DataFrame(
                    np.nan, index=[f"(failed to fit)"], columns=cols
                )

        return tables

    def table_conditional_return_values(
            self,
            grouping:typing.Literal["monthly","sectors"],
            RVE:pd.DataFrame,
    ):
        """
        Conditional return values of the secondary variable per subset.

        Parameters
        ----------
        grouping : {"monthly", "sectors"}
            Subsets to tabulate. "sectors" requires a direction variable. The
            omni model is added as "Yearly" for monthly and "Omni" for sectors.
        RVE : pandas.DataFrame
            Return values from `table_return_values`, one row per subset.

        Returns
        -------
        pandas.DataFrame
            One row per subset. Subsets without a fitted model keep the
            primary variable values and have NaN for the secondary variable.
        """
        if grouping == "sectors":
            self._require_direction("table_conditional_return_values(grouping='sectors')")

        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be monthly or sectors, got {grouping}")

        var1_cols, rp_labels = self._rve_var1_columns(RVE)

        table = {}
        for subset in all_keys:
            model = models.get(subset)
            main = RVE.loc[subset,var1_cols]
            row = {}
            if model is None:
                # Primary variable's return values come directly from RVE,
                # independent of the fit - keep those. Only the dependent
                # variable (which requires the fitted model) is blanked.
                for rp,var1 in zip(rp_labels,main):
                    row[f"{rp.replace('year','yr')} ${self.var1_symbol}$"] = var1
                    row[f"{rp.replace('year','yr')} ${self.var2_symbol}$"] = np.nan
            else:
                cond = model.get_dependent_given_marginal(main.values)
                for rp,var1,var2 in zip(rp_labels,main,cond):
                    row[f"{rp.replace('year','yr')} ${self.var1_symbol}$"] = var1
                    row[f"{rp.replace('year','yr')} ${self.var2_symbol}$"] = var2
            table[subset] = row
        return pd.DataFrame.from_dict(table,orient="index")
    
    def table_return_values(
            self,
            grouping:typing.Literal["monthly","sectors"],
            return_periods:list[float],
            state_duration:float = None,
            contour_method:str = "IFORM",
    ):
        """
        Return values of both variables per subset.

        Parameters
        ----------
        grouping : {"monthly", "sectors"}
            Subsets to tabulate. "sectors" requires a direction variable. The
            omni model is added as "Yearly" for monthly and "Omni" for sectors.
        return_periods : list of float
            Return periods, in years.
        state_duration : float, optional
            Duration of one sea state, in hours. Defaults to the mean data
            interval. A warning is issued if it differs from the mean data
            interval by more than 10 minutes.
        contour_method : str, default "IFORM"
            Contour method passed to ``get_contour_maximum``.

        Returns
        -------
        pandas.DataFrame
            One row per subset. Columns are ``"{rp}-year {symbol}"`` for each
            return period and variable. Subsets without a fitted model are NaN.
        """
        if grouping == "sectors":
            self._require_direction("table_return_values(grouping='sectors')")

        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be monthly or sectors, got {grouping}")

        data_interval = self.data_omni.sort_index().index.diff().mean().total_seconds()/3600

        if state_duration is None:
            state_duration = data_interval
        elif abs(state_duration-data_interval) > (1/6):
            warnings.warn("Average data time-interval is "
                f"{data_interval:.2f}h, not {state_duration:.2f}h.")

        col_names = [f"{rp}-year {v}" for rp in return_periods for v in (self.var1_symbol, self.var2_symbol)]

        table = {}
        for subset in all_keys:
            model = models.get(subset)
            if model is None:
                table[subset] = {c: np.nan for c in col_names}
                continue
            row = {}
            for rp in return_periods:
                hstp = model.get_contour_maximum(
                    rp,state_duration=state_duration,contour_method=contour_method)
                row[f"{rp}-year {self.var1_symbol}"] = hstp[self.var1]
                row[f"{rp}-year {self.var2_symbol}"] = hstp[self.var2]
            table[subset] = row

        return pd.DataFrame.from_dict(table,orient="index")
        
    def table_model_parameters(
            self,
            grouping:typing.Literal["monthly","sectors"],
    ):
        """
        Fitted model parameters per subset.

        Parameters
        ----------
        grouping : {"monthly", "sectors"}
            Subsets to tabulate. "sectors" requires a direction variable. The
            omni model is added as "Yearly" for monthly and "Omni" for sectors.

        Returns
        -------
        pandas.DataFrame
            One row per subset. Subsets without a fitted model are NaN.
        """
        if grouping == "sectors":
            self._require_direction("table_model_parameters(grouping='sectors')")

        if grouping == "monthly":
            models = self.models_monthly | {"Yearly":self.model_omni}
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif grouping == "sectors":
            models = self.models_sectors | {"Omni":self.model_omni}
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be monthly or sectors, got {grouping}")

        # Derive expected parameter names from any successfully fit model,
        # so a blank row has matching columns rather than being entirely empty.
        param_cols = None
        for m in models.values():
            param_cols = list(m.parameters(False).keys())
            break

        table = {}
        for subset in all_keys:
            model = models.get(subset)
            if model is None:
                table[subset] = {c: np.nan for c in (param_cols or [])}
            else:
                table[subset] = model.parameters(False)
        return pd.DataFrame.from_dict(table,orient="index")
    
    def plot_marginal_quantiles(
            self,
            grouping:typing.Literal["omni","monthly","sectors"],
            axes:list[plt.Axes] = None,
            ncols:int = 4,
            title_N_points=False,
    ):
        """
        Marginal quantile plots for the omni, monthly or sector models.

        Parameters
        ----------
        grouping : {"omni", "monthly", "sectors"}
            Models to plot. "sectors" requires a direction variable.
        axes : list of matplotlib.axes.Axes, optional
            Two axes per subset (two in total for "omni"). Created if None.
        ncols : int, default 4
            Number of subplot columns, used only if `axes` is None.
        title_N_points : bool, default False
            Add the number of observations to the subplot titles.

        Notes
        -----
        Subsets whose model failed to fit are left blank and marked
        "(failed to fit)".
        """
        if grouping == "sectors":
            self._require_direction("plot_marginal_quantiles(grouping='sectors')")

        if grouping == "omni":
            if axes is None:
                fig,axes = plt.subplots(1,2)
            elif len(axes)<2:
                raise TypeError(f"Need at least two axes to plot on.")
            return self.model_omni.plot_marginal_quantiles(axes=axes)

        if grouping == "monthly":
            models = self.models_monthly
            all_groups = self.data_monthly
        elif grouping == "sectors":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"models should be monthly or sectors, got {grouping}")

        N_plots = len(all_groups)
        owns_figure = axes is None
        if axes is None:
            fig,axes = get_n_axes(2*N_plots,max_cols=ncols)
        elif len(axes) < 2*N_plots:
            raise ValueError(f"Need at least {2*N_plots} axes, got {len(axes)}.")

        for i,(k,data) in enumerate(all_groups.items()):
            ax = axes[2*i:2*i+2]
            model = models.get(k)
            title = f"{k}"
            if title_N_points:
                title += f", N={len(data)}"

            if model is None:
                title += " (failed to fit)"
                for a in ax:
                    a.set_axis_off()
            else:
                model.plot_marginal_quantiles(axes=ax)

            ax[0].set_title(title)
            ax[1].set_title(title)

        if owns_figure:
            axes[0].get_figure().subplots_adjust(hspace=0.25)
        return axes

    def plot_dependence_functions(
        self,
        grouping:typing.Literal["omni","monthly","sectors"],
        axes:list[plt.Axes] = None,
        ncols:int = 4,
        title_N_points=False,
    ):
        """
        Dependence function plots for the omni, monthly or sector models.

        Parameters
        ----------
        grouping : {"omni", "monthly", "sectors"}
            Models to plot. "sectors" requires a direction variable.
        axes : list of matplotlib.axes.Axes, optional
            Two axes per subset (two in total for "omni"). Created if None.
        ncols : int, default 4
            Number of subplot columns, used only if `axes` is None.
        title_N_points : bool, default False
            Add the number of observations to the subplot titles.

        Notes
        -----
        Subsets whose model failed to fit are left blank and marked
        "(failed to fit)".
        """
        if grouping == "sectors":
            self._require_direction("plot_dependence_functions(grouping='sectors')")

        if grouping == "omni":
            if axes is None:
                fig,axes = plt.subplots(1,2)
            elif len(axes)<2:
                raise ValueError(f"Need at least two axes to plot on.")
            return self.model_omni.plot_dependence_functions(axes=axes)

        if grouping == "monthly":
            models = self.models_monthly
            all_groups = self.data_monthly
        elif grouping == "sectors":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"models should be monthly or sectors, got {grouping}")

        N_plots = len(all_groups)
        owns_figure = axes is None
        if axes is None:
            fig,axes = get_n_axes(2*N_plots,max_cols=ncols)
        elif len(axes) < 2*N_plots:
            raise ValueError(f"Need at least {2*N_plots} axes, got {len(axes)}.")

        for i,(k,data) in enumerate(all_groups.items()):
            ax = axes[2*i:2*i+2]
            model = models.get(k)
            title = f"{k}"
            if title_N_points:
                title += f", N={len(data)}"

            if model is None:
                title += " (failed to fit)"
                for a in ax:
                    a.set_axis_off()
            else:
                model.plot_dependence_functions(axes=ax)

            ax[0].set_title(title)
            ax[1].set_title(title)

        if owns_figure:
            axes[0].get_figure().subplots_adjust(hspace=0.25)
        return axes