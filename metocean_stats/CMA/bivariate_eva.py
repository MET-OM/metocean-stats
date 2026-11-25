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
                "Sector-based analysis is unavailable; 'all' and 'month' "
                "groupings remain fully available."
            )

    def _rve_var1_columns(self,RVE:pd.DataFrame) -> tuple[list,list]:
        """
        Pick out the marginal (var1) return-value columns of an RVE table,
        and the return-period label (e.g. "10-year") for each one.

        `BivariateEVA.table_return_values` lays out columns as one var1
        column and one var2 column per return period (e.g. "10-year $H_s$",
        "10-year $T_p$"); only the var1 half is used here. A plain,
        single-variable table (e.g. from
        `UnivariateEVA.table_return_values_final`, with columns like
        "10-year" and no variable suffix) has no such columns to match, so
        every column is assumed to already be a var1 (primary variable)
        return value instead.
        """
        suffix = f" ${self.var1_symbol}$"
        var1_cols = [c for c in RVE.columns if c.endswith(suffix)]
        if var1_cols:
            rp_labels = [c[:-len(suffix)] for c in var1_cols]
            return var1_cols, rp_labels

        var2_suffix = f" ${self.var2_symbol}$"
        if not any(c.endswith(var2_suffix) for c in RVE.columns):
            # No var1 or var2 suffixed columns anywhere - treat the whole
            # table as var1 return values keyed by return period.
            var1_cols = list(RVE.columns)
            return var1_cols, var1_cols

        raise ValueError(
            f"Could not find any '${self.var1_symbol}$' columns in RVE. "
            f"Available columns: {list(RVE.columns)}."
        )

    def _resolve_state_duration(self,state_duration:float|None) -> float:
        """
        Default `state_duration` to the mean interval of `data_omni`, warning
        if an explicitly given value differs from it by more than 10 minutes.
        Shared by every method that needs a sea-state duration for contour
        return-period calculations (`table_return_values`, `plot_contours`,
        `table_contours`).
        """
        data_interval = self.data_omni.sort_index().index.diff().mean().total_seconds()/3600
        if state_duration is None:
            return data_interval
        if abs(state_duration-data_interval) > (1/6):
            warnings.warn("Average data time-interval is "
                f"{data_interval:.2f}h, not {state_duration:.2f}h.")
        return state_duration

    def _exposure_fraction(self,by:typing.Literal["month","direction"],subset:str) -> float:
        """
        Fraction of all `data_omni` observations found in one month/sector
        subset -- i.e. that subset's empirical rate of events relative to
        the omni rate.

        This is the same correction `UnivariateEVA` applies to POT/IDM
        subset models via `exposure_fraction` (see
        `UnivariateEVA._get_return_value`): a subset's candidate-event rate
        is the omni rate scaled by its share of the data, so an event as
        rare as the nominal `return_period` *from that subset specifically*
        corresponds to an effective return period of
        ``return_period * exposure_fraction`` in the subset's own fitted
        distribution (fewer events arrive from a narrow month/sector per
        year, so a locally-X-year event there is more common than this
        subset's data would suggest if treated as if it were the whole
        year). The omni subset itself ("Yearly"/"Omni") always has
        fraction 1.0 -- no correction applies there.
        """
        if subset in ("Yearly","Omni"):
            return 1.0
        groups = self.data_monthly if by == "month" else self.data_sectors
        return len(groups[subset]) / len(self.data_omni)

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

    def _subplot_contours(
            self,
            ax:plt.Axes,
            model:JointProbabilityModel,
            periods:list[float],
            data:pd.DataFrame = None,
            state_duration:float = None,
            contour_method:str = "IFORM",
            exposure_fraction:float = 1.0,
            **kwargs,
        ):
        """
        Draw environmental contours for one subset, or its data if the
        model is missing.

        `periods` are the nominal (displayed) return periods; the actual
        contour is computed at ``period * exposure_fraction`` (see
        `_exposure_fraction`) so subset models are properly corrected for
        their empirical rate of events, while the legend/labels still show
        the nominal periods the caller asked for.
        """
        if ax is None: fig,ax = plt.subplots()

        if model is None:
            if data is not None:
                ax.scatter(data[self.var2], data[self.var1], s=4, alpha=0.4, color="grey")
                ax.set_xlabel(self.var2)
                ax.set_ylabel(self.var1)
            return ax

        effective_periods = [p*exposure_fraction for p in periods]
        kwargs.setdefault("labels", [f"{p}-year" for p in periods])

        model.reset_labels()
        model.plot_contours(ax,periods=effective_periods,state_duration=state_duration,method=contour_method,**kwargs)
        model.plot_dependent_percentiles(ax)
        model.plot_data_density(ax)
        model.plot_DNVGL_steepness_criterion(ax)
        model.plot_legend(ax)
        model.reset_labels()
        return ax

    def plot_isodensity_contours(
            self,
            by:typing.Literal["all","direction","month"],
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
        by : {"all", "month", "direction"}
            Subsets to plot. "direction" requires a direction variable.
        RVE : pandas.DataFrame
            Return values from `table_return_values`, one row per subset.
        ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes, optional
            Axes to draw on. "all" needs a single axes. The other groupings
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
            A single axes for "all", otherwise the axes array.

        Notes
        -----
        The omni row of `RVE` is looked up as "Omni", "Yearly" or "Annual",
        in that order. Subsets whose model failed to fit are plotted with
        their data only and marked "(failed to fit)".

        Examples
        --------
        .. code-block:: python

            rve = joint.table_return_values(by="direction", return_periods=[1, 10, 100])
            joint.plot_isodensity_contours(by="direction", RVE=rve)

        .. image:: /files_v2/bivariate_eva_plot_isodensity_contours_direction.png
           :width: 600
        """
        if by == "direction":
            self._require_direction("plot_isodensity_contours(by='direction')")

        contour_kwargs = {"cmap":"viridis"} | contour_kwargs

        if by == "all":
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

        elif by == "month":
            models = self.models_monthly
            all_groups = self.data_monthly
        elif by == "direction":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"by should be one of [all, direction, month], got {by}")

        N_plots = len(all_groups)
        figures, ax = resolve_axes_grid(N_plots, ax=ax, ncols=ncols, nfigs=nfigs)

        for i,(key,data) in enumerate(all_groups.items()):
            model = models.get(key)  # None if this fit failed/was skipped
            rve = RVE.loc[key] if (model is not None and key in RVE.index) else None
            self._subplot_isodensity_contours(rve, ax=ax[i], model=model, data=data, **contour_kwargs)

            title = "Month:" if by == "month" else "Sector:"
            title += f" {key}"
            if title_N_points: title += f", N={len(data)}"
            if model is None:
                title += " (failed to fit)"

            ax[i].set_title(title)

        for fig in figures:
            fig.subplots_adjust(hspace = 0.2)

        return ax

    def plot_contours(
            self,
            by:typing.Literal["all","direction","month"],
            periods:list[float],
            state_duration:float = None,
            contour_method:str = "IFORM",
            ax:plt.Axes|list[plt.Axes]=None,
            contour_kwargs:dict={},
            nfigs:int=2,
            ncols:int=2,
            title_N_points=True
            ):
        """
        Plot environmental contours (e.g. IFORM, ISORM) of both variables per subset.

        Unlike `plot_isodensity_contours`, no return-value table is needed
        first -- contours are computed directly from `periods`.

        Parameters
        ----------
        by : {"all", "month", "direction"}
            Subsets to plot. "direction" requires a direction variable.
        periods : list of float
            Return periods, in years. One contour is drawn per period.
        state_duration : float, optional
            Duration of one sea state, in hours. Defaults to the mean data
            interval. A warning is issued if it differs from the mean data
            interval by more than 10 minutes.
        contour_method : str, default "IFORM"
            Contour method, as accepted by the underlying model's
            `plot_contours`: "IFORM", "ISORM", "HighestDensity",
            "DirectSampling", "ConstantAndExceedance", "ConstantOrExceedance".
        ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes, optional
            Axes to draw on. "all" needs a single axes. The other groupings
            need one axes per subset. Created if None.
        contour_kwargs : dict, optional
            Passed to the model's contour plot, e.g. ``{"cmap": "plasma"}``.
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
            A single axes for "all", otherwise the axes array.

        Notes
        -----
        Subsets whose model failed to fit are plotted with their data only
        and marked "(failed to fit)". Each period in `periods` is corrected
        per subset by that subset's share of `data_omni` before computing
        the contour (same as `table_return_values` -- see its Notes and
        `_exposure_fraction`); the legend still shows the nominal period
        passed in, e.g. "10-year", not the corrected value used internally.

        Examples
        --------
        .. code-block:: python

            joint.plot_contours(by="direction", periods=[1, 10, 100])

        .. image:: /files_v2/bivariate_eva_plot_contours_direction.png
           :width: 600
        """
        if by == "direction":
            self._require_direction("plot_contours(by='direction')")

        state_duration = self._resolve_state_duration(state_duration)

        if by == "all":
            if ax is None:
                fig,ax = plt.subplots()
            elif not isinstance(ax,plt.Axes):
                raise TypeError(f"Expected a single matplotlib.Axes object as ax, got {type(ax)}")

            self._subplot_contours(
                ax=ax,model=self.model_omni,periods=periods,
                state_duration=state_duration,contour_method=contour_method,**contour_kwargs)
            ax.set_title("Omni")
            return ax

        elif by == "month":
            models = self.models_monthly
            all_groups = self.data_monthly
        elif by == "direction":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"by should be one of [all, direction, month], got {by}")

        N_plots = len(all_groups)
        figures, ax = resolve_axes_grid(N_plots, ax=ax, ncols=ncols, nfigs=nfigs)

        for i,(key,data) in enumerate(all_groups.items()):
            model = models.get(key)  # None if this fit failed/was skipped
            self._subplot_contours(
                ax=ax[i],model=model,periods=periods,data=data,
                state_duration=state_duration,contour_method=contour_method,
                exposure_fraction=self._exposure_fraction(by,key),**contour_kwargs)

            title = "Month:" if by == "month" else "Sector:"
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
            by:typing.Literal["month","direction"],
            RVE:pd.DataFrame,
            range_interval:float=0.5,
            replace_cols=True,
            **kwargs,
            ):
        """
        Tables of isodensity contour points per subset.

        Parameters
        ----------
        by : {"month", "direction"}
            Subsets to tabulate. "direction" requires a direction variable. The
            omni model is added as "Yearly" for month and "Omni" for direction.
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

        Examples
        --------
        .. code-block:: python

            rve = joint.table_return_values(by="direction", return_periods=[1, 10, 100])
            tables = joint.table_isodensity_contours(by="direction", RVE=rve)

        .. csv-table::
           :header-rows: 1
           :file: /files_v2/bivariate_eva_table_isodensity_contours_direction.csv
        """
        if by == "direction":
            self._require_direction("table_isodensity_contours(by='direction')")

        if by == "month":
            models = self.models_monthly
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif by == "direction":
            models = self.models_sectors
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be month or direction, got {by}")

        models = models | {("Yearly" if by == "month" else "Omni"): self.model_omni}

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
                if 2*len(var1_cols) == table.shape[1]-1:
                    table = table.set_index(table.columns[0])
                if 2*len(var1_cols) != table.shape[1]:
                    raise ValueError(f"Expected {len(var1_cols)} columns, "
                        f"2 per return value, but got {table.shape[1]}")
                table.index.name = f"${self.var1_symbol}$"
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

    def table_contours(
            self,
            by:typing.Literal["month","direction"],
            return_periods:list[float],
            range_values=None,
            range_interval:float=0.5,
            state_duration:float=None,
            n_samples:int=None,
            contour_method:str="IFORM",
            replace_cols=True,
            ):
        """
        Tables of environmental contour points per subset, for one or more
        return periods combined into a single table per subset.

        Like `table_isodensity_contours`, but for an environmental contour
        (e.g. IFORM, ISORM) instead of an isodensity contour. Unlike that
        method, no return-value table is needed first -- every contour is
        computed directly from `return_periods`.

        Parameters
        ----------
        by : {"month", "direction"}
            Subsets to tabulate. "direction" requires a direction variable. The
            omni model is added as "Yearly" for month and "Omni" for direction.
        return_periods : list of float
            Return periods, in years. One pair of columns is added per
            period, in the given order.
        range_values : float or list of float, optional
            Values of the primary variable (var1) at which to sample each
            contour. Defaults to an even grid spanning that contour, spaced
            by `range_interval` -- computed independently per return period
            (each period's contour spans a different range), so the
            combined table's index is the union of every period's grid;
            a period whose contour doesn't reach a given var1 value leaves
            NaN there rather than extrapolating.
        range_interval : float, default 0.5
            Spacing of the auto-generated `range_values` grid. Ignored if
            `range_values` is given.
        state_duration : float, optional
            Duration of one sea state, in hours. Defaults to the mean data
            interval. A warning is issued if it differs from the mean data
            interval by more than 10 minutes.
        n_samples : int, optional
            Number of contour points to generate before slicing into a
            table. Defaults to the model's own default (1e5). Increase for
            accuracy, at the cost of speed.
        contour_method : str, default "IFORM"
            Contour method, as accepted by the underlying model's
            `table_contour`: "IFORM", "ISORM", "HighestDensity",
            "DirectSampling", "ConstantAndExceedance", "ConstantOrExceedance".
        replace_cols : bool, default True
            Rename the var2 columns and the var1 index name for display
            (e.g. ``"100-yr $T_p$$_L$"``, index name ``"$H_s$"``). var1 is
            always used as the index (needed to combine periods into one
            table), regardless of this flag -- only the labelling is
            affected.

        Returns
        -------
        dict of pandas.DataFrame
            One table per subset, with var1 as the index and, for each
            return period, a low/high bound column pair for var2. A subset
            whose model failed to fit gets a single NaN row labelled
            "(failed to fit)".

        Notes
        -----
        Each return period is corrected per subset by that subset's share of
        `data_omni`, same as `table_return_values` -- see its Notes and
        `_exposure_fraction`.

        Examples
        --------
        .. code-block:: python

            joint.table_contours(by="direction", return_periods=[1, 10, 100, 10000])

        .. csv-table::
           :header-rows: 1
           :file: /files_v2/bivariate_eva_table_contours_direction.csv
        """
        if by == "direction":
            self._require_direction("table_contours(by='direction')")

        if by == "month":
            models = self.models_monthly
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif by == "direction":
            models = self.models_sectors
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be month or direction, got {by}")

        models = models | {("Yearly" if by == "month" else "Omni"): self.model_omni}

        state_duration = self._resolve_state_duration(state_duration)

        tables = {}
        template_cols = None
        for subset in all_keys:
            model = models.get(subset)
            if model is None:
                tables[subset] = None  # filled in with a blank row below
                continue
            fraction = self._exposure_fraction(by,subset)

            period_tables = []
            for return_period in return_periods:
                period_table = model.table_contour(
                    return_period=return_period*fraction, range_values=range_values,
                    range_dim=0, range_interval=range_interval,
                    state_duration=state_duration, n_samples=n_samples,
                    contour_method=contour_method,
                )
                period_table = period_table.set_index(period_table.columns[0])
                if replace_cols:
                    period_table.columns = [f"{return_period}-yr ${self.var2_symbol}$$_{lh}$" for lh in ["L","H"]]
                period_tables.append(period_table)

            table = pd.concat(period_tables, axis=1).sort_index()
            if replace_cols:
                table.index.name = f"${self.var1_symbol}$"
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
            by:typing.Literal["month","direction"],
            RVE:pd.DataFrame,
    ):
        """
        Conditional return values of the secondary variable per subset.

        Parameters
        ----------
        by : {"month", "direction"}
            Subsets to tabulate. "direction" requires a direction variable. The
            omni model is added as "Yearly" for month and "Omni" for direction.
        RVE : pandas.DataFrame
            Return values from `table_return_values`, one row per subset.

        Returns
        -------
        pandas.DataFrame
            One row per subset. Subsets without a fitted model keep the
            primary variable values and have NaN for the secondary variable.

        Examples
        --------
        .. code-block:: python

            rve = joint.table_return_values(by="direction", return_periods=[1, 10, 100])
            joint.table_conditional_return_values(by="direction", RVE=rve)

        .. csv-table::
           :header-rows: 1
           :file: /files_v2/bivariate_eva_table_conditional_return_values_direction.csv
        """
        if by == "direction":
            self._require_direction("table_conditional_return_values(by='direction')")

        if by == "month":
            models = self.models_monthly | {"Yearly":self.model_omni}
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif by == "direction":
            models = self.models_sectors | {"Omni":self.model_omni}
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be month or direction, got {by}")

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
            by:typing.Literal["month","direction"],
            return_periods:list[float],
            state_duration:float = None,
            contour_method:str = "IFORM",
    ):
        """
        Return values of both variables per subset.

        Parameters
        ----------
        by : {"month", "direction"}
            Subsets to tabulate. "direction" requires a direction variable. The
            omni model is added as "Yearly" for month and "Omni" for direction.
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
            One row per subset. Columns are ``"{rp}-year ${symbol}$"`` for
            each return period and variable. Subsets without a fitted model
            are NaN.

        Notes
        -----
        A month/sector subset only sees a fraction of all events (fewer
        candidate storms arrive from a single month or a narrow directional
        sector than from the whole record), so its return period is
        corrected by that subset's share of `data_omni` before being handed
        to the contour method -- mirroring `UnivariateEVA`'s
        `exposure_fraction` correction for POT/IDM subset models. The omni
        subset ("Yearly"/"Omni") is unaffected (fraction 1.0). See
        `_exposure_fraction`.

        Examples
        --------
        .. code-block:: python

            joint.table_return_values(by="direction", return_periods=[1, 10, 100])

        .. csv-table::
           :header-rows: 1
           :file: /files_v2/bivariate_eva_table_return_values_direction.csv
        """
        if by == "direction":
            self._require_direction("table_return_values(by='direction')")

        if by == "month":
            models = self.models_monthly | {"Yearly":self.model_omni}
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif by == "direction":
            models = self.models_sectors | {"Omni":self.model_omni}
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be month or direction, got {by}")

        state_duration = self._resolve_state_duration(state_duration)

        col_names = [f"{rp}-year ${v}$" for rp in return_periods for v in (self.var1_symbol, self.var2_symbol)]

        table = {}
        for subset in all_keys:
            model = models.get(subset)
            if model is None:
                table[subset] = {c: np.nan for c in col_names}
                continue
            fraction = self._exposure_fraction(by,subset)
            row = {}
            for rp in return_periods:
                hstp = model.get_contour_maximum(
                    rp*fraction,state_duration=state_duration,contour_method=contour_method)
                row[f"{rp}-year ${self.var1_symbol}$"] = hstp[self.var1]
                row[f"{rp}-year ${self.var2_symbol}$"] = hstp[self.var2]
            table[subset] = row

        return pd.DataFrame.from_dict(table,orient="index")
        
    def table_model_parameters(
            self,
            by:typing.Literal["month","direction"],
    ):
        """
        Fitted model parameters per subset.

        Parameters
        ----------
        by : {"month", "direction"}
            Subsets to tabulate. "direction" requires a direction variable. The
            omni model is added as "Yearly" for month and "Omni" for direction.

        Returns
        -------
        pandas.DataFrame
            One row per subset. Subsets without a fitted model are NaN.

        Examples
        --------
        .. code-block:: python

            joint.table_model_parameters(by="direction")

        .. csv-table::
           :header-rows: 1
           :file: /files_v2/bivariate_eva_table_model_parameters_direction.csv
        """
        if by == "direction":
            self._require_direction("table_model_parameters(by='direction')")

        if by == "month":
            models = self.models_monthly | {"Yearly":self.model_omni}
            all_keys = list(self.data_monthly.keys()) + ["Yearly"]
        elif by == "direction":
            models = self.models_sectors | {"Omni":self.model_omni}
            all_keys = list(self.data_sectors.keys()) + ["Omni"]
        else:
            raise ValueError(f"models should be month or direction, got {by}")

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
            by:typing.Literal["all","month","direction"],
            axes:list[plt.Axes] = None,
            ncols:int = 4,
            title_N_points=False,
            subsample:int = 1,
    ):
        """
        Marginal quantile plots for the omni, monthly or sector models.

        Parameters
        ----------
        by : {"all", "month", "direction"}
            Models to plot. "direction" requires a direction variable.
        axes : list of matplotlib.axes.Axes, optional
            Two axes per subset (two in total for "all"). Created if None.
        ncols : int, default 4
            Number of subplot columns, used only if `axes` is None.
        title_N_points : bool, default False
            Add the number of observations to the subplot titles.
        subsample : int, default 1
            Use every `subsample`-th point of each model's fitted data when
            plotting the empirical quantiles, e.g. `subsample=10` uses 1 in
            10 points. Quantile computation scales with the number of
            points, making this by far the slowest plot in the package for
            large datasets; subsampling trades a slightly coarser empirical
            quantile estimate for speed. Defaults to 1 (every point, no
            subsampling).

        Notes
        -----
        Subsets whose model failed to fit are left blank and marked
        "(failed to fit)".

        Examples
        --------
        .. code-block:: python

            joint.plot_marginal_quantiles(by="direction")

        .. image:: /files_v2/bivariate_eva_plot_marginal_quantiles_direction.png
           :width: 600
        """
        if by == "direction":
            self._require_direction("plot_marginal_quantiles(by='direction')")

        if by == "all":
            if axes is None:
                fig,axes = plt.subplots(1,2)
            elif len(axes)<2:
                raise TypeError(f"Need at least two axes to plot on.")
            sample = self.model_omni.data.values[::subsample]
            return self.model_omni.plot_marginal_quantiles(axes=axes,sample=sample)

        if by == "month":
            models = self.models_monthly
            all_groups = self.data_monthly
        elif by == "direction":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"models should be month or direction, got {by}")

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
                model.plot_marginal_quantiles(axes=ax,sample=model.data.values[::subsample])

            ax[0].set_title(title)
            ax[1].set_title(title)

        if owns_figure:
            axes[0].get_figure().subplots_adjust(hspace=0.25)
        return axes

    def plot_dependence_functions(
        self,
        by:typing.Literal["all","month","direction"],
        axes:list[plt.Axes] = None,
        ncols:int = 4,
        title_N_points=False,
    ):
        """
        Dependence function plots for the omni, monthly or sector models.

        Parameters
        ----------
        by : {"all", "month", "direction"}
            Models to plot. "direction" requires a direction variable.
        axes : list of matplotlib.axes.Axes, optional
            Two axes per subset (two in total for "all"). Created if None.
        ncols : int, default 4
            Number of subplot columns, used only if `axes` is None.
        title_N_points : bool, default False
            Add the number of observations to the subplot titles.

        Notes
        -----
        Subsets whose model failed to fit are left blank and marked
        "(failed to fit)".

        Examples
        --------
        .. code-block:: python

            joint.plot_dependence_functions(by="direction")

        .. image:: /files_v2/bivariate_eva_plot_dependence_functions_direction.png
           :width: 600
        """
        if by == "direction":
            self._require_direction("plot_dependence_functions(by='direction')")

        if by == "all":
            if axes is None:
                fig,axes = plt.subplots(1,2)
            elif len(axes)<2:
                raise ValueError(f"Need at least two axes to plot on.")
            return self.model_omni.plot_dependence_functions(axes=axes)

        if by == "month":
            models = self.models_monthly
            all_groups = self.data_monthly
        elif by == "direction":
            models = self.models_sectors
            all_groups = self.data_sectors
        else:
            raise ValueError(f"models should be month or direction, got {by}")

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