import warnings

import pandas as pd
import numpy as np
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches as mpatches
from scipy.interpolate import interp1d

import pyextremes
import virocon
from virocon import GlobalHierarchicalModel
from virocon.plotting import _get_n_axes

from .plotting import (
    plot_2D_pdf_heatmap,
    plot_DNVGL_steepness,
    get_DNVGL_steepness,
)

from .predefined import (
    get_DNVGL_Hs_Tz,
    get_DNVGL_Hs_U,
    get_OMAE2020_Hs_Tz,
    get_OMAE2020_V_Hs,
    get_cT_Hs_Tp,
    get_vonmises_wind_misalignment,
    get_LiGaoMoan_U_hs_tp,
)

from .contours import (
    get_contour,
    sort_contour,
    split_contour,
)

def _load_preset(preset:str|Callable):
    """
    Load distribution description, fit descriptions and semantics of the model.
    """
    if callable(preset):
        dist_descriptions,fit_descriptions,semantics = preset()
        return dist_descriptions,fit_descriptions,semantics
    else:
        try:
            preset = str(preset).lower()
        except:
            raise TypeError(f"Invalid input: Preset should be either callable or a string, got {type(preset)}.")
    
    if preset == 'hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_DNVGL_Hs_Tz()
    elif preset == 'hs_u':
        dist_descriptions,fit_descriptions,semantics = get_DNVGL_Hs_U()
    elif preset == 'u_hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_LiGaoMoan_U_hs_tp()
    elif preset == 'u_misalignment':
        dist_descriptions,fit_descriptions,semantics = get_vonmises_wind_misalignment()
    elif preset == 'ct_hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_cT_Hs_Tp()
    else:
        raise ValueError(f'Preset {preset} not found. See docstring for available presets.')

    return dist_descriptions,fit_descriptions,semantics

class JointProbabilityModel(GlobalHierarchicalModel):
    """
    This is a wrapper for virocon's GlobalHierarchicalModel,
    which adds to those methods and attributes by also
    storing the fitting data and metadata.
    """

    def __init__(
            self,
            model:str|Callable = "hs_tp",
            dist_descriptions:list[dict] = None,
            fit_descriptions:list[dict] = None,
            semantics:dict = None,
        ):
        """
        Initialize the joint conditional probability model.

        The simplest is to use a predefined model via a string.
        If it is necessary to configure the models more closely,
        the dist_descriptions, fit_descriptions and semantics can be input directly.
        See predefined.py to see examples of the required format for these functions.

        Parameters
        ------------
        model : str or Callable
            Choose model from any predefined model with a string:

             - Hs_Tp: DNV-RP-C205 model of significant wave height and peak wave period
             - Hs_U: DNV-RP-C205 model of wind speed and significant wave height
             - U_Hs_Tp: Model of wind speed, significant wave height and peak wave period, from a paper by Li et. al., 2015.
             - U_misalignment: Model of wind speed and wind-wave misalignment.
             - cT_Hs_Tp: Model of aligned current, significant wave height and peak wave period.

            Note that the name signifies the order of dependency of the variables in the model.
            This argument can alternatively be a custom callable (function) that
            returns the three parts - dist_descriptions, fit_descriptions and semantics.
            Examples of this function are found in predefined.py.

        dist_descriptions : list[dict], optional
            A list of dicts describing each distribution.
        fit_descriptions : list[dict], optional
            A list of dicts describing the fitting method.
        semantics : dict, optional
            Dict desribing the variables, used for plotting.
            Keys of the dict are "names","symbols","units", and each value is
            a list of two strings corresponding to the two variables.
            Additionally, the key "swap_axis" can be used to 
            swap the axes of 2D plots, e.g. to get Hs on the y-axis.
        """
        if model:
            preset_dist,preset_fit,preset_semantics=_load_preset(model)
            if dist_descriptions is None:
                dist_descriptions = preset_dist
            if fit_descriptions is None:
                fit_descriptions = preset_fit
            if semantics is None:
                semantics = preset_semantics
        else:
            if dist_descriptions is None or fit_descriptions is None:
                raise ValueError("If preset is undefined, both dist_descriptions and fit_descriptions must be provided.")
            if semantics is None:
                semantics = self.get_default_semantics()
        
        if "swap_axis" not in semantics:
            semantics["swap_axis"] = False

        self.dist_descriptions = dist_descriptions
        self.fit_descriptions = fit_descriptions
        self.semantics = semantics
        self.swap_axis = semantics["swap_axis"]

        # Store of handles and labels, for producing a legend.
        self.legend_handles = []
        self.legend_labels = []

        self.data = pd.DataFrame()

        super().__init__(self.dist_descriptions)

    def fit(
            self,
            data:pd.DataFrame,
            var1:int|str=0,
            var2:int|str=1,
            var3:int|str=2,
            ):
        """
        Fit the model to data.

        Parameters
        ----------
        data : pandas dataframe.
            The data, shape (n_samples, n_columns), with a datetime-compatible index.
        var0 : int or str, default 0 (first column)
            The data column used to fit the marginal distribution.
        var1 : int or str, default 1 (second column)
            The data column used to fit the conditional distribution.
        var2 : int or str, default 2 (third column)
            The data column used to fit the second conditional distribution.
        """

        # Check data variable
        if not isinstance(data,pd.DataFrame):
            raise ValueError(f"The data must be in a pandas dataframe, got {type(data)}.")
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Pandas could not convert index to datetime.") from e

        # Choose as many variables as the model has dimensions.
        vars = list(np.array([var1,var2,var3])[:self.n_dim])
        for i,v in enumerate(vars):
            if isinstance(v,str):
                if v not in data.columns:
                    raise ValueError(f"Var {v} not found in dataframe columns {data.columns}.")
            else:
                vars[i] = data.columns[v]
        
        # Fit model and save data.
        data = data[vars]
        self.data = data

        super().fit(data.values,self.fit_descriptions)

    def plot_marginal_quantiles(self,axes:list=None,sample=None):
        """
        Plot the theoretical (distribution) vs empirical (sample) quantiles.
        """
        if sample is None: sample = self.data.values
        elif isinstance(sample,pd.DataFrame):
            sample = sample.values
        
        return virocon.plot_marginal_quantiles(self,
            sample=sample,semantics=self.semantics,axes=axes)

    def plot_dependence_functions(self,axes=None):
        """
        Plot the dependence functions of the conditional parameters.
        """
        if axes is not None:
            axes = np.array(axes).ravel()
        return virocon.plot_dependence_functions(self,self.semantics,axes=axes)

    def plot_histograms_of_interval_distributions(self,plot_pdf=True,max_cols=4):
        """
        Plot histograms for the estimated distribution in each data interval.
        """        
        return virocon.plot_histograms_of_interval_distributions(self,
            self.data.values,self.semantics,plot_pdf=plot_pdf,max_cols=max_cols)

    def plot_isodensity_contours(
            self,
            ax=None,
            levels:list[float]=None,
            points:np.ndarray=None,
            labels:list[str]=None,
            limits=None,
            n_grid_steps=720,
            cmap=None,
            contour_kwargs=None,
            ):
        """
        Plot isodensity (constant probability density) 2D contours.
        Contours can be specified either through a density directly,
        or by choosing any 2D point of origin.

        Parameters
        -----------
        ax : matplotlib axes, optional
            Plot on existing axes object.
        levels : list[float]
            List of probabilities to plot. Either this or return_values
            can be specified.
        points : np.ndarray
            A list of marginal return values, of e.g. Hs,
            from which to draw the isodensity contours.
            This can also be a list of 2D points,
            in which case they will be considered 
            joint extremes (any point on a contour).
        labels : list of strings, optional
            Labels to use in the figure legend.
        limits : tuple
            Plot limits, e.g. ([0,10],[0,20])
        n_grid_steps : int, default 720
            The resolution of the contours (higher is better).
            Increase this number if the contours are not smooth.
        cmap : matplotlib colormap, optional
            E.g. "viridis", "Spectral", etc.
        contour_kwargs : dict, optional
            Any other keyword arguments for matplotlib.pyplot.contour().
            Example: {"linewidths": [1,2,3], "linestyles": ["solid","dotted","dashed"]}.
        """

        if ax is None:
            _,ax = plt.subplots()

        if limits is None:
            limits = [(0,2*self.data.iloc[:,d].max()) for d in range(self.n_dim)]

        if points is not None:
            if levels is not None:
                raise ValueError("Only one of levels or return_values may be specified. The other must be None.")
            points = np.array(points)
            if points.ndim == 0:
                points = np.array([points])
            if points.ndim == 1:
                marginal_labels = True
                conditional = self.get_dependent_given_marginal(points)
                points = np.array([points,conditional]).T
            elif points.ndim == 2:
                marginal_labels = False

            if points.ndim > 2 or points.shape[1] > 2:
                raise ValueError(f"Invalid shape of points {points.shape}. Should be (N) or (Nx2).")
            levels = self.pdf(points)

            if labels is None:
                sym0,sym1 = self.semantics["symbols"]
                if marginal_labels:
                    labels = [sym0+f"={p[0]:.1f}" for p in points]
                else:
                    labels = [sym0+f"={p[0]:.1f}"+", "+sym1+f"={p[1]:.1f}" for p in points]

        idx = np.argsort(levels)
        inverse_sort_idx = np.argsort(idx)
        levels = np.array(levels)[idx]

        if labels is not None: 
            if isinstance(labels,str):
                labels = [labels]
            if len(labels) != len(levels):
                raise ValueError("Number of labels does not match number of contour levels!")
            labels = np.array(labels)[idx]

        if "LoNoWe" in str(type(self.distributions[0])):
            raise NotImplementedError("Isodensity not implemented for LoNoWe.")

        # Keep track of original artists.
        old_artists = ax.get_children()

        # Produce plot.
        ax = virocon.plot_2D_isodensity(
            self,
            sample=self.data.values,
            semantics=self.semantics,
            swap_axis=self.swap_axis,
            limits=limits,
            levels=levels,
            ax=ax,
            n_grid_steps=n_grid_steps,
            cmap=cmap,
            contour_kwargs=contour_kwargs)

        # Remove scatter, and add handles and labels for the contours.
        new_artists = list(set(ax.get_children())-set(old_artists))
        for artist in new_artists:
            if "PathCollection" in str(artist): # The scatter.
                artist.remove()
            elif "ContourSet" in str(artist): # The isodensity contours.

                # The contour doesn't set correct limits, these must be manually provided.
                contour = np.concatenate([np.array(line) for seg in artist.allsegs for line in seg],axis=0)
                ax.dataLim.bounds = (*contour.min(axis=0), *contour.max(axis=0))
                self._reset_axlim(ax)

                CShandles,CSlabels = artist.legend_elements()
                self.legend_handles += list(np.array(CShandles)[inverse_sort_idx])
                if labels is None:
                    self.legend_labels += list(np.array(CSlabels)[inverse_sort_idx])
                else:
                    self.legend_labels += list(labels[inverse_sort_idx])

        return ax


    def plot_contours(
            self,
            ax=None,
            periods:list[float]=[1,10,100,1000],
            state_duration:float=1,
            method:str="IFORM",
            labels:list[str] = None,
            cmap="viridis_r",
            **kwargs
            ):
        """
        Plot 2D environmental contours.

        Parameters
        -----------
        periods : list[float]
            List of return periods to create contours of.
        state_duration : float
            The average duration (hours) of each sample 
            in the data used to fit the model.
        contour_method : str
            The method of calculating a contour. One of:

             - "IFORM", the inverse first-order method
             - "ISORM", the inverse second-order method
             - "HighestDensity"
             - "DirectSampling", monte carlo sampling
             - "ConstantAndExceedance"
             - "ConstantOrExceedance"
        
        labels : list of str, optional
            A name for each contour.
        cmap : matplotlib colormap
            Colormap to color contours. Only used if c/color keyword is not in kwargs.
        kwargs : keyword arguments
            Any matplotlib plot keyword arguments, e.g.
            "color"="blue","linewidth"=[2,3,4],"label":["1 year","10 year","100 year"]}.
        """

        periods = np.array(periods)

        if ax is None:
            _,ax = plt.subplots()

        # Check kwargs.
        if labels is None:
            labels = [f"{i}-year" for i in periods]
        else:
            if isinstance(labels,str):
                labels = []
            if len(labels) != periods.size:
                raise ValueError("Number of labels does not match number of return periods.")
        if "c" not in kwargs and "color" not in kwargs:
            if type(cmap) is str:
                cmap = plt.get_cmap(cmap)
            kwargs["color"] = cmap(np.linspace(0,1,len(periods)))

        for k,v in kwargs.items():
            if isinstance(v,str):
                kwargs[k] = [v]*periods.size
            elif len(v) != periods.size:
                raise ValueError(f"Keyword {k} should have length equal to number of return periods {len(periods)}.")

        for i,return_period in enumerate(periods):
            plot_kwargs = {k:v[i] for k,v in kwargs.items()}
            contour = get_contour(self,return_period=return_period,state_duration=state_duration,
                                  method=method,point_distribution="equal")
            old_artists = ax.get_children()
            virocon.plot_2D_contour(
                contour,
                semantics=self.semantics,
                swap_axis=self.swap_axis,ax=ax,plot_kwargs=plot_kwargs)
            new_artist = list(set(ax.get_children())-set(old_artists))
            self.legend_handles += new_artist
            self.legend_labels += [labels[i]]

        return ax

    def plot_pdf_heatmap(
            self,
            ax=None,
            limits = (25,25),
            bin_size = (1,1),
            bin_samples = (10,10),
            percentage_values = True,
            significant_digits = 4,
            range_including = True,
            marginal_values = True,
            log_norm = True,
            color_vmin = 1e-10,
            **kwargs
        ):
        """
        Visualize the PDF of 2D model as a heatmap. See options for details.

        Parameters
        ----------
        limits : tuple
            Upper limit for the plot on each variable.
        bin_size : tuple
            Bin size for each variable in the diagram.
        bin_samples : tuple(int,int)
            Number of samples to draw from pdf, for each bin. 
            A higher number should give a better estimate 
            of the true cumulative bin probability.
        percentage_values : bool
            Use percentage (of 100), rather than fraction (of 1).
        significant_digits : int
            How many significant digits to use in the scientific number notation.
        range_including : bool, default True
            Include the end of the bin range in the margins.
        marginal_values : bool, default True
            Include the marginal distribution of probability on the labels.
        log_norm : bool, default True
            Use logarithmic color norm.
        color_vmin : float, default 1e-10
            Minimum value for the color scale.
        ax : matplotlib axes object, default None
            Existing axes object, otherwise a new will be created.
        **kwargs : keyword arguments
            Any other keyword arguments will be passed directly to seaborn's heatmap.
        """
        return plot_2D_pdf_heatmap(
            self,
            semantics=self.semantics,
            limits=limits,
            bin_size=bin_size,
            bin_samples=bin_samples,
            percentage_values=percentage_values,
            significant_digits=significant_digits,
            range_including=range_including,
            marginal_values=marginal_values,
            log_norm=log_norm,
            color_vmin=color_vmin,
            ax=ax,
            **kwargs)
    

    def get_dependent_given_marginal(
            self,
            given:np.ndarray,
            percentile:np.ndarray=0.5,
            dim = 1,
        ):
        """
        Given values of dimension A, get any percentile of
        a dependent dimension B, where B is dependent on A.

        For example: Given Hs=2, get the 0.5 percentile of Tp.

        Parameters
        ----------
        given : list of floats
            List of known values.
        percentile : float
            Percentile (0 to 1).
        dim : int
            The dimension from which you want to get values.
        """
        # Check percentile
        percentile = np.squeeze(percentile)
        if np.array(percentile).size > 1: 
            raise ValueError("Only one percentile may be specified at a time.")
        elif percentile>=1 or percentile<=0:
            raise ValueError("Percentile should be between 0 and 1.")
    
        # Shape given into required (samples, n_dim) format.
        input_dim = self.conditional_on[dim]
        if input_dim is None:
            raise ValueError(f"Chosen dim {dim} is marginal, not dependent.")
        given = np.squeeze(given)
        if given.ndim > 1:
            raise ValueError("Given should be a 1D list of known values.")
        elif given.ndim == 0: # one number
            given = np.array([given])

        if given.ndim == 1: # list of numbers
            temp = np.zeros(shape=(len(given),self.n_dim))
            temp[:,input_dim] = given
            given = temp

        return self.conditional_icdf(p=percentile,dim=dim,given=given)

    def plot_dependent_percentiles(
            self,
            ax=None,
            percentiles=[0.05,0.5,0.95],
            labels:list[str] = None,
            limit:float = None,
            **kwargs
            ):
        """
        Plot lines describing percentiles of the dependent 
        variable of the joint distribution (e.g. Tp)
        as functions of the marginal variable (e.g. Hs).

        Parameters
        -----------
        ax : matplotlib axes
            Axes to plot on
        percentiles : list[float]
            List of percentiles.
        labels : list of string
            Names, for the legend.
        limit : float, default 2x data
            Limit in the marginal variable (e.g. Hs) for which to plot percentiles (of e.g. Tp/Tz).
        """
        percentiles = np.array(percentiles)
        # Set default style.
        if labels is None:
            labels = [f"${self.semantics['symbols'][1]}$ - {100*p:g}%" for p in percentiles]
        elif np.array(labels).size != percentiles.size:
            raise ValueError("Number of labels does not fit the number of percentiles.")
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["c"] = "tab:orange"
        if "linestyles" not in kwargs:
            if len(percentiles) == 3:
                kwargs["linestyles"] = ["dashed","solid","dashed"]
            else:
                kwargs["linestyles"] = "solid"
            
        # Make sure all kwargs have equal length.
        for k,v in kwargs.items():
            if len(v) != percentiles.size:
                kwargs[k] = [v]*percentiles.size

        if ax is None:
            _,ax = plt.subplots()

        if limit is None:
            limit = 1.5*np.max(self.data.values[:,0])
        marginal = np.arange(0,limit,0.001)

        # Plot percentile lines.
        for i,p in enumerate(percentiles):
            conditional = self.get_dependent_given_marginal(marginal,p)
            if self.swap_axis:
                x,y = conditional,marginal
            else:
                x,y = marginal,conditional

            handle = ax.plot(x,y,linestyle=kwargs["linestyles"][i],
                    color=kwargs["c"][i])
            self.legend_handles += handle
            self.legend_labels += [labels[i]]

        return ax

    def plot_DNVGL_steepness_criterion(
            self,
            ax=None,
            use_peak_wave_period = True,
            ylim = None,
            xlim = None,
            label = "Steepness",
            **kwargs
            ):
        """
        Add the DNVGL steepness line. By default, within the bounds of the data.
        
        Parameters
        ----------
        period_type : str, default peak
            Type of wave period: peak or mean.
        ax : matplotlib axes
            Axes to plot on. Will create new if not included.
        **kwargs : keyword arguments
            These will be passed to matplotlib.plot().

        Returns
        --------
        matplotlib axes    
        """
        if ax is None:
            _,ax = plt.subplots()
        if ylim is None:
            ylim = np.max(self.data.values[:,0])*1.5
        # Get h and t.
        wave_period_type = 'tp' if use_peak_wave_period else 'tz'
        if xlim is None and ylim is None:
            xlim,ylim = 40,20
        if ylim is not None:
            h = np.linspace(0,ylim,10000)
            t = get_DNVGL_steepness(h,"hs",wave_period_type)
        elif xlim is not None:
            t = np.linspace(0,xlim,10000)
            h = get_DNVGL_steepness(t,wave_period_type,"hs")

        # These are defaults, to be overwritten by **kwargs.
        plot_params = {**{'linestyle':'--','color':'black',},**kwargs}
        handle = ax.plot(t,h,**plot_params)
        if label is not None:
            self.legend_labels.append(label)
            self.legend_handles += handle
        return plot_DNVGL_steepness(ax=ax,peak_period_line=use_peak_wave_period,xlim=xlim,ylim=ylim,**kwargs)
    
    def plot_data_scatter(
            self,
            ax=None,
            data=None,
            label:str=None,
            **kwargs):
        """
        Plot data as a scatter plot.

        Parameters
        ----------
        data : np.ndarray, optional
            Sample to plot. If none is provided, the data 
            that was used to fit the model will be used.
        ax : matplotlib axes, optional
            Axes to plot data on.
        label : str, optional
            Label for the data. By default, 
            no label is used.
        **kwargs : keyword arguments
            These will be passed to the scatter plot function.
        """
        if ax is None:
            _,ax = plt.subplots()
        if data is None:
            data = self.data.values
        else:
            data = np.array(data)
        if data.size == 0:
            raise ValueError("Data is empty.")
        if data.ndim == 1:
            data = np.array([data])
        if data.ndim > 2:
            raise ValueError("Data should have exactly two dimensions: samples x variables.")
        if data.shape[1] != 2:
            raise NotImplementedError("Only 2-dimensional data is supported.")
        
        kwargs.setdefault("s",2)

        x,y = (1,0) if self.swap_axis else (0,1)
        handle = ax.scatter(data[:,x],data[:,y],**kwargs)
        if label is not None:
            self.legend_handles.append(handle)
            self.legend_labels.append(label)
        return ax

    def plot_data_density(
            self,
            ax=None,
            data=None,
            bins=100,
            label:str=None,
            norm=LogNorm(),
            density=False,
            **kwargs):
        """
        Plot data as a density plot.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot data on.
        data : np.ndarray, optional
            Sample to plot. If none is provided, the data 
            that was used to fit the model will be used.
        bins : array, optional
            Define histogram bins. By default, 
            each bin will be 1x1, e.g. 1Hs x 1Tp. 
            From matplotlib docs:
             - If int, the number of bins for the two dimensions (nx = ny = bins).
             - If [int, int], the number of bins in each dimension (nx, ny = bins).
             - If array-like, the bin edges for the two dimensions (x_edges = y_edges = bins).
             - If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        label : str, optional
            Label for the data. By default, 
            no label is used.
        norm : matplotlib colornorm, default LogNorm
            The color representation of density/histogram values.
        density : bool, optional
            If True, becomes a density plot (sum 1). If false, a histogram.
        **kwargs : keyword arguments
            These will be passed to the hist2d plot function.
        """
        if ax is None:
            _,ax = plt.subplots()

        if data is None:
            data = self.data.values
        else:
            data = np.array(data)
        if data.size == 0:
            raise ValueError("Data is empty.")
        if data.ndim == 1:
            data = np.array([data])
        if data.ndim > 2:
            raise ValueError("Data should have exactly two dimensions: samples x variables.")
        if data.shape[1] != 2:
            raise NotImplementedError("Only 2-dimensional data is supported.")

        x,y = (1,0) if self.swap_axis else (0,1)
        
        # if bins is None:
        #     bins = [
        #         np.arange(np.floor(np.min(data[:,x])),
        #                   np.ceil(np.max(data[:,x])),0.02),
        #         np.arange(np.floor(np.min(data[:,y])),
        #                   np.ceil(np.max(data[:,y])),0.01)
        #     ]

        handle = ax.hist2d(data[:,x],data[:,y],norm=norm,
                           bins=bins,density=density,**kwargs)
        
        if label is not None:
            if "cmap" in kwargs:
                cmap = kwargs["cmap"]
            else:
                cmap = plt.get_cmap("viridis")

            # This is a so-called "proxy handle" - a square patch
            # with color equal to the center color of the colormap.
            # Required because histogram can not be added to legend directly.
            handle = mpatches.Patch(color=cmap(0.5))
            
            self.legend_handles.append(handle)
            self.legend_labels.append(label)

        self._reset_axlim(ax)

        return ax


    def plot_3D_contour(
            self,
            ax=None,
            return_period:float=100,
            state_duration:float=1,
            method:str="IFORM",
            # surface_plot=False, # currently not working.
            n_samples = 1000,
            label:str=None,
            **kwargs
            ):
        """
        Plot a 3-dimensional environmental contour.

        Parameters
        -----------
        return_period : float
            The return period (in years).
        state_duration : float
            The average duration (hours) of each sample 
            in the data used to fit the model.
        method : str
            The method of calculating a contour. One of:

             - "IFORM", the inverse first-order method
             - "ISORM", the inverse second-order method
             - "HighestDensity"
             - "DirectSampling", monte carlo sampling
             - "ConstantAndExceedance"
             - "ConstantOrExceedance"
        labels : list of str, optional
            A name for each contour.
        kwargs : keyword arguments
            Keyword arguments will go to either ax.plot_surface or ax.scatter.
            depending on the setting of surface_plot parameter.
        """

        if ax is None:
            _,ax = plt.subplots(subplot_kw={"projection":"3d"})
        elif not isinstance(ax,plt.Axes):
            raise TypeError(f"The axis must be matplotlib axes object, got {type(ax)}.")
        elif "3d" not in ax.name:
            raise ValueError(f"The axis argument must be 3D, got {ax.name}.")

        surface_plot = False  # TODO - fix surface plot (gridded distribution in IFORMContour and ISORMContour)
        if surface_plot:
            x,y,z = get_contour(self,return_period=return_period,state_duration=state_duration,
                                method=method,point_distribution="gridded",n_samples=n_samples)
            handle = ax.plot_surface(x,y,z,**kwargs)
        else:
            contour = get_contour(self,return_period=return_period,state_duration=state_duration,
                                  method=method,point_distribution="equal", n_samples=n_samples)
            x,y,z = contour.coordinates.T
            handle = ax.scatter(x,y,z,**kwargs)
        
        if label is not None:
            self.legend_handles.append(handle)
            self.legend_labels.append(label)

        self.plot_semantics(ax)

        return ax

    def plot_3D_contour_slices(
            self,
            ax:list[plt.Axes]=None,
            subplots=True,
            return_period:float=100,
            state_duration:float=1,
            method:str="IFORM",
            slice_dim:int = 0,
            slice_values:list[float]=[1,2,3],
            slice_width:float = 0.01,
            n_samples = int(1e6),
            scatter_plot = False,
            labels:str=None,
            cmap=None,
            **kwargs
            ):
        """
        Generate a 3-dimensional point-cloud contour, and plot 2D slices of it.

        Parameters
        -----------
        ax : matplotlib axes (for one contour slice) or list of axes.
            The axes to plot on. These should be in 2D projection.
        subplots : bool, default True
            Whether to plot on subplots, or a single axes.
            Only used if axes are not given directly.
        return_period : float
            The return period (years) which the contour should represent.
        state_duration : float
            The duration (hours) of each event in the initial data.
        method : str
            Method for calculating the contour.
             - "IFORM", the inverse first-order method
             - "ISORM", the inverse second-order method
             - "HighestDensity"
             - "DirectSampling", monte carlo sampling
             - "ConstantAndExceedance"
             - "ConstantOrExceedance"
            
        slice_dim : int
            The variable/dimension to cut through.
        slice_values : list of floats
            The values of the dimension to cut at.
        slice_width : float
            The width of each slice. See notes below for how to adjust this number.
        n_samples : int
            The number of contour points on the original 3D contour.
            The 2D contours are cut from this. See notes for how to adjust this number.
        scatter_plot : bool, default False
            If True, the points are plotted as a scatter plot, instead of a line.
            Can be useful to check if the contour is well-defined.
        labels : list[str]
            A label for each contour level. If using subplots, 
            these labels will become subplot-titles instead of being added to legend entries.
        cmap : matplotlib colormap
            Used when plotting in a single figure, ignored for subplots (use color kwarg instead.)
        **kwargs : keyword arguments
            Keyword arguments will be passed to matplotlib.

        Notes
        ------
        Algorithm:
        1. Generate a full 3D contour of points using the specified contour method
        2. Slice the contour in the given dimension, value(s) and interval.
        3. Drop the dimension to get a collection of 2D points.
        4. Draw a closed contour using a greedy travelling salesman approach.

        Tips for a nice contour:
         * If the contour is jagged/zigzag-like, there are too many points, and/or the width of the slice is too large.
         * If there are noticeable straight lines and corners, there are too few points.
        """
        # Check input
        slice_values = np.array(slice_values)
        if slice_values.ndim == 0:
            slice_values = np.array([slice_values])

        # Check or create axes
        if ax is None:
            ax = _get_n_axes(len(slice_values))[1] if subplots else plt.subplots()[1]
        if hasattr(ax,"__len__"):
            ax = np.array(ax).ravel()
            subplots = True
        else:
            subplots = False
        # Check labels    
        if labels is None:
            labels = [r" $\it{"+f"{self.semantics['symbols'][slice_dim]}"+r"}$"+
                      f" = {v} {self.semantics['units'][slice_dim]}" for v in slice_values]
        elif subplots and not np.array(labels).size == len(slice_values):
            raise ValueError("The number of labels does not match number of slices.")
        
        # Check colors
        if "c" not in kwargs and "color" not in kwargs:
            if cmap is None:
                cmap = "viridis"
            if type(cmap) is str:
                cmap = plt.get_cmap(cmap)
            kwargs["color"] = cmap(np.abs(slice_values)/np.max(np.abs(slice_values)))

        # Check remaining kwargs
        for k,v in kwargs.items():
            if isinstance(v,str):
                kwargs[k] = [v]*slice_values.size
            elif len(v) != slice_values.size:
                raise ValueError(f"Keyword {k} should have length equal to number of slices {len(slice_values)}.")

        # Which dimension's semantics that should be plotted on which axis (x,y,z)
        if slice_dim == 0:
            axis_labels = (2,1) if self.swap_axis else (1,2)
        if slice_dim == 1:
            axis_labels = (0,2)
        if slice_dim == 2:
            axis_labels = (0,1)

        contour = get_contour(self,return_period=return_period,state_duration=state_duration,
                              method=method,point_distribution="random",n_samples=n_samples)

        for i,level in enumerate(slice_values):
            coords = contour.coordinates
            coords = coords[np.abs(coords[:,slice_dim]-level)<slice_width]
            if len(coords)==0:
                warnings.warn(f"Found no contour points at level {level}+-{slice_width}. Skipped.")
                continue
            coords = np.delete(coords,slice_dim,axis=1)
            coords = sort_contour(coords)

            if self.swap_axis and slice_dim==0:
                y,x = coords.T
            else:
                x,y = coords.T

            axi = ax[i] if subplots else ax

            plot_kwargs = {k:v[i] for k,v in kwargs.items()}
            if scatter_plot:
                handle = axi.scatter(x,y,**plot_kwargs)
            else:
                handle = axi.plot(x,y,**plot_kwargs)[0]

            if subplots:
                self.plot_semantics(axi,*axis_labels)
                if labels is not None:
                    axi.set_title(labels[i])
            else:
                if labels is not None:
                    self.legend_handles.append(handle)
                    self.legend_labels.append(labels[i])

        if not subplots:
            self.plot_semantics(ax,*axis_labels)

        return ax

    def plot_3D_isodensity_contour_slice(
            self,
            ax:plt.Axes=None,
            density_levels:float=None,
            marginal_values:float=None,
            slice_dim:int = 0,
            slice_value:float=5,
            grid_steps = 1000,
            limits: list[tuple]=None,
            invalid_density_policy="drop",
            labels:str=None,
            cmap=None,
            **kwargs
            ):
        """
        Using the pdf of the 3D model, create a 2D slice at a given dimension value.
        This function can plot several densities but only one dimension value.
        Alternatively, instead of densities, a list of marginal values may be given,
        e.g. wind for a wind-hs-tp model.

        Parameters
        -----------
        ax : matplotlib axes or list of axes.
            The axes to plot on. These should be in 2D projection.
        density_levels : list[float]
            Probability density level(s).
        marginal_values : list[float]
            Values of the marginal, primary variable of the model, e.g., wind. 
            These are used to get conditional values of the other variables, e.g., hs and tp.
            Finally, the 3D points are used to sample the pdf to get the densities.
        slice_dim : int
            The variable/dimension to cut through.
        slice_value : float
            Variable value to cut at.
        grid_steps : int
            How precicely the model pdf is sampled in each dimension.
        limits : list of 2 tuples of 2 floats, optional
            Two tuples defining upper and lower limits of pdf sampling,
            for the two remaining dimensions.
            If not given, they will be guessed based on the data (3x max).
        invalid_density_policy : str, default "drop"
            What to do with densities that are outside the scope of the sampled pdf,
            i.e., smaller than pdf.min() or larger than pdf.max(). Options:
             - "drop" drops these contours from the legend, default.
             - "ignore" to keep in legend
             - "raise" to raise error, good if this is unexpected.
        label : str
            A label.
        **kwargs : keyword arguments
            Keyword arguments will be passed to matplotlib's ax.contour().
        """

        if np.array(limits).size == 1:
            limits = 3 if limits is None else limits
            lim_lower = np.minimum([0,0,0],self.data.min().to_numpy())
            limits = np.array([lim_lower,self.data.max().to_numpy()*limits]).T
        elif np.array(limits).shape != (3,2):
            raise ValueError("Limits should be shape (3,2)"
                             " corresponding to lower and upper"
                             " limits per dimension.")
        limits = np.sort(limits,axis=1)

        # Create query with shape (samples x 3) and sample pdf
        gridpoints = [
            np.linspace(limits[0,0],limits[0,1],grid_steps),
            np.linspace(limits[1,0],limits[1,1],grid_steps),
            np.linspace(limits[2,0],limits[2,1],grid_steps)
        ]
        gridpoints[slice_dim] = slice_value # 3xN
        gridpoints = np.squeeze(np.meshgrid(*gridpoints,indexing="ij")) # 3xNxN
        X,Y = np.delete(gridpoints,slice_dim,0) # NxN,NxN
        gridpoints = gridpoints.reshape(3,-1).T # 3xN**2
        Z = self.pdf(gridpoints).reshape([grid_steps,grid_steps]) # NxN
        if np.any(np.isnan(Z)):
            warnings.warn("Found NaN values in pdf, these are set to 0. "\
                  "This is usually due to sampling limits being outside " \
                  "the domain of the model (distributions or dep. funcs).")
        Z = np.nan_to_num(Z)

        # Check marginal_values
        if density_levels is not None and marginal_values is not None:
            raise ValueError("Density_levels and marginal_values cannot both be defined.")
        if marginal_values is not None:
            marginal_values = np.array(marginal_values)
            if marginal_values.ndim == 0:
                marginal_values = np.array([marginal_values])
            marginal_dim = np.squeeze(np.where(np.array(self.conditional_on)==None))
            secondary_dim = np.squeeze(np.where(np.array(self.conditional_on)==marginal_dim))
            tertiary_dim = np.squeeze(np.where(np.array(self.conditional_on)==secondary_dim))
            secondary_values = self.get_dependent_given_marginal(marginal_values,dim=secondary_dim)
            tertiary_values = self.get_dependent_given_marginal(secondary_values,dim=tertiary_dim)
            points_3D = np.array([marginal_values,secondary_values,tertiary_values]).T
            density_levels = self.pdf(points_3D)

        # Check density levels
        if density_levels is None:
            density_levels = np.power(10.,np.arange(-30,0,5))
        else:
            density_levels = np.array(density_levels)
            if density_levels.ndim == 0:
                density_levels = np.array([density_levels])
            level_sort = np.argsort(density_levels)
            density_levels = density_levels[level_sort]
            if marginal_values is not None: marginal_values = marginal_values[level_sort]

        # Set labels
        if labels is None:
            if marginal_values is None:
                labels = np.array([f"p = {d:.0e}" for d in density_levels])
            else:
                labels = np.array(
                    [r"Contour $\it{"+f"{self.semantics['symbols'][slice_dim]}"+r"}$"+
                     f" = {m} {self.semantics['units'][slice_dim]}" for m in marginal_values])
        elif np.array(labels).size != density_levels.size:
            raise ValueError("Number of labels does not match levels.")
        else:
            labels = np.array(labels)[level_sort]

        # Check axes
        if ax is None:
            _,ax = plt.subplots()

        # Check colors
        if "c" not in kwargs and "color" not in kwargs:
            if cmap is None:
                cmap = "viridis"
            if type(cmap) is str:
                cmap = plt.get_cmap(cmap)
            kwargs["colors"] = cmap(np.linspace(0,1,len(density_levels)))

        # Which dimension's semantics that should be plotted on which axis (x,y,z)
        if slice_dim == 0:
            if self.swap_axis:
                axis_labels = (2,1)
                X,Y = Y,X
            else:
                axis_labels = (1,2)
        if slice_dim == 1:
            axis_labels = (0,2)
        if slice_dim == 2:
            axis_labels = (0,1)

        # Check if requested density levels exist on sampled pdf and act accordingly
        valid_density = (density_levels>=Z.min()) & (density_levels<=Z.max())
        if np.any(~valid_density):
            if invalid_density_policy == "raise":
                raise ValueError(f"Probability density {density_levels[valid_density]} "
                                 f"is outside pdf limits [{Z.min(), Z.max()}].")
            elif invalid_density_policy == "drop":
                labels = labels[valid_density]
                density_levels = density_levels[valid_density]

        # Plot contour
        CS = ax.contour(X,Y,Z,levels=density_levels,**kwargs)
        handles,_ = CS.legend_elements()
        self.plot_semantics(ax,*axis_labels)

        ax.set_title(r"Slice at $\it{"+f"{self.semantics['symbols'][slice_dim]}"+r"}$"+
            f" = {slice_value} {self.semantics['units'][slice_dim]}")

        self.legend_handles += handles
        self.legend_labels += list(labels)

        # The contour doesn't set correct limits, these must be manually provided.
        if CS.allsegs:
            contour = np.concatenate([np.array(line) for seg in CS.allsegs for line in seg],axis=0)
            ax.dataLim.bounds = (*contour.min(axis=0), *contour.max(axis=0))
            self._reset_axlim(ax)
        else:
            warnings.warn("No contours of the model exist for these values.")

        return ax

    def plot_3D_isodensity_contour(
            self,
            ax:plt.Axes=None,
            density_level:float=None,
            marginal_value:float=None,
            slice_dim:int = 0,
            slice_values:int|list[float]=20,
            grid_steps = 1000,
            contour_points = 100,
            limits:list[tuple]=None,
            cmap=None,
            **kwargs
            ):
        """
        3D contour based on isodensity. 
        Principle: Create slices of pdf, get 2D isodensity contour line
        for each slice, and connect them to get a 3D surface contour.

        Parameters
        -----------
        ax : matplotlib axes
            3D axes.
        density_level : float
            A probability density level. Either this or marginal_value can be set.
        marginal_value : float
            A value of the marginal, primary variable of the model, e.g., wind. 
            Used to get conditional values of the other variables, e.g., hs and tp.
            The 3D point is used to sample the pdf to get the density.
        slice_dim : int, default 0
            The variable/dimension to cut through.
        slice_values : int or list of float, optional
            Number of slices (resolution) in the cut dimension. Similar to grid_steps but more costly.
            Alternatively - a list of values.
        grid_steps : int, optional
            How precicely the model pdf is sampled in the two other dimensions.
        contour_points : int, optional
            Number of points on each of the 2D contours interpolated from the isodensity lines.
        limits : list of 3 tuples of 2 floats, optional
            Three tuples defining limits of pdf sampling and slicing.
            Effectively sets the limits of the plot, and the contour.
        **kwargs : keyword arguments
            Keyword arguments will be passed to matplotlib's ax.plot_surface().
        """
        # Check density
        if density_level is None and marginal_value is None:
            raise ValueError("Either density_level or marginal_value must be set.")
        if np.array(density_level).size!=1 or np.array(marginal_value).size!=1:
            raise ValueError("For the 3D contour, only one density or marginal value may be chosen.")

        # Check limits
        if np.array(limits).size == 1: # includes None
            limits = 3 if limits is None else limits
            lim_lower = np.minimum([0,0,0],self.data.min().to_numpy())
            limits = np.array([lim_lower*limits,self.data.max().to_numpy()*limits]).T
            if marginal_value is not None:
                limits[slice_dim] = [limits[slice_dim][0],marginal_value]
        elif np.array(limits).shape != (3,2):
            raise ValueError("Limits should be shape (3,2)"
                             " corresponding to lower and upper"
                             " limits per dimension.")
        limits = np.sort(limits,axis=1)

        # Not very efficient but it works
        fig,contours = plt.subplots()
        if not hasattr(slice_values,"__len__"):
            slice_values = np.linspace(limits[slice_dim][0],limits[slice_dim][1],slice_values)
        for slice_value in slice_values:
            self.plot_3D_isodensity_contour_slice(
                ax=contours,density_levels=density_level,marginal_values=marginal_value,
                slice_dim=slice_dim,slice_value=slice_value,grid_steps=grid_steps,limits=limits)
        plt.close(fig)

        # Extract and interpolate contours to grid
        contours = [c for c in contours.get_children() if "QuadContourSet" in str(c)]
        X = np.zeros(shape=(len(contours),contour_points))
        Y = np.zeros(shape=(len(contours),contour_points))
        Z = np.zeros(shape=(len(contours),contour_points))
        for i,c in enumerate(contours):
            if c.allsegs: 
                x,y = c.allsegs[0][0].T
                xi = np.linspace(0,len(x),contour_points)
                xp = np.arange(len(x))
                X[i] = np.interp(xi,xp,x)
                yi = np.linspace(0,len(x),contour_points)
                yp = np.arange(len(y))
                Y[i] = np.interp(yi,yp,y)
                Z[i] = np.ones(contour_points)*slice_values[i]
            else:
                X[i] = None
                Y[i] = None
                Z[i] = None

        if ax is None:
            _,ax = plt.subplots(subplot_kw={"projection":"3d"})
        ax.plot_surface(Z,X,Y,**kwargs)
        self.plot_semantics(ax=ax)

        return ax

    def get_contour_maximum(
            self,
            return_period,
            state_duration = 1,
            dim = 0,
            n_samples = 1000,
            contour_method="IFORM",
            ):
        """
        Method to retrieve the marginal extreme and corresponding conditional values, 
        from a contour corresponding to a given return period.

        Parameters
        -----------
        return_period : int
            Return period, in years.
        state_duration : float
            The average duration (hours) of each sample 
            in the data used to fit the model.
        primary_dim : int
            Get the point on the contour which is largest in this dimension.
        n_samples : int, default 720
            Number of contour points to generate. May need to be tuned for 
            accuracy vs speed, depending on contour method.
        contour_method : str
            The method of calculating a contour. One of:

             - "IFORM", the inverse first-order reliability method
             - "ISORM", the inverse second-order reliability method
             - "HighestDensity"
             - "DirectSampling"
             - "ConstantAndExceedance"
             - "ConstantOrExceedance"
        """

        contour = get_contour(self,return_period=return_period,state_duration=state_duration,method=contour_method,n_samples=n_samples)
        coords = contour.coordinates.T
        idxmax = np.argmax(coords[dim])
        return {p:v for p,v in zip(self.data.columns,coords[:,idxmax])}

    def table_contour(
            self,
            return_period,
            range_values = None,
            range_dim = -2,
            slice_value = None,
            slice_dim = -3,
            state_duration = 1,
            range_interval = 0.5,
            n_samples = None,
            slice_width = 0.01,
            contour_method = "IFORM"
        ):
        """
        Summarize a contour in the form of a table, i.e., a list of points lying on the contour.
        The function is general to 2 or 3 dimensions.

        For example: Get the Tp values corresponding to Hs = 1,2,3,4,5 for a 100-year return period.

        Parameters
        ----------
        return_period : float
            Return period in years.
        range_values : float or list of floats
            A value or list of values, to sample the contour along range_dim.
            This becomes the rows of the resulting table.
        range_dim : int
            The dimension along which to sample the range_values.
        fixed_value : float
            A value for the fixed dimension. Only needed if the model is 3D.
        fixed_dim : int
            The dimension to slice at.
        state_duration : float
            The average duration (hours) of each sample 
            in the data used to fit the model.
        n_samples : int, default 1e3 (2D) or 1e6 (3D)
            Number of contour points to generate. May need to be tuned for 
            accuracy vs speed, depending on contour method.
        contour_method : str
            The method of calculating a contour. One of:

             - "IFORM", the inverse first-order reliability method
             - "ISORM", the inverse second-order reliability method
             - "HighestDensity"
             - "DirectSampling"
             - "ConstantAndExceedance"
             - "ConstantOrExceedance"

        Notes
        -----
        The method relies on generating many contour points, then 
        a thin 2D "slice" at a fixed value (if the model is 3D),
        then a thin "band" at different values of the range-dimension.
        For each such band, the smallest and highest contour value of the 
        last "free" dimension is found.
        
        There is a significant tradeoff - a large number of points 
        is expensive, but a smaller number of points requires 
        wider margins to guarantee at least one small and large
        point within each slice/band, this increases error.

        A more elegant solution would be interpolation, requiring fewer points.
        """

        if not n_samples: n_samples = int(1e6) if self.n_dim==3 else int(1e5)

        contour = get_contour(
            self,return_period=return_period,state_duration=state_duration,
            method=contour_method,n_samples=n_samples,point_distribution="random")
        coords = contour.coordinates.T

        if range_dim<0:range_dim=self.n_dim+range_dim
        if slice_dim<0:slice_dim=self.n_dim+slice_dim
        free_dim = [i for i in range(self.n_dim) if (i!=range_dim) and (i!=slice_dim)]
        if len(free_dim) != 1: raise ValueError(
            "range_dim and slice_dim must be in [-3,-2,-1,0,1,2] and different.")
        free_dim = free_dim[0]

        column_names = [] if self.n_dim==2 else [f'{self.semantics['names'][slice_dim]}']
        column_names.append(f'{self.semantics['names'][range_dim]}')
        column_names.append(f'{self.semantics['names'][free_dim]}: {return_period}-year-low')
        column_names.append(f'{self.semantics['names'][free_dim]}: {return_period}-year-high')

        if self.n_dim == 3:
            coords = coords[:,np.abs(coords[slice_dim]-slice_value)<slice_width]
            coords = coords[[i for i in range(3) if i!=slice_dim]]
            if range_dim > slice_dim: range_dim -= 1
            if free_dim > slice_dim: free_dim -= 1

        if range_values is None:
            range_from = np.ceil(coords[range_dim].min()/range_interval)*range_interval
            range_to = np.ceil(coords[range_dim].max()/range_interval)*range_interval
            range_values = np.arange(range_from,range_to,range_interval)
        else:
            range_values = np.atleast_1d(range_values)

        table = []
        for range_value in range_values:
            band = coords[:,np.abs(coords[range_dim]-range_value)<slice_width]
            if len(band[0])<10: 
                warnings.warn(f"Too little data ({len(band[0])}) at {range_value}, outside contour? Or increase n_samples.")
                continue
            free_low, free_high = band[free_dim].min(),band[free_dim].max()
            row = [] if self.n_dim==2 else [slice_value]
            table.append(row+[range_value,free_low,free_high])

        return pd.DataFrame(table,columns=column_names)

    def table_isodensity_contour(
            self,
            range_values = None,
            range_dim = -2,
            slice_value = None,
            slice_dim = -3,
            range_interval = 0.5,
            levels:list[float]=None,
            points:np.ndarray=None,
            n_grid_steps=720,
            ):
        """
        Retrieve iso-density contour values in table format.

        Parameters
        -----------
        levels : list[float]
            List of probabilities to plot. Either this or return_values
            can be specified.
        points : np.ndarray
            A list of marginal return values, of e.g. Hs,
            from which to draw the isodensity contours.
            This can also be a list of 2D points,
            in which case they will be considered 
            joint extremes (any point on a contour).
        labels : list of strings, optional
            Labels to use in the figure legend.
        limits : tuple
            Plot limits, e.g. ([0,10],[0,20])
        n_grid_steps : int, default 720
            The resolution of the contours (higher is better).
            Increase this number if the contours are not smooth.
        cmap : matplotlib colormap, optional
            E.g. "viridis", "Spectral", etc.
        contour_kwargs : dict, optional
            Any other keyword arguments for matplotlib.pyplot.contour().
            Example: {"linewidths": [1,2,3], "linestyles": ["solid","dotted","dashed"]}.
        """

        # Check dimensions
        if range_dim<0:range_dim=self.n_dim+range_dim
        if slice_dim<0:slice_dim=self.n_dim+slice_dim
        free_dim = [i for i in range(self.n_dim) if (i!=range_dim) and (i!=slice_dim)]
        if len(free_dim) != 1: raise ValueError("range_dim and slice_dim must be in [-3,-2,-1,0,1,2] and different.")
        free_dim = free_dim[0]

        column_names = [] if self.n_dim==2 else [f'{self.semantics['names'][slice_dim]}']
        column_names.append(f'{self.semantics['names'][range_dim]}')

        self.reset_labels()
        swap_axis_flag = self.swap_axis
        self.swap_axis = False

        if self.n_dim == 2:
            ax = self.plot_isodensity_contours(levels=levels,points=points,n_grid_steps=n_grid_steps)

        if self.n_dim == 3:
            ax = self.plot_3D_isodensity_contour_slice(
                density_levels=levels,
                marginal_values=points,
                slice_dim=slice_dim,
                slice_value=slice_value,
                grid_steps=n_grid_steps)            
        self.swap_axis = swap_axis_flag

        contours = [c for c in ax.get_children() if "QuadContourSet" in str(c)][0].allsegs[::-1]

        for label in self.legend_labels:
            column_names += ["Contour: "+label + f', {self.semantics['symbols'][slice_dim]}=low']
            column_names += ["Contour: "+label + f', {self.semantics['symbols'][slice_dim]}=high']

        # plt.close()

        # Reduce to 2 dimensions
        if self.n_dim == 3:
            if range_dim > slice_dim: range_dim -= 1
            if free_dim > slice_dim: free_dim -= 1

        # Check range_values
        if range_values is None:
            min_val = np.min([c[0][:,range_dim].min() for c in contours])
            max_val = np.max([c[0][:,range_dim].max() for c in contours])
            range_from = np.ceil(min_val/range_interval)*range_interval
            range_to   = np.floor(max_val/range_interval)*range_interval
            range_values = np.arange(range_from.round(5),range_to.round(5)+range_interval,range_interval)
        else:
            range_values = np.atleast_1d(range_values)

        if self.n_dim == 2:
            table = []
        elif self.n_dim == 3:
            table = [[slice_value]*len(range_values)]
        table.append(range_values)
        for i,contour in enumerate(contours):
            lhs,rhs = split_contour(contour[0],split_dim=free_dim)

            tp_lhs = interp1d(lhs[:,range_dim],lhs[:,free_dim],bounds_error=False)(range_values)
            tp_rhs = interp1d(rhs[:,range_dim],rhs[:,free_dim],bounds_error=False)(range_values)

            table.append(tp_lhs)
            table.append(tp_rhs)

        return pd.DataFrame(table,index=column_names).T

    def get_default_semantics(self):
        """
        Generate default semantics.

        Returns
        -------
        semantics: dict
            Generated model description.

        References
        ----------
        Slightly modified from virocon/plotting/get_default_semantics().
        https://github.com/virocon-organization/virocon
        """
        return {
            "names": [f"Variable {c}" for c in self.data.columns],
            "symbols": [" - " for _ in range(self.n_dim)],
            "units": [" - " for _ in range(self.n_dim)],
            "swap_axis":False,
        }

    def plot_semantics(self,ax=None,x=None,y=None,z=None, names=True, symbols=True, units=True):
        """
        Plot model semantics on matplotlib axis.
        By default, all semantics are plotted.
        
        Parameters
        ----------
        ax : matplotlib axis
            Axis to add semantics to.
        x : int
            Semantics for x-axis.
        y : int
            Semantics for y-axis.
        z : int
            Semantics for z-axis.
        """
        if x is None and y is None and z is None:
            if self.n_dim == 2:
                x,y = (1,0) if self.swap_axis else (0,1)
            if self.n_dim == 3:
                x,y,z = (0,1,2)

        if ax is None:
            if self.n_dim == 3:
                _,ax = plt.subplots(subplot_kw={"projection":"3d"})
            else:
                _,ax = plt.subplots()

        def _label(dim):
            label = ""
            if names:   label += f"{self.semantics['names'][dim]}"
            if symbols: label += r" $\it{"+f"{self.semantics['symbols'][dim]}"+r"}$"
            if units:   label += f" ({self.semantics['units'][dim]})"
            return label

        if x is not None:
            ax.set_xlabel(_label(x))
        if y is not None:
            ax.set_ylabel(_label(y))
        if z is not None:
            ax.set_zlabel(_label(z))

        return ax

    def parameters(self,complete=True):
        """
        Get a dictionary containing the value of all parameters, including within dependency functions.

        Parameters
        ----------
        complete : bool, default True
             - If True, a pandas dataframe is returned containing a complete description of each value.
             - If False, a dictionary is returned, where the keys are distribution parameters if the 
                distribution is marginal, or dependency parameter if the distribution is dependent.
                It is useful if the goal is to just compare parameters between fits.
                It requires that all parameters, including dependencies, are uniquely named, to not overwrite entries.
                For example, if you have two dependency functions with parameters a1,a2,a3, refactor one to b1,b2,b3.
        """
        parameters = [] if complete else {}
        for i,dis in enumerate(self.distributions):
            # Check dependency functions
            if self.conditional_on[i] is not None:
                for param,conditional in dis.conditional_parameters.items():
                    for p,v in conditional.parameters.items():
                        if complete:
                            eq = conditional.latex # dependency function
                            parameters.append({
                                "Distribution":type(dis.distribution).__name__.removesuffix("Distribution"),
                                "Variable":self.semantics["symbols"][i],
                                "Distribution parameter":param,
                                "Dependent on":self.semantics["symbols"][self.conditional_on[i]],
                                "Dependency function":"Missing" if eq is None else eq.replace("$",""),
                                "Dependency parameter":p,
                                "Value":v})
                        else:
                            key = p
                            if key in parameters:
                                raise KeyError(f"Parameter {key} is not uniquely defined.")
                            parameters[key]=v
            # Check marginal distributions as well as fixed parameters of conditional distributions
            if self.conditional_on[i] is None:
                fixed_params = dis.parameters.items()
                dist_type = type(dis)
            elif complete:
                fixed_params = dis.fixed_parameters.items()
                dist_type = type(dis.distribution)
            else: # don't include fixed parameters in the short parameter summary.
                fixed_params = {}
            for param,value in fixed_params:
                if complete:
                    parameters.append({
                        "Distribution":dist_type.__name__.removesuffix("Distribution"),
                        "Variable":self.semantics["symbols"][i],
                        "Distribution parameter":param,
                        "Dependent on":"-" if self.conditional_on[i] is None else "Fixed",
                        "Dependency function":"-",
                        "Dependency parameter":"-",
                        "Value":value})
                else:
                    key = param
                    if key in parameters:
                        raise ValueError(f"Parameter {key} is not uniquely defined.")
                    parameters[key]=value
        return pd.DataFrame(parameters) if complete else parameters

    def __str__(self):
        """
        Printout of the model.
        """
        return self.parameters(complete=True)

    def _reset_axlim(self,ax):
        """
        Set appropriate plot limits based on content.
        """
        x0,y0,x1,y1 = ax.dataLim.bounds

        x1 *= 1.05
        y1 *= 1.05


        if x0>=0 and x1>=0:
            ax.set_xlim([0,x1])
        else:
            ax.set_xlim([x0,x1])

        if y0>=0 and y1>=0:
            ax.set_ylim([0,y1])
        else:
            ax.set_ylim([y0,y1])

    def reset_labels(self):
        """
        Reset handles and labels stored in the class instance.
        """
        self.legend_handles = []
        self.legend_labels = []
    
    def plot_legend(self,ax,**legend_kwargs):
        """
        Add legend based on stored handles and labels.
        """
        if "loc" not in legend_kwargs:
            legend_kwargs["loc"] = "upper left"
        ax.legend(handles=self.legend_handles,labels=self.legend_labels,**legend_kwargs)
        return ax