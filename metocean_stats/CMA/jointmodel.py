import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches as mpatches

import virocon
import virocon.intervals
from virocon import GlobalHierarchicalModel

from .plotting import (
    plot_2D_pdf_heatmap,
    plot_DNVGL_steepness,
)

from .extremes import (
    get_dependent_given_marginal,
    get_return_values,
)

from .predefined import (
    get_DNVGL_Hs_Tz,
    get_DNVGL_Hs_U,
    get_OMAE2020_Hs_Tz,
    get_OMAE2020_V_Hs,
    get_LoNoWe_hs_tp,
    get_windsea_hs_tp
)

def _load_preset(preset):
    """Load defaults from preset."""
    preset = preset.lower()
    if preset == 'omae_hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_OMAE2020_Hs_Tz()
        swap_axis = True
    elif preset == 'dnvgl_hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_DNVGL_Hs_Tz()
        swap_axis = True
    elif preset == 'omae_u_hs':
        dist_descriptions,fit_descriptions,semantics = get_OMAE2020_V_Hs()
        swap_axis = False
    elif preset == 'dnvgl_hs_u':
        dist_descriptions,fit_descriptions,semantics = get_DNVGL_Hs_U()
        swap_axis = True
    elif preset == 'windsea_hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_windsea_hs_tp()
        swap_axis = True
    elif preset == 'lonowe_hs_tp':
        dist_descriptions,fit_descriptions,semantics = get_LoNoWe_hs_tp()
        swap_axis = True
    else:
        raise ValueError(f'Preset {preset} not found.'
                            'See docstring for available presets.')
    return dist_descriptions,fit_descriptions,semantics,swap_axis


class JointProbabilityModel(GlobalHierarchicalModel):
    """
    This is a wrapper for virocon's GlobalHierarchicalModel,
    which adds to those methods and attributes by also
    storing the data and metadata such as semantics.
    This makes it more convenient and allows some
    higher-level operations.
    """

    def __init__(
        self,
        preset:str = "DNVGL_hs_tp",
        dist_descriptions:list[dict] = None,
        fit_descriptions:list[dict] = None,
        semantics:dict[list[str]] = None,
        intervals:virocon.intervals.IntervalSlicer=None,
        swap_axis:bool = None,

        ):
        """
        Initialize the joint conditional probability model.
        A model should have been defined in predefined.py.
        Any other input arguments following the first 
        parameter will overwrite the predefined configuration.
        
        Parameters
        ------------
        preset : str
            Choose model from any predefined model.

             - DNVGL_Hs_Tp
             - DNVGL_Hs_U
             - OMAE_Hs_Tp
             - OMAE2020_V_Hs
             - LoNoWe_Hs_Tp
             - windsea_hs_tp

        dist_descriptions : list[dict], optional
            A list of dicts describing each distribution.
        fit_descriptions : list[dict], optional
            A list of dicts describing the fitting method.
        semantics : dict[list], optional
            Dict desribing the variables, used for plotting.
            Keys of the dict are "names","symbols","units", and each value is
            a list of two strings corresponding to the two variables.
        intervals : virocon.intervals.IntervalSlicer, optional
            An interval slicer (NumberOfIntervalsSlicer or WidthOfIntervalSlicer),
            from virocon.intervals. This divides the marginal intervals into
            intervals, which are used to fit the dependent parameters 
            of the conditional distribution (e.g. tp).
        swap_axis : bool, optional
            This determines if the produced plots should have swapped axes,
            as is common with e.g. hs tp plots.
        """

        if preset:
            preset_dist,preset_fit,preset_semantics,preset_axis=_load_preset(preset)
            if dist_descriptions is None:
                dist_descriptions = preset_dist
            if fit_descriptions is None:
                fit_descriptions = preset_fit
            if semantics is None:
                semantics = preset_semantics
            if swap_axis is None:
                swap_axis = preset_axis
            if intervals is not None:
                dist_descriptions[0]["intervals"] = intervals

        else:
            if dist_descriptions is None or fit_descriptions is None:
                raise ValueError("If preset is undefined, both dist_descriptions and fit_descriptions must be provided.")
            if swap_axis is None:
                swap_axis = False
            if intervals is not None:
                dist_descriptions[0]["intervals"] = intervals
        
        self.dist_descriptions = dist_descriptions
        self.fit_descriptions = fit_descriptions
        self.semantics = semantics
        self.swap_axis = swap_axis

        # Store of handles and labels, for producing a legend.
        self.legend_handles = []
        self.legend_labels = []

        super().__init__(self.dist_descriptions)

    def fit(self,data:np.ndarray|pd.DataFrame,
            var1:int|str=0,
            var2:int|str=1):
        """
        Fit the model to data.

        Parameters
        ----------
        data : Numpy array or pandas dataframe
            The data, shape (n_samples, n_columns)
        var0 : int or str, default 0 (first column)
            The data column used to fit the marginal distribution.
        var1 : int or str, default 1 (second column)
            The data column used to fit the conditional distribution.
        """

        if isinstance(data,pd.DataFrame):
            if isinstance(var1,int):
                var1 = data.columns[var1]
            if isinstance(var2,int):
                var2 = data.columns[var2]
            data = data[[var1,var2]].values
        elif isinstance(data,np.ndarray):
            data = data[:,[var1,var2]]
        else:
            raise TypeError(f"Expected dataframe or ndarray, got {type(data)}.")

        super().fit(data,self.fit_descriptions)
        self.data = data

    def plot_marginal_quantiles(self,sample=None,axes=None):
        """
        Plot the theoretical (distribution) vs empirical (sample) quantiles.
        """
        sample = self.data if sample is None else sample
        return virocon.plot_marginal_quantiles(self,sample=self.data,semantics=self.semantics,axes=axes)

    def plot_dependence_functions(self,axes=None):
        """
        Plot the dependence functions of the conditional parameters.
        """
        return virocon.plot_dependence_functions(self,self.semantics,axes=axes)

    def plot_histograms_of_interval_distributions(self,plot_pdf=True,max_cols=4):
        """
        Plot histograms for the estimated distribution in each data interval.
        """        
        return virocon.plot_histograms_of_interval_distributions(self,self.data,self.semantics,plot_pdf=plot_pdf,max_cols=max_cols)

    def plot_2D_isodensity(self,
                           levels:list[float]=None,
                           points:np.ndarray=None,
                           labels:list[str]=None,
                           ax=None,
                           limits=None,
                           n_grid_steps=720,
                           cmap=None,
                           contour_kwargs=None,
                           ):
        """
        Plot isodensity (constant probability density) 2D contours.
        Contours can be specified either through a density directly,
        or by choosing any 2D point of origin.

        Parameters:
        -----------
        levels : list[float], optional
            List of probabilities to plot. Either this or return_values
            can be specified. Not both.
        points : np.ndarray
            A list of 2D points, from which to draw isodensity contours.
            Either this or levels may be specified. Not both.
        labels : list of strings
            Labels to use in the figure legend.
        ax : matplotlib axes
            Plot on existing axes object.
        """

        if ax is None:
            _,ax = plt.subplots()

        if points is not None:
            if levels is not None:
                raise ValueError("Only one of levels or return_values may be specified. The other must be None.")
            if points.ndim == 1:
                points = np.array([points])
            levels = self.pdf(points)
            idx = np.argsort(levels)
            levels = levels[idx]

            if labels is None:
                sym0,sym1 = self.semantics["symbols"]
                labels = [sym0+f"={p[0]}"+", "+sym1+f"={p[1]}" for p in points]

        if labels is not None: 
            if isinstance(labels,str):
                labels = [labels]
            if len(labels) != len(levels):
                raise ValueError("Number of labels does not match number of contour levels!")

        if "LoNoWe" in str(type(self.distributions[0])):
            raise NotImplementedError("Isodensity not implemented for LoNoWe.")

        # Keep track of original artists.
        old_artists = ax.get_children()

        # Produce plot.
        ax = virocon.plot_2D_isodensity(
            self,
            sample=self.data,
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
                print("yeah")
            elif "ContourSet" in str(artist): # The contours.
                CShandles,CSlabels = artist.legend_elements()
                self.legend_handles += CShandles
                if labels is None:
                    self.legend_labels += CSlabels
                else:
                    self.legend_labels += labels

        return ax


    def plot_2D_contours(self,
                         periods:list[float]=[1,10,100,1000],
                         state_duration:float=1,
                         contour_method:str="IFORM",
                         labels:list[str] = None,
                         ax=None,
                         **kwargs
                         ):
        """
        Plot 2D environmental contours.

        Parameters
        -----------
        periods : list[float]
            List of return periods to create contours of.
        state_duration : float
            The average duration of each sample 
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
        for k,v in kwargs.items():
            if isinstance(v,str):
                kwargs[k] = [v]*periods.size
            elif len(v) != periods.size:
                raise ValueError(f"Keyword {k} should have length equal to number of return periods {len(periods)}.")


        contour_method = contour_method.lower()
        if contour_method == "iform":
            ContourMethod = virocon.IFORMContour
        elif contour_method == "isorm":
            ContourMethod = virocon.ISORMContour
        elif contour_method == "highestdensity":
            ContourMethod = virocon.HighestDensityContour
        elif contour_method == "directsampling":
            ContourMethod = virocon.DirectSamplingContour
        elif contour_method in ["constantandexceedance","and"]:
            ContourMethod = virocon.AndContour
        elif contour_method in ["constantorexceedance","or"]:
            ContourMethod = virocon.OrContour
        else:
            raise ValueError(f"Unknown contour method: {contour_method}")            

        for i,return_period in enumerate(periods):
            plot_kwargs = {k:v[i] for k,v in kwargs.items()}
            alpha = virocon.calculate_alpha(state_duration,return_period)
            contour = ContourMethod(self,alpha)
            old_artists = ax.get_children()
            virocon.plot_2D_contour(contour,semantics=self.semantics,swap_axis=self.swap_axis,ax=ax,plot_kwargs=plot_kwargs)
            new_artist = list(set(ax.get_children())-set(old_artists))
            self.legend_handles += new_artist
            self.legend_labels += [labels[i]]

        return ax

    def plot_2D_pdf_heatmap(
            self,
            limits = (25,25),
            bin_size = (1,1),
            bin_samples = (10,10),
            percentage_values = True,
            significant_digits = 4,
            range_including = True,
            marginal_values = True,
            log_norm = True,
            color_vmin = 1e-10,
            ax=None,
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
    
    def plot_dependent_percentiles(
            self,
            percentiles=[0.05,0.5,0.95],
            labels:list[str] = None,
            ax=None,
            **kwargs
            ):
        """
        Plot lines describing percentiles of the dependent 
        variable of the joint distribution (e.g. Tp)
        as functions of the marginal variable (e.g. Hs).
        """
        percentiles = np.array(percentiles)
        
        # Set default style.
        if labels is None:
            labels = [f"{self.semantics["symbols"][1]} - {int(100*p)}%" for p in percentiles]
        elif np.array(labels).size != percentiles.size:
            raise ValueError("Number of labels does not fit the number of percentiles.")
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["c"] = "tab:green"
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

        marginal = np.arange(0,np.max(self.data[:,0]),0.001)
        for i,p in enumerate(percentiles):
            conditional = get_dependent_given_marginal(self,marginal,p)
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
            use_peak_wave_period = True,
            ax=None,
            ylim = None,
            xlim = None,
            **kwargs
            ):
        """
        Add the DNVGL steepness line.
        
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
            ylim = np.max(self.data[:,0])
        return plot_DNVGL_steepness(ax=ax,peak_period_line=use_peak_wave_period,ylim=ylim,**kwargs)
    
    def plot_data_scatter(self,
                          data=None,
                          ax=None,
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
            data = self.data
        if data.ndim == 1:
            data = np.array([data])
        if data.ndim > 2:
            raise ValueError("Data should have exactly two dimensions: samples x variables.")
        if data.shape[1] != 2:
            raise NotImplementedError("Only 2-dimensional data is supported.")
        
        x,y = (1,0) if self.swap_axis else (0,1)
        handle = ax.scatter(data[:,x],data[:,y],**kwargs)
        if label is not None:
            self.legend_handles.append(handle)
            self.legend_labels.append(label)
        return ax

    def plot_data_density(self,
                          ax=None,
                          data=None,
                          bins=None,
                          label:str=None,
                          norm=LogNorm(),
                          density=True,
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
        density : bool, default True
            If True, becomes a density plot (sum 1). If false, a histogram..
        **kwargs : keyword arguments
            These will be passed to the hist2d plot function.
        """
        if ax is None:
            _,ax = plt.subplots()
        if data is None:
            data = self.data
        if data.ndim == 1:
            data = np.array([data])
        if data.ndim > 2:
            raise ValueError("Data should have exactly two dimensions: samples x variables.")
        if data.shape[1] != 2:
            raise NotImplementedError("Only 2-dimensional data is supported.")

        x,y = (1,0) if self.swap_axis else (0,1)
        
        if bins is None:
            bins = [
                np.arange(np.floor(np.min(data[:,x])),
                          np.ceil(np.max(data[:,x])),0.1),
                np.arange(np.floor(np.min(data[:,y])),
                          np.ceil(np.max(data[:,y])),0.1)
            ]

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
        return ax


    def reset_labels(self):
        """
        Reset handles and labels stored in the class instance.
        """
        self.legend_handles = []
        self.legend_labels = []
    
    def plot_legend(self,ax):
        """
        Add legend based on stored handles and labels.
        """
        ax.legend(handles=self.legend_handles,labels=self.legend_labels)
        return ax