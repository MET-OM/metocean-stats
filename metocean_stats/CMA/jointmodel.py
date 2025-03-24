import pandas as pd
import virocon
import numpy as np
from virocon import GlobalHierarchicalModel
import virocon.intervals
import matplotlib.pyplot as plt

from .plotting import (
    plot_2D_pdf_heatmap,
    plot_DNVGL_steepness
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
    elif preset == 'omae_U_hs':
        dist_descriptions,fit_descriptions,semantics = get_OMAE2020_V_Hs()
        swap_axis = False
    elif preset == 'dnvgl_hs_U':
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
        model_config : str
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
        
        self.dist_descriptions = dist_descriptions
        self.fit_descriptions = fit_descriptions
        self.semantics = semantics
        self.swap_axis = swap_axis

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

    def plot_histograms_of_interval_distributions(self,plot_pdf=True):
        """
        Plot histograms for the estimated distribution in each data interval.
        """        
        return virocon.plot_histograms_of_interval_distributions(self,self.data,self.semantics,plot_pdf=plot_pdf)

    def plot_2D_isodensity(self,levels:list[float]=None,ax=None):
        """
        Plot isodensity lines.

        Parameters:
        -----------
        levels : list[float], optional
            List of probabilities to plot. If None, appropriate values are found.
        ax : matplotlib axes
            Plot on existing axes object.
        """
        return virocon.plot_2D_isodensity(self,self.data,self.semantics,self.swap_axis,ax=ax, levels=levels)

    def plot_2D_contours(self,
                         periods:list[float]=[1,10,100,1000],
                         state_duration:float=1,
                         contour_method:str="IFORM",
                         ax=None):
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
        """

        if ax is None:
            _,ax = plt.subplots()

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

        for return_period in periods:
            alpha = virocon.calculate_alpha(state_duration,return_period)
            contour = ContourMethod(self,alpha)
            virocon.plot_2D_contour(contour,semantics=self.semantics,swap_axis=self.swap_axis,ax=ax)

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
    