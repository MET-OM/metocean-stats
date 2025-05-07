import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns


def plot_DNVGL_steepness(ax,peak_period_line=True,xlim=None,ylim=None,**kwargs):
    """
    Plot dnv gl RP 2017/2019 recommended steepness line, within the axis limits.
    Assumes t is x-axis, and h is y-axis.

    Parameters
    ----------
    ax : matplotlib axes
        The figure to plot on. Works best if data has already been plotted. 
    peak_period : bool, default True
        Use the line corresponding to peak wave period (Tp). False will give the steepness line of mean wave period (Tz).
    **kwargs : keyword arguments
        Any matplotlib.plot() keyword arguments (color, linestyle, etc.).
    """

    # Get h and t.
    wave_period_type = 'tp' if peak_period_line else 'tz'
    if xlim is None and ylim is None:
        xlim,ylim = 40,20
    if ylim is not None:
        h = np.linspace(0,ylim,10000)
        t = get_DNVGL_steepness(h,"hs",wave_period_type)
    elif xlim is not None:
        t = np.linspace(0,xlim,10000)
        h = get_DNVGL_steepness(t,wave_period_type,"hs")

    # These are defaults, to be overwritten by **kwargs.
    plot_params = {**{
        'linestyle':'--',
        'color':'black',
    },**kwargs}
    ax.plot(t,h,**plot_params)

    return ax

def get_DNVGL_steepness(x,input='tp',output='hs'):
    if np.any(x<0): 
        raise ValueError('Values must be real and positive.')
    request = input+output

    # Define constants (dependng on tp or tz)
    g = 9.81
    if request in ['hstp','tphs']:
        sl, su = 1/15, 1/25
        tl, tu = 8, 15
    elif request in ['hstz','tzhs']:
        sl, su = 1/10, 1/15
        tl, tu = 6, 12
    else:
        raise ValueError('Unknown input-output request.')

    inverse = True if input=='hs' else False

    # Define functions for t->h or h->t
    def t_to_h(t):
        def low(t): return sl*(t**2)*g/(2*np.pi)
        def high(t):return su*(t**2)*g/(2*np.pi)
        def mid(t): return np.interp(t,[tl,tu],[low(tl),high(tu)])
        return np.piecewise(t,[t<=tl,t>=tu],[low,high,mid])

    if inverse:
        t = np.linspace(0,100,10000)
        h = t_to_h(t)
        return np.interp(x,h,t)
    else:
        return t_to_h(x)

def plot_2D_pdf_heatmap(
        model,
        semantics,
        limits = (25,25),
        bin_size = (1,1),
        bin_samples = (10,10),
        percentage_values = True,
        significant_digits = 4,
        range_including = True,
        marginal_values = True,
        log_norm = True,
        include_cbar = True,
        color_vmin = 1e-9,
        annot_vmin = 1e-12,
        linewidths=0.1,
        linecolor="tab:orange",
        ax=None,
        **kwargs
):
    """
    Visualize the PDF of 2D model as a heatmap. See options for details.

    Parameters
    ----------
    model : GlobalHierarchicalModel
        The model with a .pdf() method.
    semantics : dict
        Dictionary describing the variables - see predefined.py for examples.
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
    include_cbar : bool, default True
    color_vmin : float, default 1e-10
        Minimum value for the color scale.
    annot_vmin : float, default 1e-12
        Minimum value of annotated values.
    linewidths : float, default 0.1
        Width of the gridlines.
    linecolor : str, default "tab:orange"
    ax : matplotlib axes object, default None
        Existing axes object, otherwise a new will be created.
    **kwargs : keyword arguments
        Any other keyword arguments will be passed directly to seaborn's heatmap.
    """
    # Number of boxes, samples per box
    max_hs, max_tp = limits
    step_hs, step_tp = bin_size
    N_hs = int(max_hs/step_hs)
    N_tp = int(max_tp/step_tp)
    n_hs, n_tp = bin_samples

    # PDF query matrix
    h = np.linspace(0,max_hs,n_hs*N_hs,endpoint=False)
    t = np.linspace(0,max_tp,n_tp*N_tp,endpoint=False)
    ht = np.array(np.meshgrid(h,t,indexing='ij')).reshape(2,-1).swapaxes(0,1)

    # PDF query
    pdf = model.pdf(ht).reshape(len(h),len(t))
    pdf_int = pdf.reshape(N_hs,n_hs,N_tp,n_tp)
    pdf_int = pdf_int.sum(axis=(1,3))
    pdf_int = pdf_int/pdf_int.sum()

    if percentage_values:
        pdf_int = pdf_int*100

    marginal_hs = pdf_int.sum(axis=1)
    marginal_tp = pdf_int.sum(axis=0)

    # Writer for the x and y labels/ticks.
    def _tick_writer(a,b,c):
        a = int(a) if np.isclose(int(a),a) else np.round(a,1)
        b = int(b) if np.isclose(int(b),b) else np.round(b,1)
        c = str(int(c)) if not percentage_values else f'{c:.{significant_digits}f}'

        tick = f'{a}'
        if range_including: 
            tick = tick + f'-{b}'
        if marginal_values:
            if percentage_values:
                tick = tick + f' | {c}%'
            else:
                tick = tick + f' | {c}'
        return tick

    H = h[::n_hs]+step_hs
    T = t[::n_tp]+step_tp
    yticks = [_tick_writer(a,b,c) for a,b,c in zip(H-step_hs,H,marginal_hs)]
    xticks = [_tick_writer(a,b,c) for a,b,c in zip(T-step_tp,T,marginal_tp)]

    norm = LogNorm(vmin=color_vmin) if log_norm else None
    
    if ax is None:
        _,ax = plt.subplots()
    
    mask = pdf_int < annot_vmin

    sns.heatmap(
        data = pdf_int,
        xticklabels=xticks,
        yticklabels=yticks,
        linewidths=linewidths,
        norm=norm,
        cbar=include_cbar,
        linecolor=linecolor,
        annot=True,
        ax=ax,
        mask=mask,
        **kwargs)
    ax.invert_yaxis()

    x_idx, y_idx = 1,0
    x_name = semantics["names"][x_idx]
    x_symbol = semantics["symbols"][x_idx]
    x_unit = semantics["units"][x_idx]
    x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
    y_name = semantics["names"][y_idx]
    y_symbol = semantics["symbols"][y_idx]
    y_unit = semantics["units"][y_idx]
    y_label = f"{y_name}," + r" $\it{" + f"{y_symbol}" + r"}$" + f" ({y_unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax
