import numpy as np
import xarray as xr
import scipy
import pandas as pd

def jonswap(f,hs,tp,gamma='fit', sigma_low=.07, sigma_high=.09):
    """
    Purpose:
        To determine spectral density based on the JONSWAP spectrum 

    Input:
        hs  - Significant wave height
        tp  - Spectral peak period
        f   - array of Wave frequency

    Output:
        sf  - Spectral density
    """
    g = 9.82
    fp = 1/tp
    if gamma == 'fit':
       gamma = min(np.exp(3.484*(1-0.1975*(0.036-0.0056*tp/np.sqrt(hs))*(tp**4)/hs**2)),5) # see MET-report_03-2021.pdf, max value should not exceed 5
    else:
       pass
    
    #print('gamma-JONSWAP is',gamma)
    alpha  = 5.061*(hs**2/tp**4)*(1-0.287*np.log(gamma)) # see MET-report_03-2021.pdf
    E_pm = alpha*(g**2)*((2*np.pi)**-4)*f**-5*np.exp((-5/4)*((fp/f)**4))
    sigma = np.ones(f.shape)*sigma_low
    sigma[f > 1./tp] = sigma_high
    E_js = E_pm*gamma**np.exp(-0.5*(((f/fp)-1)/sigma)**2)   # see MET-report_03-2021.pdf
    sf = np.nan_to_num(E_js)
    return sf

def velocity_spectrum(f, S_w, depth, ref_depth):
    # g = 9.82
    h = depth 
    z = ref_depth
    #k = (1/g)*(2*np.pi/f)**2
    k, ik = waveno(t=1/f,h =depth) 
    k = np.nan_to_num(k)
    G = 2*np.pi *f* np.cosh(k*(depth-ref_depth))/np.sinh(k*ref_depth)
    G = 2*np.pi*f*np.exp(-k*z)*(1+np.exp(-2*(k*h-k*z)))/(1.-np.exp(-2*k*h))
    G = np.nan_to_num(G)
    S_uu = S_w*G**2
    return S_uu



def torsethaugen(f, hs, tp):
    """
    Purpose:
        To determine spectral density based on the Torsethaugen double peaked spectrum 

    Input:
        hs  - Significant wave height
        tp  - Spectral peak period
        f   - array of Wave frequency

    Output:
        sf  - Spectral density
    """
    # Constants
    pi = np.pi
    g = 9.81
    af, ae, au = 6.6, 2.0, 25.0
    a10, a1, rkg = 0.7, 0.5, 35.0
    b1, a20, a2, a3 = 2.0, 0.6, 0.3, 6.0
    g0 = 3.26

    tpf = af * hs ** (1.0 / 3.0)
    tl = ae * hs ** (1.0 / 2.0)
    el = (tpf - tp) / (tpf - tl)

    if tp <= tpf:
        rw = (1.0 - a10) * np.exp(-(el / a1) ** 2) + a10
        hw1 = rw * hs
        tpw1 = tp
        sp = (2.0 * pi / g) * hw1 / tpw1 ** 2
        gam1 = max(1.0, rkg * sp ** (6.0 / 7.0))

        hw2 = np.sqrt(1.0 - rw ** 2) * hs
        tpw2 = tpf + b1
        h1, tp1 = hw1, tpw1
        h2, tp2 = hw2, tpw2
    else:
        tu = au
        eu = (tp - tpf) / (tu - tpf)
        rs = (1.0 - a20) * np.exp(-(eu / a2) ** 2) + a20
        hs1 = rs * hs
        tps1 = tp
        sf = (2.0 * pi / g) * hs / tpf ** 2
        gam1 = max(1.0, rkg * sf ** (6.0 / 7.0) * (1.0 + a3 * eu))
        hs2 = np.sqrt(1.0 - rs ** 2) * hs
        tps2 = af * hs2 ** (1.0 / 3.0)
        h1, tp1 = hs1, tps1
        h2, tp2 = hs2, tps2

    e1 = (1.0 / 16.0) * (h1 ** 2) * tp1
    e2 = (1.0 / 16.0) * (h2 ** 2) * tp2
    ag = (1.0 + 1.1 * np.log(gam1) ** 1.19) / gam1

    f1n, f2n = f * tp1, f * tp2
    
    sigma1 = np.where(f1n > 1.0, 0.09, 0.07)
    
    fnc1 = f1n ** (-4) * np.exp(-f1n ** (-4))
    fnc2 = gam1 ** (np.exp(-1.0 / (2.0 * sigma1 ** 2) * (f1n - 1.0) ** 2))
    s1 = g0 * ag * fnc1 * fnc2

    fnc3 = f2n ** (-4) * np.exp(-f2n ** (-4))
    s2 = g0 * fnc3

    sf = e1 * s1 + e2 * s2
    return np.nan_to_num(sf)


def waveno(t, h):
    """
    Purpose:
        To compute wave number

    Input:
        t  - Wave period
        h  - Water depth (can be an array or a single value)

    Output:
        k - Wave number 
        nier - Negative depth values: nier = 1
    """
    # Set value of constants
    g = 9.82
    A = [0.66667, 0.35550, 0.16084, 0.06320, 0.02174, 0.00654, 0.00171, 0.00039, 0.00011]

    # Initialize output
    nier = 0
    sigma = 2.0 * np.pi / t

    # Function to compute wave number for a single depth
    def compute_k(depth):
        nonlocal nier
        b = g * depth
        if b < 0:
            nier = 1
            return 0.0
        y = sigma * sigma * depth / g
        x = A[4] + y * (A[5] + y * (A[6] + y * (A[7] + y * A[8])))
        x = 1.0 + y * (A[0] + y * (A[1] + y * (A[2] + y * (A[3] + y * x))))
        c = np.sqrt(b / (y + 1.0 / x))
        return sigma / c

    # Check if h is a single value or a list
    if isinstance(h, (int, float)):
        k = compute_k(h)
    else:
        k = [compute_k(depth) for depth in h]

    return k, nier

def _interpolate_linear(fp,n):
    '''
    Linear interpolation.
    '''
    L = len(fp)
    x = np.linspace(0,L-1,n)
    xp = np.arange(L)
    return np.interp(x,xp,fp)
    
def _interpolate_cubic(fp,n):
    '''
    Cubic spline interpolation.
    '''
    L = len(fp)
    x = np.linspace(0,L-1,n)
    xp = np.arange(L)
    spl = scipy.interpolate.CubicSpline(xp,fp)
    return spl(x)

def interpolate_2D_spec(spec  : np.ndarray,
                        freq0 : np.ndarray,
                        dir0  : np.ndarray,
                        freq1 : np.ndarray,
                        dir1  : np.ndarray,
                        method: str="linear"):
    '''
    Interpolate 2D wave spectra from fre0 and dir0 to freq1 and dir1.
    
    Arguments
    ---------
    spec : np.ndarray
        N-D array of spectra, must have dimensions [..., frequencies, directions].
    freq0 : np.ndarray
        Array of frequencies.
    dir0 : np.ndarray
        Array of directions.
    freq1 : np.ndarray
        Array of new frequencies.
    dir1 : np.ndarray
        Array of new directions.
    method : str
        The interpolation method used by scipy.interpolate.RegularGridInterpolator(),
        e.g. "nearest", "linear", "cubic", "quintic".
        
    Returns
    -------
    spec : np.ndarray
        The interpolated spectra.
    '''
    # Sort on directions, required for interpolation.
    sorted_indices = np.argsort(dir0)
    dir0 = dir0[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Create current and new interpolation points.
    points = tuple(np.arange(s) for s in spec.shape[:-2]) + (freq0,dir0)
    coords = tuple(np.arange(s) for s in spec.shape[:-2]) + (freq1,dir1)
    reorder = tuple(np.arange(1,len(coords)+1))+(0,)
    coords = np.transpose(np.meshgrid(*coords,indexing="ij"),reorder)

    # Define interpolator and interpolate.
    grid_interp = scipy.interpolate.RegularGridInterpolator(points=points,values=spec)
    return grid_interp(coords,method=method)

def interpolate_dataarray_spec( spec: xr.DataArray,
                                new_frequencies: np.ndarray | int = 20,
                                new_directions: np.ndarray | int = 20,
                                method="linear"):
    '''
    Interpolate 2D wave spectra to a new shape.
    The last two dimensions of spec must represent frequencies and directions.
    
    Arguments
    ---------
    spec : xr.DataArray
        Array of spectra. Must have dimensions [..., frequencies, directions].
    new_frequencies : xr.DataArray or np.ndarray or int
        Either an array of new frequences, or an integer for the number of new frequencies.
        If integer, new frequencies will be created with cubic interpolation.
    new_directions : xr.DataArray or np.ndarray or int
        Either an array of new directions, or an integer for the number of new directions.
        If integer, new directions will be created with linear interpolation.
    method : str
        The interpolation method used by scipy.interpolate.RegularGridInterpolator(),
        e.g. "nearest", "linear", "cubic", "quintic".
    
    Returns
    -------
    spec : xr.DataArray
        The 2D-interpolated spectra.
    '''

    # Extract dimension labels and coordinate arrays from spec.
    spec_coords = spec.coords
    spec_dims = list(spec.dims)
    freq_var = spec_dims[-2]
    dir_var = spec_dims[-1]
    free_dims = spec_dims[:-2]

    frequencies = spec_coords[freq_var]
    directions = spec_coords[dir_var]

    # Create freqs and dirs through interpolation if not supplied directly
    if hasattr(new_frequencies, "__len__"):
        new_frequencies = np.array(new_frequencies)
    else:
        new_frequencies = _interpolate_cubic(frequencies,new_frequencies)
    if hasattr(new_directions,"__len__"):
        new_directions = np.array(new_directions)
    else:
        new_directions = _interpolate_linear(directions,new_directions)

    new_spec = interpolate_2D_spec(spec.data,frequencies,directions,
                                   new_frequencies,new_directions,method)
    
    new_coordinates = {k:spec_coords[k] for k in free_dims}
    new_coordinates[freq_var] = new_frequencies
    new_coordinates[dir_var] = new_directions
    return xr.DataArray(new_spec,new_coordinates)

def integrated_parameters_dict(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray) -> dict:
    """
    Calculate the integrated parameters of a 2D wave spectrum, 
    or some array/list of spectra. Uses simpsons integration rule.

    Implemented: Hs, peak dir, peak freq.
    
    Arguments
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
        
    Returns
    -------
    spec_parameters : dict[str, np.ndarray]
        A dict with keys Hs, peak_freq, peak_dir, and values are arrays
        of the integrated parameter.
    """
    
    # Make sure all arrays are numpy.
    if isinstance(spec, xr.DataArray):
        spec = spec.data
    if isinstance(frequencies, xr.DataArray):
        frequencies = frequencies.data
    if isinstance(directions, xr.DataArray):
        directions = directions.data

    # Check if spec values and shape are OK
    if np.any(spec < 0):
        print("Warning: negative spectra values set to 0")
        spec = np.clip(spec, a_min=0, a_max=None)

    flat_check = (len(spec.shape)<2)
    freq_check = (len(frequencies) != spec.shape[-2])
    dir_check = (len(directions) != spec.shape[-1])
    if flat_check or freq_check or dir_check:
        try:
            spec = spec.reshape(spec.shape[:-1]+(len(frequencies),len(directions)))
        except Exception:
            raise IndexError("Spec shape does not match frequencies and directions.")

    # Use argmax to find indices of largest value of each spectrum.
    peak_dir_freq = np.array([np.unravel_index(s.argmax(),s.shape) 
        for s in spec.reshape(-1,len(frequencies),len(directions))])
    peak_dir_freq = peak_dir_freq.reshape(spec.shape[:-2]+(2,))
    peak_freq = frequencies[peak_dir_freq[...,0]]
    peak_dir = directions[peak_dir_freq[...,1]]
    
    # Integration requires radians
    if np.max(directions) > 2*np.pi: 
        directions = np.deg2rad(directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Integration with simpson's rule
    S_f = scipy.integrate.simpson(spec, x=directions)
    m0 = scipy.integrate.simpson(S_f, x=frequencies)
    Hs = 4 * np.sqrt(m0)

    spec_parameters = {
        "Hs":Hs,
        "peak_freq":peak_freq,
        "peak_dir":peak_dir
    }

    return spec_parameters


def integrated_parameters(data, var='SPEC', params=None):
    '''
    Calculate significant wave height (Hm0), peak frequency, and peak direction 
    from a directional wave spectrum. You can select which parameters to compute.

    Parameters
    - data : xarray.Dataset or xarray.DataArray
        Wave spectrum with dimensions including 'freq' and 'direction'. If a Dataset,
        the spectral variable is specified by `var`.
    - var : str, optional, default = 'SPEC'
        Name of the spectral variable in `data` if `data` is a Dataset.
    - params : list of str or None, optional, default = None
        List of parameters to compute. Possible values: 'hm0', 'peak_freq', 'peak_dir'.
        If None, all parameters are computed.

    Returns
    - tuple or single xarray.DataArray
        The requested parameters, in order ('hm0', 'peak_freq', 'peak_dir'). If only
        one parameter is requested, returns it directly.
    '''
    if params is None:
        params = ['hm0', 'peak_freq', 'peak_dir']
    else:
        # Validate params list
        valid_params = {'hm0', 'peak_freq', 'peak_dir'}
        if not set(params).issubset(valid_params):
            raise ValueError(f"Invalid params {params}. Allowed values are {valid_params}")

    try:
        data = data[var]
    except KeyError:
        pass

    directions = data['direction']
    frequencies = data['freq']

    # Convert directions to radians if in degrees
    if directions.max().item() > 2 * np.pi:
        directions = np.deg2rad(directions)

    # Assign radian direction coordinates to spectrum (for correct integration)
    data = data.assign_coords(direction=directions)

    results = {}

    # Compute peak_freq and/or peak_dir only if requested
    if 'peak_freq' in params or 'peak_dir' in params:
        spec_flat = data.stack(freq_dir=("freq", "direction"))                                      # Combine frequency and direction dimensions into a single dimension to find global peak
        peak_flat_idx = spec_flat.argmax(dim="freq_dir")                                            # Find index of maximum spectral energy along the combined freq-direction dimension
        freq_idx, dir_idx = np.unravel_index(peak_flat_idx, (len(frequencies), len(directions)))    # Convert the flat index back into separate frequency and direction indices
        
        if 'peak_freq' in params:                                                                   # Select frequencies and directions at the peak indices
            results['peak_freq'] = frequencies.isel(freq=freq_idx)
        if 'peak_dir' in params:
            results['peak_dir'] = directions.isel(direction=dir_idx)

    # Compute hm0 only if requested
    if 'hm0' in params:
        spec_1d = data.integrate(coord='direction')
        m0 = spec_1d.integrate(coord='freq')
        results['hm0'] = 4 * np.sqrt(m0)

    # Return in fixed order, but only requested params
    output_order = ['hm0', 'peak_freq', 'peak_dir']
    output = tuple(results[param] for param in output_order if param in params)

    # If only one param requested, return it directly (not tuple)
    if len(output) == 1:
        return output[0]
    else:
        return output


def combine_spec_wind(dataspec, datawind):
    '''
    Merge spectral and wind datasets after dropping conflicting spatial coordinates ('x' and 'y').
    Both input datasets must correspond to a single geographic location.

    Parameters
    - dataspec : xarray.Dataset
        Spectral data.
    - datawind : xarray.Dataset
        Wind data.

    Returns
    - xarray.Dataset
        Combined dataset with variables from both inputs.
    '''

    datawind = datawind.drop_vars(['x', 'y'], errors='ignore')  
    dataspec = dataspec.drop_vars('x', errors='ignore')         
    return xr.merge([dataspec, datawind])


def from_2dspec_to_1dspec(data, var='SPEC', dataframe=True, hm0=False):
    '''
    Converts a 2D directional wave spectrum to a 1D frequency spectrum by integrating over direction.
    Optionally includes significant wave height (Hm0) if present in the input dataset.

    Parameters
    - data : xarray.Dataset or xarray.DataArray
        Input containing 2D spectral data with dimensions:
        - (time, frequency, direction), or
        - (frequency, direction) for single time steps.
    - var : str, optional, default='SPEC'
        Name of the spectral variable in the dataset. Ignored if data is already a DataArray.
    - dataframe : bool, optional, default=True
        If True, return a pandas DataFrame; otherwise, return an xarray.DataArray.
    - hm0 : bool, optional, default=False
        If True, include the Hm0 column in the output DataFrame when present.

    Returns
    - pandas.DataFrame or xarray.DataArray
        1D frequency spectrum. Hm0 included if requested and present.
    '''

    try:                                                                                     # Try to extract the 2D spectral variable from the dataset; if not found, assume input is already the DataArray
        spec2d = data[var]
    except KeyError:
        spec2d = data

    if spec2d['direction'].max() > 2 * np.pi:                                                # Convert direction coordinates from degrees to radians if needed (assuming max direction value > 2π)
        spec2d = spec2d.assign_coords(direction=np.deg2rad(spec2d['direction']))

    spec_1d = spec2d.integrate(coord='direction')                                            # Integrate over the 'direction' coordinate to collapse 2D spectrum into 1D frequency spectrum

    if dataframe:
        df = spec_1d.to_pandas().reset_index()                                               # Convert xarray.DataArray to pandas DataFrame and reset index so 'time' becomes a column
        df.columns.name = None

        if hm0 and isinstance(data, xr.Dataset) and 'hm0' in data:                           # Optionally add 'Hm0' column if requested and present in the input dataset
            df['Hm0'] = data['hm0'].values

        return df

    if hm0 and isinstance(data, xr.Dataset) and 'hm0' in data:                               # If returning xarray.DataArray and hm0 requested, assign Hm0 as a coordinate if present
        spec_1d = spec_1d.assign_coords(hm0=data['hm0'])

    return spec_1d


def standardize_wave_dataset(data):
    '''
    Standardize a 2D wave spectrum dataset to match the WINDSURFER/NORA3 format.

    Parameters:
    - data : xarray.Dataset
        Input dataset to standardize.

    Returns:
    - xarray.Dataset
        The standardized dataset.
    '''

    # Detect NORAC product
    if 'product_name' in data.attrs and data.attrs['product_name'].startswith("ww3"):
        # Rename dims and vars
        data = data.rename({
            'frequency': 'freq',
            'efth': 'SPEC'})

    return data


def filter_period(data, period):
    '''
    Filters the dataset to a specified time period.

    Parameters
    - data : xarray.Dataset or xarray.DataArray
        The input dataset containing a 'time' dimension.
    period : tuple or None
        A tuple specifying the desired time range.
        - (start_time, end_time): Filters between start_time and end_time.
        - (start_time,): Filters to a single timestamp.
        - None: Uses the full time range available in data.
        Both start_time and end_time may be strings or datetime-like objects.

    Returns
    filtered_data : xarray.Dataset or xarray.DataArray
        Subset of the data within the specified time range.
    period_label : str
        A string label describing the filtered time period.
        If start and end times are equal, only that time is returned.
    '''

    # Finds data time range
    data_start = pd.to_datetime(data.time.min().values)
    data_end = pd.to_datetime(data.time.max().values)

    # Uses the full dataset if period is None
    if period is None:
        start_time, end_time = data_start, data_end
    
    # Uses the choosen period range
    elif isinstance(period,tuple):
        start_time, end_time = pd.to_datetime(period[0]), pd.to_datetime(period[1])

        if start_time < data_start or end_time > data_end:
            raise ValueError(f"Period {start_time} to {end_time} is outside data range {data_start} to {data_end}.")

    # Uses one timestamp if choosen
    else:
        start_time = end_time = pd.to_datetime(period)
        if start_time not in data.time.values:
            raise ValueError(f"({start_time}) is outside data range {data_start} to {data_end}.")

    filtered_data = data.sel(time=slice(start_time, end_time))

    if start_time == end_time:
        period_label = f"{start_time.strftime('%Y-%m-%dT%H')}Z"
    else:
        period_label = f"{start_time.strftime('%Y-%m-%dT%H')}Z to {end_time.strftime('%Y-%m-%dT%H')}Z"

    return filtered_data, period_label

def wrap_directional_spectrum(data, var='SPEC'):
    '''
    Normalize and wrap directional spectral data to 0-360°.

    Parameters:
    - data : xarray.Dataset or DataArray
        Input spectral dataset.
    - var : str, optional, default='SPEC'
        Variable name. 

    Returns:
    - spec : xarray.DataArray
        Direction-sorted and wrapped spectral data.
    - dirc : xarray.DataArray
        Corresponding direction coordinates.
    '''
    try:
        datspec=data[var]
    except KeyError:
        datspec=data

    directions = datspec.coords['direction']

    # Ensure directions are 0–360 and sorted
    directions = directions % 360                               # Normalize direction values to [0, 360)
    sorted_idx = np.argsort(directions.values)                  # Sort direction values in increasing order
    directions = directions[sorted_idx]
    datspec = datspec.isel({'direction': sorted_idx})           # Reorder spectrum data accordingly

    # Ensure continuity in the directional spectrum by appending the initial directional value (e.g., 0°) at the 360° position
    if directions[0] != directions[-1]:
        if directions[0] < 360:
            dirc = xr.concat([directions, directions[0:1] + 360], dim='direction')      # Add 360° copy of first value
        else:
            dirc = xr.concat([directions, directions[0:1]], dim='direction')            # Fallback: just repeat first value
        spec = xr.concat([datspec, datspec.isel({'direction': 0})], dim='direction')    # Add same data point at the end
    else:
        dirc = directions
        spec = datspec
    return spec, dirc


def aggregate_spectrum(data, hm0, var='SPEC', method='mean', month=None):
    '''
    Aggregate a variable over time using the specified method, with optional filtering by month.

    Parameters
    - data : xarray.Dataset or xarray.DataArray  
        Input data to aggregate.  
        Can be:  
        - A 2D directional-frequency variable (e.g., wave spectrum with 'freq' and 'direction'), or  
        - A simpler variable (e.g., wind speed) without those dimensions.  
        If a Dataset, `var` specifies which variable to use.  
        If a DataArray, it's used directly.
    - hm0 : xarray.DataArray  
        Significant wave height used for percentile-based and max-based methods.
    - var : str, optional, default='SPEC'  
        Variable name to aggregate (ignored if `data` is a DataArray).
    - method : str, optional, default='mean'  
        Aggregation method:  
        - 'mean'            : Average over time.  
        - 'top_1_percent_mean'   : Average over times where Hm0 ≥ 99th percentile.  
        - 'hm0_max'         : Use time step with maximum Hm0.  
        - 'hm0_top3_mean'   : Average of Hm0 over the three time steps with the highest values.
    - month : int, optional, default=None  
        If set (1-12), filters data to that month. Otherwise, uses all available data.

    Returns
    - data_aggregated : xarray.DataArray  
        The aggregated result with time collapsed.
    '''
    
    try:
        data=data[var]
    except KeyError:
        data = data

    # Apply monthly filtering if needed
    if month is not None:
        time_mask = data['time'].dt.month == month
        data = data.sel(time=time_mask)
        hm0 = hm0.sel(time=time_mask)

    if method == 'mean':
        data_aggregated = data.mean(dim='time')              

    elif method == 'top_1_percent_mean':
        # Get time steps where hm0 >= 99th percentile
        p99 = hm0.quantile(0.99)
        high_hm0_times = hm0['time'][hm0 >= p99]

        # Select those time steps from data
        spec_p99 = data.sel(time=high_hm0_times)
        data_aggregated = spec_p99.mean(dim='time')
    
    elif method == 'hm0_max':
        # Find time coordinate where hm0 is maximum
        hm0_max_time = hm0.idxmax(dim='time')

        # Select data at that max hm0 time step
        spec_hm0_max = data.sel(time=hm0_max_time)
        data_aggregated = spec_hm0_max

    elif method == 'hm0_top3_mean':
        # Sort hm0 descending and get top 3 time indices
        top3_times = hm0.sortby(hm0, ascending=False)['time'][:3]

        # Select those times from data
        spec_top3 = data.sel(time=top3_times)
        data_aggregated = spec_top3.mean(dim='time')

        # Store top 3 times as metadata
        data_aggregated.attrs['hm0_top3_timesteps'] = [str(t) for t in top3_times.values]

    return data_aggregated


def compute_mean_wave_direction(data, var='SPEC', method='mean', month=None, hm0=None):
    '''
    Compute the mean wave direction from a directional wave energy spectrum.

    This function calculates the mean wave direction dir_mean based on the 
    discrete approximation of the integrals:

        a = ∫∫ cos(dir) * F(freq, dir) dfreq ddir
        b = ∫∫ sin(dir) * F(freq, dir) dfreq ddir
        dir_mean = arctan2(b, a)

    where:
        - F(freq, dir) is the spectral energy density as a function of frequency and direction.
        - The integrals are approximated by summations over the frequency and
          direction bins weighted by the bin widths.

    Parameters:
    - data : xarray.Dataset or xarray.DataArray
        Wave spectrum with dimensions including 'freq' and 'direction'. If a Dataset,
        the spectral variable is specified by `var`.
    - var : str, optional, default = 'SPEC'
        Name of the spectral variable in `data` if `data` is a Dataset.
    - method : str, optional, default = 'mean'
        Aggregation method to apply over time ('mean', 'top_1_percent_mean', 'hm0_max', 'hm0_top3_mean') after computing a and b.
    month : int or None, optional, default = None
        If specified, selects data only for that month (1-12) before computing direction.
    hm0 : xarray.DataArray or None, optional, default = None
        Significant wave height time series. This is required for weighted aggregation.
        It must either be passed explicitly or be present as `data[var]['hm0']`.


    Returns:
    - mean_dir_rad : xarray.DataArray
        Mean wave direction in radians using the mathematical convention:
        0 = East, positive counter-clockwise (CCW).

    Notes:
    - Direction angles in the input are assumed to be oceanographic (0° = North, increasing clockwise).
      They are converted to mathematical radians (0 = East, increasing CCW).
    - Frequency and direction bin widths are computed using gradients and used to weight the integration.
    - The final direction is based on vector summation (a, b) and converted using arctangent.
    - Based on the method in the WAVEWATCH III User Manual (v6.07, NOAA/NCEP, 2019).
    '''

    try:
        spectrum=data[var]
    except KeyError:
        spectrum = data                                                                             # Extract the spectrum variable (dimensions expected: ['time', 'freq', 'direction'])

    directions_rad = np.deg2rad((450 - spectrum['direction']) % 360)                                # Convert oceanographic directions (degrees, coming from North clockwise) 
                                                                                                    # to mathematical directions (radians, pointing to East counterclockwise)

    delta_freq = np.gradient(spectrum.freq.values)                                                  # Calculate frequency bin widths

    delta_dir = np.gradient(spectrum['direction'])                                                  # Calculate direction bin widths in degrees and convert to radians
    delta_dir_rad = np.deg2rad(delta_dir)

    dfreq_2d = xr.DataArray(delta_freq, dims=['freq'])                                              # Create DataArrays for bin widths to broadcast over spectrum dims
    ddir_2d = xr.DataArray(delta_dir_rad, dims=['direction'])

    area_element = dfreq_2d.broadcast_like(spectrum) * ddir_2d.broadcast_like(spectrum)             # Compute the area element dfreq ddir for each frequency-direction bin by outer product

    a = (xr.ufuncs.cos(directions_rad) * spectrum * area_element).sum(dim=['freq', 'direction'])    # Compute weighted sums a and b over freq and direction dimensions
    b = (xr.ufuncs.sin(directions_rad) * spectrum * area_element).sum(dim=['freq', 'direction'])

    if hm0 is None:
        try:
            hm0 = data['hm0']
        except KeyError:
            hm0 = integrated_parameters(data=data, var=var, params=['hm0'])

    a = aggregate_spectrum(data=a, hm0=hm0, method=method, month=month)                             # Aggregate based on method
    b = aggregate_spectrum(data=b, hm0=hm0, method=method, month=month)                     

    mean_dir_rad = np.arctan2(b, a)                                                                 # Compute mean wave direction as arctangent of vector sum components

    return mean_dir_rad




def Spectral_Partition_wind(data, beta =1.3, method='mean', month=None):
    '''
    Partition wave spectrum into wind sea and swell using the dimensionless parameter A.

    Based on the formulation by Komen et al. (1984). 
    A threshold of beta ≤ 1.3 is typically used to identify pure wind sea (Hasselmann et al. 1996; Voorrips et al. 1997; Bidlot 2001).

    Parameters
    - data : xarray.Dataset
        Must contain 'SPEC', 'freq', 'direction', 'wind_speed', 'wind_direction' (all with time).
    - beta : float, optional, default = 1.3
        Partition threshold (typically ≤ 1.3).
    - method : str, default 'mean'
        Aggregation method for computing mean direction.
    - month : int or None
        If set, compute mean direction only for that month.

    Returns
    - data : xarray.Dataset
        Dataset with added variables for each partition:
        - 'SPEC_swell', 'SPEC_windsea'
        - 'Hm0_*', 'Tp_*', 'pdir_*', 'mean_dir_*'
    '''

    # Calculate the phase speed (cp) using the dispersion relation for deep water waves
    data['cp'] = 9.81/(2*np.pi*data['freq']) 
    
    # Initialize an array to store the directional difference between wave direction and wind direction
    data['diff_dir'] = xr.DataArray(np.zeros((data.time.size, data.direction.size)), 
                            dims=['time', 'direction'],
                            coords={'time': data.time, 'direction': data.direction})

    # Loop over each time step to compute the angular difference between wave direction and wind direction
    for i in range (len(data['diff_dir'].time)):
        data['diff_dir'][i,:] = angular_difference((data.direction), ((data['wind_direction'].sel(height=10)[i].item() - 180)))

    # Calculate the dimensionless parameter A, which is used to distinguish between swell and windsea
    data['A']  = beta*(data['wind_speed'].sel(height=10)/data['cp'])*np.cos(np.deg2rad(data['diff_dir']))

    # Partition the wave spectrum into swell and windsea components based on the value of A
    data['SPEC_swell'] = data['SPEC'].where(data['A']<=1,0)
    data['SPEC_windsea'] = data['SPEC'].where(data['A']>1,0)

    # Estimate integrated parameters
    part = ['swell','windsea']
    for k in range(len(part)):
        data['Hm0_'+part[k]] = (4*(data['SPEC_'+part[k]].integrate("freq").integrate("direction"))**0.5).assign_attrs(units='m', standard_name = 'significant_wave_height_from_spectrum_'+part[k])
        data['Tp_'+part[k]] = (1/data['SPEC_'+part[k]].integrate('direction').idxmax(dim='freq')).assign_attrs(units='s', standard_name = 'peak_wave_period_'+part[k])
        data['pdir_'+part[k]] = (data['SPEC_'+part[k]].integrate('freq').idxmax(dim='direction'))
        data['mean_dir_'+part[k]+'_rad'] = compute_mean_wave_direction(data = data['SPEC_' + part[k]], hm0 = data['hm0'], method = method, month=month)
        data['mean_dir_'+part[k]+'_deg'] = (450 - np.rad2deg(data['mean_dir_'+part[k]+'_rad'].values)) % 360                                            # Meteorological convention (0° = North, clockwise)

    # Uncomment the following lines to print the mean direction
    # print('Mean wave direction swell:', data['mean_dir_swell_deg'].values)
    # print('Mean wave direction sea', data['mean_dir_windsea_deg'].values)

    return data

def angular_difference(deg1, deg2):
    """
    Calculate the smallest difference between two angles in degrees.
    
    Parameters:
    deg1 (float or array-like): The first angle(s) in degrees.
    deg2 (float or array-like): The second angle(s) in degrees.
    
    Returns:
    float or np.ndarray: The smallest difference(s) between the two angles in degrees.
    """
    # Convert inputs to numpy arrays for element-wise operations
    deg1 = np.asarray(deg1)
    deg2 = np.asarray(deg2)
    
    # Calculate the absolute difference
    diff = np.abs(deg1 - deg2)
    
    # Normalize the difference to the range [0, 360)
    diff = diff % 360
    
    # Adjust if the difference is greater than 180 to get the shortest path
    diff = np.where(diff > 180, 360 - diff, diff)
    
    return diff

