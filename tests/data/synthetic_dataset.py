import numpy as np
import pandas as pd
import xarray as xr

def synthetic_dataset_spectra():
    '''
    Create a synthetic xarray.Dataset representing a 2D wave spectrum time series
    at a single geographical point.

    The dataset contains:
    - Time: Hourly timestamps over 2 years (2021-2022)
    - Frequency: 30 values from 0.035 to 0.55 Hz
    - Direction: 24 values from 7.5° to 352.5°
    - SPEC: Randomly generated 2D wave spectral data (time x freq x direction)
    - Latitude and Longitude: Constant single-point coordinates (59°N, 4°E)

    Returns
    - xr.Dataset
        Synthetic wave spectrum dataset.
    '''
    # Define dimensions
    n_time = 24 * 365 * 2  # Number of time steps: 2 years of hourly data
    n_freq = 30        
    n_dir = 24           

    # Create coordinate arrays
    time = pd.date_range('2021-01-01', periods=n_time, freq='h')                      # Hourly time steps from Jan 1, 2021
    freq = np.linspace(0.03, 0.55, n_freq).astype(np.float32)                         # Frequency values from 0.03 to 0.55 Hz
    direction = np.linspace(7.5, 352.5, n_dir).astype(np.float32)                     # Direction values evenly spaced in 360° range

    # Define a base 2D spectrum shape: Gaussian frequency profile × uniform direction profile
    freq_profile = np.exp(-((freq - 0.15) / 0.05) ** 2).astype(np.float32)            # Gaussian centered at 0.15 Hz
    dir_profile = np.ones(n_dir, dtype=np.float32)                                    # Flat directional distribution
    base_spectrum = np.outer(freq_profile, dir_profile)                               # Create base 2D spectrum (freq × dir)

    # Create a seasonal modulation factor that varies smoothly over time (between 0.7 and 1.3)
    time_hours = np.arange(n_time)                                                   
    hours_per_year = 24 * 365                                                         
    seasonal_cycle = 1 + 0.3 * np.sin(2 * np.pi * time_hours / hours_per_year)        

    # Initialize 3D SPEC array (time × freq × dir) with seasonal variations and random noise
    np.random.seed(42)                                                                
    SPEC = np.empty((n_time, n_freq, n_dir), dtype=np.float32)                        

    for t in range(n_time):
        noise = 0.1 * np.random.randn(n_freq, n_dir).astype(np.float32)              # Add small Gaussian noise
        SPEC[t] = base_spectrum * seasonal_cycle[t] + noise                          # Apply seasonal modulation

    SPEC = np.clip(SPEC, 0, None)                                                    # Ensure all values are non-negative

    # Wrap SPEC array in a DataArray with units and metadata
    spec = xr.DataArray(
        SPEC,
        dims=("time", "freq", "direction"),
        coords={"time": time, "freq": freq, "direction": direction},
        attrs={
            "units": "m**2 s",                                                       
            "long_name": "Variance density spectrum"
        }
    )

    # Construct the full Dataset with SPEC and metadata
    ds = xr.Dataset(
        {
            "SPEC": spec
        },
        coords={
            "time": time,
            "freq": freq,
            "direction": direction,
            "x": 4,                                                                   # Dummy x-dimension for compatibility
            "longitude": 4.0,                                                         # Constant longitude (as coordinate)
            "latitude": 60.0                                                          # Constant latitude (as coordinate)
        },
        attrs={
            "title": "Simplified synthetic wave dataset, single point, 2 years",
        }
    )

    return ds
