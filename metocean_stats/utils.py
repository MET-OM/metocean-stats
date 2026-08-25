import numpy as np
import pandas as pd

DEFAULT_SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

def groupby_season(
    data: pd.DataFrame,
    seasons: dict[str, list[int]] = None,
    var: str = None,
):
    """
    Group dataframe by meteorological season.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    seasons : dict[str, list[int]], optional
        Map of season name → list of month numbers.
        Defaults to DJF / MAM / JJA / SON.
    var : str, optional
        If set, return a Series for that column only.
    """
    seasons = seasons or DEFAULT_SEASONS

    result = {}
    for name, months in seasons.items():
        mask = data.index.month.isin(months)
        grp  = data.loc[mask]
        result[name] = grp[var] if var else grp

    return result

def groupby_month(
        data:pd.DataFrame,
        var:str=None,
        ) -> dict[str,pd.Series|pd.DataFrame]:
    """
    Group dataframe by month and return as dictionary.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    var : str
        If this is set, a dict of Series will be returned. Otherwise, DataFrames are returned.
        
    """
    data.index = pd.to_datetime(data.index)
    return {pd.to_datetime(k,format="%m").month_name()[:3]:g[var] if var else g
            for k,g in data.groupby(data.index.month,observed=True)}

def groupby_sector(
        data:pd.DataFrame,
        var_dir:str,
        sectors:int|list[float]=12,
        var:str=None
        ) -> dict[str,pd.Series|pd.DataFrame]:
    """
    Group dataframe by direction sector.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    var_dir : str
        Column name of the direction variable.
    sectors : int or list of floats
        If int, equally-sized sectors are created, with the first centered on north.
        If a list of floats, the floats define the boundaries of the sectors.
    var : str
        If this is set, a dict of Series will be returned. Otherwise, DataFrames are returned.
    """

    data = data.copy()

    if not hasattr(sectors,"__len__"):
        sectors = np.linspace(0, 360, sectors+1,dtype=int)

    dir_offset = (sectors[1]-sectors[0])/2
    labels = [f"{(sectors[i]-dir_offset)%360:.0f}-{(sectors[i+1]-dir_offset)%360:.0f}°" for i in range(len(sectors)-1)]
    data["sector"] = pd.cut((data[var_dir]+dir_offset)%360, bins=sectors, labels=labels, right=False)
    return {k:g[var] if var else g for k,g in data.groupby("sector",observed=True)}


def infer_step(s: pd.Series, target: int = 10) -> float:
    """Infer a round step size from the data range, snapped to a nice value."""
    NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10]
    raw        = (s.max() - s.min()) / target
    magnitude  = 10 ** np.floor(np.log10(raw))
    scaled     = raw / magnitude
    nice       = min(NICE_STEPS, key=lambda x: abs(x - scaled))
    return nice * magnitude

def bin_edges(s: pd.Series, step: float) -> np.ndarray:
    """
    Compute bin edges for a series snapped to a given step size.
    Starts from zero if all values are non-negative, otherwise
    fits outer edges to the data range.
    """
    lo = 0.0 if s.min() >= 0 else np.floor(s.min() / step) * step
    hi = np.ceil(s.max() / step) * step
    n  = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


def aggregate_statistics(series: pd.Series, func: list[str]) -> dict:
    """
    Apply a list of statistic names to a series and return a dict of results.
    Accepts: "min", "mean", "max", "std", "count", and percentiles as "p90", "P90", "90%" etc.
    """
    result = {}
    for f in func:
        fl = f.lower().strip("%")
        if fl == "min":
            result[f] = series.min()
        elif fl == "mean":
            result[f] = series.mean()
        elif fl == "max":
            result[f] = series.max()
        elif fl == "std":
            result[f] = series.std()
        elif fl == "count":
            result[f] = series.count()
        elif fl.startswith("p") and fl[1:].isdigit():
            result[f] = series.quantile(int(fl[1:]) / 100)
        elif fl.isdigit():
            result[f] = series.quantile(int(fl) / 100)
        else:
            raise ValueError(f"Unknown aggregation function: '{f}'")
    return result


def table_to_latex(
        table:pd.DataFrame,
        filename:str,
        float_format,
    ):
    """
    Script to write a pandas dataframe to a latex .tex file.
    Allows setting a per-column float format.
    The top left cell will be the index name, if set,
    otherwise the column name, if set, otherwise empty.
    """
    if not filename.endswith(".tex"): filename += ".tex"
    with open(filename, "w") as f:
        f.write("\\begin{tabular}{l"+"r"*len(table.columns)+"}\n")
        f.write("\\toprule \n")
        if table.index.name: f.write(f"{table.index.name}")
        elif table.columns.name: f.write(f"{table.columns.name}")
        columns = [c.replace("%",r"\%") for c in table.columns]
        f.write(" & " + " & ".join(columns)+" \\\\\n")
        f.write("\\midrule\n")
        for index, row in table.iterrows():
            if isinstance(float_format,str):
                if isinstance(index,str):
                    f.write(f"{index}")
                else:
                    f.write(f"{index:{float_format}}")
                for i,x in enumerate(row.values):
                    if isinstance(x, str):
                        f.write(f" & {x}")
                    elif np.isnan(x):
                        f.write(" & ")
                    else:
                        f.write(f" & {x:{float_format}}")
                f.write(" \\\\\n")
            elif hasattr(float_format,"__len__") and (len(float_format)==len(columns)+1):
                f.write(f"{index:{float_format[0]}}")
                for i,x in enumerate(row.values):
                    if isinstance(x, str):
                        f.write(f" & {x}")
                    elif np.isnan(x):
                        f.write(" & ")
                    else:
                        f.write(f" & {x:{float_format[i+1]}}")
                f.write(" \\\\\n")
            else: raise ValueError("float_format has incorrect length: "
                f"Expected {len(columns)+1} and got {len(float_format)}.")
        f.write("\\bottomrule \n")
        f.write("\\end{tabular}")


def dirmag_to_uv(wind_direction, wind_speed, going_to=True):
    '''
    Get wind x (east) and y (north) component 
    from speed and direction (degrees, default: going to).

    Parameters
    ---------
    wind_direction : np.ndarray
        Wind direction (degrees)
    wind_speed : np.ndarray
        Wind Speed (degrees)
    going_to : bool, default True (oceanographic)
        Controls direction convention, False gives "from" direction.
    '''
    
    wind_direction = np.radians(wind_direction)
    
    if not going_to: 
        wind_direction = (wind_direction+np.pi)%(2*np.pi)
    
    u = wind_speed*np.sin(wind_direction)
    v = wind_speed*np.cos(wind_direction)
    return u,v
