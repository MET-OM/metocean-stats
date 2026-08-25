import warnings
import itertools
import typing

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from tqdm import tqdm
import pandas as pd
import numpy as np

from ..utils import groupby_month, groupby_season, groupby_sector, bin_edges, infer_step, aggregate_statistics

class WeatherWindow:
    """
    Weather window analysis module.
    
    A weather window is a sequence of stages that must be completed in order.
    Each stage has a duration (fixed or flexible) and optional criteria.
    
    Example
    -------
    >>> # Simple: constant criteria for 24 timesteps
    >>> ts.WW.fit(stages=[{"duration": 24, "criteria": {"hs": 2.0, "ws": 12.0}}])
    >>>
    >>> # Staged: varying criteria
    >>> ts.WW.fit(stages=[
    ...     {"duration": 12, "criteria": {"hs": 4.0}},
    ...     {"duration": 12, "criteria": {"hs": 6.0}},
    ... ])
    >>>
    >>> # Flexible: with pause/safe state
    >>> ts.WW.fit(stages=[
    ...     {"duration": 12, "criteria": {"hs": 4.0}},
    ...     {"duration": (0, 24)},  # flexible pause, no criteria
    ...     {"duration": 12, "criteria": {"hs": (2.0, 4.0)}},  # lower and upper bounds
    ... ])
    """
        
    def __init__(self, data, variables, timestep, direction=None):
        """
        Initialize WeatherWindow analyzer.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data with DatetimeIndex.
        variables : dict or list
            Variable objects or names.
        direction : str, optional
            Direction variable (degrees). Used for directional statistics.
        timestep : int, float, str, or pd.Timedelta
            Time resolution of the data.
            - int/float: interpreted as hours
            - str: pandas-compatible timedelta string (e.g., '1h', '3H', '30min')
            - pd.Timedelta: used directly
        """
        if direction and direction not in data.columns:
            raise ValueError(f"Direction var {direction} is not a column of the data.")
        
        self.data = data
        self.variables = variables
        self.direction = direction
        
        # Standardize timestep to pd.Timedelta
        if isinstance(timestep, pd.Timedelta):
            self.timestep = timestep
        elif isinstance(timestep, (int, float)):
            self.timestep = pd.Timedelta(hours=timestep)
        elif isinstance(timestep, str):
            self.timestep = pd.Timedelta(timestep)
        else:
            raise TypeError(
                f"timestep must be int, float, str, or pd.Timedelta, "
                f"got {type(timestep).__name__}"
            )
        
        # State (populated by .fit())
        self.stages = None           # List of standardized stage definitions
        self.result = None           # DataFrame with all data + computed metrics

    def fit(self, stages: list[dict]):
        """
        Find all feasible weather windows for the given staged operation.
        
        Parameters
        ----------
        stages : list[dict]
            Ordered list of operation stages. Each stage is a dict with:
            
            - "duration" : int or tuple[int, int]
                Number of timesteps required (fixed) or range (min, max) for
                flexible pauses.
                
            - "criteria" : dict[str, float or tuple], optional
                Variable name → threshold(s) mapping during this stage.
                
                Single value = upper limit (must be ≤ value)
                Tuple (lower, upper) = must be within range [lower, upper]
                Use None, NaN, or ±inf for unbounded limits
                
                Examples:
                {"hs": 2.0}           # Hs must be ≤ 2.0
                {"hs": (1.0, 3.0)}    # Hs must be in [1.0, 3.0]
                {"hs": (2.0, None)}   # Hs must be ≥ 2.0 (no upper limit)
                {"hs": (None, 4.0)}   # Hs must be ≤ 4.0 (explicit)
                
                If omitted, no criteria (always satisfied).
        
        Returns
        -------
        self
            For method chaining.
        
        Notes
        -----
        After fitting, results are available as:
        - `self.result`: DataFrame with original data plus computed metrics:
            - `can_start`: Boolean indicating feasible start times
            - `duration`: Operation duration in timesteps
            - `waiting_time`: Time until next window in timesteps
            - `characteristic_duration`: waiting_time + duration
        
        Examples
        --------
        >>> # Single stage, 24 timesteps, Hs ≤ 2m and Ws ≤ 12 m/s
        >>> ww.fit(stages=[{"duration": 24, "criteria": {"hs": 2.0, "ws": 12.0}}])
        >>>
        >>> # Two stages with different criteria
        >>> ww.fit(stages=[
        ...     {"duration": 12, "criteria": {"hs": 4.0}},
        ...     {"duration": 12, "criteria": {"hs": 6.0}},
        ... ])
        >>>
        >>> # Range criteria: Hs must be between 1m and 3m
        >>> ww.fit(stages=[
        ...     {"duration": 12, "criteria": {"hs": (1.0, 3.0)}},
        ... ])
        >>>
        >>> # Lower bound only: Tp must be at least 5 seconds
        >>> ww.fit(stages=[
        ...     {"duration": 12, "criteria": {"tp": (5.0, None)}},
        ... ])
        >>>
        >>> # Flexible pause: wait 0-24 timesteps with no criteria
        >>> ww.fit(stages=[
        ...     {"duration": 12, "criteria": {"hs": 4.0}},
        ...     {"duration": (0, 24)},  # no criteria during pause
        ...     {"duration": 12, "criteria": {"hs": 4.0}},
        ... ])
        """
        # Validate and standardize input
        self.stages = self._validate_and_standardize_stages(stages)
        
        # Verify data continuity
        self._verify_data_continuity()
        
        # Compute feasibility
        can_start, duration = self._compute_feasibility()
        
        # Compute waiting time
        waiting_time = self._compute_waiting_time(can_start)
        
        # Build result DataFrame with all data + computed metrics
        self._build_result(can_start, duration, waiting_time)
        
        return self

    def _build_result(self, can_start, duration, waiting_time):
        """
        Build result DataFrame with original data, criteria columns, direction, and computed metrics.
        
        Parameters
        ----------
        can_start : pd.Series
            Boolean Series indicating feasible start times.
        duration : pd.Series
            Series with operation duration in timesteps.
        waiting_time : pd.Series
            Series with time until next window in timesteps.
        """
        self.result = self.data.copy()
        
        # Add computed metrics
        self.result["can_start"] = can_start
        self.result["duration"] = duration
        self.result["waiting_time"] = waiting_time

        # For characteristic_duration, backward-fill duration to get the duration of the next possible window.
        duration_filled = duration.bfill()
        self.result["characteristic_duration"] = waiting_time + duration_filled

        # Add direction column if present (for debugging/grouping)
        if self.direction is not None and self.direction not in self.result.columns:
            self.result[self.direction] = self.data[self.direction]
        
        # Add criteria columns mentioned in stages (for debugging)
        criteria_vars = set()
        for stage in self.stages:
            criteria_vars.update(stage["criteria"].keys())
        
        for var in criteria_vars:
            if var not in self.result.columns:
                self.result[var] = self.data[var]
    
    def _require_fit(self):
        """Raise error if fit() hasn't been called."""
        if self.stages is None:
            raise RuntimeError(
                "Weather window not fitted. Call .fit(stages=...) first."
            )
        if self.result is None:
            raise RuntimeError(
                "Weather window not fitted. Call .fit(stages=...) first."
            )

    def _validate_and_standardize_stages(self, stages: list[dict]) -> list[dict]:
        """
        Validate and standardize stage definitions.
        
        Performs the following checks and transformations:
        1. stages must be a non-empty list
        2. Each stage must be a dict with "duration" key
        3. Duration must be int or tuple of (min, max)
        4. Criteria must be dict mapping column names to float or tuple thresholds
        5. All criteria variables must exist in the data
        6. Standardize to consistent format
        
        Parameters
        ----------
        stages : list[dict]
            User-provided stage definitions.
        
        Returns
        -------
        list[dict]
            Standardized stages with guaranteed keys:
            - "duration" : tuple[int, int] (min, max) - always a tuple
            - "criteria" : dict[str, tuple[float|None, float|None]] (lower, upper)
        
        Raises
        ------
        TypeError
            If stages is not a list or contains non-dict elements.
        ValueError
            If required keys are missing, values have wrong type, or
            variable names don't exist in data.
        """
        # 1. Check stages is a non-empty list
        if not isinstance(stages, list):
            raise TypeError(
                f"stages must be a list of dicts, got {type(stages).__name__}"
            )
        if len(stages) == 0:
            raise ValueError("stages must contain at least one stage")
        
        standardized = []
        
        for i, stage in enumerate(stages):
            # 2. Each stage must be a dict
            if not isinstance(stage, dict):
                raise TypeError(
                    f"Stage {i} must be a dict, got {type(stage).__name__}"
                )
            
            # 3. Duration is required
            if "duration" not in stage:
                raise ValueError(f"Stage {i} missing required key 'duration'")
            
            duration = stage["duration"]
            
            # Validate and standardize duration
            if isinstance(duration, int):
                if duration <= 0:
                    raise ValueError(
                        f"Stage {i}: duration must be positive, got {duration}"
                    )
                duration_std = (duration, duration)
                
            elif isinstance(duration, tuple):
                if len(duration) != 2:
                    raise ValueError(
                        f"Stage {i}: duration tuple must have exactly 2 elements "
                        f"(min, max), got {len(duration)}"
                    )
                
                min_dur, max_dur = duration
                
                # Check min
                if not isinstance(min_dur, int) or min_dur < 0:
                    raise ValueError(
                        f"Stage {i}: duration min must be non-negative int, "
                        f"got {min_dur}"
                    )
                
                # Check max
                if not isinstance(max_dur, int) or max_dur < min_dur:
                    raise ValueError(
                        f"Stage {i}: duration max must be int >= min "
                        f"({min_dur}), got {max_dur}"
                    )
                
                duration_std = (min_dur, max_dur)
                
            else:
                raise TypeError(
                    f"Stage {i}: duration must be int or tuple[int, int], "
                    f"got {type(duration).__name__}"
                )
            
            # 4. Criteria (optional)
            criteria = stage.get("criteria", {})
            
            if not isinstance(criteria, dict):
                raise TypeError(
                    f"Stage {i}: criteria must be a dict, "
                    f"got {type(criteria).__name__}"
                )
            
            # 5. Validate and standardize criteria
            criteria_std = {}
            
            for var, threshold in criteria.items():
                # Check variable exists
                if var not in self.data.columns:
                    raise ValueError(
                        f"Stage {i}: criteria variable '{var}' not found in data. "
                        f"Available: {list(self.data.columns)}"
                    )
                
                # Standardize threshold to (lower, upper) tuple
                if isinstance(threshold, (int, float)):
                    # Single value = upper limit
                    if np.isnan(threshold) or np.isinf(threshold):
                        criteria_std[var] = (None, None)
                    else:
                        criteria_std[var] = (None, float(threshold))
                    
                elif isinstance(threshold, tuple):
                    if len(threshold) != 2:
                        raise ValueError(
                            f"Stage {i}: criteria threshold for '{var}' must be "
                            f"a single value or 2-tuple (lower, upper), got {len(threshold)} values"
                        )
                    
                    lower, upper = threshold
                    
                    # Convert None/NaN/inf to None
                    def normalize_limit(val):
                        if val is None:
                            return None
                        if not isinstance(val, (int, float)):
                            raise TypeError(
                                f"Stage {i}: criteria limit for '{var}' must be numeric or None, "
                                f"got {type(val).__name__}"
                            )
                        if np.isnan(val) or np.isinf(val):
                            return None
                        return float(val)
                    
                    lower_std = normalize_limit(lower)
                    upper_std = normalize_limit(upper)
                    
                    # Check lower <= upper if both are specified
                    if lower_std is not None and upper_std is not None:
                        if lower_std > upper_std:
                            raise ValueError(
                                f"Stage {i}: criteria for '{var}' has lower ({lower_std}) > "
                                f"upper ({upper_std})"
                            )
                    
                    criteria_std[var] = (lower_std, upper_std)
                    
                else:
                    raise TypeError(
                        f"Stage {i}: criteria threshold for '{var}' must be "
                        f"numeric or tuple, got {type(threshold).__name__}"
                    )
            
            # 6. Build standardized stage
            standardized.append({
                "duration": duration_std,
                "criteria": criteria_std,
            })
        
        return standardized
    
    def _verify_data_continuity(self):
        """Check for data gaps that would invalidate window calculations."""
        if not isinstance(self.data.index, pd.DatetimeIndex):
            return  # Can't check continuity without datetime index
        
        expected_freq = self.timestep
        actual_diffs = self.data.index.to_series().diff()
        
        # Allow small tolerance for floating point
        gaps = actual_diffs[actual_diffs > expected_freq * 1.01]
        
        if len(gaps) > 0:
            warnings.warn(
                f"Data contains {len(gaps)} gaps larger than expected timestep "
                f"({self.timestep}h). Weather window calculations may be incorrect. "
                f"First gap at index {gaps.index[0]}"
            )
    
    def _compute_feasibility(self):
        """
        Compute can_start and duration for all timestamps.
 
        Also populates ``self.stage_durations``: a dict mapping each feasible
        start timestamp to the list of per-stage durations (in timesteps) for
        the shortest valid configuration found at that timestamp. Used by
        plot_timeseries() to reconstruct the forward stage timeline.
        """
        n = len(self.data)
 
        # Find all flexible stages and their ranges
        flexible_stages = [
            (i, range(stage["duration"][0], stage["duration"][1] + 1))
            for i, stage in enumerate(self.stages)
            if stage["duration"][0] != stage["duration"][1]
        ]
 
        if not flexible_stages:
            durations = [stage["duration"][0] for stage in self.stages]
            all_combinations = [tuple(durations)]
            n_combinations = 1
        else:
            flex_indices, flex_ranges = zip(*flexible_stages)
            all_combinations = list(itertools.product(*flex_ranges))
            n_combinations = len(all_combinations)
 
        max_total_duration = sum(stage["duration"][1] for stage in self.stages)
 
        can_start = np.full(n, np.nan)
        duration  = np.full(n, np.nan)
 
        # Per-timestamp winning stage durations: index → [d0, d1, ...]
        stage_durations_arr = [None] * n
 
        for combo in tqdm(all_combinations, total=n_combinations,
                          desc="Checking weather windows", disable=(n_combinations == 1)):
            # Build full duration list for this combination
            durations = []
            if not flexible_stages:
                durations = list(combo)
            else:
                flex_indices, _ = zip(*flexible_stages)
                flex_idx = 0
                for i, stage in enumerate(self.stages):
                    if i in flex_indices:
                        durations.append(combo[flex_idx])
                        flex_idx += 1
                    else:
                        durations.append(stage["duration"][0])
 
            total_dur = sum(durations)
            valid     = self._check_configuration(durations)
 
            for t in range(n):
                if t + total_dur > n:
                    continue
                if valid[t]:
                    if np.isnan(duration[t]) or total_dur < duration[t]:
                        can_start[t]           = True
                        duration[t]            = total_dur
                        stage_durations_arr[t] = list(durations)  # store winning split
 
        # Mark timestamps where all configs failed
        for t in range(n - max_total_duration + 1):
            if np.isnan(can_start[t]):
                can_start[t] = False
 
        # Expose as dict keyed by timestamp for O(1) lookup in plot_timeseries
        self.stage_durations = {
            self.data.index[t]: stage_durations_arr[t]
            for t in range(n)
            if stage_durations_arr[t] is not None
        }
 
        can_start_series = pd.Series(can_start, index=self.data.index, name="can_start")
        duration_series  = pd.Series(duration,  index=self.data.index, name="duration")
 
        return can_start_series, duration_series

    
    def _check_configuration(self, durations):
        """
        Vectorized check for a specific duration configuration.
        
        Parameters
        ----------
        durations : list[int]
            Duration for each stage in this configuration.
        
        Returns
        -------
        np.ndarray (bool)
            Boolean array of length T indicating which timestamps can start
            a window with this configuration.
        """
        n = len(self.data)
        total_dur = sum(durations)
        
        # Build criteria array for this configuration
        # Map: position in window -> {var: (lower, upper)}
        position_criteria = []
        
        for stage_idx, dur in enumerate(durations):
            stage = self.stages[stage_idx]
            
            for _ in range(dur):
                step_criteria = {}
                for var, (lower, upper) in stage["criteria"].items():
                    step_criteria[var] = (lower, upper)
                position_criteria.append(step_criteria)
        
        # If no criteria at all
        if all(len(pc) == 0 for pc in position_criteria):
            return np.ones(n, dtype=bool)
        
        # Get all variables
        all_vars = set()
        for pc in position_criteria:
            all_vars.update(pc.keys())
        
        can_start = np.ones(n, dtype=bool)
        
        for var in all_vars:
            values = self.data[var].values
            
            # Build validation matrix: result[t, k] = True if data[t+k] meets criteria[k]
            window_valid = np.ones((n, total_dur), dtype=bool)
            
            for k in range(total_dur):
                if var in position_criteria[k]:
                    lower, upper = position_criteria[k][var]
                    
                    # Check if values[t+k] meets the criteria
                    if k == 0:
                        # Check current values
                        if lower is not None:
                            window_valid[:, k] &= (values >= lower)
                        if upper is not None:
                            window_valid[:, k] &= (values <= upper)
                    else:
                        # Check values k steps ahead
                        if k < n:
                            if lower is not None:
                                window_valid[:-k, k] &= (values[k:] >= lower)
                            if upper is not None:
                                window_valid[:-k, k] &= (values[k:] <= upper)
                            # Last k positions can't look forward enough
                            window_valid[-k:, k] = False
                        else:
                            window_valid[:, k] = False
            
            # Window starting at t is valid if all positions satisfy criteria
            can_start_var = np.all(window_valid, axis=1)
            can_start &= can_start_var
        
        return can_start
    
    def _compute_waiting_time(self, can_start):
        """
        Compute waiting time until next feasible weather window.
        
        For each timestamp t:
        - If can_start[t] is True: waiting_time[t] = 0 (can start now)
        - If can_start[t] is False: waiting_time[t] = time until next True
        - If can_start[t] is NaN: waiting_time[t] = NaN (unknown)
        
        Returns
        -------
        pd.Series
            Waiting time in timesteps until next feasible window.
        """
        n = len(can_start)
        waiting_time = np.full(n, np.nan, dtype=float)
        
        # Get boolean array (NaN treated separately)
        can_start_values = can_start.values
        is_nan = pd.isna(can_start_values)
        can_start_bool = np.where(is_nan, False, can_start_values).astype(bool)  # ← ADD .astype(bool)
        
        # Find indices where we CAN start
        can_start_indices = np.where(can_start_bool)[0]
        
        if len(can_start_indices) == 0:
            # No feasible windows at all - leave as NaN
            return pd.Series(waiting_time, index=self.data.index, name="waiting_time")
        
        # Vectorized: for each position, find next can_start position
        # Use searchsorted to find insertion point (next window)
        all_positions = np.arange(n)
        next_window_idx = np.searchsorted(can_start_indices, all_positions, side='left')
        
        # Calculate waiting times
        has_future_window = next_window_idx < len(can_start_indices)
        waiting_time[has_future_window] = (
            can_start_indices[next_window_idx[has_future_window]] - all_positions[has_future_window]
        )
        
        # Set to 0 where we can start immediately
        waiting_time[can_start_bool] = 0  # ← Now works because can_start_bool is bool type
        
        # Preserve NaN where can_start was NaN
        waiting_time[is_nan] = np.nan
        
        return pd.Series(waiting_time, index=self.data.index, name="waiting_time")
    
    
    def statistics(
        self,
        by: typing.Literal["month", "season", "direction", "year"] | str = "month",
        metric: typing.Literal["waiting_time", "duration", "characteristic_duration"] = "waiting_time",
        func: list[str] = None,
        sectors: int = None,
        seasons: dict[str, list[int]] = None,
        step: float = None,
    ) -> pd.DataFrame:
        """
        Aggregated statistics for weather window metrics grouped by month, season, direction, or year.
        
        Parameters
        ----------
        by : {"month", "season", "direction", "year"} or str
            Grouping strategy. Can also be a column name in self.result.
            Defaults to "month".
        metric : {"waiting_time", "duration", "characteristic_duration"}
            Which metric to analyze:
            - 'waiting_time': Time until next feasible window (in timesteps)
            - 'duration': Operation duration if starting now (in timesteps)
            - 'characteristic_duration': waiting_time + duration (total timesteps)
        func : list[str], optional
            Aggregation functions to apply. Accepts: "min", "mean", "max", "std", 
            "count", "%", and percentiles as "p90", "P90", "90%" etc.
            Defaults to ["min", "mean", "max", "std", "count", "%"].
        sectors : int, optional
            Number of directional sectors (only used when by="direction").
            Defaults to 12.
        seasons : dict[str, list[int]], optional
            Custom season definitions mapping season names to month lists 
            (only used when by="season").
            Defaults to {"DJF": [12,1,2], "MAM": [3,4,5], "JJA": [6,7,8], "SON": [9,10,11]}.
        step : float, optional
            Bin width for variable grouping (only used when by is a column name).
            If not provided, is inferred automatically.
        
        Returns
        -------
        pd.DataFrame
            Statistics aggregated by groups (columns) and statistics/percentiles (rows).
            Includes an "All" column with statistics across all data.
        
        Raises
        ------
        RuntimeError
            If .fit() hasn't been called.
        ValueError
            If `by` is not recognized or metric is invalid.
        
        Examples
        --------
        >>> ww.fit(stages=[{"duration": 12, "criteria": {"hs": 4.0}}])
        >>>
        >>> # Statistics by month
        >>> stats_month = ww.statistics(by="month", metric="waiting_time")
        >>>
        >>> # Statistics by season with custom seasons
        >>> stats_season = ww.statistics(
        ...     by="season",
        ...     metric="characteristic_duration",
        ...     seasons={"Winter": [12, 1, 2], "Summer": [6, 7, 8]}
        ... )
        >>>
        >>> # Statistics by wind speed bins
        >>> stats_by_ws = ww.statistics(by="ws", metric="duration", step=2.0)
        >>>
        >>> # Statistics by direction sector
        >>> stats_dir = ww.statistics(by="direction", metric="waiting_time", sectors=8)
        """
        self._require_fit()
        
        # Validate metric
        if metric not in ["waiting_time", "duration", "characteristic_duration"]:
            raise ValueError(
                f"metric must be one of ['waiting_time', 'duration', 'characteristic_duration'], "
                f"got {metric}"
            )
        
        func = func or ["min", "mean", "max", "std", "count", "%"]
        
        # Get groups
        groups = self._get_groups(
            by=by,
            metric=metric,
            sectors=sectors,
            seasons=seasons,
            step=step,
        )
        groups["All"] = self.result[metric].dropna()
        
        # Aggregation
        n_total = len(self.result)

        include_ratio = "%" in func
        func = [f for f in func if f != "%"]
        def _agg_row(s: pd.Series) -> dict:
            result = aggregate_statistics(s, func=func)
            # Handle "%" function if present
            if include_ratio:
                result["%"] = 100 * len(s) / n_total
            return result
        
        df = pd.DataFrame({label: _agg_row(s) for label, s in groups.items()}).T
        df.index.name = by
        df.columns.name = metric
        return df
    
    def _get_groups(
        self,
        by: str,
        metric: str = None,
        sectors: int = None,
        seasons: dict = None,
        step: float = None,
    ) -> dict[str, pd.Series]:
        """
        Create groupings for statistics aggregation.
        
        Parameters
        ----------
        by : str
            Grouping dimension: "month", "season", "direction", "year", or column name.
        metric : str, optional
            Column name to group. If None, uses the entire result DataFrame.
        sectors : int, optional
            Number of directional sectors (for by="direction").
        seasons : dict[str, list[int]], optional
            Custom season definitions (for by="season").
        step : float, optional
            Bin width for variable grouping (for by=column_name).
        
        Returns
        -------
        dict[str, pd.Series]
            Mapping of group label to metric values for that group.
        
        Raises
        ------
        ValueError
            If `by` is not recognized.
        """
        if by == "direction":
            if self.direction is None:
                raise ValueError("Direction not set. Provide direction parameter to __init__.")
            
            data_clean = self.result[[self.direction, metric]].dropna() if metric else self.result[[self.direction]].dropna()
            
            return groupby_sector(
                data_clean,
                var_dir=self.direction,
                sectors=sectors or 12,
                var=metric
            )
        
        # For temporal groupings
        if metric:
            data_clean = self.result[[metric]].dropna()
        else:
            data_clean = self.result.dropna()
        
        if by == "month":
            return groupby_month(data_clean, var=metric)
        
        elif by == "season":
            return groupby_season(data_clean, seasons=seasons, var=metric)
        
        elif by == "year":
            return {
                str(y): g[metric] if metric else g
                for y, g in data_clean.groupby(data_clean.index.year, observed=False)
            }
        
        elif by in self.result.columns:
            # Group by a variable (e.g., wind speed bins)
            s_by = self.result[by].dropna()
            _step = step or infer_step(s_by)
            edges = bin_edges(s_by, _step)
            labels = [f"{e:.10g}" for e in edges[:-1]]
            bins = pd.cut(self.result[by], bins=edges, labels=labels, right=False)
            
            if metric:
                return {
                    k: g[metric].dropna() 
                    for k, g in data_clean.groupby(bins, observed=False)
                }
            else:
                return {
                    k: g.dropna()
                    for k, g in self.result.groupby(bins, observed=False)
                }
        
        else:
            raise ValueError(
                f"'by' must be one of 'month', 'season', 'direction', 'year', "
                f"or a column name in result. Got '{by}'. "
                f"Available columns: {list(self.result.columns)}"
            )

    def _require_direction(self, method_name: str) -> None:
        """Raise a clear error if direction was not provided at initialisation."""
        if self.direction is None:
            raise AttributeError(
                f"{method_name}() requires a direction variable, "
                "but none was provided at initialisation."
            )

    def _get_climatological_date_index(self, freq, n):
        """Return a DatetimeIndex of length n for climatological plotting, anchored to a dummy leap year (2000)."""
        year = 2000  # leap year so day 366 is valid
        if freq in ("h", "hour"):
            return pd.date_range(f"{year}-01-01", periods=n, freq="h")
        elif freq in ("D", "day"):
            return pd.date_range(f"{year}-01-01", periods=n, freq="D")
        elif freq in ("W", "week"):
            return pd.date_range(f"{year}-01-01", periods=n, freq="W")
        elif freq in ("M", "month"):
            return pd.date_range(f"{year}-01-01", periods=n, freq="MS")
        else:
            raise ValueError(f"Unknown freq: {freq}")

    def plot_operability(
        self,
        freqs: list[str] = None,
        ax=None,
        xlim: tuple = None,
    ):
        """
        Plot the frequency of weather window starts at multiple climatological scales.
        For each frequency, groups data by year and that period (hour-of-day, day-of-year, etc.),
        takes .any() within each (year, period) bin, then means across years — giving the
        fraction of years in which at least one window started in that period.

        Parameters
        ----------
        freqs : list[str], optional
            List of frequencies to plot. Options: "h"/"hour", "D"/"day", "W"/"week", "M"/"month".
            Defaults to ["M", "W", "D", "h"].
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        xlim : tuple[str,str], optional
            Limit the plot by providing month-day strings, for example ("06-01","07-31").

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> ww.fit(stages=[{"duration": 12, "criteria": {"hs": 4.0}}])
        >>> fig, ax = plt.subplots(figsize=(15, 6))
        >>> ww.plot_operability(freqs=["M", "W", "D", "h"], ax=ax)
        """
        self._require_fit()
        import matplotlib.pyplot as plt

        freqs = freqs or ["M", "W", "D", "h"]
        if ax is None:
            fig, ax = plt.subplots()

        if xlim is not None:
            x0 = pd.Timestamp(f"2000-{xlim[0]}")
            x1 = pd.Timestamp(f"2000-{xlim[1]}")
        else:
            x0, x1 = None, None

        # Get can_start as boolean (NaN -> False)
        can_start = self.result["can_start"].fillna(False).astype(bool)
        idx = can_start.index  # DatetimeIndex

        colors  = {"h": "steelblue", "hour": "steelblue",
                "D": "orange",    "day":  "orange",
                "W": "green",     "week": "green",
                "M": "red",       "month": "red"}
        markers = {"h": None,  "hour": None,
                "D": ".",   "day":  ".",
                "W": "o",   "week": "o",
                "M": "s",   "month": "s"}
        labels  = {"h": "Hour",  "hour": "Hour",
                "D": "Day",   "day":  "Day",
                "W": "Week",  "week": "Week",
                "M": "Month", "month": "Month"}

        # Maps each freq key to a (period_values, period_level_name) pair
        def _period_index(f):
            if f in ("h", "hour"):
                return idx.dayofyear * 24 + idx.hour, "hour"
            elif f in ("D", "day"):
                return idx.dayofyear, "dayofyear"
            elif f in ("W", "week"):
                return idx.isocalendar().week.values, "week"
            elif f in ("M", "month"):
                return idx.month, "month"
            else:
                raise ValueError(f"Unknown freq: {f!r}")

        for freq in freqs:
            period_vals, level_name = _period_index(freq)

            # Double groupby: any() within (year, period), then mean() across years
            grouped = (
                can_start
                .groupby([idx.year, period_vals]).any()
                .groupby(level=1).mean()
            )

            # Create a date index for plotting
            date_idx = self._get_climatological_date_index(freq, len(grouped))
            grouped.index = date_idx

            grouped.plot(
                ax=ax,
                linewidth=1,
                label=labels[freq],
                color=colors[freq],
                marker=markers[freq],
                legend=False,
            )

            # if freq in ("W", "week"):
            #     for date in date_idx:
            #         ax.axvline(date, color="gray", lw=0.5, linestyle="--", alpha=0.5)

        # for y in np.arange(0, 1.05, 0.05):
        #     ax.axhline(y, color="gray", lw=0.5, linestyle="--", alpha=0.5)

        ax.set_yticks(np.arange(0, 1.05, 0.05))
        ax.set_yticklabels([f"{100*v:.0f}%" for v in np.arange(0, 1.05, 0.05)])
        ax.set_ylim([0, 1.01])
        ax.set_ylabel("Fraction of years with ≥1 window start")
        ax.set_xlabel("")

        ax.legend(title="Frequency of weather\nwindow starts in each", loc="best")
        ax.grid(which="both",alpha=0.3)
        
        ax.set_xticks(ax.get_xticks())
        labels = [t.get_text().replace("2000", "").strip() for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        ax.set_xlim(x0, x1)

        plt.tight_layout()

        return ax
    
    def plot_statistics(
        self,
        by: typing.Literal["month", "season", "direction", "year"] | str = "month",
        metric: typing.Literal["waiting_time", "duration", "characteristic_duration"] = "characteristic_duration",
        func: list[str] = None,
        sectors: int = None,
        seasons: dict[str, list[int]] = None,
        step: float = None,
        unit: typing.Literal["hours", "days"] = "hours",
        ax=None,
        figsize: tuple = None,
        colormap: str = "tab10",
        show_table: bool = True,
    ):
        """
        Line plot of aggregated weather window statistics grouped by month,
        season, direction, year, or any column — mirroring the flexibility
        of :meth:`statistics`.

        Each line represents one statistic (e.g. mean, p90, max). The x-axis
        shows the categories defined by ``by`` (e.g. months Jan–Dec). An
        optional summary table is attached directly below the plot via
        ``ax.table()``, so it always matches the plot width exactly.

        Parameters
        ----------
        by : {"month", "season", "direction", "year"} or str
            Grouping strategy — passed directly to :meth:`statistics`.
        metric : {"waiting_time", "duration", "characteristic_duration"}
            Which metric to plot. Defaults to ``"characteristic_duration"``.
        func : list[str], optional
            Statistics to compute. Accepts: "min", "mean", "max", "std",
            "count", "%" and percentile strings like "p50", "p90", "90%" etc.
            "count" and "%" are included in the table but omitted from the
            line plot (no time unit). Defaults to ["mean", "max", "p50", "p90"].
        sectors : int, optional
            Number of directional sectors (only used when ``by="direction"``).
        seasons : dict[str, list[int]], optional
            Custom season definitions (only used when ``by="season"``).
        step : float, optional
            Bin width for variable grouping (only used when ``by`` is a column name).
        unit : {"hours", "days"}
            Display unit for the y-axis and table. Raw values are in timesteps.
            Defaults to ``"hours"``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, optional
            Figure size. Defaults are chosen automatically.
        colormap : str, optional
            Matplotlib colormap name for line colours. Defaults to ``"tab10"``.
        show_table : bool, optional
            Attach a summary table below the plot via ``ax.table()``.
            Defaults to ``True``.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ww.fit(stages=[{"duration": 12, "criteria": {"hs": 4.0}}])

        >>> # Default: waiting time by month
        >>> ww.plot_statistics()

        >>> # Characteristic duration by season in days
        >>> ww.plot_statistics(
        ...     by="season",
        ...     metric="characteristic_duration",
        ...     unit="days",
        ... )

        >>> # Custom statistics, by year
        >>> ww.plot_statistics(
        ...     by="year",
        ...     metric="duration",
        ...     func=["mean", "p50", "p90", "max"],
        ... )
        """
        self._require_fit()

        # ------------------------------------------------------------------
        # Defaults
        # ------------------------------------------------------------------
        func        = func        or ["p50", "mean", "P90", "P95", "P99"]

        # ------------------------------------------------------------------
        # Unit conversion
        # ------------------------------------------------------------------
        unit_factors = {
            "hours": self.timestep / pd.Timedelta(hours=1),
            "days":  self.timestep / pd.Timedelta(hours=24),
        }
        if unit not in unit_factors:
            raise ValueError(f"unit must be one of {list(unit_factors)}, got {unit!r}")
        scale = unit_factors[unit]

        metric_labels = {
            "waiting_time":            "Waiting time",
            "duration":                "Operation duration",
            "characteristic_duration": "Characteristic duration",
        }

        # ------------------------------------------------------------------
        # Statistics DataFrame
        #   shape: (n_stats × n_categories+1)
        #   index  = stat labels  (mean, max, p50, …, count, %)
        #   columns = categories  (Jan, Feb, … + All)
        # ------------------------------------------------------------------
        stats_df = self.statistics(
            by=by,
            metric=metric,
            func=func,
            sectors=sectors,
            seasons=seasons,
            step=step,
        ).T

        # Transpose so: index = categories (Jan, Feb, …), columns = stats
        # This is the natural shape for plotting: x = categories, lines = stats
        stats_df = stats_df.T

        # Split "All" off — shown in table but not plotted on x-axis
        cat_rows  = [r for r in stats_df.index if r != "All"]
        all_row   = stats_df.loc["All"] if "All" in stats_df.index else None
        plot_df   = stats_df.loc[cat_rows]

        # Columns that have a time unit → plotted as lines
        NON_TIME_COLS = {"count", "%", "std"}
        line_cols  = [c for c in plot_df.columns if str(c).lower() not in NON_TIME_COLS]

        # Scale time columns
        plot_df_scaled = plot_df.copy().astype(float)
        plot_df_scaled[line_cols] *= scale

        # ------------------------------------------------------------------
        # Figure / axes
        # ------------------------------------------------------------------
        n_cats = len(cat_rows)
        if ax is None:
            fw = max(8, n_cats * 0.75)
            fh = (9 if show_table else 5)
            fig, ax = plt.subplots(figsize=figsize or (fw, fh))

        # ------------------------------------------------------------------
        # Line plot  — iterate over stat columns, categories on x-axis
        # ------------------------------------------------------------------
        cmap    = plt.get_cmap(colormap, max(len(line_cols), 1))
        x       = np.arange(n_cats)
        xticks  = [str(r) for r in cat_rows]

        plot_df_scaled.plot(ax=ax,marker="o",cmap="viridis")

        ax.set_xticks(x)
        ax.set_xticklabels(xticks)
        ax.set_ylabel(f"{metric_labels[metric]} [{unit}]")
        ax.set_xlim(-0.5, n_cats - 0.5)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{metric_labels[metric]}  ·  grouped by {by}", fontsize=11, pad=10)
        ax.legend(title="Statistic", fontsize=8, title_fontsize=9)
        ax.grid(True, alpha=0.3, axis="both")

        # ------------------------------------------------------------------
        # Table via ax.table() — automatically matches plot width
        #   columns = categories (+ All)
        #   rows    = all statistics including count / %
        # ------------------------------------------------------------------
        if show_table:
            # display_df: index = stats, columns = categories (+ All)
            # i.e. the original stats_df orientation (before the .T above)
            display_df = plot_df_scaled.T.copy()  # stats as index, categories as columns
            # if all_row is not None:
            #     display_df["All"] = all_row.astype(float)
            #     # scale All column time rows too
            #     display_df.loc[line_cols, "All"] *= scale

            def _fmt(val, row_label):
                if pd.isna(val):
                    return "—"
                rl = str(row_label).lower()
                if rl == "count":
                    return f"{int(val)}"
                if rl == "%":
                    return f"{val:.1f}%"
                return f"{val:.1f}"

            cell_text  = [
                [_fmt(display_df.loc[stat, col], stat) for col in display_df.columns]
                for stat in display_df.index
            ]
            col_labels = [str(c) for c in display_df.columns]
            row_labels = [str(r) for r in display_df.index]

            tbl = ax.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=col_labels,
                loc="bottom",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.3)

            # Style: shade header row and row-label column
            for (r, c), cell in tbl.get_celld().items():
                if r == 0 or c == -1:
                    cell.set_facecolor("#e8eaf0")
                    cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#cccccc")

            # Push x-axis up to make room for the table
            n_table_rows = len(display_df) + 1  # +1 for header
            ax.set_xlabel("")
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            bottom_margin = 0.045 * n_table_rows
            plt.subplots_adjust(bottom=bottom_margin)

        plt.tight_layout()
        return ax

    def __repr__(self) -> str:
        """
        Concise summary of the WeatherWindow object.

        Shows setup info always; adds stage definitions and key results
        once :meth:`fit` has been called.

        Examples
        --------
        >>> ww = WeatherWindow(data, variables, timestep=1)
        >>> ww                          # before fit — shows setup only
        WeatherWindow | 8760 records  2015-01-01 → 2015-12-31  Δt=1h

        >>> ww.fit(stages=[...])
        >>> ww                          # after fit — full summary
        WeatherWindow | 8760 records  2015-01-01 → 2015-12-31  Δt=1h
        Stages
          #0  duration: 12h       Hs ≤ 2.0,  Ws ≤ 12.0
          #1  duration: 12h       Hs ≤ 4.0
        Results
          Feasible windows  : 1 423  (16.2% of evaluated timestamps)
          Median wait       : 18h
          Mean wait         : 34h
          P90 wait          : 96h
          Duration          : 24h (fixed)
          Data coverage     : 97.3%
        """
        ts_hours = self.timestep / pd.Timedelta(hours=1)

        # ---- header (always shown) ----
        n_records = len(self.data)
        if isinstance(self.data.index, pd.DatetimeIndex):
            date_str = f"{self.data.index[0]:%Y-%m-%d} → {self.data.index[-1]:%Y-%m-%d}"
        else:
            date_str = "unknown dates"

        lines = [
            f"WeatherWindow | {n_records:,} records  {date_str}  Δt={ts_hours:g}h"
        ]

        # ---- not yet fitted ----
        if self.stages is None:
            lines.append("  (not fitted — call .fit(stages=...) to analyse windows)")
            return "\n".join(lines)

        # ---- stages ----
        lines.append("Stages")
        for i, stage in enumerate(self.stages):
            lo, hi = stage["duration"]
            dur_str = (
                f"{int(lo * ts_hours)}h"
                if lo == hi
                else f"{int(lo * ts_hours)}–{int(hi * ts_hours)}h"
            )
            criteria = stage["criteria"]
            if criteria:
                parts = []
                for var, (lower, upper) in criteria.items():
                    if lower is None and upper is not None:
                        parts.append(f"{var} ≤ {upper:g}")
                    elif lower is not None and upper is None:
                        parts.append(f"{var} ≥ {lower:g}")
                    elif lower is not None and upper is not None:
                        parts.append(f"{lower:g} ≤ {var} ≤ {upper:g}")
                    else:
                        parts.append(f"{var} (unbounded)")
                crit_str = ",  ".join(parts)
            else:
                crit_str = "no criteria"
            lines.append(f"  #{i}  duration: {dur_str:<10}  {crit_str}")

        # ---- results ----
        can_start    = self.result["can_start"]
        known_mask   = can_start.notna()
        n_known      = known_mask.sum()
        n_windows    = int(can_start.sum())
        avail_pct    = 100.0 * n_windows / n_known if n_known > 0 else float("nan")
        coverage_pct = 100.0 * n_known / n_records if n_records > 0 else float("nan")

        wt = self.result["waiting_time"].dropna() * ts_hours
        du = self.result["duration"].dropna()     * ts_hours

        def _fmt(v): return f"{v:.0f}h" if not np.isnan(v) else "n/a"

        wait_median = float(wt.median())     if len(wt) else float("nan")
        wait_mean   = float(wt.mean())       if len(wt) else float("nan")
        wait_p90    = float(wt.quantile(.9)) if len(wt) else float("nan")
        dur_min     = float(du.min())        if len(du) else float("nan")
        dur_max     = float(du.max())        if len(du) else float("nan")
        dur_median  = float(du.median())     if len(du) else float("nan")

        dur_summary = (
            f"{_fmt(dur_median)} (fixed)"
            if dur_min == dur_max
            else f"{_fmt(dur_min)}–{_fmt(dur_max)}  (median {_fmt(dur_median)})"
        )

        lines += [
            "Results",
            f"  Feasible windows  : {n_windows:,}  ({avail_pct:.1f}% of evaluated timestamps)",
            f"  Median / mean wait: {_fmt(wait_median)} / {_fmt(wait_mean)}",
            f"  P90 wait          : {_fmt(wait_p90)}",
            f"  Duration          : {dur_summary}",
            f"  Data coverage     : {coverage_pct:.1f}%",
        ]

        return "\n".join(lines)
    
    def availability_table(
        self,
        row: typing.Literal["month", "season", "direction", "year"] | str = "direction",
        col: typing.Literal["month", "season", "direction", "year"] | str = "month",
        step_row: float = None,
        step_col: float = None,
        metric: typing.Literal["availability", "waiting_time", "duration", "characteristic_duration"] = "availability",
        func: str = None,
        margins: bool = True,
        sectors: int = None,
        seasons_row: dict[str, list[int]] = None,
        seasons_col: dict[str, list[int]] = None,
    ) -> pd.DataFrame:
        """
        2D table of a weather-window metric grouped by two axes.

        Each axis can be a grouping keyword ("month", "season", "direction",
        "year") or any column name in ``self.result`` (binned by value).

        Parameters
        ----------
        row : str
            Row axis — grouping keyword or column name. Defaults to
            ``"direction"``.
        col : str
            Column axis — grouping keyword or column name. Defaults to
            ``"month"``.
        step_row : float, optional
            Bin width for row axis when ``row`` is a column name.
        step_col : float, optional
            Bin width for col axis when ``col`` is a column name.
        metric : {"availability", "waiting_time", "duration", "characteristic_duration"}
            Which quantity to tabulate.

            - ``"availability"`` (default): percentage of timestamps in each
              bin where a window can start. ``func`` is ignored; the result is
              always a ratio.
            - ``"waiting_time"``, ``"duration"``, ``"characteristic_duration"``:
              aggregated value of that column (in timesteps) per bin, using
              ``func``.
        func : str, optional
            Aggregation function applied to continuous metrics. Accepts
            ``"min"``, ``"mean"``, ``"max"``, ``"std"``, and percentile
            strings such as ``"p50"``, ``"p90"``, ``"90%"``.
            Ignored when ``metric="availability"``.
            Defaults to ``"mean"``.
        margins : bool
            Append an ``"All"`` row and column with marginal values.
            Defaults to ``True``.
        sectors : int, optional
            Number of directional sectors. Used when ``row`` or ``col`` is
            ``"direction"``. Defaults to 12.
        seasons_row : dict[str, list[int]], optional
            Custom season definitions for the row axis.
        seasons_col : dict[str, list[int]], optional
            Custom season definitions for the col axis.

        Returns
        -------
        pd.DataFrame
            Row groups as index, column groups as columns.
            - ``metric="availability"``: values are availability % rounded to
              1 decimal, or integer counts when ``func="count"``.
            - other metrics: values are aggregated in timesteps, rounded to
              1 decimal.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``row == col``, ``metric`` is invalid, or a direction axis is
            requested without a direction variable set.

        Examples
        --------
        >>> ww.fit(stages=[{"duration": 12, "criteria": {"hs": 4.0}}])

        >>> # Default: availability by direction x month
        >>> ww.availability_table()

        >>> # Mean waiting time by month x year
        >>> ww.availability_table(
        ...     row="month", col="year",
        ...     metric="waiting_time", func="mean",
        ... )

        >>> # P90 characteristic duration by season x direction
        >>> ww.availability_table(
        ...     row="season", col="direction",
        ...     metric="characteristic_duration", func="p90",
        ... )

        >>> # Raw window counts instead of availability
        >>> ww.availability_table(metric="availability", func="count")
        """
        self._require_fit()

        # ------------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------------
        valid_metrics = ("availability", "waiting_time", "duration", "characteristic_duration")
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got {metric!r}")

        if row == col:
            raise ValueError("'row' and 'col' must be different.")

        if "direction" in (row, col) and self.direction is None:
            raise ValueError(
                "availability_table() with by='direction' requires a direction "
                "variable to be set at initialisation."
            )

        # Default aggregation function
        if func is None:
            func = "mean"

        # ------------------------------------------------------------------
        # Parse func into a callable that operates on a pd.Series
        # ------------------------------------------------------------------
        def _parse_func(f):
            """Return a (label, callable) pair for an aggregation function string."""
            fl = f.strip().lower()
            if fl == "min":
                return lambda s: s.min()
            if fl in ("mean", "average", "avg"):
                return lambda s: s.mean()
            if fl == "max":
                return lambda s: s.max()
            if fl == "std":
                return lambda s: s.std()
            if fl == "count":
                return lambda s: int(s.count())
            # Percentile: p90, P90, 90%, 0.9 etc.
            for pattern, factor in (
                (r"^p(\d+(?:\.\d+)?)$",    1.0),
                (r"^(\d+(?:\.\d+)?)%$",    1.0),
            ):
                import re
                m = re.match(pattern, fl)
                if m:
                    q = float(m.group(1)) / 100.0 * factor
                    return lambda s, q=q: s.quantile(q)
            raise ValueError(
                f"Unrecognised func {f!r}. Use 'min', 'mean', 'max', 'std', "
                f"'count', or a percentile string like 'p90' or '90%'."
            )

        agg_fn = _parse_func(func)

        # ------------------------------------------------------------------
        # Get groups for each axis
        # ------------------------------------------------------------------
        def _groups(axis, step, seasons):
            return self._get_groups(
                by=axis,
                metric="can_start" if metric == "availability" else metric,
                sectors=sectors,
                seasons=seasons,
                step=step,
            )

        row_groups = _groups(row, step_row, seasons_row)
        col_groups = _groups(col, step_col, seasons_col)

        # ------------------------------------------------------------------
        # Core series to aggregate
        # ------------------------------------------------------------------
        if metric == "availability":
            # Boolean (NaN excluded): True = window can start
            core = self.result["can_start"].dropna().astype(bool)
        else:
            core = self.result[metric].dropna()

        # ------------------------------------------------------------------
        # Helper: compute one cell value
        # ------------------------------------------------------------------
        def _cell(row_idx, col_idx):
            shared_idx = col_idx.intersection(row_idx)
            s = core.reindex(shared_idx).dropna()

            if metric == "availability":
                if func == "count":
                    return int(s.sum())
                n = len(s)
                if n == 0:
                    return float("nan")
                return round(100.0 * s.sum() / n, 1)
            else:
                if len(s) == 0:
                    return float("nan")
                return round(float(agg_fn(s)), 1)

        # ------------------------------------------------------------------
        # Build matrix
        # ------------------------------------------------------------------
        records = {}
        for col_key, col_series in col_groups.items():
            c_idx = col_series.dropna().index
            records[col_key] = {
                row_key: _cell(row_series.dropna().index, c_idx)
                for row_key, row_series in row_groups.items()
            }

        df = pd.DataFrame(records)

        # ------------------------------------------------------------------
        # Margins
        # ------------------------------------------------------------------
        if margins:
            all_idx = core.index

            # "All" column: aggregate over the full row group
            for row_key, row_series in row_groups.items():
                r_idx = row_series.dropna().index.intersection(all_idx)
                df.loc[row_key, "All"] = _cell(r_idx, all_idx)

            # "All" row: aggregate over the full col group
            for col_key, col_series in col_groups.items():
                c_idx = col_series.dropna().index.intersection(all_idx)
                df.loc["All", col_key] = _cell(all_idx, c_idx)

            # Bottom-right corner: aggregate over everything
            df.loc["All", "All"] = _cell(all_idx, all_idx)

            if metric == "availability" and func == "count":
                df = df.astype(int)

        df.index.name   = row
        df.columns.name = col

        return df
    
    def plot_temporal_raster(
        self,
        years: list[int] = None,
        ax=None,
        figsize=None,
        color_open: str = "#2ecc71",
        color_closed: str = "#e74c3c",
        color_unknown: str = "#bdc3c7",
        show_pct: bool = True,
        bar_height: float = 0.9,
        xlim: tuple[str, str] = None,
    ):
        """
        Temporal raster plot with years on the y-axis and date-within-year
        on the x-axis.

        Green bars mark timestamps where a window can start, red where it
        cannot, and grey where the result is unknown (typically the tail of
        the record where insufficient data remain to evaluate the full
        operation duration).

        Parameters
        ----------
        years : list[int], optional
            Years to include, one row each. Defaults to all years in the
            record.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when ``None``.
        figsize : tuple, optional
            Figure size. Defaults to ``(14, n_years * 0.5 + 2)``.
        color_open : str, optional
            Colour for feasible timestamps. Defaults to green.
        color_closed : str, optional
            Colour for infeasible timestamps. Defaults to red.
        color_unknown : str, optional
            Colour for unevaluated timestamps. Defaults to light grey.
        show_pct : bool, optional
            Annotate each year row with its availability percentage.
            Defaults to ``True``.
        bar_height : float, optional
            The fraction of space filled by each bar. Default 0.9.
        xlim : tuple[str, str], optional
            Restrict the x-axis to a date range given as ``("MM-DD", "MM-DD")``,
            e.g. ``("06-01", "07-31")`` for June–July only.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ww.fit(stages=[{"duration": 12, "criteria": {"hs": 2.0}}])

        >>> # All years
        >>> ww.plot_temporal_raster()

        >>> # Subset of years
        >>> ww.plot_temporal_raster(years=[2019, 2020, 2021])

        >>> # Summer window only
        >>> ww.plot_temporal_raster(xlim=("05-01", "09-30"))
        """
        self._require_fit()
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        DUMMY_YEAR = 2000  # leap year — day 366 always valid

        # ------------------------------------------------------------------
        # Resolve years
        # ------------------------------------------------------------------
        available_years = sorted(self.result.index.year.unique())
        if years is None:
            years = available_years
        else:
            missing = [y for y in years if y not in available_years]
            if missing:
                raise ValueError(
                    f"Years {missing} not found in data. "
                    f"Available: {available_years}"
                )

        n_years = len(years)

        # ------------------------------------------------------------------
        # xlim
        # ------------------------------------------------------------------
        if xlim is not None:
            x0 = pd.Timestamp(f"{DUMMY_YEAR}-{xlim[0]}")
            x1 = pd.Timestamp(f"{DUMMY_YEAR}-{xlim[1]}")
        else:
            x0 = pd.Timestamp(f"{DUMMY_YEAR}-01-01")
            x1 = pd.Timestamp(f"{DUMMY_YEAR}-12-31")

        # ------------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ts_hours = self.timestep / pd.Timedelta(hours=1)

        color_map = {
            "open":    color_open,
            "closed":  color_closed,
            "unknown": color_unknown,
        }

        # ------------------------------------------------------------------
        # Draw one row per year (top to bottom)
        # ------------------------------------------------------------------
        for row_idx, year in enumerate(years):
            y = n_years - row_idx  # top = highest y value

            df_year = self.result[self.result.index.year == year]

            # Map can_start → state label
            states = df_year["can_start"].map(
                lambda v: "open" if v == True else ("unknown" if pd.isna(v) else "closed")
            )

            # Collapse consecutive equal states into runs
            runs = []
            if len(states):
                cur_state = states.iloc[0]
                cur_start = df_year.index[0]
                for ts, state in zip(df_year.index[1:], states.iloc[1:]):
                    if state != cur_state:
                        runs.append((cur_start, ts, cur_state))
                        cur_state = state
                        cur_start = ts
                runs.append((
                    cur_start,
                    df_year.index[-1] + self.timestep,
                    cur_state,
                ))

            for t_start, t_end, state in runs:
                # Re-anchor to dummy year for a common x-axis
                x_start = pd.Timestamp(f"{DUMMY_YEAR}-{t_start.month:02d}-{t_start.day:02d}") \
                          + pd.Timedelta(hours=t_start.hour)
                x_end   = pd.Timestamp(f"{DUMMY_YEAR}-{t_end.month:02d}-{t_end.day:02d}") \
                          + pd.Timedelta(hours=t_end.hour)

                # Guard against Dec 31 → Jan 1 wrap-around spilling into next year
                if x_end < x_start:
                    x_end = pd.Timestamp(f"{DUMMY_YEAR}-12-31 23:59")

                duration_days = (x_end - x_start).total_seconds() / 86400

                ax.barh(
                    y=y,
                    width=duration_days,
                    left=mdates.date2num(x_start),
                    height=bar_height,
                    color=color_map[state],
                    linewidth=0,
                )

            # Per-year availability annotation
            if show_pct:
                n_known = states.isin(["open", "closed"]).sum()
                n_open  = (states == "open").sum()
                pct = 100.0 * n_open / n_known if n_known > 0 else float("nan")
                if not np.isnan(pct):
                    ax.text(
                        mdates.date2num(x1) + 3, y,
                        f"{pct:.0f}%",
                        va="center", ha="left", fontsize=8, color="black",
                    )

        # ------------------------------------------------------------------
        # Axes formatting
        # ------------------------------------------------------------------
        ax.set_yticks(range(1, n_years + 1))
        ax.set_yticklabels(reversed(years), fontsize=9)
        ax.set_ylim(0.5, n_years + 0.5)

        ax.xaxis_date()
        ax.set_xlim(mdates.date2num(x0), mdates.date2num(x1) + (2 if show_pct else 0))

        ax.grid(axis="x", color="white", linewidth=0.8, zorder=2)
        ax.tick_params(axis="x", labelsize=8)

        ax.set_xticks(ax.get_xticks())
        labels = [t.get_text().replace("2000-", "").strip() for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        # ------------------------------------------------------------------
        # Legend + title
        # ------------------------------------------------------------------
        legend_patches = [
            mpatches.Patch(color=color_open,    label="Window available"),
            mpatches.Patch(color=color_closed,  label="Window unavailable"),
            mpatches.Patch(color=color_unknown, label="Unknown"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

        # Overall availability across all plotted years
        data_plotted = self.result[self.result.index.year.isin(years)]["can_start"]
        n_known_total = data_plotted.notna().sum()
        overall_pct = 100.0 * data_plotted.sum() / n_known_total if n_known_total > 0 else float("nan")

        ax.set_title(
            "Weather window availability"
            + (f"  ·  overall {overall_pct:.1f}%" if not np.isnan(overall_pct) else ""),
            fontsize=11, pad=10,
        )

        plt.tight_layout()
        return ax
    
    def plot_timeseries(
        self,
        xlim: tuple[str, str] = None,
        unit: typing.Literal["hours", "days"] = "hours",
        start_times: list = None,
        figsize: tuple = None,
        criteria_color: str = "steelblue",
        threshold_color: str = "tomato",
    ):
        """
        Diagnostic time series plot for a user-defined date window.

        Produces stacked subplots:

        1. One subplot per criteria variable (e.g. Hs, Ws), with threshold
           lines drawn for each stage that constrains that variable.
        2. ``can_start`` -- step plot showing feasible start times.
        3. ``waiting_time``, ``duration``, ``characteristic_duration``.

        If ``start_times`` is provided, each submitted timestamp is overlaid
        on every criteria subplot as a stepped threshold line -- drawn at the
        active limit during constrained stages, and broken (NaN gap) during
        free/pause stages. Static threshold lines are suppressed when any
        valid start_times are provided.

        If ``xlim`` is not set but ``start_times`` are provided, the x-axis
        is automatically set to one day before the earliest window start and
        one day after the end of the latest window.

        Parameters
        ----------
        xlim : tuple[str, str], optional
            Date range as ``("YYYY-MM-DD HH", "YYYY-MM-DD HH")``.
            Defaults to the first month of the record, or -- when start_times
            are provided -- one day either side of the window span.
        unit : {"hours", "days"}
            Display unit for time-based metrics. Defaults to ``"hours"``.
        start_times : list[str or pd.Timestamp], optional
            One or more window start timestamps to overlay. For each, a
            stepped threshold line is drawn per criteria variable showing
            the active limit across all stages. Static threshold lines are
            suppressed when any valid start_times are provided.
            Non-feasible timestamps trigger a warning and are skipped.
        figsize : tuple, optional
            Figure size. Defaults to ``(14, 2.5 * n_subplots)``.
        criteria_color : str, optional
            Line colour for criteria variable time series. Defaults to
            ``"steelblue"``.
        threshold_color : str, optional
            Colour for static threshold lines (shown only when no valid
            start_times are provided). Defaults to ``"tomato"``.

        Returns
        -------
        list[matplotlib.axes.Axes]

        Examples
        --------
        >>> ww.fit(stages=[
        ...     {"duration": 6,      "criteria": {"hs": 3.0}},
        ...     {"duration": (0, 12)},
        ...     {"duration": 6,      "criteria": {"hs": 3.0}},
        ... ])

        >>> # Default: first month, no overlays
        >>> ww.plot_timeseries()

        >>> # Overlay two specific windows -- xlim set automatically
        >>> ww.plot_timeseries(
        ...     start_times=["1960-01-19 23:00", "1960-01-20 06:00"],
        ... )
        """
        self._require_fit()
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.cm as cm
        from matplotlib.lines import Line2D

        # ------------------------------------------------------------------
        # Unit conversion
        # ------------------------------------------------------------------
        unit_factors = {
            "hours": self.timestep / pd.Timedelta(hours=1),
            "days":  self.timestep / pd.Timedelta(hours=24),
        }
        if unit not in unit_factors:
            raise ValueError(f"unit must be one of {list(unit_factors)}, got {unit!r}")
        scale = unit_factors[unit]

        # ------------------------------------------------------------------
        # Parse and validate start_times
        # Must happen before xlim resolution so the window-aware fallback
        # can use the resolved timestamps.
        # ------------------------------------------------------------------
        parsed_starts = []
        if start_times is not None:
            for t in start_times:
                ts = pd.Timestamp(t)
                if ts not in self.stage_durations:
                    warnings.warn(
                        f"start_time {ts} is not a feasible window start "
                        f"(can_start is False or NaN). Stage overlay skipped."
                    )
                else:
                    parsed_starts.append(ts)

        # ------------------------------------------------------------------
        # xlim
        # ------------------------------------------------------------------
        if xlim is not None:
            x0 = pd.Timestamp(xlim[0])
            x1 = pd.Timestamp(xlim[1])
        elif parsed_starts:
            # Span from one day before the earliest start to one day after
            # the end of the latest window (start + total stage duration).
            first_start = min(parsed_starts)
            last_end    = max(
                t + sum(self.stage_durations[t]) * self.timestep
                for t in parsed_starts
            )
            x0 = first_start - pd.Timedelta(days=1)
            x1 = last_end    + pd.Timedelta(days=1)
        else:
            x0 = self.result.index[0]
            x1 = x0 + pd.DateOffset(months=1)

        mask = (self.result.index >= x0) & (self.result.index <= x1)
        df   = self.result.loc[mask]

        if len(df) == 0:
            raise ValueError(
                f"No data found between {x0.date()} and {x1.date()}. "
                f"Record spans {self.result.index[0].date()} to "
                f"{self.result.index[-1].date()}."
            )

        n_windows     = len(parsed_starts)
        # tab10.colors is a plain tuple of 10 (R,G,B) values -- safe to index directly.
        # Index 0 is blue (same family as criteria_color), so we skip it and cycle
        # through the remaining 9 colours: orange, green, red, purple, brown, ...
        _tab10_non_blue = [c for i, c in enumerate(plt.cm.tab10.colors) if i != 0]
        window_colors = [_tab10_non_blue[i % len(_tab10_non_blue)] for i in range(n_windows)]

        show_static_thresholds = (n_windows == 0)

        # ------------------------------------------------------------------
        # Criteria variables and their stage thresholds
        # ------------------------------------------------------------------
        criteria_info: dict[str, list] = {}
        for i, stage in enumerate(self.stages):
            for var, (lower, upper) in stage["criteria"].items():
                criteria_info.setdefault(var, []).append((i, lower, upper))

        criteria_vars = list(criteria_info.keys())

        # ------------------------------------------------------------------
        # Helper: build a stepped threshold line for one window x one variable.
        #
        # Returns ((times_u, vals_u), (times_l, vals_l)) where each list
        # contains pd.Timestamp / float / NaN values for ax.plot().
        # Gaps in free/pause stages are represented by NaN so the line breaks,
        # producing the "3 3 3 - - - 3 3 3" pattern.
        # Each constrained stage contributes exactly two points (start, end)
        # so the line is perfectly flat within the stage.
        # ------------------------------------------------------------------
        def _threshold_line(t_start, per_stage_durs, var):
            times_u, vals_u = [], []
            times_l, vals_l = [], []
            cursor = t_start

            for stage_idx, stage_dur in enumerate(per_stage_durs):
                t_end = cursor + stage_dur * self.timestep
                stage = self.stages[stage_idx]

                if var in stage["criteria"] and stage_dur > 0:
                    lower, upper = stage["criteria"][var]
                    if upper is not None:
                        times_u += [cursor, t_end]
                        vals_u  += [upper,  upper]
                    if lower is not None:
                        times_l += [cursor, t_end]
                        vals_l  += [lower,  lower]
                else:
                    # Free/pause stage: insert a NaN break so the line gaps
                    if times_u:
                        times_u.append(cursor)
                        vals_u.append(np.nan)
                    if times_l:
                        times_l.append(cursor)
                        vals_l.append(np.nan)

                cursor = t_end

            return (times_u, vals_u), (times_l, vals_l)

        # ------------------------------------------------------------------
        # Helper: build the legend label for one window, listing each stage
        # duration in human-readable form.
        #   e.g.  "Window 1  (2024-01-03 06:00)  |  12h - 6h - 12h"
        # ------------------------------------------------------------------
        def _window_label(win_idx, t_start, per_stage_durs):
            ts_hours = self.timestep / pd.Timedelta(hours=1)
            stage_parts = [f"{stage_dur * ts_hours:g}h" for stage_dur in per_stage_durs]
            dur_str = " - ".join(stage_parts)
            return f"Window {win_idx + 1}  ({t_start})  |  {dur_str}"

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        metric_cols = ["can_start", "waiting_time", "duration", "characteristic_duration"]
        metric_labels = {
            "can_start":               "Can start",
            "waiting_time":            f"Waiting time [{unit}]",
            "duration":                f"Duration [{unit}]",
            "characteristic_duration": f"Char. duration [{unit}]",
        }
        n_subplots = len(criteria_vars) + len(metric_cols)

        fig, axes = plt.subplots(
            n_subplots, 1,
            figsize=figsize or (14, 2.5 * n_subplots),
            sharex=True,
        )
        if n_subplots == 1:
            axes = [axes]

        # ------------------------------------------------------------------
        # 1. Criteria variable subplots
        # ------------------------------------------------------------------
        for ax_idx, var in enumerate(criteria_vars):
            ax = axes[ax_idx]
            ax.plot(df.index, df[var], color=criteria_color, linewidth=1, zorder=2)

            # Static threshold lines (only when no windows are submitted)
            if show_static_thresholds:
                threshold_handles = []
                seen_labels = set()
                for stage_idx, lower, upper in criteria_info[var]:
                    if upper is not None:
                        lbl = f"<= {upper:g}  (stage {stage_idx})"
                        if lbl not in seen_labels:
                            l = ax.axhline(upper, color=threshold_color, linewidth=1.2,
                                           linestyle="--", zorder=3, label=lbl)
                            threshold_handles.append(l)
                            seen_labels.add(lbl)
                    if lower is not None:
                        lbl = f">= {lower:g}  (stage {stage_idx})"
                        if lbl not in seen_labels:
                            l = ax.axhline(lower, color=threshold_color, linewidth=1.2,
                                           linestyle=":", zorder=3, label=lbl)
                            threshold_handles.append(l)
                            seen_labels.add(lbl)
                if threshold_handles:
                    ax.legend(handles=threshold_handles, fontsize=7, loc="upper right",
                              framealpha=0.8)

            # Stepped threshold lines for each submitted window
            for win_idx, (t_start, color) in enumerate(zip(parsed_starts, window_colors)):
                per_stage_durs = self.stage_durations[t_start]
                (times_u, vals_u), (times_l, vals_l) = _threshold_line(
                    t_start, per_stage_durs, var
                )
                if times_u:
                    ax.plot(times_u, vals_u, color=color, linewidth=1.8,
                            linestyle="--", zorder=3)
                if times_l:
                    ax.plot(times_l, vals_l, color=color, linewidth=1.8,
                            linestyle=":", zorder=3)

            ax.set_ylabel(var, fontsize=9)
            ax.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # 2. Metric subplots
        # ------------------------------------------------------------------
        for ax_idx, col in enumerate(metric_cols):
            ax = axes[len(criteria_vars) + ax_idx]
            series = df[col].copy()

            if col == "can_start":
                s_num = series.map(
                    lambda v: 1.0 if v == True else (np.nan if pd.isna(v) else 0.0)
                )
                ax.fill_between(df.index, s_num, step="pre", alpha=0.5,
                                color=criteria_color, zorder=2)
                ax.step(df.index, s_num, color=criteria_color, linewidth=0.8,
                        where="pre", zorder=3)
                ax.set_ylim(-0.05, 1.15)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["No", "Yes"], fontsize=8)
            else:
                scaled = series * scale
                ax.plot(df.index, scaled, color=criteria_color, linewidth=1, zorder=2)
                ax.fill_between(df.index, scaled, alpha=0.15, color=criteria_color)
                ax.set_ylim(bottom=0)

            # Vertical start-time markers
            for t, color in zip(parsed_starts, window_colors):
                ax.axvline(t, color=color, linewidth=1.5, linestyle="--", zorder=4)

            ax.set_ylabel(metric_labels[col], fontsize=9)
            ax.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # Figure-level title and legend
        #
        # When windows are provided:
        #   - title goes top-left (via fig.text so the legend can occupy top-right)
        #   - a single legend is placed above the figure at the top-right,
        #     outside all axes, so it never overlaps any subplot
        # When no windows:
        #   - title is centred as normal via fig.suptitle
        # ------------------------------------------------------------------
        title_str = f"Weather window diagnostics  {x0.date()} to {x1.date()}"

        if n_windows > 0:
            # Title: top-left in figure coordinates, just above the axes
            fig.text(
                0.01, 1.0, title_str,
                fontsize=11, va="bottom", ha="left",
                transform=fig.transFigure,
            )

            # Legend: one entry per window, placed top-right outside the axes.
            # We attach it to the figure rather than any single axes so it is
            # truly shared across all subplots and never clips the data.
            handles = [
                Line2D(
                    [0], [0],
                    color=window_colors[i],
                    linewidth=2,
                    linestyle="--",
                    label=_window_label(i, t, self.stage_durations[t]),
                )
                for i, t in enumerate(parsed_starts)
            ]
            fig.legend(
                handles=handles,
                fontsize=7,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=fig.transFigure,
                title="Submitted windows  (start  |  stage durations)",
                title_fontsize=7,
                framealpha=0.9,
            )
        else:
            fig.suptitle(title_str, fontsize=11)

        plt.tight_layout()
        return axes