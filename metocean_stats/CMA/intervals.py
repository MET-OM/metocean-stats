from virocon.intervals import IntervalSlicer
import numpy as np


class WidthSlicer(IntervalSlicer):
    """
    Custom width interval slicer based on virocon's WidthOfIntervalSlicer,
    but with the added option to specify the start value of slicing.
    """

    def __init__(self, 
                 width, 
                 reference="center", 
                 right_open=True, 
                 value_range=None, 
                 drop_first_n=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.width = width
        self.reference = reference
        self.right_open = right_open
        self.value_range = value_range
        self.drop_first_n = drop_first_n

    def _slice(self, data):
        if self.value_range is None:
            data_min = 0
            data_max = np.max(data)
        else:
            if self.value_range[0] is not None:
                data_min = self.value_range[0]
            else:
                data_min = 0
            if self.value_range[1] is not None:
                data_max = self.value_range[1]
            else:
                data_max = np.max(data)

        width = self.width
        interval_references = np.arange(data_min, data_max + width, width) + 0.5 * width

        if self.right_open:
            interval_slices = [
                ((int_cent - 0.5 * width <= data) & (data < int_cent + 0.5 * width))
                for int_cent in interval_references
            ]
        else:
            interval_slices = [
                ((int_cent - 0.5 * width < data) & (data <= int_cent + 0.5 * width))
                for int_cent in interval_references
            ]

        interval_boundaries = [
            (c - width / 2, c + width / 2) for c in interval_references
        ]

        if isinstance(self.reference, str):
            if self.reference.lower() == "center":
                pass  # interval_references are already center of intervals
            elif self.reference.lower() == "right":
                interval_references += 0.5 * width
            elif self.reference.lower() == "left":
                interval_references -= 0.5 * width
            else:
                raise ValueError(
                    "Unknown value for 'reference'. "
                    "Supported values are 'center', 'left', "
                    f"and 'right', but got '{self.reference}'."
                )
        elif callable(self.reference):
            pass  #  handled in super class
        else:
            raise TypeError(
                "Wrong type for reference. Expected str or callable, "
                f"but got {type(self.reference)}."
            )

        interval_slices,interval_references,interval_boundaries = self._drop_too_small_intervals(
            interval_slices, interval_references, interval_boundaries)

        interval_slices = interval_slices[self.drop_first_n:]
        interval_references = interval_references[self.drop_first_n:]
        interval_boundaries = interval_boundaries[self.drop_first_n:]

        return interval_slices, interval_references, interval_boundaries


class NumberSlicer(IntervalSlicer):
    """
    """

    def __init__(
        self,
        n_intervals,
        reference="center",
        include_max=True,
        value_range=None,
        drop_first_n = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_intervals < self.min_n_intervals:
            self.min_n_intervals = n_intervals
        self.n_intervals = n_intervals
        self.reference = reference
        self.include_max = include_max
        self.value_range = value_range
        self.drop_first_n = drop_first_n

    def _slice(self, data):
        if self.value_range is not None:
            value_range = self.value_range
        else:
            value_range = (min(data), max(data))

        interval_starts, interval_width = np.linspace(
            value_range[0],
            value_range[1],
            num=self.n_intervals,
            endpoint=False,
            retstep=True,
        )
        interval_references = interval_starts + 0.5 * interval_width

        interval_boundaries = [
            (c - interval_width / 2, c + interval_width / 2)
            for c in interval_references
        ]

        if isinstance(self.reference, str):
            if self.reference.lower() == "center":
                pass  # default
            elif self.reference.lower() == "right":
                interval_references = interval_starts + interval_width
            elif self.reference.lower() == "left":
                interval_references = interval_starts
            else:
                raise ValueError(
                    "Unknown value for 'reference'. "
                    "Supported values are 'center', 'left', "
                    f"and 'right', but got '{self.reference}'."
                )
        elif callable(self.reference):
            pass  #  handled in super class
        else:
            raise TypeError(
                "Wrong type for reference. Expected str or callable, "
                f"but got {type(self.reference)}."
            )

        interval_slices = [
            ((data >= int_start) & (data < int_start + interval_width))
            for int_start in interval_starts[:-1]
        ]

        # include max in last interval ?
        int_start = interval_starts[-1]
        if self.include_max:
            interval_slices.append(
                ((data >= int_start) & (data <= int_start + interval_width))
            )
        else:
            interval_slices.append(
                ((data >= int_start) & (data < int_start + interval_width))
            )

        (
            interval_slices,
            interval_references,
            interval_boundaries,
        ) = self._drop_too_small_intervals(
            interval_slices, interval_references, interval_boundaries
        )

        interval_slices = interval_slices[self.drop_first_n:]
        interval_references = interval_references[self.drop_first_n:]
        interval_boundaries = interval_boundaries[self.drop_first_n:]

        return interval_slices, interval_references, interval_boundaries

