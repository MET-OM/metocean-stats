import functools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from metocean_stats.stats.aux_funcs import readNora10File

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@functools.lru_cache(maxsize=None)
def _spectra_base():
    from tests.data.synthetic_dataset import synthetic_dataset_spectra
    return synthetic_dataset_spectra().isel(time=slice(0, 96), height=0, drop=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def nora():
    return readNora10File(os.path.join(DATA_DIR, "NORA_test.txt"))


@pytest.fixture
def spectra_ds():
    return _spectra_base().copy()
