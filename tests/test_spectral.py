import pytest

from metocean_stats.spectral import SpectralTimeSeries, SpectralTimeSeriesCollection


@pytest.fixture
def sts(spectra_ds):
    return SpectralTimeSeries(spectra_ds)


@pytest.fixture
def coll(sts):
    return sts.partition_ptm1("wind_speed", "wind_direction")


def test_init(spectra_ds):
    SpectralTimeSeries(spectra_ds)


def test_properties(sts):
    sts.has_2d
    sts.has_frequency
    sts.has_directional
    sts.spectrum("S")
    sts.time
    sts.freqs
    sts.dirs
    sts.duration
    sts.timestep
    sts.n_years


def test_compute_frequency_spectrum(sts):
    sts.compute_frequency_spectrum()


def test_compute_directional_spectrum(sts):
    sts.compute_directional_spectrum()


def test_integrate(sts):
    sts.integrate()


def test_aggregate(sts):
    sts.aggregate(by="month")


def test_groupby(sts):
    sts.groupby(by="month").aggregate()


def test_partition_ptm1(sts):
    sts.partition_ptm1("wind_speed", "wind_direction")


def test_partition_ptm2(sts):
    sts.partition_ptm2("wind_speed", "wind_direction")


def test_partition_ptm3(sts):
    sts.partition_ptm3()


def test_partition_ptm4(sts):
    sts.partition_ptm4("wind_speed", "wind_direction")


def test_partition_ptm5(sts):
    sts.partition_ptm5(fcut=0.1)


def test_partition_hp01(sts):
    sts.partition_hp01("wind_speed", "wind_direction")


def test_denoise_gaussian(sts):
    sts.denoise_gaussian()


def test_plot(sts):
    sts.plot(time=sts.time[0])


def test_plot_overview(sts):
    sts.plot_overview(wspd_var="wind_speed", wdir_var="wind_direction")


def test_collection_init(coll):
    SpectralTimeSeriesCollection(dict(coll.items()))


def test_collection_properties(coll):
    coll.labels()
    coll.items()
    coll.aligned


def test_collection_align(coll):
    coll.align(target=coll.labels()[0])


def test_collection_track(sts):
    sts.partition_ptm1("wind_speed", "wind_direction").track(wspd_var="wind_speed")


def test_collection_integrate(coll):
    coll.integrate()


def test_collection_plot_timeseries(coll):
    coll.integrate()
    coll.plot_timeseries("hs")


def test_collection_plot(coll, sts):
    coll.plot(time=sts.time[0])
