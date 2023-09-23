#!/usr/bin/env python

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'version.py')).read())

setuptools.setup(
    name        = 'metocean-engine',
    description = 'metocean-engine - Tool for generation of metocean statistics based on NORA3 data',
    author      = 'Konstantinos Christakos MET Norway & NTNU',
    url         = 'https://github.com/MET-OM/metocean-engine',
    download_url = 'https://github.com/MET-OM/metocean-engine',
    version = __version__,
    license = 'GPLv2',
    install_requires = [
        'numpy',
        'python',
        'matplotlib',
        'pandas',
        'xarray',
        'pip',
        'nco',
        'scipy',
        'virocon',
        'xclim',
        'netcdf4',
        'time',
        'pip'
    ],
    packages = setuptools.find_packages(),
    include_package_data = True,
    setup_requires = ['setuptools_scm'],
    tests_require = ['pytest'],
    scripts = []
)
